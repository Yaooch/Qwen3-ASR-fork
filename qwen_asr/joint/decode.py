from typing import Dict, List, Optional, Sequence, Union

import torch
from qwen_asr.inference.utils import parse_asr_output

from .defaults import (
    DEFAULT_PROMPT,
    ENCODER_BATCH_SIZE,
    RNNT_MAX_SYMBOLS,
    STREAM_CHUNK_SEC,
    STREAM_ENCODER_BATCH,
    STREAM_FIRST_PAD_SEC,
    STREAM_LEFT_SEC,
    STREAM_RIGHT_SEC,
    STREAM_WINDOW_BATCH,
    hotword_prompt,
)
from .tokens import ids_to_text


HOTWORD_SOURCE_ORDER = ("ctc", "rnnt")


class DecodeMixin:
    """CTC/RNNT/LLM 推理入口。"""

    def _modes(self, modes) -> tuple:
        return self._clean_names(modes, {"llm", "ctc", "rnnt"}, "mode")

    def _one_or_many(self, audio, results: List):
        return results[0] if isinstance(audio, str) else results

    def _waveforms(self, audios):
        import librosa
        import numpy as np

        wavs = []
        for item in audios:
            if isinstance(item, str):
                wavs.append(librosa.load(item, sr=16000, mono=True)[0].astype(np.float32, copy=False))
            elif isinstance(item, np.ndarray):
                wavs.append(item.astype(np.float32, copy=False))
            else:
                raise TypeError(f"不支持的音频项类型：{type(item)}")
        return wavs

    def _broadcast(self, value, size: int, default=None) -> List:
        if value is None:
            return [default] * size
        if isinstance(value, str):
            return [value] * size
        return list(value)

    def _stream_kwargs(self) -> Dict:
        return {
            "chunk_sec": STREAM_CHUNK_SEC,
            "left_context_sec": STREAM_LEFT_SEC,
            "right_context_sec": STREAM_RIGHT_SEC,
            "first_chunk_left_pad_sec": STREAM_FIRST_PAD_SEC,
            "window_batch_size": STREAM_WINDOW_BATCH,
            "window_encoder_batch_size": STREAM_ENCODER_BATCH,
        }

    def decode(self, log_probs: torch.Tensor, output_lengths: torch.Tensor) -> List[str]:
        if self.ctc is None:
            raise RuntimeError("decode(log_probs) 需要 CTC 头。")

        pred_ids = log_probs.argmax(dim=2)
        results = []
        for b in range(pred_ids.shape[0]):
            ids = pred_ids[b, : int(output_lengths[b].item())].cpu().tolist()
            kept, prev = [], -1
            for idx in ids:
                if idx != self.ctc.blank_id and idx != prev:
                    kept.append(idx)
                prev = idx
            results.append(ids_to_text(kept, self._id_to_token))
        return results

    def _decode_head(
        self,
        name: str,
        hs_pad: torch.Tensor,
        out_lens: torch.Tensor,
        max_symbols_per_step: int = RNNT_MAX_SYMBOLS,
    ) -> List[str]:
        head = self._head(name)
        hs_pad = hs_pad.to(next(head.parameters()).dtype)
        if name == "ctc":
            ids_batch = head.greedy_decode(hs_pad, out_lens)
        else:
            ids_batch = head.greedy_decode(
                hs_pad,
                out_lens,
                max_symbols_per_step=max_symbols_per_step,
            )
        return [ids_to_text(ids, self._id_to_token) for ids in ids_batch]

    @torch.no_grad()
    def decode_feats(
        self,
        input_features: torch.Tensor,
        feature_attention_mask: Optional[torch.Tensor] = None,
        head: str = "ctc",
        max_symbols_per_step: int = RNNT_MAX_SYMBOLS,
        encoder_batch_size: Optional[int] = None,
    ) -> List[str]:
        ref = next(self.qwen_model.parameters())
        input_features = input_features.to(device=ref.device, dtype=ref.dtype)
        if feature_attention_mask is not None:
            feature_attention_mask = feature_attention_mask.to(device=ref.device)

        hs_pad, _, out_lens, _ = self._enc_joint(
            input_features,
            feature_attention_mask,
            need_llm_features=False,
            encoder_batch_size=encoder_batch_size,
        )
        return self._decode_head(head, hs_pad, out_lens, max_symbols_per_step=max_symbols_per_step)

    def _asr_fields(self, obj):
        if obj is None:
            return {"text": "", "language": None}
        if isinstance(obj, str):
            return {"text": obj, "language": None}
        if isinstance(obj, dict):
            return {
                "text": obj.get("text") or obj.get("prediction") or obj.get("transcription") or "",
                "language": obj.get("language"),
            }
        text = getattr(obj, "text", None)
        language = getattr(obj, "language", None)
        if text is not None:
            return {"text": str(text), "language": language}
        return {"text": str(obj), "language": None}

    def _norm_outputs(self, outputs, batch_size: int):
        if isinstance(outputs, list):
            if len(outputs) == batch_size:
                return outputs
            if batch_size == 1:
                return [outputs]
            raise ValueError(f"底座输出数量不匹配：expect {batch_size}, got {len(outputs)}")
        return [outputs]

    def _audio_prompt(self, num_audio_tokens: int, context: str, language: Optional[str]) -> str:
        if self._asr_wrapper is None:
            raise RuntimeError("模型未正确初始化，请使用 from_pretrained 加载。")
        prompt = self._asr_wrapper._build_text_prompt(context=context or "", force_language=language)
        audio_token = self.processor.audio_token
        if audio_token not in prompt:
            raise RuntimeError(f"Prompt does not contain audio token {audio_token!r}: {prompt!r}")
        return prompt.replace(audio_token, audio_token * num_audio_tokens, 1)

    @torch.no_grad()
    def _gen_llm_feats(
        self,
        audio_features_list: List[torch.Tensor],
        contexts: List[Optional[str]],
        languages: List[Optional[str]],
        max_new_tokens: Optional[int] = None,
    ) -> List[Dict[str, Optional[str]]]:
        thinker = self.qwen_model.thinker
        processor = self.processor
        device = next(self.qwen_model.parameters()).device
        dtype = next(thinker.parameters()).dtype
        prompts = [
            self._audio_prompt(int(feats.shape[0]), context or "", language)
            for feats, context, language in zip(audio_features_list, contexts, languages)
        ]

        old_padding_side = processor.tokenizer.padding_side
        processor.tokenizer.padding_side = "left"
        try:
            tok = processor.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        finally:
            processor.tokenizer.padding_side = old_padding_side

        input_ids = tok["input_ids"].to(device)
        attention_mask = tok["attention_mask"].to(device)
        inputs_embeds = thinker.get_input_embeddings()(input_ids)
        audio_mask = thinker.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds)
        audio_features = torch.cat([x.to(device=device, dtype=dtype) for x in audio_features_list], dim=0)
        placeholder_count = int(audio_mask[..., 0].sum().item())
        if placeholder_count != int(audio_features.shape[0]):
            raise RuntimeError(f"audio placeholder mismatch: prompt={placeholder_count}, features={audio_features.shape[0]}")

        inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)
        max_new_tokens = max_new_tokens or getattr(self._asr_wrapper, "max_new_tokens", 512)
        generated = self.qwen_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
        )
        sequences = generated.sequences if hasattr(generated, "sequences") else generated
        raws = processor.batch_decode(
            sequences[:, input_ids.shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        results = []
        for raw, language in zip(raws, languages):
            parsed_language, text = parse_asr_output(raw, user_language=language)
            results.append({"text": text, "language": parsed_language or language})
        return results

    def _split_llm_feats(self, llm_features: torch.Tensor, out_lens: torch.Tensor) -> List[torch.Tensor]:
        out, offset = [], 0
        for length in out_lens.tolist():
            cur_len = int(length)
            out.append(llm_features[offset: offset + cur_len])
            offset += cur_len
        return out

    def _encode_batch(self, wavs: List, need_llm: bool):
        ref = next(self.qwen_model.parameters())
        batch = self._feature_batch(wavs)
        mask = batch.get("feature_attention_mask", None)
        input_features = batch["input_features"].to(device=ref.device, dtype=ref.dtype)
        if mask is not None:
            mask = mask.to(device=ref.device)
        return self._enc_joint(
            input_features,
            mask,
            need_llm_features=need_llm,
            encoder_batch_size=ENCODER_BATCH_SIZE,
        )

    def _decode_stream_one(self, wav, modes: Sequence[str], need_llm: bool, stream_kwargs: Dict, max_symbols: int):
        if need_llm:
            chunks, llm_features = self._stream_joint_feats(wav, **stream_kwargs)
        else:
            chunks = self._stream_aux_chunks(wav, **stream_kwargs)
            llm_features = None

        texts = {}
        if "ctc" in modes:
            texts["ctc_text"] = ids_to_text(self._ctc_stream_decode(chunks), self._id_to_token)
        if "rnnt" in modes:
            texts["rnnt_text"] = ids_to_text(
                self._rnnt_stream_decode(chunks, max_symbols_per_step=max_symbols),
                self._id_to_token,
            )
        return texts, llm_features

    @torch.no_grad()
    def transcribe(
        self,
        audio,
        modes: Union[str, Sequence[str]] = "llm",
        language: Optional[Union[str, List[str]]] = None,
        prompt: Optional[str] = None,
        hotword_retriever=None,
        hotword_topk: int = 10,
        stream: bool = False,
        max_symbols_per_step: int = RNNT_MAX_SYMBOLS,
        **kwargs,
    ):
        modes = self._modes(modes)
        for name in ("ctc", "rnnt"):
            if name in modes:
                self._head(name)

        audios = self._audio_list(audio)
        wavs = self._waveforms(audios)
        languages = self._broadcast(language, len(audios), default=None)
        base_prompt = prompt or DEFAULT_PROMPT
        records = [{"text": "", "language": lang, "hotwords": []} for lang in languages]
        need_llm = "llm" in modes

        if stream:
            llm_features_list = []
            feature_len_cache: Dict[int, int] = {}
            stream_kwargs = self._stream_kwargs()
            stream_kwargs["feature_len_cache"] = feature_len_cache
            for i, wav in enumerate(wavs):
                texts, llm_features = self._decode_stream_one(wav, modes, need_llm, stream_kwargs, max_symbols_per_step)
                records[i].update(texts)
                if need_llm:
                    llm_features_list.append(llm_features)
        else:
            hs_pad, llm_features, out_lens, _ = self._encode_batch(wavs, need_llm)
            if "ctc" in modes:
                for record, text in zip(records, self._decode_head("ctc", hs_pad, out_lens)):
                    record["ctc_text"] = text
            if "rnnt" in modes:
                rnnt_texts = self._decode_head("rnnt", hs_pad, out_lens, max_symbols_per_step=max_symbols_per_step)
                for record, text in zip(records, rnnt_texts):
                    record["rnnt_text"] = text
            llm_features_list = self._split_llm_feats(llm_features, out_lens) if need_llm else []

        if need_llm:
            llm_outputs = self._gen_llm_feats(
                llm_features_list,
                contexts=[base_prompt] * len(records),
                languages=languages,
                max_new_tokens=kwargs.get("max_new_tokens", None),
            )
            for record, out in zip(records, llm_outputs):
                record["llm_text"] = out["text"]
                record["language"] = out["language"] or record["language"]

        if need_llm and hotword_retriever is not None:
            sources, hotword_contexts = [], []
            for record in records:
                source_name = next((name for name in HOTWORD_SOURCE_ORDER if record.get(f"{name}_text")), "")
                query = record.get(f"{source_name}_text", "") if source_name else ""
                words = hotword_retriever.retrieve(query, topk=hotword_topk) if query else []
                record["hotwords"] = words
                record["hotword_source"] = source_name
                hotword_contexts.append(hotword_prompt(words, base_prompt))
                sources.append(source_name)
            hot_outputs = self._gen_llm_feats(
                llm_features_list,
                contexts=hotword_contexts,
                languages=languages,
                max_new_tokens=kwargs.get("max_new_tokens", None),
            )
            for record, out in zip(records, hot_outputs):
                record["hotword_llm_text"] = out["text"]
                record["language"] = out["language"] or record["language"]

        for record in records:
            record["text"] = (
                record.get("hotword_llm_text")
                or record.get("llm_text")
                or record.get("ctc_text")
                or record.get("rnnt_text")
                or ""
            )
        return self._one_or_many(audio, records)
