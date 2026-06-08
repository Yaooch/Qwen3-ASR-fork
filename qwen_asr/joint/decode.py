from typing import Dict, List, Optional, Sequence, Union

import torch
from qwen_asr.inference.utils import parse_asr_output

from .defaults import (
    DEFAULT_PROMPT,
    ENCODER_BATCH_SIZE,
    RNNT_MAX_SYMBOLS,
    STREAM_CHUNK_SEC,
    hotword_prompt,
)


HOTWORD_SOURCE_ORDER = ("ctc", "rnnt")
ENCODER_MODES = {"offline", "stream", "train_mask"}


def ids_to_text(ids: List[int], id_to_token: Dict[int, str]) -> str:
    tokens = [id_to_token.get(i, "") for i in ids]
    return "".join(tokens).replace("▁", " ").strip().lower()


class DecodeMixin:
    """CTC/RNNT/LLM 推理入口。"""

    def _modes(self, modes) -> tuple:
        return self._clean_names(modes, {"llm", "ctc", "rnnt"}, "mode")

    def _encoder_mode(self, encoder_mode: Optional[str], stream: bool) -> str:
        mode = (encoder_mode or ("stream" if stream else "offline")).strip().lower()
        if mode not in ENCODER_MODES:
            raise ValueError(f"不支持的 encoder_mode: {mode}")
        if stream and mode != "stream":
            raise ValueError("stream=True 与 encoder_mode 冲突。")
        return mode

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

    def _decode_head(
        self,
        name: str,
        hs_pad: torch.Tensor,
        out_lens: torch.Tensor,
        max_symbols_per_step: int = RNNT_MAX_SYMBOLS,
        ctc_mask_mode: str = "offline",
    ) -> List[str]:
        head = self._head(name)
        hs_pad = hs_pad.to(next(head.parameters()).dtype)
        if name == "ctc":
            ids_batch = head.greedy_decode(
                hs_pad,
                out_lens,
                **self._ctc_mask_kwargs(ctc_mask_mode),
            )
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

        ctc_mask_mode = "offline"
        if getattr(self, "stream_train", False):
            hs_pad, out_lens, _ = self._stream_train_mask(input_features, feature_attention_mask)
            ctc_mask_mode = "chunk"
        else:
            hs_pad, _, out_lens, _ = self._enc_joint(
                input_features,
                feature_attention_mask,
                need_llm_features=False,
                encoder_batch_size=encoder_batch_size,
            )
        return self._decode_head(
            head,
            hs_pad,
            out_lens,
            max_symbols_per_step=max_symbols_per_step,
            ctc_mask_mode=ctc_mask_mode,
        )

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

    def _feature_batch(self, wavs: List):
        feature_extractor = self.processor.feature_extractor
        sr = int(getattr(feature_extractor, "sampling_rate", 16000) or 16000)
        batch = feature_extractor(
            wavs,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
            truncation=False,
            return_attention_mask=True,
        )
        if "feature_attention_mask" not in batch and "attention_mask" in batch:
            batch["feature_attention_mask"] = batch["attention_mask"]
        return batch

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

    def _encode_train_mask_batch(self, wavs: List, need_llm: bool):
        ref = next(self.qwen_model.parameters())
        batch = self._feature_batch(wavs)
        mask = batch.get("feature_attention_mask", None)
        input_features = batch["input_features"].to(device=ref.device, dtype=ref.dtype)
        if mask is not None:
            mask = mask.to(device=ref.device)
        hs_pad, out_lens, feat_lens = self._stream_train_mask(input_features, mask)
        llm_features = self._project_llm_features(hs_pad, out_lens) if need_llm else None
        return hs_pad, llm_features, out_lens, feat_lens

    def _stream_hs_batch(self, chunks_list: List[List[torch.Tensor]]):
        seqs = []
        lengths = []
        for idx, chunks in enumerate(chunks_list):
            if not chunks:
                raise RuntimeError(f"No streaming auxiliary features were produced for item {idx}.")
            seq = torch.cat(chunks, dim=0)
            seqs.append(seq)
            lengths.append(int(seq.shape[0]))
        hs_pad = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)
        out_lens = torch.tensor(lengths, dtype=torch.long, device=hs_pad.device)
        return hs_pad, out_lens

    def _decode_ctc_stream_batch(self, chunks_list: List[List[torch.Tensor]]) -> List[str]:
        hs_pad, out_lens = self._stream_hs_batch(chunks_list)
        return self._decode_head("ctc", hs_pad, out_lens, ctc_mask_mode="causal")

    def _decode_stream_batch(
        self,
        chunks_list: List[List[torch.Tensor]],
        modes: Sequence[str],
        max_symbols: int,
    ) -> List[Dict[str, str]]:
        records = [{} for _ in chunks_list]
        if "ctc" not in modes and "rnnt" not in modes:
            return records
        hs_pad, out_lens = self._stream_hs_batch(chunks_list)
        if "ctc" in modes:
            for record, text in zip(records, self._decode_head("ctc", hs_pad, out_lens, ctc_mask_mode="causal")):
                record["ctc_text"] = text
        if "rnnt" in modes:
            texts = self._decode_head("rnnt", hs_pad, out_lens, max_symbols_per_step=max_symbols)
            for record, text in zip(records, texts):
                record["rnnt_text"] = text
        return records

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
        encoder_mode: Optional[str] = None,
        max_symbols_per_step: int = RNNT_MAX_SYMBOLS,
        **kwargs,
    ):
        modes = self._modes(modes)
        encoder_mode = self._encoder_mode(encoder_mode, stream)
        for name in ("ctc", "rnnt"):
            if name in modes:
                self._head(name)

        audios = self._audio_list(audio)
        wavs = self._waveforms(audios)
        languages = self._broadcast(language, len(audios), default=None)
        base_prompt = prompt or DEFAULT_PROMPT
        records = [{"text": "", "language": lang, "hotwords": []} for lang in languages]
        need_llm = "llm" in modes

        if encoder_mode == "train_mask" and not ({"ctc", "rnnt"} & set(modes)):
            raise RuntimeError("train_mask 需要同时启用 CTC 或 RNNT，以复用流式训练 Encoder 路径。")

        if encoder_mode == "stream":
            if need_llm and "ctc" not in modes:
                raise RuntimeError("流式 LLM 需要同时启用 CTC，以复用 CTC 流式 Encoder 输出。")
            chunks_list, llm_features_list = self._encode_stream_waveforms(wavs, need_llm)
            if "ctc" in modes:
                for record, text in zip(records, self._decode_ctc_stream_batch(chunks_list)):
                    record["ctc_text"] = text
            rest_modes = tuple(name for name in modes if name != "ctc")
            if "rnnt" in rest_modes:
                for record, texts in zip(records, self._decode_stream_batch(chunks_list, rest_modes, max_symbols_per_step)):
                    record.update(texts)
        else:
            encode = self._encode_train_mask_batch if encoder_mode == "train_mask" else self._encode_batch
            ctc_mask_mode = "chunk" if encoder_mode == "train_mask" else "offline"
            hs_pad, llm_features, out_lens, _ = encode(wavs, need_llm)
            if "ctc" in modes:
                for record, text in zip(records, self._decode_head("ctc", hs_pad, out_lens, ctc_mask_mode=ctc_mask_mode)):
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
