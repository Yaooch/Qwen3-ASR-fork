# qwen_asr/joint/decode.py
from typing import Dict, List, Optional, Union

import torch
from qwen_asr.inference.utils import parse_asr_output

from .tokens import ids_to_text


class DecodeMixin:
    """CTC/RNNT/LLM/joint 推理入口。"""

    def decode(self, log_probs: torch.Tensor, output_lengths: torch.Tensor) -> List[str]:
        """CTC-only decode from precomputed log probabilities."""
        if self.aux_loss_type != "ctc":
            raise RuntimeError("decode(log_probs) 只适用于 CTC 模式。")

        preds = log_probs.argmax(dim=2)
        results = []

        for b in range(preds.shape[0]):
            cur_len = int(output_lengths[b].item())
            ids = preds[b, :cur_len].cpu().tolist()

            dedup_ids = []
            prev_id = -1
            for idx in ids:
                if idx != self.ctc.blank_id and idx != prev_id:
                    dedup_ids.append(idx)
                prev_id = idx

            results.append(ids_to_text(dedup_ids, self._id_to_token))

        return results

    @torch.no_grad()
    def decode_feats(
        self,
        input_features: torch.Tensor,
        feature_attention_mask: Optional[torch.Tensor] = None,
        max_symbols_per_step: int = 5,
        aux_encoder_batch_size: Optional[int] = None,
    ) -> List[str]:
        """用已整理好的音频特征做 CTC/RNNT 解码。"""
        ref = next(self.qwen_model.parameters())
        input_features = input_features.to(device=ref.device, dtype=ref.dtype)
        if feature_attention_mask is not None:
            feature_attention_mask = feature_attention_mask.to(device=ref.device)

        hs_pad, _, out_lens, _ = self._enc_joint(
            input_features,
            feature_attention_mask,
            need_llm_features=False,
            encoder_batch_size=aux_encoder_batch_size,
        )
        hs_pad = hs_pad.to(next(self.aux_head.parameters()).dtype)

        if self.aux_loss_type == "ctc":
            pred_ids_batch = self.ctc.greedy_decode(hs_pad, out_lens)
        elif self.aux_loss_type == "rnnt":
            pred_ids_batch = self.rnnt.greedy_decode(
                hs_pad,
                out_lens,
                max_symbols_per_step=max_symbols_per_step,
            )
        else:
            raise ValueError(f"不支持的 aux_loss_type: {self.aux_loss_type}")

        return [ids_to_text(ids, self._id_to_token) for ids in pred_ids_batch]

    def _aux_decode(
        self,
        audio,
        max_symbols_per_step: int = 5,
        aux_encoder_batch_size: int = 1,
    ) -> List[str]:
        """辅助头推理：CTC/RNNT 共用入口。"""
        import librosa
        import numpy as np

        audios = self._audio_list(audio)
        waveform_list = []

        for item in audios:
            if isinstance(item, str):
                waveform_list.append(librosa.load(item, sr=16000, mono=True)[0])
            elif isinstance(item, np.ndarray):
                waveform_list.append(item)
            else:
                raise TypeError(f"不支持的音频项类型：{type(item)}")

        if aux_encoder_batch_size <= 0:
            aux_encoder_batch_size = len(waveform_list)

        results = []
        ref = next(self.qwen_model.parameters())
        for start in range(0, len(waveform_list), aux_encoder_batch_size):
            sub_wavs = waveform_list[start: start + aux_encoder_batch_size]
            batch = self._feature_batch(sub_wavs)
            input_features = batch["input_features"].to(device=ref.device, dtype=ref.dtype)
            feature_attention_mask = batch.get("feature_attention_mask", None)
            if feature_attention_mask is not None:
                feature_attention_mask = feature_attention_mask.to(device=ref.device)

            results.extend(
                self.decode_feats(
                    input_features,
                    feature_attention_mask,
                    max_symbols_per_step=max_symbols_per_step,
                    aux_encoder_batch_size=aux_encoder_batch_size,
                )
            )
        return results

    @torch.no_grad()
    def _rnnt_stream(
        self,
        audio,
        max_symbols_per_step: int = 5,
        chunk_sec: float = 0.64,
        left_context_sec: float = 0.64,
        right_context_sec: float = 0.07,
        first_chunk_left_pad_sec: float = 0.0,
        window_batch_size: int = 4,
        window_encoder_batch_size: int = 1,
    ) -> List[str]:
        """RNNT chunk-wise streaming decode.

        This is a streaming-behavior inference path: each chunk only decodes
        newly arrived encoder frames, while the RNNT predictor state is carried
        across chunks. The encoder windows still include configured left/right
        context and are recomputed per window for correctness and simplicity.
        """
        if self.aux_loss_type != "rnnt":
            raise RuntimeError("当前 checkpoint 不是 RNNT 模式。")

        import librosa
        import numpy as np

        audios = self._audio_list(audio)
        waveform_list = []
        for item in audios:
            if isinstance(item, str):
                waveform_list.append(librosa.load(item, sr=16000, mono=True)[0].astype(np.float32, copy=False))
            elif isinstance(item, np.ndarray):
                waveform_list.append(item.astype(np.float32, copy=False))
            else:
                raise TypeError(f"不支持的音频项类型：{type(item)}")

        results = []
        feature_len_cache: Dict[int, int] = {}
        for wav in waveform_list:
            chunks = self._stream_aux_chunks(
                wav,
                chunk_sec=chunk_sec,
                left_context_sec=left_context_sec,
                right_context_sec=right_context_sec,
                first_chunk_left_pad_sec=first_chunk_left_pad_sec,
                window_batch_size=window_batch_size,
                window_encoder_batch_size=window_encoder_batch_size,
                feature_len_cache=feature_len_cache,
            )
            ids = self._rnnt_stream_decode(
                chunks,
                max_symbols_per_step=max_symbols_per_step,
            )
            results.append(ids_to_text(ids, self._id_to_token))
        return results

    @torch.no_grad()
    def _ctc_stream(
        self,
        audio,
        chunk_sec: float = 0.64,
        left_context_sec: float = 0.64,
        right_context_sec: float = 0.07,
        first_chunk_left_pad_sec: float = 0.0,
        window_batch_size: int = 4,
        window_encoder_batch_size: int = 1,
    ) -> List[str]:
        if self.aux_loss_type != "ctc":
            raise RuntimeError("当前 checkpoint 不是 CTC 模式。")

        import librosa
        import numpy as np

        audios = self._audio_list(audio)
        waveform_list = []
        for item in audios:
            if isinstance(item, str):
                waveform_list.append(librosa.load(item, sr=16000, mono=True)[0].astype(np.float32, copy=False))
            elif isinstance(item, np.ndarray):
                waveform_list.append(item.astype(np.float32, copy=False))
            else:
                raise TypeError(f"不支持的音频项类型：{type(item)}")

        results = []
        feature_len_cache: Dict[int, int] = {}
        for wav in waveform_list:
            chunks = self._stream_aux_chunks(
                wav,
                chunk_sec=chunk_sec,
                left_context_sec=left_context_sec,
                right_context_sec=right_context_sec,
                first_chunk_left_pad_sec=first_chunk_left_pad_sec,
                window_batch_size=window_batch_size,
                window_encoder_batch_size=window_encoder_batch_size,
                feature_len_cache=feature_len_cache,
            )
            ids = self._ctc_stream_decode(chunks)
            results.append(ids_to_text(ids, self._id_to_token))
        return results

    @torch.no_grad()
    def transcribe_ctc(self, audio, aux_encoder_batch_size: int = 1) -> Union[str, List[str]]:
        if self.aux_loss_type != "ctc":
            raise RuntimeError("当前 checkpoint 不是 CTC 模式。")
        results = self._aux_decode(audio, aux_encoder_batch_size=aux_encoder_batch_size)
        return results[0] if isinstance(audio, str) else results

    @torch.no_grad()
    def transcribe_ctc_streaming(
        self,
        audio,
        chunk_sec: float = 0.64,
        left_context_sec: float = 0.64,
        right_context_sec: float = 0.07,
        first_chunk_left_pad_sec: float = 0.0,
        window_batch_size: int = 4,
        window_encoder_batch_size: int = 1,
    ) -> Union[str, List[str]]:
        results = self._ctc_stream(
            audio,
            chunk_sec=chunk_sec,
            left_context_sec=left_context_sec,
            right_context_sec=right_context_sec,
            first_chunk_left_pad_sec=first_chunk_left_pad_sec,
            window_batch_size=window_batch_size,
            window_encoder_batch_size=window_encoder_batch_size,
        )
        return results[0] if isinstance(audio, str) else results

    @torch.no_grad()
    def transcribe_rnnt(
        self,
        audio,
        max_symbols_per_step: int = 5,
        aux_encoder_batch_size: int = 1,
    ) -> Union[str, List[str]]:
        if self.aux_loss_type != "rnnt":
            raise RuntimeError("当前 checkpoint 不是 RNNT 模式。")
        results = self._aux_decode(
            audio,
            max_symbols_per_step=max_symbols_per_step,
            aux_encoder_batch_size=aux_encoder_batch_size,
        )
        return results[0] if isinstance(audio, str) else results

    @torch.no_grad()
    def transcribe_rnnt_streaming(
        self,
        audio,
        max_symbols_per_step: int = 5,
        chunk_sec: float = 0.64,
        left_context_sec: float = 0.64,
        right_context_sec: float = 0.07,
        first_chunk_left_pad_sec: float = 0.0,
        window_batch_size: int = 4,
        window_encoder_batch_size: int = 1,
    ) -> Union[str, List[str]]:
        results = self._rnnt_stream(
            audio,
            max_symbols_per_step=max_symbols_per_step,
            chunk_sec=chunk_sec,
            left_context_sec=left_context_sec,
            right_context_sec=right_context_sec,
            first_chunk_left_pad_sec=first_chunk_left_pad_sec,
            window_batch_size=window_batch_size,
            window_encoder_batch_size=window_encoder_batch_size,
        )
        return results[0] if isinstance(audio, str) else results

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
        if isinstance(obj, (list, tuple)):
            if len(obj) == 0:
                return {"text": "", "language": None}
            if len(obj) == 1:
                return self._asr_fields(obj[0])
            items = [self._asr_fields(x) for x in obj]
            text = " ".join([x["text"] for x in items if x["text"]]).strip()
            language = next((x["language"] for x in items if x["language"]), None)
            return {"text": text, "language": language}

        text = getattr(obj, "text", None)
        language = getattr(obj, "language", None)
        if text is not None:
            return {"text": str(text), "language": language}

        return {"text": str(obj), "language": None}

    def _waveforms(self, audios):
        """读取音频为 16k mono waveform。"""
        import librosa
        import numpy as np

        waveform_list = []
        for item in audios:
            if isinstance(item, str):
                waveform_list.append(librosa.load(item, sr=16000, mono=True)[0].astype(np.float32, copy=False))
            elif isinstance(item, np.ndarray):
                waveform_list.append(item.astype(np.float32, copy=False))
            else:
                raise TypeError(f"不支持的音频项类型：{type(item)}")
        return waveform_list

    def _decode_aux_hs(
        self,
        hs_pad: torch.Tensor,
        out_lens: torch.Tensor,
        max_symbols_per_step: int = 5,
    ) -> List[str]:
        """从已编码的 aux 特征解出 CTC/RNNT 文本。"""
        hs_pad = hs_pad.to(next(self.aux_head.parameters()).dtype)
        if self.aux_loss_type == "ctc":
            pred_ids_batch = self.ctc.greedy_decode(hs_pad, out_lens)
        elif self.aux_loss_type == "rnnt":
            pred_ids_batch = self.rnnt.greedy_decode(
                hs_pad,
                out_lens,
                max_symbols_per_step=max_symbols_per_step,
            )
        else:
            raise ValueError(f"不支持的 aux_loss_type: {self.aux_loss_type}")
        return [ids_to_text(ids, self._id_to_token) for ids in pred_ids_batch]

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
        if num_audio_tokens <= 0:
            raise ValueError(f"num_audio_tokens must be positive, got {num_audio_tokens}")

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
            self._audio_prompt(
                num_audio_tokens=int(audio_features.shape[0]),
                context=context or "",
                language=language,
            )
            for audio_features, context, language in zip(audio_features_list, contexts, languages)
        ]

        old_padding_side = processor.tokenizer.padding_side
        processor.tokenizer.padding_side = "left"
        try:
            tok = processor.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
        finally:
            processor.tokenizer.padding_side = old_padding_side

        input_ids = tok["input_ids"].to(device)
        attention_mask = tok["attention_mask"].to(device)
        inputs_embeds = thinker.get_input_embeddings()(input_ids)
        audio_mask = thinker.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds)

        audio_features = torch.cat(
            [x.to(device=device, dtype=dtype) for x in audio_features_list],
            dim=0,
        )
        placeholder_count = int(audio_mask[..., 0].sum().item())
        if placeholder_count != int(audio_features.shape[0]):
            raise RuntimeError(
                f"audio placeholder count mismatch: prompt={placeholder_count}, "
                f"features={audio_features.shape[0]}"
            )

        inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)
        if max_new_tokens is None:
            max_new_tokens = getattr(self._asr_wrapper, "max_new_tokens", 512)

        generated = self.qwen_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
        )
        sequences = generated.sequences if hasattr(generated, "sequences") else generated
        gen_ids = sequences[:, input_ids.shape[1]:]
        raws = processor.batch_decode(
            gen_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        results = []
        for raw, language in zip(raws, languages):
            parsed_language, text = parse_asr_output(raw, user_language=language)
            results.append({"text": text, "language": parsed_language or language})
        return results

    @torch.no_grad()
    def transcribe_llm(
        self,
        audio,
        language: Optional[Union[str, List[str]]] = None,
        context: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ):
        if self._asr_wrapper is None:
            raise RuntimeError("模型未正确初始化，请使用 from_pretrained 加载。")

        audios = self._audio_list(audio)
        batch_size = len(audios)

        call_kwargs = dict(kwargs)
        if language is not None:
            call_kwargs["language"] = language
        if context is not None:
            call_kwargs["context"] = context

        raw_outputs = self._asr_wrapper.transcribe(audios, **call_kwargs)
        raw_outputs = self._norm_outputs(raw_outputs, batch_size)

        results = []
        for idx, raw_out in enumerate(raw_outputs):
            fields = self._asr_fields(raw_out)
            if isinstance(language, list):
                input_lang = language[idx]
            elif isinstance(language, str):
                input_lang = language
            else:
                input_lang = None

            results.append({
                "text": fields["text"],
                "language": fields["language"] or input_lang,
            })

        return results[0] if isinstance(audio, str) else results

    @torch.no_grad()
    def transcribe_llm_streaming(
        self,
        audio,
        language: Optional[Union[str, List[str]]] = None,
        context: Optional[Union[str, List[str]]] = None,
        chunk_sec: float = 0.64,
        left_context_sec: float = 0.64,
        right_context_sec: float = 0.07,
        first_chunk_left_pad_sec: float = 0.0,
        window_batch_size: int = 4,
        window_encoder_batch_size: int = 1,
        max_new_tokens: Optional[int] = None,
    ):
        import librosa
        import numpy as np

        audios = self._audio_list(audio)
        waveform_list = []
        for item in audios:
            if isinstance(item, str):
                waveform_list.append(librosa.load(item, sr=16000, mono=True)[0].astype(np.float32, copy=False))
            elif isinstance(item, np.ndarray):
                waveform_list.append(item.astype(np.float32, copy=False))
            else:
                raise TypeError(f"不支持的音频项类型：{type(item)}")

        if language is None:
            languages = [None] * len(audios)
        elif isinstance(language, str):
            languages = [language] * len(audios)
        else:
            languages = language

        if context is None:
            contexts = [""] * len(audios)
        elif isinstance(context, str):
            contexts = [context] * len(audios)
        else:
            contexts = [c or "" for c in context]

        feature_len_cache: Dict[int, int] = {}
        audio_features_list = [
            self._stream_llm_feats(
                wav,
                chunk_sec=chunk_sec,
                left_context_sec=left_context_sec,
                right_context_sec=right_context_sec,
                first_chunk_left_pad_sec=first_chunk_left_pad_sec,
                window_batch_size=window_batch_size,
                window_encoder_batch_size=window_encoder_batch_size,
                feature_len_cache=feature_len_cache,
            )
            for wav in waveform_list
        ]
        results = self._gen_llm_feats(
            audio_features_list,
            contexts=contexts,
            languages=languages,
            max_new_tokens=max_new_tokens,
        )
        return results[0] if isinstance(audio, str) else results

    def _joint_context(
        self,
        aux_text: str,
        prompt: Optional[str] = None,
        hotword_retriever=None,
        hotword_topk: int = 10,
    ):
        """构造 joint LLM 上下文。

        粗识别结果只用于热词检索，不再直接注入 prompt。
        """
        hotwords = []
        if hotword_retriever is not None and aux_text:
            hotwords = hotword_retriever.retrieve(aux_text, topk=hotword_topk)

        parts = []
        if prompt:
            parts.append(prompt)
        if hotwords:
            parts.append("专属名词列表如下：[" + "，".join(hotwords) + "]" )

        return ("\n".join(parts) if parts else None), hotwords

    @torch.no_grad()
    def transcribe_joint(
        self,
        audio,
        language: Optional[Union[str, List[str]]] = None,
        prompt: Optional[str] = None,
        hotword_retriever=None,
        hotword_topk: int = 10,
        aux_max_symbols_per_step: int = 5,
        aux_encoder_batch_size: int = 1,
        stream_aux: bool = False,
        stream_chunk_sec: float = 0.64,
        stream_left_context_sec: float = 0.64,
        stream_right_context_sec: float = 0.07,
        stream_first_chunk_left_pad_sec: float = 0.0,
        stream_window_batch_size: int = 4,
        stream_window_encoder_batch_size: int = 1,
        **kwargs,
    ):
        if self._asr_wrapper is None:
            raise RuntimeError("模型未正确初始化，请使用 from_pretrained 加载。")

        audios = self._audio_list(audio)
        if language is None:
            languages = [None] * len(audios)
        elif isinstance(language, str):
            languages = [language] * len(audios)
        else:
            languages = language

        if stream_aux:
            waveform_list = self._waveforms(audios)
            feature_len_cache: Dict[int, int] = {}
            aux_texts = []
            audio_features_list = []
            for wav in waveform_list:
                aux_chunks, llm_features = self._stream_joint_feats(
                    wav,
                    chunk_sec=stream_chunk_sec,
                    left_context_sec=stream_left_context_sec,
                    right_context_sec=stream_right_context_sec,
                    first_chunk_left_pad_sec=stream_first_chunk_left_pad_sec,
                    window_batch_size=stream_window_batch_size,
                    window_encoder_batch_size=stream_window_encoder_batch_size,
                    feature_len_cache=feature_len_cache,
                )
                if self.aux_loss_type == "rnnt":
                    ids = self._rnnt_stream_decode(
                        aux_chunks,
                        max_symbols_per_step=aux_max_symbols_per_step,
                    )
                elif self.aux_loss_type == "ctc":
                    ids = self._ctc_stream_decode(aux_chunks)
                else:
                    raise ValueError(f"不支持的 aux_loss_type: {self.aux_loss_type}")
                aux_texts.append(ids_to_text(ids, self._id_to_token))
                audio_features_list.append(llm_features)
        else:
            waveform_list = self._waveforms(audios)
            ref = next(self.qwen_model.parameters())
            batch = self._feature_batch(waveform_list)
            input_features = batch["input_features"].to(device=ref.device, dtype=ref.dtype)
            feature_attention_mask = batch.get("feature_attention_mask", None)
            if feature_attention_mask is not None:
                feature_attention_mask = feature_attention_mask.to(device=ref.device)

            hs_pad, llm_features, out_lens, _ = self._enc_joint(
                input_features,
                feature_attention_mask,
                need_llm_features=True,
                encoder_batch_size=aux_encoder_batch_size,
            )
            aux_texts = self._decode_aux_hs(
                hs_pad,
                out_lens,
                max_symbols_per_step=aux_max_symbols_per_step,
            )
            audio_features_list = []
            offset = 0
            for length in out_lens.tolist():
                cur_len = int(length)
                audio_features_list.append(llm_features[offset: offset + cur_len])
                offset += cur_len

        contexts = []
        hotwords_list = []
        for aux_text in aux_texts:
            context, hotwords = self._joint_context(
                aux_text=aux_text,
                prompt=prompt,
                hotword_retriever=hotword_retriever,
                hotword_topk=hotword_topk,
            )
            contexts.append(context)
            hotwords_list.append(hotwords)

        raw_outputs = self._gen_llm_feats(
            audio_features_list,
            contexts=contexts,
            languages=languages,
            max_new_tokens=kwargs.pop("max_new_tokens", None),
        )
        raw_outputs = self._norm_outputs(raw_outputs, len(audios))

        results = []
        for raw_out, lang, aux_text, hotwords, context in zip(
            raw_outputs, languages, aux_texts, hotwords_list, contexts
        ):
            fields = self._asr_fields(raw_out)
            aux_key = "ctc_text" if self.aux_loss_type == "ctc" else "rnnt_text"
            results.append({
                aux_key: aux_text,
                "aux_text": aux_text,
                "aux_stream_text": aux_text,
                "llm_refined_text": fields["text"],
                "llm_text": fields["text"],
                "text": fields["text"],
                "language": fields["language"] or lang,
                "hotwords": hotwords,
                "context": context,
                "prompt": context,
            })

        return results[0] if isinstance(audio, str) else results

    @torch.no_grad()
    def transcribe(self, audio, mode: str = "joint", **kwargs):
        if mode == "llm":
            if kwargs.get("stream", False):
                return self.transcribe_llm_streaming(
                    audio,
                    language=kwargs.get("language", None),
                    context=kwargs.get("context", kwargs.get("prompt", None)),
                    chunk_sec=kwargs.get("stream_chunk_sec", 0.64),
                    left_context_sec=kwargs.get("stream_left_context_sec", 0.64),
                    right_context_sec=kwargs.get("stream_right_context_sec", 0.07),
                    first_chunk_left_pad_sec=kwargs.get("stream_first_chunk_left_pad_sec", 0.0),
                    window_batch_size=kwargs.get("stream_window_batch_size", 4),
                    window_encoder_batch_size=kwargs.get("stream_window_encoder_batch_size", 1),
                    max_new_tokens=kwargs.get("max_new_tokens", None),
                )
            return self.transcribe_llm(audio, **kwargs)
        if mode == "ctc":
            if kwargs.get("stream", False):
                return self.transcribe_ctc_streaming(
                    audio,
                    chunk_sec=kwargs.get("stream_chunk_sec", 0.64),
                    left_context_sec=kwargs.get("stream_left_context_sec", 0.64),
                    right_context_sec=kwargs.get("stream_right_context_sec", 0.07),
                    first_chunk_left_pad_sec=kwargs.get("stream_first_chunk_left_pad_sec", 0.0),
                    window_batch_size=kwargs.get("stream_window_batch_size", 4),
                    window_encoder_batch_size=kwargs.get("stream_window_encoder_batch_size", 1),
                )
            return self.transcribe_ctc(
                audio,
                aux_encoder_batch_size=kwargs.get("aux_encoder_batch_size", 1),
            )
        if mode == "rnnt":
            max_symbols_per_step = kwargs.get(
                "max_symbols_per_step",
                kwargs.get("rnnt_max_symbols_per_step", 5),
            )
            if kwargs.get("stream", False):
                return self.transcribe_rnnt_streaming(
                    audio,
                    max_symbols_per_step=max_symbols_per_step,
                    chunk_sec=kwargs.get("stream_chunk_sec", 0.64),
                    left_context_sec=kwargs.get("stream_left_context_sec", 0.64),
                    right_context_sec=kwargs.get("stream_right_context_sec", 0.07),
                    first_chunk_left_pad_sec=kwargs.get("stream_first_chunk_left_pad_sec", 0.0),
                    window_batch_size=kwargs.get("stream_window_batch_size", 4),
                    window_encoder_batch_size=kwargs.get("stream_window_encoder_batch_size", 1),
                )
            return self.transcribe_rnnt(
                audio,
                max_symbols_per_step=max_symbols_per_step,
                aux_encoder_batch_size=kwargs.get("aux_encoder_batch_size", 1),
            )
        if mode == "joint":
            return self.transcribe_joint(audio, **kwargs)
        raise ValueError(f"不支持的推理模式：{mode}")
