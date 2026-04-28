# qwen_joint/joint_model.py
import json
import os
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from qwen_asr import Qwen3ASRModel

from .ctc import CTC
from .rnnt import RNNT
from .tokenize_utils import build_id_to_token, ids_to_text


class Qwen3ASRJointModel(nn.Module):
    """Qwen3-ASR + CTC/RNNT 联合模型。

    aux_loss_type:
    - ctc: 兼容原来的 CTC 方案
    - rnnt: 使用 RNNT 作为辅助 ASR loss

    ctc_position:
    - pre_proj:  接在 audio encoder proj 前，维度 d_model
    - post_proj: 接在 audio encoder proj 后，维度 output_dim
    """

    def __init__(
        self,
        qwen_model,
        vocab_size: int,
        vocab: Dict[str, int],
        ctc_weight: float = 0.3,
        blank_id: int = 0,
        ctc_layer_idx: Optional[int] = None,
        ctc_position: str = "pre_proj",
        ctc_only: bool = False,
        aux_loss_type: str = "ctc",
        aux_encoder_batch_size: int = 1,
    ):
        super().__init__()
        self.qwen_model = qwen_model
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.blank_id = blank_id
        self._id_to_token = build_id_to_token(vocab)

        self.ctc_weight = ctc_weight
        self.ctc_layer_idx = ctc_layer_idx
        self.ctc_position = ctc_position
        self.ctc_only = ctc_only
        self.aux_loss_type = aux_loss_type
        self.aux_encoder_batch_size = aux_encoder_batch_size

        audio_config = qwen_model.thinker.audio_tower.config
        if ctc_position == "pre_proj":
            self.encoder_output_size = audio_config.d_model
        elif ctc_position == "post_proj":
            self.encoder_output_size = audio_config.output_dim
        else:
            raise ValueError(f"不支持的 ctc_position: {ctc_position}")

        self.ctc = None
        self.rnnt = None
        if aux_loss_type == "ctc":
            self.ctc = CTC(vocab_size, self.encoder_output_size, blank_id=blank_id)
        elif aux_loss_type == "rnnt":
            self.rnnt = RNNT(vocab_size, self.encoder_output_size, blank_id=blank_id)
        else:
            raise ValueError(f"不支持的 aux_loss_type: {aux_loss_type}")

        self.processor = None
        self._asr_wrapper = None

        for p in self.parameters():
            p.requires_grad = True

    @property
    def aux_head(self):
        return self.ctc if self.aux_loss_type == "ctc" else self.rnnt

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        dtype: torch.dtype = torch.bfloat16,
        device_map: Optional[Union[str, dict]] = "auto",
        ctc_weight: Optional[float] = None,
        ctc_only: Optional[bool] = None,
        load_ctc: bool = True,
        **kwargs,
    ) -> "Qwen3ASRJointModel":
        """从 joint checkpoint 加载：底座 HF 权重 + 辅助头 + ctc_config.json。"""
        base = Qwen3ASRModel.from_pretrained(
            model_path,
            dtype=dtype,
            device_map=device_map,
            **kwargs,
        )

        cfg_path = os.path.join(model_path, "ctc_config.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"未找到配置：{cfg_path}")

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        aux_loss_type = cfg.get("aux_loss_type", "ctc")

        instance = cls(
            qwen_model=base.model,
            vocab_size=cfg["vocab_size"],
            vocab=cfg.get("vocab", {}),
            ctc_weight=ctc_weight if ctc_weight is not None else cfg.get("ctc_weight", 0.3),
            blank_id=cfg.get("blank_id", 0),
            ctc_layer_idx=cfg.get("ctc_layer_idx", None),
            ctc_position=cfg.get("ctc_position", "pre_proj"),
            ctc_only=ctc_only if ctc_only is not None else cfg.get("ctc_only", False),
            aux_loss_type=aux_loss_type,
            aux_encoder_batch_size=cfg.get("aux_encoder_batch_size", 1),
        )

        instance.processor = base.processor
        instance._asr_wrapper = base

        if load_ctc:
            if aux_loss_type == "ctc":
                state_path = os.path.join(model_path, "ctc_head.pt")
                if not os.path.exists(state_path):
                    raise FileNotFoundError(f"未找到 CTC 权重：{state_path}")
                print(f"正在加载 CTC 头：{state_path}")
                instance.ctc.load_state_dict(torch.load(state_path, map_location="cpu"), strict=True)
            elif aux_loss_type == "rnnt":
                state_path = os.path.join(model_path, "rnnt_head.pt")
                if not os.path.exists(state_path):
                    raise FileNotFoundError(f"未找到 RNNT 权重：{state_path}")
                print(f"正在加载 RNNT 头：{state_path}")
                instance.rnnt.load_state_dict(torch.load(state_path, map_location="cpu"), strict=True)

            ref_param = next(base.model.parameters())
            # Keep the auxiliary head in its checkpoint dtype. During training the
            # RNNT/CTC head is initialized and saved in fp32, while the Qwen base
            # may be bf16/fp16; casting the head can change greedy RNNT paths.
            instance.aux_head.to(device=ref_param.device)

        if hasattr(instance.qwen_model, "tie_weights"):
            instance.qwen_model.tie_weights()

        return instance

    def save_ctc(self, output_dir: str) -> None:
        """保存辅助头和配置。文件名保持 ctc_config.json，兼容旧流程。"""
        os.makedirs(output_dir, exist_ok=True)

        if self.aux_loss_type == "ctc":
            torch.save(self.ctc.state_dict(), os.path.join(output_dir, "ctc_head.pt"))
        elif self.aux_loss_type == "rnnt":
            torch.save(self.rnnt.state_dict(), os.path.join(output_dir, "rnnt_head.pt"))

        cfg = {
            "vocab_size": self.vocab_size,
            "encoder_output_size": self.encoder_output_size,
            "blank_id": self.blank_id,
            "ctc_weight": self.ctc_weight,
            "ctc_layer_idx": self.ctc_layer_idx,
            "ctc_position": self.ctc_position,
            "ctc_only": self.ctc_only,
            "aux_loss_type": self.aux_loss_type,
            "aux_encoder_batch_size": self.aux_encoder_batch_size,
            "vocab": self.vocab,
        }
        with open(os.path.join(output_dir, "ctc_config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)

    def _get_feat_extract_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        leave = input_lengths % 100
        feat_lengths = (leave - 1) // 2 + 1
        return ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13

    def _encode_audio_for_joint(
        self,
        input_features: torch.Tensor,
        feature_attention_mask: Optional[torch.Tensor] = None,
        need_llm_features: bool = True,
        encoder_batch_size: Optional[int] = None,
    ):
        """只跑一次 audio_tower，同时得到辅助头特征和 LLM audio embedding。"""
        batch_size = input_features.shape[0]
        device = input_features.device
        if encoder_batch_size is None:
            encoder_batch_size = self.aux_encoder_batch_size
        if encoder_batch_size is not None and encoder_batch_size > 0 and batch_size > encoder_batch_size:
            hs_chunks = []
            llm_feature_chunks = []
            out_len_chunks = []
            feat_len_chunks = []

            for start in range(0, batch_size, encoder_batch_size):
                end = start + encoder_batch_size
                sub_mask = feature_attention_mask[start:end] if feature_attention_mask is not None else None
                sub_hs, sub_llm_features, sub_out_lens, sub_feat_lens = self._encode_audio_for_joint(
                    input_features[start:end],
                    sub_mask,
                    need_llm_features=need_llm_features,
                    encoder_batch_size=0,
                )
                hs_chunks.append(sub_hs)
                if sub_llm_features is not None:
                    llm_feature_chunks.append(sub_llm_features)
                out_len_chunks.append(sub_out_lens)
                feat_len_chunks.append(sub_feat_lens)

            out_lens = torch.cat(out_len_chunks, dim=0)
            feat_lens = torch.cat(feat_len_chunks, dim=0)
            max_len = int(out_lens.max().item())
            hs_pad = torch.zeros(
                batch_size,
                max_len,
                self.encoder_output_size,
                dtype=hs_chunks[0].dtype,
                device=device,
            )
            offset = 0
            for hs_chunk, lens_chunk in zip(hs_chunks, out_len_chunks):
                for i in range(hs_chunk.shape[0]):
                    cur_len = int(lens_chunk[i].item())
                    hs_pad[offset + i, :cur_len] = hs_chunk[i, :cur_len]
                offset += hs_chunk.shape[0]

            audio_features_for_llm = torch.cat(llm_feature_chunks, dim=0) if llm_feature_chunks else None
            return hs_pad, audio_features_for_llm, out_lens, feat_lens

        if feature_attention_mask is not None:
            feat_lens = feature_attention_mask.sum(dim=1).long()
        else:
            feat_lens = torch.full((batch_size,), input_features.shape[2], dtype=torch.long, device=device)

        valid_features = [input_features[b, :, : feat_lens[b]] for b in range(batch_size)]
        concat_features = torch.cat(valid_features, dim=1)
        audio_tower = self.qwen_model.thinker.audio_tower

        if self.ctc_position == "post_proj":
            enc = audio_tower(
                input_features=concat_features,
                feature_lens=feat_lens,
                return_pre_proj=False,
            )
            audio_features_for_llm = enc.last_hidden_state
            aux_features = audio_features_for_llm
        else:
            enc, aux_hidden = audio_tower(
                input_features=concat_features,
                feature_lens=feat_lens,
                return_pre_proj=True,
                ctc_layer_idx=self.ctc_layer_idx,
            )
            pre_final = enc.last_hidden_state
            aux_features = aux_hidden
            audio_features_for_llm = None
            if need_llm_features:
                audio_features_for_llm = audio_tower.proj2(audio_tower.act(audio_tower.proj1(pre_final)))

        out_lens = self._get_feat_extract_output_lengths(feat_lens)
        max_len = int(out_lens.max().item())

        hs_pad = torch.zeros(
            batch_size,
            max_len,
            self.encoder_output_size,
            dtype=aux_features.dtype,
            device=device,
        )

        idx = 0
        for b in range(batch_size):
            cur_len = int(out_lens[b].item())
            hs_pad[b, :cur_len] = aux_features[idx: idx + cur_len]
            idx += cur_len

        return hs_pad, audio_features_for_llm, out_lens, feat_lens

    def _compute_aux_loss(self, hs_pad, out_lens, target_ids, target_lengths):
        if target_ids is None:
            return torch.tensor(0.0, device=hs_pad.device)

        if self.aux_loss_type == "ctc":
            return self.ctc(hs_pad, out_lens, target_ids, target_lengths)

        if self.aux_loss_type == "rnnt":
            return self.rnnt(hs_pad, out_lens, target_ids, target_lengths)

        raise ValueError(f"不支持的 aux_loss_type: {self.aux_loss_type}")

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_features=None,
        feature_attention_mask=None,
        labels=None,
        ctc_target_ids=None,
        ctc_target_lengths=None,
        texts=None,
        **kwargs,
    ):
        hs_pad, audio_features_for_llm, out_lens, _ = self._encode_audio_for_joint(
            input_features,
            feature_attention_mask,
            need_llm_features=not self.ctc_only,
        )

        aux_loss = self._compute_aux_loss(hs_pad, out_lens, ctc_target_ids, ctc_target_lengths)

        if self.ctc_only:
            llm_loss = torch.zeros_like(aux_loss)
            outputs = {
                "loss": aux_loss,
                "llm_loss": llm_loss,
                "aux_loss": aux_loss,
                "output_lengths": out_lens,
            }
            if self.aux_loss_type == "ctc":
                outputs["ctc_loss"] = aux_loss
                outputs["log_probs"] = self.ctc.log_softmax(hs_pad)
            else:
                outputs["rnnt_loss"] = aux_loss
            return outputs

        embeds = self.qwen_model.thinker.get_input_embeddings()(input_ids)
        audio_mask = self.qwen_model.thinker.get_placeholder_mask(input_ids, embeds)
        embeds = embeds.masked_scatter(audio_mask, audio_features_for_llm.to(embeds.dtype))

        llm_out = self.qwen_model.thinker(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        llm_loss = llm_out.loss
        loss = (1.0 - self.ctc_weight) * llm_loss + self.ctc_weight * aux_loss

        outputs = {
            "loss": loss,
            "llm_loss": llm_loss,
            "aux_loss": aux_loss,
            "output_lengths": out_lens,
        }
        if self.aux_loss_type == "ctc":
            outputs["ctc_loss"] = aux_loss
            outputs["log_probs"] = self.ctc.log_softmax(hs_pad)
        else:
            outputs["rnnt_loss"] = aux_loss
        return outputs

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
    def decode_aux_features(
        self,
        input_features: torch.Tensor,
        feature_attention_mask: Optional[torch.Tensor] = None,
        max_symbols_per_step: int = 5,
        rnnt_decode_strategy: str = "cached",
        aux_encoder_batch_size: Optional[int] = None,
    ) -> List[str]:
        """Decode the active auxiliary head from already-collated audio features."""
        ref = next(self.qwen_model.parameters())
        input_features = input_features.to(device=ref.device, dtype=ref.dtype)
        if feature_attention_mask is not None:
            feature_attention_mask = feature_attention_mask.to(device=ref.device)

        hs_pad, _, out_lens, _ = self._encode_audio_for_joint(
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
                decode_strategy=rnnt_decode_strategy,
            )
        else:
            raise ValueError(f"不支持的 aux_loss_type: {self.aux_loss_type}")

        return [ids_to_text(ids, self._id_to_token) for ids in pred_ids_batch]

    def _normalize_audio_to_list(self, audio) -> List:
        if isinstance(audio, str):
            return [audio]
        if isinstance(audio, list):
            return audio

        import numpy as np
        if isinstance(audio, np.ndarray):
            return [audio]

        raise TypeError(f"不支持的音频类型：{type(audio)}")

    def _build_ctc_feature_batch(self, waveform_list: List):
        feature_extractor = self.processor.feature_extractor
        batch = feature_extractor(
            waveform_list,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            truncation=False,
            return_attention_mask=True,
        )
        if "feature_attention_mask" not in batch and "attention_mask" in batch:
            batch["feature_attention_mask"] = batch.pop("attention_mask")
        return batch

    @torch.no_grad()
    def _aux_decode_from_audio(
        self,
        audio,
        max_symbols_per_step: int = 5,
        rnnt_decode_strategy: str = "cached",
        aux_encoder_batch_size: int = 1,
    ) -> List[str]:
        """辅助头推理：CTC/RNNT 共用入口。"""
        import librosa
        import numpy as np

        audios = self._normalize_audio_to_list(audio)
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
            batch = self._build_ctc_feature_batch(sub_wavs)
            input_features = batch["input_features"].to(device=ref.device, dtype=ref.dtype)
            feature_attention_mask = batch.get("feature_attention_mask", None)
            if feature_attention_mask is not None:
                feature_attention_mask = feature_attention_mask.to(device=ref.device)

            results.extend(
                self.decode_aux_features(
                    input_features,
                    feature_attention_mask,
                    max_symbols_per_step=max_symbols_per_step,
                    rnnt_decode_strategy=rnnt_decode_strategy,
                    aux_encoder_batch_size=aux_encoder_batch_size,
                )
            )
        return results

    @torch.no_grad()
    def _ctc_decode_from_audio(self, audio, aux_encoder_batch_size: int = 1) -> List[str]:
        if self.aux_loss_type != "ctc":
            raise RuntimeError("当前 checkpoint 不是 CTC 模式，请使用 transcribe_rnnt 或 joint。")
        return self._aux_decode_from_audio(audio, aux_encoder_batch_size=aux_encoder_batch_size)

    @torch.no_grad()
    def _rnnt_decode_from_audio(
        self,
        audio,
        max_symbols_per_step: int = 5,
        rnnt_decode_strategy: str = "cached",
        aux_encoder_batch_size: int = 1,
    ) -> List[str]:
        if self.aux_loss_type != "rnnt":
            raise RuntimeError("当前 checkpoint 不是 RNNT 模式。")
        return self._aux_decode_from_audio(
            audio,
            max_symbols_per_step=max_symbols_per_step,
            rnnt_decode_strategy=rnnt_decode_strategy,
            aux_encoder_batch_size=aux_encoder_batch_size,
        )

    @torch.no_grad()
    def transcribe_ctc(self, audio, aux_encoder_batch_size: int = 1) -> Union[str, List[str]]:
        results = self._ctc_decode_from_audio(audio, aux_encoder_batch_size=aux_encoder_batch_size)
        return results[0] if isinstance(audio, str) else results

    @torch.no_grad()
    def transcribe_rnnt(
        self,
        audio,
        max_symbols_per_step: int = 5,
        rnnt_decode_strategy: str = "cached",
        aux_encoder_batch_size: int = 1,
    ) -> Union[str, List[str]]:
        results = self._rnnt_decode_from_audio(
            audio,
            max_symbols_per_step=max_symbols_per_step,
            rnnt_decode_strategy=rnnt_decode_strategy,
            aux_encoder_batch_size=aux_encoder_batch_size,
        )
        return results[0] if isinstance(audio, str) else results

    def _extract_asr_fields(self, obj):
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
                return self._extract_asr_fields(obj[0])
            items = [self._extract_asr_fields(x) for x in obj]
            text = " ".join([x["text"] for x in items if x["text"]]).strip()
            language = next((x["language"] for x in items if x["language"]), None)
            return {"text": text, "language": language}

        text = getattr(obj, "text", None)
        language = getattr(obj, "language", None)
        if text is not None:
            return {"text": str(text), "language": language}

        return {"text": str(obj), "language": None}

    def _normalize_transcribe_outputs(self, outputs, batch_size: int):
        if isinstance(outputs, list):
            if len(outputs) == batch_size:
                return outputs
            if batch_size == 1:
                return [outputs]
            raise ValueError(f"底座输出数量不匹配：expect {batch_size}, got {len(outputs)}")
        return [outputs]

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

        audios = self._normalize_audio_to_list(audio)
        batch_size = len(audios)

        call_kwargs = dict(kwargs)
        if language is not None:
            call_kwargs["language"] = language
        if context is not None:
            call_kwargs["context"] = context

        raw_outputs = self._asr_wrapper.transcribe(audios, **call_kwargs)
        raw_outputs = self._normalize_transcribe_outputs(raw_outputs, batch_size)

        results = []
        for idx, raw_out in enumerate(raw_outputs):
            fields = self._extract_asr_fields(raw_out)
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

    def _build_joint_context(
        self,
        aux_text: str,
        prompt: Optional[str] = None,
        hotword_retriever=None,
        hotword_topk: int = 10,
        inject_ctc_into_prompt: bool = True,
    ):
        hotwords = []
        if hotword_retriever is not None and aux_text:
            hotwords = hotword_retriever.retrieve(aux_text, topk=hotword_topk)

        parts = []
        if prompt:
            parts.append(prompt)
        if inject_ctc_into_prompt and aux_text:
            parts.append(f"参考粗识别结果：{aux_text}")
        if hotwords:
            parts.append("相关热词：" + "，".join(hotwords))

        return ("\n".join(parts) if parts else None), hotwords

    @torch.no_grad()
    def transcribe_joint(
        self,
        audio,
        language: Optional[Union[str, List[str]]] = None,
        prompt: Optional[str] = None,
        hotword_retriever=None,
        hotword_topk: int = 10,
        inject_ctc_into_prompt: bool = True,
        aux_max_symbols_per_step: int = 5,
        rnnt_decode_strategy: str = "cached",
        aux_encoder_batch_size: int = 1,
        **kwargs,
    ):
        if self._asr_wrapper is None:
            raise RuntimeError("模型未正确初始化，请使用 from_pretrained 加载。")

        audios = self._normalize_audio_to_list(audio)
        if language is None:
            languages = [None] * len(audios)
        elif isinstance(language, str):
            languages = [language] * len(audios)
        else:
            languages = language

        aux_texts = self._aux_decode_from_audio(
            audios,
            max_symbols_per_step=aux_max_symbols_per_step,
            rnnt_decode_strategy=rnnt_decode_strategy,
            aux_encoder_batch_size=aux_encoder_batch_size,
        )

        contexts = []
        hotwords_list = []
        for aux_text in aux_texts:
            context, hotwords = self._build_joint_context(
                aux_text=aux_text,
                prompt=prompt,
                hotword_retriever=hotword_retriever,
                hotword_topk=hotword_topk,
                inject_ctc_into_prompt=inject_ctc_into_prompt,
            )
            contexts.append(context)
            hotwords_list.append(hotwords)

        raw_outputs = self._asr_wrapper.transcribe(
            audios,
            language=languages,
            context=contexts,
            **kwargs,
        )
        raw_outputs = self._normalize_transcribe_outputs(raw_outputs, len(audios))

        results = []
        for raw_out, lang, aux_text, hotwords, context in zip(
            raw_outputs, languages, aux_texts, hotwords_list, contexts
        ):
            fields = self._extract_asr_fields(raw_out)
            results.append({
                "ctc_stream_text": aux_text,
                "aux_stream_text": aux_text,
                "llm_refined_text": fields["text"],
                "text": fields["text"],
                "language": fields["language"] or lang,
                "hotwords": hotwords,
                "prompt": context,
            })

        return results[0] if isinstance(audio, str) else results

    @torch.no_grad()
    def transcribe(self, audio, mode: str = "joint", **kwargs):
        if mode == "llm":
            return self.transcribe_llm(audio, **kwargs)
        if mode == "ctc":
            return self.transcribe_ctc(
                audio,
                aux_encoder_batch_size=kwargs.get("aux_encoder_batch_size", 1),
            )
        if mode == "rnnt":
            max_symbols_per_step = kwargs.get(
                "max_symbols_per_step",
                kwargs.get("rnnt_max_symbols_per_step", 5),
            )
            return self.transcribe_rnnt(
                audio,
                max_symbols_per_step=max_symbols_per_step,
                rnnt_decode_strategy=kwargs.get("rnnt_decode_strategy", "cached"),
                aux_encoder_batch_size=kwargs.get("aux_encoder_batch_size", 1),
            )
        if mode == "joint":
            return self.transcribe_joint(audio, **kwargs)
        raise ValueError(f"不支持的推理模式：{mode}")
