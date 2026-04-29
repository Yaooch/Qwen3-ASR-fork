# qwen_joint/joint_model.py
import json
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from qwen_asr import Qwen3ASRModel
from qwen_asr.inference.utils import parse_asr_output

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
        aux_streaming_train: bool = False,
        aux_stream_chunk_frames: int = 64,
        aux_stream_left_context_frames: int = 64,
        aux_stream_right_context_frames: int = 7,
        aux_stream_random_left: bool = True,
        aux_stream_window_batch_size: int = 4,
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
        self.aux_streaming_train = aux_streaming_train
        self.aux_stream_chunk_frames = aux_stream_chunk_frames
        self.aux_stream_left_context_frames = aux_stream_left_context_frames
        self.aux_stream_right_context_frames = aux_stream_right_context_frames
        self.aux_stream_random_left = aux_stream_random_left
        self.aux_stream_window_batch_size = aux_stream_window_batch_size

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
            aux_streaming_train=cfg.get("aux_streaming_train", False),
            aux_stream_chunk_frames=cfg.get("aux_stream_chunk_frames", 64),
            aux_stream_left_context_frames=cfg.get("aux_stream_left_context_frames", 64),
            aux_stream_right_context_frames=cfg.get("aux_stream_right_context_frames", 7),
            aux_stream_random_left=cfg.get("aux_stream_random_left", True),
            aux_stream_window_batch_size=cfg.get("aux_stream_window_batch_size", 4),
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
            "aux_streaming_train": self.aux_streaming_train,
            "aux_stream_chunk_frames": self.aux_stream_chunk_frames,
            "aux_stream_left_context_frames": self.aux_stream_left_context_frames,
            "aux_stream_right_context_frames": self.aux_stream_right_context_frames,
            "aux_stream_random_left": self.aux_stream_random_left,
            "aux_stream_window_batch_size": self.aux_stream_window_batch_size,
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

    def _build_feature_stream_windows(
        self,
        input_features: torch.Tensor,
        feat_lens: torch.Tensor,
    ):
        """Build feature-level streaming windows for aux loss training.

        Each window contains a current chunk, a random amount of left context,
        and a fixed right context. Only the current chunk frames are kept for
        the final aux sequence.
        """
        windows = []
        chunk_frames = max(1, int(self.aux_stream_chunk_frames))
        max_left_frames = max(0, int(self.aux_stream_left_context_frames))
        right_frames = max(0, int(self.aux_stream_right_context_frames))
        random_left = bool(self.aux_stream_random_left)
        device = input_features.device

        for b in range(input_features.shape[0]):
            total = int(feat_lens[b].item())
            start = 0
            while start < total:
                end = min(total, start + chunk_frames)
                allowed_left = min(max_left_frames, start)
                if random_left and allowed_left > 0 and self.training:
                    left = int(torch.randint(0, allowed_left + 1, (1,), device=device).item())
                else:
                    left = allowed_left
                enc_start = start - left
                enc_end = min(total, end + right_frames)
                windows.append(
                    {
                        "sample_idx": b,
                        "start": start,
                        "end": end,
                        "enc_start": enc_start,
                        "features": input_features[b, :, enc_start:enc_end],
                    }
                )
                start = end
        return windows

    def _pad_feature_windows(self, windows: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        if not windows:
            raise RuntimeError("No feature windows were produced.")

        batch_size = len(windows)
        channels = windows[0]["features"].shape[0]
        max_len = max(int(w["features"].shape[1]) for w in windows)
        ref = windows[0]["features"]
        padded = torch.zeros(batch_size, channels, max_len, dtype=ref.dtype, device=ref.device)
        mask = torch.zeros(batch_size, max_len, dtype=torch.long, device=ref.device)
        for i, item in enumerate(windows):
            cur = item["features"]
            cur_len = int(cur.shape[1])
            padded[i, :, :cur_len] = cur
            mask[i, :cur_len] = 1
        return padded, mask

    def _encode_aux_streaming_train_features(
        self,
        input_features: torch.Tensor,
        feature_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode aux features with streaming-style random left context for training."""
        if feature_attention_mask is not None:
            feat_lens = feature_attention_mask.sum(dim=1).long()
        else:
            feat_lens = torch.full(
                (input_features.shape[0],),
                input_features.shape[2],
                dtype=torch.long,
                device=input_features.device,
            )

        windows = self._build_feature_stream_windows(input_features, feat_lens)
        per_sample_chunks = [[] for _ in range(input_features.shape[0])]
        window_batch_size = max(1, int(self.aux_stream_window_batch_size))
        device = input_features.device

        for batch_start in range(0, len(windows), window_batch_size):
            batch_windows = windows[batch_start: batch_start + window_batch_size]
            window_features, window_mask = self._pad_feature_windows(batch_windows)
            hs_pad, _, out_lens, _ = self._encode_audio_for_joint(
                window_features,
                window_mask,
                need_llm_features=False,
                encoder_batch_size=self.aux_encoder_batch_size,
            )
            hs_pad = hs_pad.to(next(self.aux_head.parameters()).dtype)

            for i, item in enumerate(batch_windows):
                keep_start_frames = item["start"] - item["enc_start"]
                keep_end_frames = item["end"] - item["enc_start"]
                keep_start_idx = self._encoder_len_from_feature_len(keep_start_frames, device)
                keep_end_idx = self._encoder_len_from_feature_len(keep_end_frames, device)
                cur_len = int(out_lens[i].item())
                keep_start_idx = max(0, min(keep_start_idx, cur_len))
                keep_end_idx = max(keep_start_idx, min(keep_end_idx, cur_len))
                kept = hs_pad[i, keep_start_idx:keep_end_idx]
                if kept.numel() > 0:
                    per_sample_chunks[item["sample_idx"]].append(kept)

        seqs = []
        lens = []
        for chunks in per_sample_chunks:
            if not chunks:
                raise RuntimeError("Streaming aux training produced an empty sample.")
            seq = torch.cat(chunks, dim=0)
            seqs.append(seq)
            lens.append(seq.shape[0])

        max_len = max(lens)
        hs_out = torch.zeros(
            len(seqs),
            max_len,
            self.encoder_output_size,
            dtype=seqs[0].dtype,
            device=input_features.device,
        )
        for i, seq in enumerate(seqs):
            hs_out[i, : seq.shape[0]] = seq
        out_lens = torch.tensor(lens, dtype=torch.long, device=input_features.device)
        return hs_out, out_lens

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
        use_streaming_aux_train = bool(self.training and self.aux_streaming_train)
        if use_streaming_aux_train:
            hs_pad, out_lens = self._encode_aux_streaming_train_features(
                input_features,
                feature_attention_mask,
            )
            audio_features_for_llm = None
            if not self.ctc_only:
                _, audio_features_for_llm, _, _ = self._encode_audio_for_joint(
                    input_features,
                    feature_attention_mask,
                    need_llm_features=True,
                )
        else:
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

    def _feature_len_for_num_samples(self, num_samples: int, cache: Optional[Dict[int, int]] = None) -> int:
        """Feature extractor valid length for a waveform with this many samples."""
        if num_samples <= 0:
            return 0
        num_samples = int(num_samples)
        if cache is not None and num_samples in cache:
            return cache[num_samples]

        import numpy as np

        feat = self.processor.feature_extractor(
            np.zeros((num_samples,), dtype=np.float32),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            truncation=False,
            return_attention_mask=True,
        )
        value = int(feat["attention_mask"].sum().item())
        if cache is not None:
            cache[num_samples] = value
        return value

    def _encoder_len_from_feature_len(self, feature_len: int, device: torch.device) -> int:
        if feature_len <= 0:
            return 0
        x = torch.tensor([feature_len], dtype=torch.long, device=device)
        return int(self._get_feat_extract_output_lengths(x)[0].item())

    def _build_stream_windows(
        self,
        wav,
        chunk_sec: float,
        left_context_sec: float,
        right_context_sec: float,
        first_chunk_left_pad_sec: float,
    ):
        """Build current-chunk windows for streaming aux decoding."""
        if chunk_sec <= 0:
            raise ValueError(f"chunk_sec must be > 0, got {chunk_sec}")

        import numpy as np

        sr = 16000
        total_samples = int(wav.shape[0])
        chunk_samples = max(1, int(round(chunk_sec * sr)))
        left_samples = max(0, int(round(left_context_sec * sr)))
        right_samples = max(0, int(round(right_context_sec * sr)))
        first_left_pad_samples = max(0, int(round(first_chunk_left_pad_sec * sr)))
        if first_left_pad_samples >= chunk_samples:
            raise ValueError(
                "first_chunk_left_pad_sec must be smaller than chunk_sec, got "
                f"{first_chunk_left_pad_sec} >= {chunk_sec}"
            )

        windows = []
        start = 0
        first_chunk = True
        while start < total_samples:
            cur_chunk_samples = chunk_samples
            left_pad_samples = 0
            if first_chunk and first_left_pad_samples > 0:
                cur_chunk_samples = chunk_samples - first_left_pad_samples
                left_pad_samples = first_left_pad_samples

            end = min(total_samples, start + cur_chunk_samples)
            enc_start = max(0, start - left_samples)
            enc_end = min(total_samples, end + right_samples)
            window_wav = wav[enc_start:enc_end]
            if left_pad_samples > 0:
                pad = np.zeros(left_pad_samples, dtype=window_wav.dtype)
                window_wav = np.concatenate([pad, window_wav], axis=0)

            windows.append((start, end, enc_start, enc_end, left_pad_samples, window_wav))
            start = end
            first_chunk = False
        return windows

    @torch.no_grad()
    def _encode_aux_feature_windows(
        self,
        window_wavs: List,
        encoder_batch_size: int = 1,
    ) -> List[torch.Tensor]:
        """Encode audio windows and return one aux-feature tensor per window."""
        if not window_wavs:
            return []

        ref = next(self.qwen_model.parameters())
        batch = self._build_ctc_feature_batch(window_wavs)
        input_features = batch["input_features"].to(device=ref.device, dtype=ref.dtype)
        feature_attention_mask = batch["feature_attention_mask"].to(device=ref.device)

        hs_pad, _, out_lens, _ = self._encode_audio_for_joint(
            input_features,
            feature_attention_mask,
            need_llm_features=False,
            encoder_batch_size=encoder_batch_size,
        )
        hs_pad = hs_pad.to(next(self.aux_head.parameters()).dtype)
        return [hs_pad[i, : int(length)] for i, length in enumerate(out_lens.tolist())]

    @torch.no_grad()
    def _encode_joint_feature_windows(
        self,
        window_wavs: List,
        encoder_batch_size: int = 1,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Encode audio windows and return aux + LLM features per window."""
        if not window_wavs:
            return [], []

        ref = next(self.qwen_model.parameters())
        batch = self._build_ctc_feature_batch(window_wavs)
        input_features = batch["input_features"].to(device=ref.device, dtype=ref.dtype)
        feature_attention_mask = batch["feature_attention_mask"].to(device=ref.device)

        hs_pad, llm_features, out_lens, _ = self._encode_audio_for_joint(
            input_features,
            feature_attention_mask,
            need_llm_features=True,
            encoder_batch_size=encoder_batch_size,
        )
        hs_pad = hs_pad.to(next(self.aux_head.parameters()).dtype)

        aux_results = []
        llm_results = []
        offset = 0
        for i, length in enumerate(out_lens.tolist()):
            cur_len = int(length)
            aux_results.append(hs_pad[i, :cur_len])
            llm_results.append(llm_features[offset: offset + cur_len])
            offset += cur_len
        return aux_results, llm_results

    @torch.no_grad()
    def _build_streaming_aux_chunks(
        self,
        wav,
        chunk_sec: float = 0.64,
        left_context_sec: float = 0.64,
        right_context_sec: float = 0.07,
        first_chunk_left_pad_sec: float = 0.0,
        window_batch_size: int = 4,
        window_encoder_batch_size: int = 1,
        feature_len_cache: Optional[Dict[int, int]] = None,
    ) -> List[torch.Tensor]:
        """Encode overlap windows and keep only the newly arrived chunk frames."""
        device = next(self.qwen_model.parameters()).device
        windows = self._build_stream_windows(
            wav,
            chunk_sec=chunk_sec,
            left_context_sec=left_context_sec,
            right_context_sec=right_context_sec,
            first_chunk_left_pad_sec=first_chunk_left_pad_sec,
        )
        if window_batch_size is None or window_batch_size <= 0:
            window_batch_size = len(windows)

        kept_chunks = []
        for batch_start in range(0, len(windows), window_batch_size):
            batch_windows = windows[batch_start: batch_start + window_batch_size]
            window_features_batch = self._encode_aux_feature_windows(
                [x[5] for x in batch_windows],
                encoder_batch_size=window_encoder_batch_size,
            )

            for (start, end, enc_start, _enc_end, left_pad_samples, _), window_features in zip(
                batch_windows, window_features_batch
            ):
                keep_start_samples = left_pad_samples + start - enc_start
                keep_end_samples = left_pad_samples + end - enc_start
                keep_start_feat = self._feature_len_for_num_samples(keep_start_samples, feature_len_cache)
                keep_end_feat = self._feature_len_for_num_samples(keep_end_samples, feature_len_cache)
                keep_start_idx = self._encoder_len_from_feature_len(keep_start_feat, device)
                keep_end_idx = self._encoder_len_from_feature_len(keep_end_feat, device)

                keep_start_idx = max(0, min(keep_start_idx, window_features.shape[0]))
                keep_end_idx = max(keep_start_idx, min(keep_end_idx, window_features.shape[0]))
                kept = window_features[keep_start_idx:keep_end_idx]
                if kept.numel() > 0:
                    kept_chunks.append(kept)

        if not kept_chunks:
            raise RuntimeError("No streaming auxiliary features were produced.")
        return kept_chunks

    @torch.no_grad()
    def _build_streaming_llm_features(
        self,
        wav,
        chunk_sec: float = 0.64,
        left_context_sec: float = 0.64,
        right_context_sec: float = 0.07,
        first_chunk_left_pad_sec: float = 0.0,
        window_batch_size: int = 4,
        window_encoder_batch_size: int = 1,
        feature_len_cache: Optional[Dict[int, int]] = None,
    ) -> torch.Tensor:
        """Encode overlap windows and concatenate current-chunk LLM audio features."""
        device = next(self.qwen_model.parameters()).device
        windows = self._build_stream_windows(
            wav,
            chunk_sec=chunk_sec,
            left_context_sec=left_context_sec,
            right_context_sec=right_context_sec,
            first_chunk_left_pad_sec=first_chunk_left_pad_sec,
        )
        if window_batch_size is None or window_batch_size <= 0:
            window_batch_size = len(windows)

        kept_chunks = []
        for batch_start in range(0, len(windows), window_batch_size):
            batch_windows = windows[batch_start: batch_start + window_batch_size]
            _aux_features_batch, llm_features_batch = self._encode_joint_feature_windows(
                [x[5] for x in batch_windows],
                encoder_batch_size=window_encoder_batch_size,
            )

            for (start, end, enc_start, _enc_end, left_pad_samples, _), window_features in zip(
                batch_windows, llm_features_batch
            ):
                keep_start_samples = left_pad_samples + start - enc_start
                keep_end_samples = left_pad_samples + end - enc_start
                keep_start_feat = self._feature_len_for_num_samples(keep_start_samples, feature_len_cache)
                keep_end_feat = self._feature_len_for_num_samples(keep_end_samples, feature_len_cache)
                keep_start_idx = self._encoder_len_from_feature_len(keep_start_feat, device)
                keep_end_idx = self._encoder_len_from_feature_len(keep_end_feat, device)

                keep_start_idx = max(0, min(keep_start_idx, window_features.shape[0]))
                keep_end_idx = max(keep_start_idx, min(keep_end_idx, window_features.shape[0]))
                kept = window_features[keep_start_idx:keep_end_idx]
                if kept.numel() > 0:
                    kept_chunks.append(kept)

        if not kept_chunks:
            raise RuntimeError("No streaming LLM audio features were produced.")
        return torch.cat(kept_chunks, dim=0)

    @torch.no_grad()
    def _rnnt_streaming_greedy_decode(
        self,
        chunks: List[torch.Tensor],
        max_symbols_per_step: int = 5,
    ) -> List[int]:
        """Stateful RNNT greedy decode. Predictor state is carried across chunks."""
        if self.aux_loss_type != "rnnt":
            raise RuntimeError(f"当前 checkpoint 的 aux_loss_type={self.aux_loss_type!r}，不是 rnnt。")
        if max_symbols_per_step <= 0:
            raise ValueError(f"max_symbols_per_step must be positive, got {max_symbols_per_step}")

        rnnt = self.rnnt
        device = next(rnnt.parameters()).device
        rnnt_dtype = next(rnnt.parameters()).dtype
        state_dtype = rnnt.embed.weight.dtype
        hidden_size = rnnt.pred_rnn.hidden_size
        num_layers = rnnt.pred_rnn.num_layers

        h = torch.zeros(num_layers, 1, hidden_size, device=device, dtype=state_dtype)
        c = torch.zeros(num_layers, 1, hidden_size, device=device, dtype=state_dtype)

        start_token = torch.full((1, 1), rnnt.blank_id, dtype=torch.long, device=device)
        pred_step, (h, c) = rnnt.pred_rnn(rnnt.embed(start_token), (h, c))
        pred = rnnt.pred_proj(pred_step[:, -1, :])

        emitted = []
        for chunk in chunks:
            enc = rnnt.enc_proj(chunk.to(device=device, dtype=rnnt_dtype).unsqueeze(0))[0]
            for t in range(enc.shape[0]):
                symbols = 0
                while symbols < max_symbols_per_step:
                    joint = torch.tanh(enc[t].unsqueeze(0) + pred)
                    next_id = int(rnnt.joiner(joint).argmax(dim=-1).item())
                    if next_id == rnnt.blank_id:
                        break

                    emitted.append(next_id)
                    token = torch.tensor([[next_id]], dtype=torch.long, device=device)
                    pred_step, (h, c) = rnnt.pred_rnn(rnnt.embed(token), (h, c))
                    pred = rnnt.pred_proj(pred_step[:, -1, :])
                    symbols += 1

        return emitted

    @torch.no_grad()
    def _ctc_streaming_greedy_decode(self, chunks: List[torch.Tensor]) -> List[int]:
        """Streaming CTC greedy decode with CTC collapse state carried across chunks."""
        if self.aux_loss_type != "ctc":
            raise RuntimeError(f"当前 checkpoint 的 aux_loss_type={self.aux_loss_type!r}，不是 ctc。")

        emitted = []
        prev_id = -1
        for chunk in chunks:
            hs = chunk.to(device=next(self.ctc.parameters()).device, dtype=next(self.ctc.parameters()).dtype)
            log_probs = self.ctc.log_softmax(hs.unsqueeze(0))[0]
            pred_ids = log_probs.argmax(dim=-1).tolist()
            for idx in pred_ids:
                if idx != self.ctc.blank_id and idx != prev_id:
                    emitted.append(idx)
                prev_id = idx
        return emitted

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
    def _rnnt_streaming_decode_from_audio(
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

        audios = self._normalize_audio_to_list(audio)
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
            chunks = self._build_streaming_aux_chunks(
                wav,
                chunk_sec=chunk_sec,
                left_context_sec=left_context_sec,
                right_context_sec=right_context_sec,
                first_chunk_left_pad_sec=first_chunk_left_pad_sec,
                window_batch_size=window_batch_size,
                window_encoder_batch_size=window_encoder_batch_size,
                feature_len_cache=feature_len_cache,
            )
            ids = self._rnnt_streaming_greedy_decode(
                chunks,
                max_symbols_per_step=max_symbols_per_step,
            )
            results.append(ids_to_text(ids, self._id_to_token))
        return results

    @torch.no_grad()
    def _ctc_streaming_decode_from_audio(
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

        audios = self._normalize_audio_to_list(audio)
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
            chunks = self._build_streaming_aux_chunks(
                wav,
                chunk_sec=chunk_sec,
                left_context_sec=left_context_sec,
                right_context_sec=right_context_sec,
                first_chunk_left_pad_sec=first_chunk_left_pad_sec,
                window_batch_size=window_batch_size,
                window_encoder_batch_size=window_encoder_batch_size,
                feature_len_cache=feature_len_cache,
            )
            ids = self._ctc_streaming_greedy_decode(chunks)
            results.append(ids_to_text(ids, self._id_to_token))
        return results

    @torch.no_grad()
    def transcribe_ctc(self, audio, aux_encoder_batch_size: int = 1) -> Union[str, List[str]]:
        results = self._ctc_decode_from_audio(audio, aux_encoder_batch_size=aux_encoder_batch_size)
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
        results = self._ctc_streaming_decode_from_audio(
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
        results = self._rnnt_streaming_decode_from_audio(
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

    def _build_manual_audio_prompt(self, num_audio_tokens: int, context: str, language: Optional[str]) -> str:
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
    def _generate_llm_from_audio_features(
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
            self._build_manual_audio_prompt(
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

        audios = self._normalize_audio_to_list(audio)
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
            self._build_streaming_llm_features(
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
        results = self._generate_llm_from_audio_features(
            audio_features_list,
            contexts=contexts,
            languages=languages,
            max_new_tokens=max_new_tokens,
        )
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

        audios = self._normalize_audio_to_list(audio)
        if language is None:
            languages = [None] * len(audios)
        elif isinstance(language, str):
            languages = [language] * len(audios)
        else:
            languages = language

        if stream_aux and self.aux_loss_type in ("ctc", "rnnt"):
            if self.aux_loss_type == "rnnt":
                aux_texts = self._rnnt_streaming_decode_from_audio(
                    audios,
                    max_symbols_per_step=aux_max_symbols_per_step,
                    chunk_sec=stream_chunk_sec,
                    left_context_sec=stream_left_context_sec,
                    right_context_sec=stream_right_context_sec,
                    first_chunk_left_pad_sec=stream_first_chunk_left_pad_sec,
                    window_batch_size=stream_window_batch_size,
                    window_encoder_batch_size=stream_window_encoder_batch_size,
                )
            else:
                aux_texts = self._ctc_streaming_decode_from_audio(
                    audios,
                    chunk_sec=stream_chunk_sec,
                    left_context_sec=stream_left_context_sec,
                    right_context_sec=stream_right_context_sec,
                    first_chunk_left_pad_sec=stream_first_chunk_left_pad_sec,
                    window_batch_size=stream_window_batch_size,
                    window_encoder_batch_size=stream_window_encoder_batch_size,
                )
        else:
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

        if stream_aux:
            raw_outputs = self.transcribe_llm_streaming(
                audios,
                language=languages,
                context=contexts,
                chunk_sec=stream_chunk_sec,
                left_context_sec=stream_left_context_sec,
                right_context_sec=stream_right_context_sec,
                first_chunk_left_pad_sec=stream_first_chunk_left_pad_sec,
                window_batch_size=stream_window_batch_size,
                window_encoder_batch_size=stream_window_encoder_batch_size,
                max_new_tokens=kwargs.pop("max_new_tokens", None),
            )
        else:
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
                rnnt_decode_strategy=kwargs.get("rnnt_decode_strategy", "cached"),
                aux_encoder_batch_size=kwargs.get("aux_encoder_batch_size", 1),
            )
        if mode == "joint":
            return self.transcribe_joint(audio, **kwargs)
        raise ValueError(f"不支持的推理模式：{mode}")
