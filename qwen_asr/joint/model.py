# qwen_asr/joint/model.py
import json
import os
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from qwen_asr import Qwen3ASRModel
from .ctc import CTC
from .rnnt import RNNT
from .decode import DecodeMixin
from .stream import StreamMixin
from .tokens import build_id_to_token
from .train_utils import TrainMixin


class Qwen3ASRJointModel(TrainMixin, StreamMixin, DecodeMixin, nn.Module):
    """Qwen3-ASR + CTC/RNNT 联合模型。

    aux_loss_type:
    - ctc: 使用 CTC 作为辅助 ASR loss
    - rnnt: 使用 RNNT 作为辅助 ASR loss
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

    def save_aux(self, output_dir: str) -> None:
        """保存辅助头和配置。配置文件名沿用 ctc_config.json。"""
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

    def _out_lens(self, input_lengths: torch.Tensor) -> torch.Tensor:
        leave = input_lengths % 100
        feat_lengths = (leave - 1) // 2 + 1
        return ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13

    def _enc_joint(
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
                sub_hs, sub_llm_features, sub_out_lens, sub_feat_lens = self._enc_joint(
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

        out_lens = self._out_lens(feat_lens)
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

    def _aux_loss(self, hs_pad, out_lens, target_ids, target_lengths):
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
        use_streaming_aux_train = bool(self.training and self.aux_streaming_train)
        if use_streaming_aux_train:
            hs_pad, out_lens = self._enc_train_stream(
                input_features,
                feature_attention_mask,
            )
            audio_features_for_llm = None
            if not self.ctc_only:
                _, audio_features_for_llm, _, _ = self._enc_joint(
                    input_features,
                    feature_attention_mask,
                    need_llm_features=True,
                )
        else:
            hs_pad, audio_features_for_llm, out_lens, _ = self._enc_joint(
                input_features,
                feature_attention_mask,
                need_llm_features=not self.ctc_only,
            )

        aux_loss = self._aux_loss(hs_pad, out_lens, ctc_target_ids, ctc_target_lengths)

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
