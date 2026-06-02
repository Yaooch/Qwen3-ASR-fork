# qwen_asr/joint/model.py
import json
import os
import shutil
from typing import Dict, Iterable, Optional, Union

import torch
import torch.nn as nn
from qwen_asr import Qwen3ASRModel
from .ctc import CTC
from .rnnt import RNNT
from .decode import DecodeMixin
from .stream import StreamMixin
from .defaults import (
    ENCODER_BATCH_SIZE,
    JOINT_CONFIG,
    STREAM_CHUNK_SEC,
    STREAM_LEFT_CHUNKS,
)


def ctc_cfg(cfg: Dict) -> Dict:
    """从 joint_config 还原 CTC adapter，旧 checkpoint 默认使用 MLP。"""
    return {"adapter_type": cfg.get("ctc_adapter", "mlp")}


def read_joint_cfg(path: Optional[str]) -> Dict:
    if not path:
        return {}
    cfg_path = os.path.join(path, JOINT_CONFIG)
    if not os.path.exists(cfg_path):
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_ctc_cfg(cfg: Dict, ctc=None, source_cfg: Optional[Dict] = None) -> None:
    if ctc is None:
        cfg["ctc_adapter"] = (source_cfg or {}).get("ctc_adapter", "mlp")
        return

    cfg["ctc_adapter"] = ctc.adapter_type


class Qwen3ASRJointModel(StreamMixin, DecodeMixin, nn.Module):
    """Qwen3-ASR + 可选 CTC/RNNT 辅助头。"""

    def __init__(
        self,
        qwen_model,
        vocab_size: int,
        vocab: Dict[str, int],
        blank_id: int = 0,
        heads: Iterable[str] = ("ctc",),
        train_tasks: Iterable[str] = ("llm", "ctc"),
        loss_weights: Optional[Dict[str, float]] = None,
        ctc_config: Optional[Dict] = None,
        stream_train: bool = False,
    ):
        super().__init__()
        self.qwen_model = qwen_model
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.blank_id = blank_id
        self._id_to_token = {v: k for k, v in vocab.items()}

        self.heads = self._clean_names(heads, allowed={"ctc", "rnnt"}, name="heads")
        self.train_tasks = self._clean_names(
            train_tasks,
            allowed={"llm", "ctc", "rnnt"},
            name="train_tasks",
        )
        self.loss_weights = {"llm": 1.0, "ctc": 1.0, "rnnt": 1.0}
        if loss_weights:
            self.loss_weights.update({k: float(v) for k, v in loss_weights.items()})
        self.encoder_batch_size = ENCODER_BATCH_SIZE
        self.stream_train = stream_train

        self.encoder_output_size = qwen_model.thinker.audio_tower.config.d_model

        self.ctc = None
        self.rnnt = None
        if "ctc" in self.heads:
            self.ctc = CTC(
                vocab_size,
                self.encoder_output_size,
                blank_id=blank_id,
                **(ctc_config or {}),
            )
        if "rnnt" in self.heads:
            self.rnnt = RNNT(vocab_size, self.encoder_output_size, blank_id=blank_id)

        self.processor = None
        self._asr_wrapper = None

        for p in self.parameters():
            p.requires_grad = True

    @staticmethod
    def _clean_names(values, allowed, name: str):
        if isinstance(values, str):
            values = values.split(",")
        out = []
        for value in values:
            item = str(value).strip().lower()
            if not item:
                continue
            if item not in allowed:
                raise ValueError(f"不支持的 {name}: {item}")
            if item not in out:
                out.append(item)
        return tuple(out)

    def _head(self, name: str):
        head = getattr(self, name, None)
        if head is None:
            raise RuntimeError(f"当前 checkpoint 没有 {name.upper()} 头。")
        return head

    def _head_dtype(self, names=None):
        for name in names or ("ctc", "rnnt"):
            head = getattr(self, name, None)
            if head is not None:
                return next(head.parameters()).dtype
        return next(self.qwen_model.parameters()).dtype

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        dtype: torch.dtype = torch.bfloat16,
        device_map: Optional[Union[str, dict]] = "auto",
        load_heads: bool = True,
        **kwargs,
    ) -> "Qwen3ASRJointModel":
        """从 joint checkpoint 加载：底座 HF 权重 + 辅助头 + joint_config.json。"""
        base = Qwen3ASRModel.from_pretrained(
            model_path,
            dtype=dtype,
            device_map=device_map,
            **kwargs,
        )

        cfg_path = os.path.join(model_path, JOINT_CONFIG)
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"未找到配置：{cfg_path}")

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if "heads" not in cfg:
            raise ValueError(f"joint checkpoint 配置缺少 heads：{cfg_path}")

        audio_tower = base.model.thinker.audio_tower
        audio_config = base.model.config.thinker_config.audio_config
        if cfg.get("audio_n_window", 0) > 0:
            audio_tower.n_window = cfg["audio_n_window"]
            audio_tower.config.n_window = cfg["audio_n_window"]
            audio_config.n_window = cfg["audio_n_window"]
        if cfg.get("audio_n_window_infer", 0) > 0:
            audio_tower.n_window_infer = cfg["audio_n_window_infer"]
            audio_tower.config.n_window_infer = cfg["audio_n_window_infer"]
            audio_config.n_window_infer = cfg["audio_n_window_infer"]

        instance = cls(
            qwen_model=base.model,
            vocab_size=cfg["vocab_size"],
            vocab=cfg.get("vocab", {}),
            blank_id=cfg.get("blank_id", 0),
            heads=cfg["heads"],
            train_tasks=("llm", *cfg["heads"]),
            ctc_config=ctc_cfg(cfg),
        )

        instance.processor = base.processor
        instance._asr_wrapper = base

        if load_heads:
            for name in instance.heads:
                state_path = os.path.join(model_path, f"{name}_head.pt")
                if not os.path.exists(state_path):
                    raise FileNotFoundError(f"未找到 {name.upper()} 权重：{state_path}")
                print(f"正在加载 {name.upper()} 头：{state_path}")
                getattr(instance, name).load_state_dict(
                    torch.load(state_path, map_location="cpu"),
                    strict=True,
                )

            ref_param = next(base.model.parameters())
            for name in instance.heads:
                getattr(instance, name).to(device=ref_param.device)

        if hasattr(instance.qwen_model, "tie_weights"):
            instance.qwen_model.tie_weights()

        return instance

    def save_aux(self, output_dir: str, heads=None, copy_heads_from: Optional[str] = None) -> None:
        """保存辅助头和 joint 结构配置。"""
        os.makedirs(output_dir, exist_ok=True)
        heads = self._clean_names(self.heads if heads is None else heads, {"ctc", "rnnt"}, "heads")

        for name in heads:
            dst = os.path.join(output_dir, f"{name}_head.pt")
            head = getattr(self, name, None)
            if head is not None:
                torch.save(head.state_dict(), dst)
                continue

            src = os.path.join(copy_heads_from or "", f"{name}_head.pt")
            if not os.path.exists(src):
                raise FileNotFoundError(f"无法保存 {name.upper()} 头，未找到可复制权重：{src}")
            shutil.copy2(src, dst)

        cfg = {
            "vocab_size": self.vocab_size,
            "encoder_output_size": self.encoder_output_size,
            "blank_id": self.blank_id,
            "heads": list(heads),
            "audio_n_window": self.qwen_model.thinker.audio_tower.n_window,
            "audio_n_window_infer": self.qwen_model.thinker.audio_tower.n_window_infer,
            "vocab": self.vocab,
        }
        if "ctc" in heads:
            save_ctc_cfg(cfg, self.ctc, read_joint_cfg(copy_heads_from))
        with open(os.path.join(output_dir, JOINT_CONFIG), "w", encoding="utf-8") as f:
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
            encoder_batch_size = self.encoder_batch_size
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

        enc, aux_features = audio_tower(
            input_features=concat_features,
            feature_lens=feat_lens,
            return_pre_proj=True,
        )
        pre_final = enc.last_hidden_state
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

    def _stream_train_mask(
        self,
        input_features: torch.Tensor,
        feature_attention_mask: Optional[torch.Tensor] = None,
    ):
        """按 WeNet 式 chunk mask 编码整条音频，当前 chunk 只看左侧固定 chunks。"""
        batch_size = input_features.shape[0]
        device = input_features.device
        if feature_attention_mask is not None:
            feat_lens = feature_attention_mask.sum(dim=1).long()
        else:
            feat_lens = torch.full((batch_size,), input_features.shape[2], dtype=torch.long, device=device)

        chunk = self._sec_to_feature_count(STREAM_CHUNK_SEC, min_value=1)
        audio_tower = self.qwen_model.thinker.audio_tower
        hs_pad, out_lens = audio_tower.forward_stream_mask(
            input_features,
            feat_lens,
            chunk_size=chunk,
            left_chunks=STREAM_LEFT_CHUNKS,
        )
        return hs_pad, out_lens, feat_lens

    def _project_llm_features(self, hs_pad: torch.Tensor, out_lens: torch.Tensor) -> torch.Tensor:
        audio_tower = self.qwen_model.thinker.audio_tower
        pieces = [hs_pad[b, : int(out_lens[b].item())] for b in range(hs_pad.shape[0])]
        pre_final = torch.cat(pieces, dim=0)
        return audio_tower.proj2(audio_tower.act(audio_tower.proj1(pre_final)))

    def _aux_loss(self, name: str, hs_pad, out_lens, target_ids, target_lengths):
        if target_ids is None:
            return torch.tensor(0.0, device=hs_pad.device)
        return self._head(name)(hs_pad, out_lens, target_ids, target_lengths)

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
        tasks=None,
        **kwargs,
    ):
        tasks = self._clean_names(tasks or self.train_tasks, {"llm", "ctc", "rnnt"}, "tasks")
        aux_tasks = [name for name in ("ctc", "rnnt") if name in tasks]
        need_llm = "llm" in tasks

        audio_features_for_llm = None
        if self.stream_train and aux_tasks:
            hs_pad, out_lens, _ = self._stream_train_mask(input_features, feature_attention_mask)
            if need_llm:
                audio_features_for_llm = self._project_llm_features(hs_pad, out_lens)
        else:
            hs_pad, audio_features_for_llm, out_lens, _ = self._enc_joint(
                input_features,
                feature_attention_mask,
                need_llm_features=need_llm,
            )

        outputs = {"output_lengths": out_lens}
        losses = []
        for name in aux_tasks:
            head_hs = hs_pad.to(self._head_dtype([name]))
            cur_loss = self._aux_loss(name, head_hs, out_lens, ctc_target_ids, ctc_target_lengths)
            outputs[f"{name}_loss"] = cur_loss
            losses.append(self.loss_weights.get(name, 1.0) * cur_loss)
            if name == "ctc" and not self.training:
                outputs["log_probs"] = self.ctc.log_softmax(head_hs, out_lens)

        if need_llm:
            embeds = self.qwen_model.thinker.get_input_embeddings()(input_ids)
            audio_mask = self.qwen_model.thinker.get_placeholder_mask(input_ids, embeds)
            embeds = embeds.masked_scatter(audio_mask, audio_features_for_llm.to(embeds.dtype))
            llm_out = self.qwen_model.thinker(
                inputs_embeds=embeds,
                attention_mask=attention_mask,
                labels=labels,
            )
            outputs["llm_loss"] = llm_out.loss
            losses.append(self.loss_weights.get("llm", 1.0) * llm_out.loss)

        if not losses:
            raise RuntimeError("tasks 为空，无法计算 loss。")

        outputs["loss"] = sum(losses)
        return outputs
