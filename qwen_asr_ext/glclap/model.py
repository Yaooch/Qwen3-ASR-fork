# coding: utf-8
"""GLCLAP 音频-文本全局/局部对比学习。"""
import math
from contextlib import nullcontext
from typing import Dict, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from qwen_asr_ext.joint.defaults import (
    TRAIN_MASK_CURRENT_FRAMES,
    TRAIN_MASK_LEFT_FRAMES,
    TRAIN_MASK_RIGHT_FRAMES,
)


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """沿序列维做 masked mean。"""
    weight = mask.to(dtype=x.dtype).unsqueeze(-1)
    return (x * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)


def feature_mask(
    attention_mask: torch.Tensor,
    output_length: int,
    conv_kernel,
    conv_stride,
) -> torch.Tensor:
    """把 waveform mask 换算为 Data2Vec encoder 帧 mask。"""
    lengths = attention_mask.long().sum(dim=-1)
    for kernel, stride in zip(conv_kernel, conv_stride):
        lengths = torch.div(lengths - kernel, stride, rounding_mode="floor") + 1
    lengths = lengths.clamp(min=0, max=output_length)
    steps = torch.arange(output_length, device=attention_mask.device)
    return steps.unsqueeze(0) < lengths.unsqueeze(1)


def _pad_batch(x: torch.Tensor, size: int) -> torch.Tensor:
    if x.shape[0] == size:
        return x
    padding = x.new_zeros((size - x.shape[0], *x.shape[1:]))
    return torch.cat((x, padding), dim=0)


def _gather_grad(x: torch.Tensor, sizes) -> torch.Tensor:
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return x
    from torch.distributed.nn.functional import all_gather

    padded = _pad_batch(x, max(sizes))
    return torch.cat(
        [part[:size] for part, size in zip(all_gather(padded), sizes)],
        dim=0,
    )


def _gather_mask(mask: torch.Tensor, sizes) -> torch.Tensor:
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return mask
    padded = _pad_batch(mask, max(sizes))
    parts = [torch.empty_like(padded) for _ in range(dist.get_world_size())]
    dist.all_gather(parts, padded)
    return torch.cat([part[:size] for part, size in zip(parts, sizes)], dim=0)


def gather_embeddings(
    text_global: torch.Tensor,
    text_local: torch.Tensor,
    audio_global: torch.Tensor,
    audio_local: torch.Tensor,
    audio_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """带梯度聚合 embedding；时间序列先跨卡补到同一长度。"""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return text_global, text_local, audio_global, audio_local, audio_mask

    batch = torch.tensor([text_global.shape[0]], device=text_global.device)
    batches = [torch.empty_like(batch) for _ in range(dist.get_world_size())]
    dist.all_gather(batches, batch)
    sizes = [int(x.item()) for x in batches]

    max_time = torch.tensor(audio_local.shape[1], device=audio_local.device)
    dist.all_reduce(max_time, op=dist.ReduceOp.MAX)
    pad = int(max_time.item()) - audio_local.shape[1]
    if pad:
        audio_local = F.pad(audio_local, (0, 0, 0, pad))
        audio_mask = F.pad(audio_mask, (0, pad), value=False)

    return (
        _gather_grad(text_global, sizes),
        _gather_grad(text_local, sizes),
        _gather_grad(audio_global, sizes),
        _gather_grad(audio_local, sizes),
        _gather_mask(audio_mask, sizes),
    )


def glclap_loss(
    text_global: torch.Tensor,
    text_local: torch.Tensor,
    audio_global: torch.Tensor,
    audio_local: torch.Tensor,
    audio_mask: torch.Tensor,
    logit_scale: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """计算论文的双向 global + local InfoNCE。"""
    scale = logit_scale.exp().clamp(max=100.0)
    global_logits = scale * text_global @ audio_global.transpose(0, 1)

    local_sim = torch.einsum("id,jtd->ijt", text_local, audio_local)
    local_sim = local_sim.masked_fill(~audio_mask.unsqueeze(0), torch.finfo(local_sim.dtype).min)
    local_logits = scale * local_sim.amax(dim=-1)

    labels = torch.arange(global_logits.shape[0], device=global_logits.device)
    global_loss = 0.5 * (
        F.cross_entropy(global_logits, labels)
        + F.cross_entropy(global_logits.transpose(0, 1), labels)
    )
    local_loss = 0.5 * (
        F.cross_entropy(local_logits, labels)
        + F.cross_entropy(local_logits.transpose(0, 1), labels)
    )
    return {
        "loss": global_loss + local_loss,
        "global_loss": global_loss,
        "local_loss": local_loss,
        "global_logits": global_logits,
        "local_logits": local_logits,
    }


class GLCLAPHead(nn.Module):
    """接收已有音频 Encoder 输出的 GLCLAP 训练头。"""

    def __init__(
        self,
        text_model: str,
        audio_dim: int,
        embed_dim: int = 512,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        from transformers import BertModel

        self.text_model = text_model
        self.embed_dim = embed_dim
        self.text_encoder = BertModel.from_pretrained(
            text_model, local_files_only=True, add_pooling_layer=False
        )
        self.audio_projection = nn.Linear(audio_dim, embed_dim, bias=False)
        self.text_projection = nn.Linear(
            self.text_encoder.config.hidden_size, embed_dim, bias=False
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))
        if gradient_checkpointing:
            self.text_encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        pooled = masked_mean(hidden, attention_mask.bool())
        return F.normalize(self.text_projection(pooled), dim=-1)

    def forward(
        self,
        audio_hidden: torch.Tensor,
        audio_mask: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        subtext_input_ids: torch.Tensor,
        subtext_attention_mask: torch.Tensor,
        gather: bool = True,
    ) -> Dict[str, torch.Tensor]:
        projected = self.audio_projection(audio_hidden)
        audio_local = F.normalize(projected, dim=-1)
        audio_global = F.normalize(masked_mean(projected, audio_mask), dim=-1)
        text_global = self.encode_text(text_input_ids, text_attention_mask)
        text_local = self.encode_text(subtext_input_ids, subtext_attention_mask)
        if gather:
            text_global, text_local, audio_global, audio_local, audio_mask = gather_embeddings(
                text_global, text_local, audio_global, audio_local, audio_mask
            )
        return glclap_loss(
            text_global,
            text_local,
            audio_global,
            audio_local,
            audio_mask,
            self.logit_scale,
        )


class GLCLAPModel(nn.Module):
    """Data2Vec 或 Qwen3-ASR Audio Encoder + multilingual BERT。"""

    def __init__(
        self,
        audio_model: str,
        text_model: str,
        embed_dim: int = 512,
        unfreeze_audio_layers: int = 0,
        unfreeze_text_layers: int = 0,
        gradient_checkpointing: bool = True,
        audio_backend: str = "data2vec",
        stream_left_frames: int = TRAIN_MASK_LEFT_FRAMES,
        stream_current_frames: int = TRAIN_MASK_CURRENT_FRAMES,
        stream_right_frames: int = TRAIN_MASK_RIGHT_FRAMES,
    ):
        super().__init__()
        from transformers import BertModel, Data2VecAudioModel

        self.audio_backend = audio_backend
        self.stream_left_frames = stream_left_frames
        self.stream_current_frames = stream_current_frames
        self.stream_right_frames = stream_right_frames
        if audio_backend == "data2vec":
            self.audio_encoder = Data2VecAudioModel.from_pretrained(
                audio_model, local_files_only=True
            )
            audio_dim = self.audio_encoder.config.hidden_size
        elif audio_backend == "qwen":
            from qwen_asr import Qwen3ASRModel
            from qwen_asr_ext.joint.defaults import DEFAULT_ATTN_IMPLEMENTATION

            wrapper = Qwen3ASRModel.from_pretrained(
                audio_model,
                dtype=torch.bfloat16,
                device_map=None,
                local_files_only=True,
                attn_implementation=DEFAULT_ATTN_IMPLEMENTATION,
            )
            self.audio_encoder = wrapper.model.thinker.audio_tower.float()
            audio_dim = self.audio_encoder.config.d_model
            del wrapper
        else:
            raise ValueError(f"不支持的 audio_backend: {audio_backend}")

        self.text_encoder = BertModel.from_pretrained(
            text_model, local_files_only=True, add_pooling_layer=False
        )
        self.audio_projection = nn.Linear(audio_dim, embed_dim, bias=False)
        self.text_projection = nn.Linear(self.text_encoder.config.hidden_size, embed_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

        self._set_trainable_layers(unfreeze_audio_layers, unfreeze_text_layers)
        self.audio_frozen = not any(p.requires_grad for p in self.audio_encoder.parameters())
        self.text_frozen = not any(p.requires_grad for p in self.text_encoder.parameters())
        if gradient_checkpointing and not self.audio_frozen:
            self.audio_encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        if gradient_checkpointing and not self.text_frozen:
            self.text_encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        self.train()

    @staticmethod
    def _unfreeze(layers, count: int) -> None:
        if count < 0:
            selected = layers
        elif count > 0:
            selected = layers[-count:]
        else:
            selected = []
        for layer in selected:
            for param in layer.parameters():
                param.requires_grad = True

    def _audio_layers(self):
        if self.audio_backend == "qwen":
            return self.audio_encoder.layers
        return self.audio_encoder.encoder.layers

    def _audio_norm(self):
        if self.audio_backend == "qwen":
            return self.audio_encoder.ln_post
        return self.audio_encoder.encoder.layer_norm

    def _set_trainable_layers(self, audio_layers: int, text_layers: int) -> None:
        for param in self.audio_encoder.parameters():
            param.requires_grad = audio_layers < 0
        for param in self.text_encoder.parameters():
            param.requires_grad = text_layers < 0
        if audio_layers >= 0:
            self._unfreeze(self._audio_layers(), audio_layers)
            if audio_layers > 0:
                for param in self._audio_norm().parameters():
                    param.requires_grad = True
        if self.audio_backend == "qwen":
            for module in (self.audio_encoder.proj1, self.audio_encoder.proj2):
                for param in module.parameters():
                    param.requires_grad = False
        if text_layers >= 0:
            self._unfreeze(self.text_encoder.encoder.layer, text_layers)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.audio_frozen:
            self.audio_encoder.eval()
        if self.text_frozen:
            self.text_encoder.eval()
        return self

    def extract_audio_features(
        self,
        input_values: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        input_features: torch.Tensor = None,
        feature_attention_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        context = torch.no_grad() if self.audio_frozen else nullcontext()
        with context:
            if self.audio_backend == "data2vec":
                hidden = self.audio_encoder(
                    input_values=input_values,
                    attention_mask=attention_mask,
                ).last_hidden_state
                mask = feature_mask(
                    attention_mask,
                    hidden.shape[1],
                    self.audio_encoder.config.conv_kernel,
                    self.audio_encoder.config.conv_stride,
                )
            else:
                from qwen_asr_ext.joint.encoder import encode_train_mask, feature_lens

                lengths = feature_lens(input_features, feature_attention_mask)
                hidden, _, lengths = encode_train_mask(
                    self.audio_encoder,
                    input_features,
                    lengths,
                    self.stream_left_frames,
                    self.stream_current_frames,
                    self.stream_right_frames,
                    need_llm=False,
                )
                steps = torch.arange(hidden.shape[1], device=hidden.device)
                mask = steps.unsqueeze(0) < lengths.unsqueeze(1)
        return hidden, mask

    def encode_audio_features(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        projected = self.audio_projection(hidden)
        local = F.normalize(projected, dim=-1)
        global_ = F.normalize(masked_mean(projected, mask), dim=-1)
        return global_, local, mask

    def encode_audio(
        self,
        input_values: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        input_features: torch.Tensor = None,
        feature_attention_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, mask = self.extract_audio_features(
            input_values,
            attention_mask,
            input_features,
            feature_attention_mask,
        )
        return self.encode_audio_features(hidden, mask)

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        context = torch.no_grad() if self.text_frozen else nullcontext()
        with context:
            hidden = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state
        return F.normalize(self.text_projection(masked_mean(hidden, attention_mask.bool())), dim=-1)

    def forward(
        self,
        input_values: torch.Tensor = None,
        audio_attention_mask: torch.Tensor = None,
        input_features: torch.Tensor = None,
        feature_attention_mask: torch.Tensor = None,
        text_input_ids: torch.Tensor = None,
        text_attention_mask: torch.Tensor = None,
        subtext_input_ids: torch.Tensor = None,
        subtext_attention_mask: torch.Tensor = None,
        gather: bool = True,
    ) -> Dict[str, torch.Tensor]:
        audio_global, audio_local, audio_mask = self.encode_audio(
            input_values,
            audio_attention_mask,
            input_features,
            feature_attention_mask,
        )
        text_global = self.encode_text(text_input_ids, text_attention_mask)
        text_local = self.encode_text(subtext_input_ids, subtext_attention_mask)
        if gather:
            text_global, text_local, audio_global, audio_local, audio_mask = gather_embeddings(
                text_global, text_local, audio_global, audio_local, audio_mask
            )
        return glclap_loss(
            text_global,
            text_local,
            audio_global,
            audio_local,
            audio_mask,
            self.logit_scale,
        )
