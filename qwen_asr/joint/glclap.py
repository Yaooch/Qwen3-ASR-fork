# coding: utf-8
"""GLCLAP 音频-文本全局/局部对比学习。"""
import math
from contextlib import nullcontext
from typing import Dict, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


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


def _gather_grad(x: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return x
    from torch.distributed.nn.functional import all_gather

    return torch.cat(all_gather(x), dim=0)


def _gather_mask(mask: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return mask
    parts = [torch.empty_like(mask) for _ in range(dist.get_world_size())]
    dist.all_gather(parts, mask)
    return torch.cat(parts, dim=0)


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
    if any(int(x.item()) != int(batch.item()) for x in batches):
        raise RuntimeError("GLCLAP 多卡训练要求每个 rank 的 batch size 相同。")

    max_time = torch.tensor(audio_local.shape[1], device=audio_local.device)
    dist.all_reduce(max_time, op=dist.ReduceOp.MAX)
    pad = int(max_time.item()) - audio_local.shape[1]
    if pad:
        audio_local = F.pad(audio_local, (0, 0, 0, pad))
        audio_mask = F.pad(audio_mask, (0, pad), value=False)

    return (
        _gather_grad(text_global),
        _gather_grad(text_local),
        _gather_grad(audio_global),
        _gather_grad(audio_local),
        _gather_mask(audio_mask),
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


class GLCLAPModel(nn.Module):
    """Data2Vec Audio Large + multilingual BERT 的 GLCLAP。"""

    def __init__(
        self,
        audio_model: str,
        text_model: str,
        embed_dim: int = 512,
        unfreeze_audio_layers: int = 0,
        unfreeze_text_layers: int = 0,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        from transformers import BertModel, Data2VecAudioModel

        self.audio_encoder = Data2VecAudioModel.from_pretrained(audio_model, local_files_only=True)
        self.text_encoder = BertModel.from_pretrained(
            text_model, local_files_only=True, add_pooling_layer=False
        )
        self.audio_projection = nn.Linear(self.audio_encoder.config.hidden_size, embed_dim, bias=False)
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

    def _set_trainable_layers(self, audio_layers: int, text_layers: int) -> None:
        for param in self.audio_encoder.parameters():
            param.requires_grad = audio_layers < 0
        for param in self.text_encoder.parameters():
            param.requires_grad = text_layers < 0
        if audio_layers >= 0:
            self._unfreeze(self.audio_encoder.encoder.layers, audio_layers)
            if audio_layers > 0:
                for param in self.audio_encoder.encoder.layer_norm.parameters():
                    param.requires_grad = True
        if text_layers >= 0:
            self._unfreeze(self.text_encoder.encoder.layer, text_layers)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.audio_frozen:
            self.audio_encoder.eval()
        if self.text_frozen:
            self.text_encoder.eval()
        return self

    def encode_audio(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context = torch.no_grad() if self.audio_frozen else nullcontext()
        with context:
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
        projected = self.audio_projection(hidden)
        local = F.normalize(projected, dim=-1)
        global_ = F.normalize(masked_mean(projected, mask), dim=-1)
        return global_, local, mask

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
        input_values: torch.Tensor,
        audio_attention_mask: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        subtext_input_ids: torch.Tensor,
        subtext_attention_mask: torch.Tensor,
        gather: bool = True,
    ) -> Dict[str, torch.Tensor]:
        audio_global, audio_local, audio_mask = self.encode_audio(
            input_values, audio_attention_mask
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
