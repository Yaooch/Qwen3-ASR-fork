# finetuning/grpo_math.py
"""GRPO 纯数学：组内优势 + clip surrogate + KL。"""
import torch

EPS_CLIP = 0.2


def group_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """rewards: (B, G) → 组内归一化优势 (B, G)。"""
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True, unbiased=False)
    return (rewards - mean) / (std + 1e-8)


def grpo_loss(
    logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    ref_logp: torch.Tensor,
    beta: float = 0.04,
) -> torch.Tensor:
    """token-level：logp/old_logp/ref_logp/advantages 同 shape (T,)。
    返回标量 loss = -mean(clip_surrogate) + beta * mean(KL)。
    KL 用 (logp - ref_logp) 作为惩罚项（简化估计，带梯度）。
    on-policy 单步训练中 old_logp = logp.detach()，ratio≡1、clip 为多 epoch 占位；
    KL 项把 LoRA 拉回基线 talker，护住非热词能力。"""
    ratio = torch.exp(logp - old_logp)
    clipped = torch.clamp(ratio, 1.0 - EPS_CLIP, 1.0 + EPS_CLIP)
    surrogate = torch.min(ratio * advantages, clipped * advantages)
    kl = logp - ref_logp
    return -surrogate.mean() + beta * kl.mean()
