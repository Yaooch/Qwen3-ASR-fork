# finetuning/grpo_core.py
"""GRPO 共用轻量工具：训练样本读写、组内优势与 clip surrogate 损失、LoRA 装配。

不在模块顶部 import 重模型，便于单测快速加载。apply_lora 内部惰性 import peft。
"""
import json
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set, Tuple

import torch

from qwen_asr.tools.hotword_reward import parse_hotword_list, parse_text_field

# --------------------------------------------------------------------------
# 数据
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GrpoSample:
    audio: str
    gt_text: str  # parse + normalize 后纯转写
    hotwords: List[str]  # normalize 后热词列表
    prompt: str  # 原 prompt 字段（rollout 时原样用作 context）


def load_samples(jsonl_path: str, limit: Optional[int] = None) -> List[GrpoSample]:
    """读 ContextASR jsonl → GRPO 训练样本。"""
    samples: List[GrpoSample] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            samples.append(
                GrpoSample(
                    audio=r["audio"],
                    gt_text=parse_text_field(r.get("text", "")),
                    hotwords=parse_hotword_list(r.get("prompt", "")),
                    prompt=r.get("prompt", ""),
                )
            )
    return samples


def split_eval(
    samples: List[GrpoSample], eval_ratio: float = 0.02, seed: int = 42
) -> Tuple[List[GrpoSample], List[GrpoSample]]:
    """随机留 eval_ratio 比例作 eval，其余作 train。"""
    rng = random.Random(seed)
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    n_eval = int(len(idx) * eval_ratio)
    eval_idx = set(idx[:n_eval])
    train = [samples[i] for i in range(len(samples)) if i not in eval_idx]
    eval_ = [samples[i] for i in idx[:n_eval]]
    return train, eval_


# --------------------------------------------------------------------------
# GRPO 数学
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# LoRA 装配
# --------------------------------------------------------------------------

# thinker 文本解码器的注意力与 MLP 投影。audio_tower 的注意力层同名(q_proj 等)，
# 故用正则限定到 thinker.model.layers 下，避免误挂音频侧。
TEXT_DECODER_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
# peft 用 re.fullmatch 匹配整条模块路径。前缀可选：训练时包 joint（有 base_model.model.qwen_model. 前缀），
# 评测时包 qwen_model（无前缀，键为 thinker.model...），两种都需命中；audio_tower 同名层靠 `.model.layers` 排除。
TEXT_DECODER_TARGET_REGEX = (
    r"(?:.*\.)?thinker\.model\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)"
    r"|mlp\.(?:gate_proj|up_proj|down_proj))"
)


def apply_lora(
    joint,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
):
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=TEXT_DECODER_TARGET_REGEX,
        bias="none",
        task_type=None,  # Qwen3-ASR 自带 generate，不走 peft 的 CausalLM 生成路径
    )
    # 先冻结全部，peft 再解冻 LoRA
    for p in joint.parameters():
        p.requires_grad_(False)
    peft_model = get_peft_model(joint, cfg)
    assert_only_text_decoder_trainable(peft_model)
    return peft_model


def assert_only_text_decoder_trainable(peft_model) -> None:
    """可训参数必须全在 thinker.model 下（文本解码器），不得误挂 audio_tower / heads。"""
    bad = []
    for name, p in peft_model.named_parameters():
        if p.requires_grad and "thinker.model" not in name:
            bad.append(name)
    if bad:
        raise RuntimeError(
            f"LoRA 误挂到非文本解码器: {bad[:5]}（共 {len(bad)} 个）"
        )
