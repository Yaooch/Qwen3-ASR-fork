"""热词 GRPO 的样本、奖励与优化数学。"""

import json
import re
from dataclasses import dataclass
from typing import List, Sequence, Set, Tuple

import editdistance
import torch

# --------------------------------------------------------------------------
# 数据
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GrpoSample:
    audio: str
    gt_text: str  # parse + normalize 后纯转写
    hotwords: List[str]  # normalize 后热词列表
    prompt: str  # 原 prompt 字段（rollout 时原样用作 context）


def load_samples(jsonl_path: str) -> List[GrpoSample]:
    """读 ContextASR jsonl → GRPO 训练样本。"""
    samples: List[GrpoSample] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
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
    KL 用 GRPO 的非负采样估计：exp(ref-logp) - (ref-logp) - 1。
    old_logp 固定为 rollout behavior policy；多轮更新后 ratio 变化，clip 才实际生效；
    KL 项把 LoRA 拉回基线 talker，护住非热词能力。"""
    ratio = torch.exp(logp - old_logp)
    clipped = torch.clamp(ratio, 1.0 - EPS_CLIP, 1.0 + EPS_CLIP)
    surrogate = torch.min(ratio * advantages, clipped * advantages)
    ref_minus_logp = ref_logp - logp
    kl = torch.exp(ref_minus_logp) - ref_minus_logp - 1.0
    return -surrogate.mean() + beta * kl.mean()


# --------------------------------------------------------------------------
# 奖励
# --------------------------------------------------------------------------
_PUNCT_RE = re.compile(r"[!-/:-@\[-`{-~　-〿＀-￯ -⁯]")
_WS_RE = re.compile(r"\s+")
_CJK_RE = re.compile(r"^[\u4e00-\u9fff]+$")

DEFAULT_WEIGHTS = {"w_r": 1.0, "w_f": 1.0, "w_mix": 0.5, "w_c": 1.0, "w_fmt": 0.5}


def normalize(text: str) -> str:
    if not text:
        return ""
    s = _PUNCT_RE.sub(" ", str(text))
    s = _WS_RE.sub(" ", s).strip()
    return s.lower()


def parse_text_field(raw: str) -> str:
    """从训练数据 text 字段剥 language X<asr_text> 前缀，返回归一化纯转写。"""
    if raw is None:
        return ""
    s = str(raw).strip()
    tag = "<asr_text>"
    if tag in s:
        s = s.split(tag, 1)[1]
    return normalize(s)


def parse_hotword_list(prompt: str) -> List[str]:
    """从 prompt 抠 专属名词：[a，b，c] 词表，去标点归一。"""
    if not prompt:
        return []
    m = re.search(r"专属名词：\[([^\]]*)\]", str(prompt))
    if not m:
        return []
    inner = m.group(1)
    parts = re.split(r"[，,]", inner)
    words = [normalize(p) for p in parts]
    return [w for w in words if w]


def _substring_match(hotword: str, text: str) -> bool:
    return bool(hotword) and hotword in text


def split_truth(
    hotwords: Sequence[str], gt_text: str
) -> Tuple[Set[str], Set[str]]:
    """T = 列表中出现在 gt 的；D = 其余。基于 normalize 后子串。"""
    gt = normalize(gt_text)
    true_set, dist = set(), set()
    for h in hotwords:
        h = normalize(h)
        (true_set if _substring_match(h, gt) else dist).add(h)
    return true_set, dist


def hotword_recall(output: str, true_hw: Set[str]) -> float:
    if not true_hw:
        return 0.0
    out = normalize(output)
    hit = sum(1 for h in true_hw if _substring_match(h, out))
    return hit / max(1, len(true_hw))


def false_injection_rate(output: str, distractors: Set[str]) -> float:
    if not distractors:
        return 0.0
    out = normalize(output)
    hit = sum(1 for h in distractors if _substring_match(h, out))
    return hit / max(1, len(distractors))


def _compact(text: str) -> str:
    return _WS_RE.sub("", normalize(text))


def _cjk_word(text: str) -> str:
    s = _compact(text)
    return s if len(s) >= 2 and _CJK_RE.fullmatch(s) else ""


def _hybrid_words(a: str, b: str) -> Set[str]:
    if len(a) != len(b):
        return set()
    words = set()
    for i in range(1, len(a)):
        words.add(a[:i] + b[i:])
        words.add(b[:i] + a[i:])
    words.discard(a)
    words.discard(b)
    return words


def hybrid_injection_rate(output: str, true_hw: Set[str], distractors: Set[str]) -> float:
    """惩罚真热词和干扰词片段拼出的中文混合词。"""
    true_words = [_cjk_word(w) for w in true_hw]
    dist_words = [_cjk_word(w) for w in distractors]
    true_words = [w for w in true_words if w]
    dist_words = [w for w in dist_words if w]
    if not true_words or not dist_words:
        return 0.0
    out = _compact(output)
    hit = 0
    for t in true_words:
        mixed = set()
        for d in dist_words:
            mixed.update(_hybrid_words(t, d))
        hit += int(any(w in out for w in mixed))
    return hit / max(1, len(true_words))


def _mask_all(text: str, words: Sequence[str]) -> str:
    s = normalize(text)
    # 按长度降序避免短词遮蔽长词
    for w in sorted({normalize(w) for w in words if normalize(w)}, key=len, reverse=True):
        if w:
            s = s.replace(w, " ")
    return _WS_RE.sub(" ", s).strip()


def non_hotword_cer(output: str, gt_text: str, hotwords: Sequence[str]) -> float:
    pred = _mask_all(output, hotwords)
    ref = _mask_all(gt_text, hotwords)
    if not ref:
        return 0.0
    return editdistance.eval(pred, ref) / len(ref)


def _format_ok(output: str) -> bool:
    s = str(output or "").strip()
    if not s:
        return False
    if "专属名词" in s:
        return False
    if "[" in s and "]" in s:
        return False
    return True


def compute_reward(
    output: str,
    gt_text: str,
    hotwords: Sequence[str],
    weights: dict | None = None,
) -> float:
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    T, D = split_truth(hotwords, gt_text)
    recall = hotword_recall(output, T)
    fp = false_injection_rate(output, D)
    mix = hybrid_injection_rate(output, T, D)
    cer_nh = non_hotword_cer(output, gt_text, hotwords)
    fmt = 1.0 if _format_ok(output) else 0.0
    return (
        w["w_r"] * recall
        - w["w_f"] * fp
        - w["w_mix"] * mix
        - w["w_c"] * cer_nh
        - w["w_fmt"] * (1.0 - fmt)
    )
