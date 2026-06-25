# finetuning/grpo_data.py
"""读 ContextASR jsonl → GRPO 训练样本，切 eval。"""
import json
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

from qwen_asr.tools.hotword_reward import parse_hotword_list, parse_text_field


@dataclass(frozen=True)
class GrpoSample:
    audio: str
    gt_text: str  # parse + normalize 后纯转写
    hotwords: List[str]  # normalize 后热词列表
    prompt: str  # 原 prompt 字段（rollout 时原样用作 context）


def load_samples(jsonl_path: str, limit: Optional[int] = None) -> List[GrpoSample]:
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
    rng = random.Random(seed)
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    n_eval = int(len(idx) * eval_ratio)
    eval_idx = set(idx[:n_eval])
    train = [samples[i] for i in range(len(samples)) if i not in eval_idx]
    eval_ = [samples[i] for i in idx[:n_eval]]
    return train, eval_
