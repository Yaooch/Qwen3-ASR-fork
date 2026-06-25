# finetuning/grpo_eval.py
"""评测 GRPO LoRA：热词召回 / 误注入率 / 非热词 CER / 整体 CER。

不传 --lora 评基线；传 --lora 评 RL 后。对比两者判断成功标准是否达成。
"""
import argparse

import editdistance
import torch

from qwen_asr.joint import Qwen3ASRJointModel
from qwen_asr.joint.defaults import DEFAULT_ATTN_IMPLEMENTATION
from qwen_asr.tools.hotword_reward import (
    false_injection_rate,
    hotword_recall,
    non_hotword_cer,
    normalize,
    split_truth,
)
from finetuning.grpo_core import load_samples

CKPT_DEFAULT = "/cfs/data/private/WangYaoChi/model/qwen3-asr-ctc-joint-14-hotword-1/checkpoint-228"
DATA_DEFAULT = "/cfs/data/private/WangYaoChi/train_data/all/contextasr/train_contextasr2.jsonl"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=CKPT_DEFAULT)
    p.add_argument("--data", default=DATA_DEFAULT)
    p.add_argument("--lora", default=None, help="RL 训出的 LoRA 目录；不传则评基线")
    p.add_argument("--limit", type=int, default=200)
    return p.parse_args()


def overall_cer(output: str, gt: str) -> float:
    o, g = normalize(output), normalize(gt)
    return editdistance.eval(o, g) / max(1, len(g))


def main():
    args = parse_args()
    joint = Qwen3ASRJointModel.from_pretrained(
        args.ckpt,
        dtype=torch.bfloat16,
        device_map=None,
        load_heads=False,
        attn_implementation=DEFAULT_ATTN_IMPLEMENTATION,
    ).to("cuda")
    joint.eval()
    if args.lora:
        from peft import PeftModel

        # 与训练一致：包整个 joint，使 adapter key 与 target_modules 都对齐
        joint = PeftModel.from_pretrained(joint, args.lora)
        joint.eval()

    samples = load_samples(args.data, limit=args.limit)
    agg = {"recall": [], "fp": [], "cer_nh": [], "cer": []}
    for i, s in enumerate(samples, 1):
        rec = joint.transcribe(s.audio, modes="llm", prompt=s.prompt)
        out = rec.get("text", "")
        T, D = split_truth(s.hotwords, s.gt_text)
        agg["recall"].append(hotword_recall(out, T))
        agg["fp"].append(false_injection_rate(out, D))
        agg["cer_nh"].append(non_hotword_cer(out, s.gt_text, s.hotwords))
        agg["cer"].append(overall_cer(out, s.gt_text))
        if i % 10 == 0:
            print(f"... {i}/{len(samples)}")

    label = "RL" if args.lora else "baseline"
    print(f"=== {label} (n={len(samples)}) ===")
    for k in ("recall", "fp", "cer_nh", "cer"):
        v = agg[k]
        print(f"{k}: mean={sum(v) / len(v):.4f} n={len(v)}")


if __name__ == "__main__":
    main()
