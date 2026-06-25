# finetuning/grpo_rollout_smoke.py
"""手动跑：python -m finetuning.grpo_rollout_smoke --ckpt <ckpt> --data <jsonl>
验证 rollout 采样 + ref logp + 训练 logp 三条路径都通。"""
import argparse

import torch

from qwen_asr.joint import Qwen3ASRJointModel
from qwen_asr.joint.defaults import DEFAULT_ATTN_IMPLEMENTATION
from finetuning.grpo_data import load_samples
from finetuning.grpo_lora import apply_lora
from finetuning.grpo_rollout import RolloutSampler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        default="/cfs/data/private/WangYaoChi/model/qwen3-asr-ctc-joint-14-hotword-1/checkpoint-228",
    )
    ap.add_argument(
        "--data",
        default="/cfs/data/private/WangYaoChi/train_data/all/contextasr/train_contextasr2.jsonl",
    )
    ap.add_argument("--group_size", type=int, default=2)
    args = ap.parse_args()

    joint = Qwen3ASRJointModel.from_pretrained(
        args.ckpt,
        dtype=torch.bfloat16,
        device_map=None,
        load_heads=False,
        attn_implementation=DEFAULT_ATTN_IMPLEMENTATION,
    ).to("cuda")
    joint.eval()
    asr_wrapper = joint._asr_wrapper
    peft = apply_lora(joint)
    sampler = RolloutSampler(
        peft, joint.processor, asr_wrapper, group_size=args.group_size
    )
    samples = load_samples(args.data, limit=1)
    s = samples[0]
    print("audio:", s.audio.split("/")[-1])
    print("gt:", s.gt_text[:80])
    print("hotwords:", s.hotwords[:5], "...")

    rs = sampler.sample(s)
    for i, r in enumerate(rs):
        print(f"--- rollout {i} ---")
        print("text:", repr(r.text[:120]))
        print("ref_logp[:4]:", tuple(float(x) for x in r.logp_ref[:4]))
        print("ref_logp finite:", bool(torch.isfinite(r.logp_ref).all()))

    logp = sampler.token_logp(s, rs[0].ids)
    print("train logp shape:", tuple(logp.shape))
    print("train logp finite:", bool(torch.isfinite(logp).all()))
    print("train logp requires_grad:", logp.requires_grad)


if __name__ == "__main__":
    main()
