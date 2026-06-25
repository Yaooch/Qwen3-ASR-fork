# finetuning/grpo_train.py
"""GRPO 训练主循环。

每条样本采样 G 个 rollout → 可验证奖励 → 组内归一化优势 →
clip surrogate + KL(π_θ ‖ π_ref) 更新 LoRA。π_ref = 关 LoRA 的基线 talker。
"""
import argparse
import os

import torch
from torch.optim import AdamW

from qwen_asr.joint import Qwen3ASRJointModel
from qwen_asr.joint.defaults import DEFAULT_ATTN_IMPLEMENTATION
from qwen_asr.tools.hotword_reward import compute_reward
from finetuning.grpo_data import load_samples, split_eval
from finetuning.grpo_lora import apply_lora
from finetuning.grpo_math import group_advantages, grpo_loss
from finetuning.grpo_rollout import RolloutSampler

CKPT_DEFAULT = "/cfs/data/private/WangYaoChi/model/qwen3-asr-ctc-joint-14-hotword-1/checkpoint-228"
DATA_DEFAULT = "/cfs/data/private/WangYaoChi/train_data/all/contextasr/train_contextasr2.jsonl"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=CKPT_DEFAULT)
    p.add_argument("--data", default=DATA_DEFAULT)
    p.add_argument("--output_dir", required=False, default="/cfs/data/private/WangYaoChi/model/grpo_lora_out")
    p.add_argument("--group_size", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--beta", type=float, default=0.04)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--eval_ratio", type=float, default=0.02)
    p.add_argument("--smoke", action="store_true", help="少量样本 1 step 自检")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

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
    peft.eval()

    group_size = 4 if args.smoke else args.group_size
    temperature = 1.0 if args.smoke else args.temperature
    sampler = RolloutSampler(
        peft,
        joint.processor,
        asr_wrapper,
        group_size=group_size,
        temperature=temperature,
        max_new_tokens=args.max_new_tokens,
    )
    trainable = [p for p in peft.parameters() if p.requires_grad]
    opt = AdamW(trainable, lr=args.lr)

    limit = 8 if args.smoke else None
    samples = load_samples(args.data, limit=limit)
    train, _eval = split_eval(samples, eval_ratio=args.eval_ratio)
    max_steps = 1 if args.smoke else args.max_steps

    step = 0
    for sample in train:
        if step >= max_steps:
            break
        with torch.no_grad():
            rollouts = sampler.sample(sample)
        rewards = torch.tensor(
            [compute_reward(r.text, sample.gt_text, sample.hotwords) for r in rollouts],
            device="cuda",
            dtype=torch.float32,
        )
        if float(rewards.std().item()) < 1e-6:
            # 组内无区分（如简单样本 4 路全对），无学习信号，跳过且不计入 max_steps
            continue
        adv = group_advantages(rewards.unsqueeze(0)).squeeze(0)  # (G,)

        opt.zero_grad()
        loss_acc = 0.0
        for r, a in zip(rollouts, adv):
            logp = sampler.token_logp(sample, r.ids)  # (T,) LoRA-on 带梯度
            old_logp = logp.detach()
            loss_acc = loss_acc + grpo_loss(
                logp, old_logp, a.expand_as(logp), r.logp_ref.to(logp.dtype), beta=args.beta
            )
        loss = loss_acc / len(rollouts)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()

        if step % 10 == 0 or args.smoke:
            print(
                f"step {step} loss {float(loss.detach()):.4f} "
                f"reward mean {float(rewards.mean()):.3f} std {float(rewards.std()):.3f}"
            )
        step += 1

    if step == 0:
        print("warning: 未发生梯度步（所有样本组内奖励方差为 0）")
    lora_dir = os.path.join(args.output_dir, "lora")
    peft.save_pretrained(lora_dir)
    print("saved LoRA to", lora_dir)


if __name__ == "__main__":
    main()
