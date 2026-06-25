# finetuning/grpo_train.py
"""GRPO 训练主循环（支持多卡数据并行）。

每条样本采样 G 个 rollout → 可验证奖励 → 组内归一化优势 →
clip surrogate + KL(π_θ ‖ π_ref) 更新 LoRA。π_ref = 关 LoRA 的基线 talker。

多卡（torchrun 启动）：每卡取 rank-strided 样本，各自做 G 路 rollout 与 backward，
LoRA 梯度跨卡 all-reduce 求平均后同步步进；apply_lora 前固定种子使各卡 LoRA 初始化一致，
保证参数始终同步。effective batch = world_size × batch_size_per_rank。
"""
import argparse
import os

import torch
import torch.distributed as dist
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
    p.add_argument("--output_dir", default="/cfs/data/private/WangYaoChi/model/grpo_lora_out")
    p.add_argument("--group_size", type=int, default=8, help="每条样本采样的 rollout 数 G")
    p.add_argument("--batch_size_per_rank", type=int, default=1, help="每卡每步累积样本数；effective batch = world_size × 此值")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--beta", type=float, default=0.04)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--save_steps", type=int, default=100, help="每 N 步保存一次 LoRA 到 lora-step{N}；0 表示不周期保存")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--eval_ratio", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true", help="少量样本 1 step 自检")
    return p.parse_args()


def init_distributed():
    """torchrun 注入 RANK/WORLD_SIZE/LOCAL_RANK；否则单卡。"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group("nccl", rank=rank, world_size=world)
        torch.cuda.set_device(local_rank)
        return rank, world, local_rank
    return 0, 1, 0


def main():
    args = parse_args()
    rank, world, local_rank = init_distributed()
    device = f"cuda:{local_rank}" if world > 1 else "cuda"
    is_main = rank == 0
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)

    joint = Qwen3ASRJointModel.from_pretrained(
        args.ckpt,
        dtype=torch.bfloat16,
        device_map=None,
        load_heads=False,
        attn_implementation=DEFAULT_ATTN_IMPLEMENTATION,
    ).to(device)
    joint.eval()
    asr_wrapper = joint._asr_wrapper

    # apply_lora 前固定种子：LoRA 的 B 用 kaiming 随机初始化，各卡必须一致才能保持参数同步
    torch.manual_seed(args.seed)
    peft = apply_lora(joint)
    peft.eval()
    # 之后各卡用不同种子，使 rollout 采样各异（不同样本本身也已 strided）
    torch.manual_seed(args.seed + rank)

    group_size = 4 if args.smoke else args.group_size
    temperature = 1.0 if args.smoke else args.temperature
    sampler = RolloutSampler(
        peft,
        joint.processor,
        asr_wrapper,
        group_size=group_size,
        temperature=temperature,
        max_new_tokens=args.max_new_tokens,
        device=device,
    )
    trainable = [p for p in peft.parameters() if p.requires_grad]
    opt = AdamW(trainable, lr=args.lr)

    limit = 8 if args.smoke else None
    samples = load_samples(args.data, limit=limit)
    train, _eval = split_eval(samples, eval_ratio=args.eval_ratio)
    # 多卡按 rank stride 切分，各卡样本数一致以保持步数对齐
    train_local = train[rank::world] if world > 1 else train
    max_steps = 1 if args.smoke else args.max_steps
    accum = args.batch_size_per_rank

    step = 0
    idx = 0
    while step < max_steps and idx < len(train_local):
        opt.zero_grad()
        last_loss = 0.0
        last_rewards = None
        for _ in range(accum):
            if idx >= len(train_local):
                break
            sample = train_local[idx]
            idx += 1
            with torch.no_grad():
                rollouts = sampler.sample(sample)
            rewards = torch.tensor(
                [compute_reward(r.text, sample.gt_text, sample.hotwords) for r in rollouts],
                device=device,
                dtype=torch.float32,
            )
            if float(rewards.std().item()) < 1e-6:
                # 组内无区分，无学习信号：跳过 backward（仍消耗该样本，保持各卡步数对齐）
                sampler.clear_audio_cache()
                continue
            adv = group_advantages(rewards.unsqueeze(0)).squeeze(0)  # (G,)
            loss_acc = 0.0
            for r, a in zip(rollouts, adv):
                # on-policy 单步：old_logp 取当前策略 logp 的 detach，ratio≡1、clip 为多 epoch 占位
                logp = sampler.token_logp(sample, r.ids)  # (T,) LoRA-on 带梯度
                old_logp = logp.detach()
                loss_acc = loss_acc + grpo_loss(
                    logp, old_logp, a.expand_as(logp), r.logp_ref.to(logp.dtype), beta=args.beta
                )
            # 累积：按 accum 与 world 同时归一，等价于对所有样本梯度求平均
            (loss_acc / len(rollouts) / accum).backward()
            last_loss = float(loss_acc.detach() / len(rollouts))
            last_rewards = rewards
            sampler.clear_audio_cache()

        # 确保各卡 LoRA 都有 grad 张量（某卡若全部 skip 则为 None），再 all-reduce 求平均
        if world > 1:
            for p in trainable:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
            for p in trainable:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad.mul_(1.0 / world)

        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()

        if is_main and (step % 10 == 0 or args.smoke):
            if last_rewards is not None:
                print(
                    f"step {step} loss {last_loss:.4f} "
                    f"reward mean {float(last_rewards.mean()):.3f} std {float(last_rewards.std()):.3f} "
                    f"(world={world} accum={accum} eff_batch={world * accum})"
                )
            else:
                print(f"step {step} all-skipped (world={world})")

        # 周期保存：崩溃不丢进度，可从最近 checkpoint 评测/续训
        if is_main and args.save_steps > 0 and step % args.save_steps == 0:
            ckpt_dir = os.path.join(args.output_dir, f"lora-step{step}")
            peft.save_pretrained(ckpt_dir)
            print(f"saved LoRA checkpoint at step {step} to {ckpt_dir}")
        step += 1

    if is_main:
        if step == 0:
            print("warning: 未发生梯度步（所有样本组内奖励方差为 0）")
        lora_dir = os.path.join(args.output_dir, "lora")
        peft.save_pretrained(lora_dir)
        print("saved LoRA to", lora_dir)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
