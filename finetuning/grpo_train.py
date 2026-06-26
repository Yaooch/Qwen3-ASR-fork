# finetuning/grpo_train.py
"""GRPO 训练主循环（支持多卡数据并行）+ rollout 采样器。

每条样本采样 G 个 rollout → 可验证奖励 → 组内归一化优势 →
clip surrogate + KL(π_θ ‖ π_ref) 更新 LoRA。π_ref = 关 LoRA 的基线 talker。

多卡（torchrun 启动）：每卡取 rank-strided 样本，各自做 G 路 rollout 与 backward，
LoRA 梯度跨卡 all-reduce 求平均后同步步进；apply_lora 前固定种子使各卡 LoRA 初始化一致，
保证参数始终同步。effective batch = world_size × batch_size_per_rank。
"""
import argparse
import os
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from typing import List

import librosa
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import AdamW

from qwen_asr.inference.utils import parse_asr_output
from qwen_asr.joint import Qwen3ASRJointModel
from qwen_asr.joint.defaults import DEFAULT_ATTN_IMPLEMENTATION
from qwen_asr.joint.encoder import encode_offline, feature_lens
from qwen_asr.tools.hotword_reward import compute_reward
from finetuning.grpo_core import (
    apply_lora,
    assert_only_text_decoder_trainable,
    group_advantages,
    grpo_loss,
    load_samples,
    split_eval,
)

CKPT_DEFAULT = "/cfs/data/private/WangYaoChi/model/qwen3-asr-ctc-joint-14-hotword-1/checkpoint-228"
DATA_DEFAULT = "/cfs/data/private/WangYaoChi/train_data/all/contextasr/train_contextasr2.jsonl"


# --------------------------------------------------------------------------
# Rollout：音频 embedding 缓存 + G 路采样 + token logprob
# 与 qwen_asr/joint/model.py 的 decode_llm 路径对齐：音频 embedding 由 encode_offline
# 产出；prompt 由 _asr_wrapper._build_text_prompt 构造，audio placeholder 用 processor.audio_token；
# 采样走 qwen_model.generate(do_sample)，logprob 走 thinker.forward 的 logits。
# --------------------------------------------------------------------------


@dataclass
class RolloutResult:
    ids: torch.LongTensor   # (T_gen,) 生成 token
    text: str
    logp_ref: torch.Tensor  # (T_gen,) base(LoRA-off) logprob，detach


class RolloutSampler:
    def __init__(
        self,
        joint_peft,
        processor,
        asr_wrapper,
        group_size: int = 8,
        temperature: float = 0.8,
        max_new_tokens: int = 512,
        device: str = "cuda",
    ):
        self.joint = joint_peft
        self.processor = processor
        self.asr_wrapper = asr_wrapper
        self.G = group_size
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.device = device
        self._audio_cache = {}
        # 显式 pad_token_id，避免 generate 时 "Setting pad_token_id to eos" 告警刷屏
        eos = getattr(self.joint.qwen_model.generation_config, "eos_token_id", None)
        self.pad_token_id = eos[0] if isinstance(eos, (list, tuple)) else eos
        self._eos_set = set(eos if isinstance(eos, (list, tuple)) else [eos])

    def clear_audio_cache(self) -> None:
        """跨样本清理音频 embedding 缓存，避免长训练 OOM。"""
        self._audio_cache = {}

    def audio_embedding(self, audio_path: str) -> torch.Tensor:
        """返回该音频的 LLM 输入 embedding (n_audio_tokens, hidden)，缓存。"""
        if audio_path in self._audio_cache:
            return self._audio_cache[audio_path]
        wav, _ = librosa.load(audio_path, sr=16000, mono=True)
        wav = wav.astype("float32")
        fe = self.processor.feature_extractor
        batch = fe(
            [wav],
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            truncation=False,
            return_attention_mask=True,
        )
        feats = batch["input_features"]
        mask = batch.get("feature_attention_mask", batch.get("attention_mask"))
        ref = next(self.joint.parameters())
        feats = feats.to(device=ref.device, dtype=ref.dtype)
        if mask is not None:
            mask = mask.to(device=ref.device)
        lens = feature_lens(feats, mask)
        tower = self.joint.qwen_model.thinker.audio_tower
        _, llm, _ = encode_offline(tower, feats, lens, need_llm=True)
        emb = llm[: int(lens[0])]  # 单条音频：llm 即 (n_audio, hidden) 的 LLM 音频 embedding
        self._audio_cache[audio_path] = emb
        return emb

    def build_batch(self, sample):
        """构造 G 份相同 prompt 的 batch（同音频同 prompt，仅 rollout 生成不同）。
        返回 input_ids (G,T)、attn (G,T)、audio_embeds (G, n_audio, hidden)。"""
        thinker = self.joint.qwen_model.thinker
        processor = self.processor
        token = processor.audio_token
        context = sample.prompt or ""
        text = self.asr_wrapper._build_text_prompt(context=context, force_language=None)
        audio_embeds_1d = self.audio_embedding(sample.audio)  # (n_audio, hidden)
        text = text.replace(token, token * int(audio_embeds_1d.shape[0]), 1)
        old = processor.tokenizer.padding_side
        processor.tokenizer.padding_side = "left"
        try:
            tok = processor.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        finally:
            processor.tokenizer.padding_side = old
        input_ids_1d = tok["input_ids"][0]   # (T,)
        attn_1d = tok["attention_mask"][0]   # (T,)
        G = self.G
        input_ids = input_ids_1d.unsqueeze(0).expand(G, -1).contiguous()
        attn = attn_1d.unsqueeze(0).expand(G, -1).contiguous()
        audio_embeds = audio_embeds_1d.unsqueeze(0).expand(G, -1, -1).contiguous()
        return input_ids.to(self.device), attn.to(self.device), audio_embeds

    def _strip_gen(self, gen_full_1d) -> torch.Tensor:
        """从 generate 输出截出生成段（到第一个 eos 前，不含 eos/pad）。"""
        arr = gen_full_1d.tolist()
        cut = len(arr)
        for j, t in enumerate(arr):
            if t in self._eos_set:
                cut = j
                break
        return gen_full_1d[:cut]

    @torch.no_grad()
    def _generate_batch(self, input_ids, attn, audio_embeds):
        """一次 batch=G 的采样生成，返回 list[(gen_ids, text)]。"""
        thinker = self.joint.qwen_model.thinker
        embeds = thinker.get_input_embeddings()(input_ids)  # (G, T, hidden)
        audio_mask = thinker.get_placeholder_mask(input_ids, inputs_embeds=embeds)
        inputs_embeds = embeds.masked_scatter(audio_mask, audio_embeds.to(dtype=embeds.dtype))
        gen = self.joint.qwen_model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            inputs_embeds=inputs_embeds,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=0.95,
            pad_token_id=self.pad_token_id,
        )
        seqs = gen.sequences  # (G, T + max_gen)
        prompt_len = input_ids.shape[1]
        gen_part = seqs[:, prompt_len:]
        results = []
        for i in range(self.G):
            gen_ids = self._strip_gen(gen_part[i])
            raw = self.processor.batch_decode(gen_ids.unsqueeze(0), skip_special_tokens=True)[0]
            _, text = parse_asr_output(raw)
            results.append((gen_ids, text))
        return results

    def _logp_batch(self, input_ids, audio_embeds, gen_ids_list, use_lora: bool) -> List[torch.Tensor]:
        """一次 batch=G 前向算各 rollout 生成 token 的 logprob。use_lora=False 关 LoRA（ref）。"""
        thinker = self.joint.qwen_model.thinker
        G = len(gen_ids_list)
        prompt_len = input_ids.shape[1]
        max_gen = max((g.numel() for g in gen_ids_list), default=0)
        if max_gen == 0:
            return [torch.zeros(0, device=self.device, dtype=torch.float32) for _ in range(G)]
        pad = self.pad_token_id
        full = torch.full((G, prompt_len + max_gen), pad, dtype=input_ids.dtype, device=input_ids.device)
        full[:, :prompt_len] = input_ids
        attn = torch.zeros((G, prompt_len + max_gen), dtype=torch.long, device=input_ids.device)
        attn[:, :prompt_len] = 1
        for i, g in enumerate(gen_ids_list):
            li = g.numel()
            full[i, prompt_len:prompt_len + li] = g
            attn[i, prompt_len:prompt_len + li] = 1
        embeds = thinker.get_input_embeddings()(full)  # (G, T+max, hidden)
        audio_mask = thinker.get_placeholder_mask(full, inputs_embeds=embeds)
        inputs_embeds = embeds.masked_scatter(audio_mask, audio_embeds.to(dtype=embeds.dtype))
        ctx = (
            self.joint.disable_adapter()
            if (not use_lora and hasattr(self.joint, "disable_adapter"))
            else nullcontext()
        )
        with ctx:
            logits = thinker(
                input_ids=full,
                attention_mask=attn,
                inputs_embeds=inputs_embeds,
            ).logits  # (G, T+max, V)
        logp_list = []
        for i, g in enumerate(gen_ids_list):
            li = g.numel()
            if li == 0:
                logp_list.append(torch.zeros(0, device=logits.device, dtype=logits.dtype))
                continue
            # 位置 prompt_len-1+k 的 logits 预测 gen token k
            row_logits = logits[i, prompt_len - 1:prompt_len - 1 + li, :]
            logp = F.log_softmax(row_logits, dim=-1).gather(-1, g.to(logits.device).unsqueeze(-1)).squeeze(-1)
            logp_list.append(logp)
        return logp_list

    @torch.no_grad()
    def sample(self, sample):
        """batched rollout：1 次 generate + 1 次 ref logp。
        返回 (inputs, list[RolloutResult])，inputs 供 token_logp_batch 复用避免重 tokenize。"""
        input_ids, attn, audio_embeds = self.build_batch(sample)
        gen_results = self._generate_batch(input_ids, attn, audio_embeds)
        gen_ids_list = [g for g, _ in gen_results]
        ref_logps = self._logp_batch(input_ids, audio_embeds, gen_ids_list, use_lora=False)
        results = [
            RolloutResult(ids=g.detach(), text=t, logp_ref=ref.detach())
            for (g, t), ref in zip(gen_results, ref_logps)
        ]
        return (input_ids, audio_embeds), results

    def token_logp_batch(self, inputs, gen_ids_list) -> List[torch.Tensor]:
        """训练时 LoRA-on 带梯度重算各 rollout logp（1 次 batch 前向）。"""
        input_ids, audio_embeds = inputs
        return self._logp_batch(input_ids, audio_embeds, gen_ids_list, use_lora=True)


# --------------------------------------------------------------------------
# 训练主循环
# --------------------------------------------------------------------------


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
    p.add_argument("--resume", default=None, help="从 LoRA 目录续训（加载 LoRA 权重 + 优化器状态 + step）；max_steps 为总目标步数")
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


def save_checkpoint(peft, opt, step: int, out_dir: str):
    """保存 LoRA 权重 + 优化器状态 + step，供续训。idx = step * accum 可由 step 推出，不单独存。"""
    peft.save_pretrained(out_dir)
    torch.save(
        {"step": step, "opt": opt.state_dict()},
        os.path.join(out_dir, "train_state.pt"),
    )


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

    if args.resume:
        from peft import PeftModel

        # 续训：各卡从同一目录加载 LoRA 权重，结构/权重天然一致
        peft = PeftModel.from_pretrained(joint, args.resume)
        assert_only_text_decoder_trainable(peft)
    else:
        peft = apply_lora(joint)
        # 多卡下各卡 LoRA 随机初始化不同，广播 rank0 参数使各卡一致；
        # 之后 all-reduce 平均梯度才能保持各卡参数同步（不依赖随机种子）
        if world > 1:
            for p in peft.parameters():
                if p.requires_grad:
                    dist.broadcast(p.data, src=0)
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
        device=device,
    )
    trainable = [p for p in peft.parameters() if p.requires_grad]
    opt = AdamW(trainable, lr=args.lr)

    # 续训：恢复优化器状态与步数；idx = step * accum（每步每卡固定消费 accum 个样本）
    start_step = 0
    if args.resume:
        state_path = os.path.join(args.resume, "train_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location="cpu")
            opt.load_state_dict(state["opt"])
            start_step = int(state["step"])
            if is_main:
                print(f"resumed from {args.resume}: step={start_step} (with optimizer state)")
        elif is_main:
            print(
                f"resumed from {args.resume}: 无 train_state.pt，仅加载 LoRA 权重，"
                "优化器重置、step 从 0 起（旧版 checkpoint 兜底）"
            )

    limit = 8 if args.smoke else None
    samples = load_samples(args.data, limit=limit)
    train, _eval = split_eval(samples, eval_ratio=args.eval_ratio)
    # 多卡按 rank stride 切分，各卡样本数一致以保持步数对齐
    train_local = train[rank::world] if world > 1 else train
    max_steps = 1 if args.smoke else args.max_steps
    accum = args.batch_size_per_rank

    step = start_step
    idx = start_step * accum
    reward_history = deque(maxlen=50)  # 滚动 reward 均值，平滑样本难度噪声看真趋势
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
                inputs, rollouts = sampler.sample(sample)
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
            # 1 次 batch 带梯度前向算所有 rollout 的 logp（复用 inputs，避免重 tokenize）
            logps = sampler.token_logp_batch(inputs, [r.ids for r in rollouts])
            loss_acc = 0.0
            n_valid = 0
            for r, a, logp in zip(rollouts, adv, logps):
                if logp.numel() == 0:
                    continue  # 退化空输出，无梯度可算
                # on-policy 单步：old_logp 取当前策略 logp 的 detach，ratio≡1、clip 为多 epoch 占位
                old_logp = logp.detach()
                loss_acc = loss_acc + grpo_loss(
                    logp, old_logp, a.expand_as(logp), r.logp_ref.to(logp.dtype), beta=args.beta
                )
                n_valid += 1
            if n_valid == 0:
                sampler.clear_audio_cache()
                continue
            # 累积：按 accum 与 world 同时归一，等价于对所有样本梯度求平均
            (loss_acc / n_valid / accum).backward()
            last_loss = float(loss_acc.detach() / n_valid)
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

        # clip_grad_norm_ 返回裁剪前的总范数，用来确认梯度在流动（loss 前向值恒≈0，看 grad_norm 才有意义）
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
        opt.step()
        if last_rewards is not None:
            reward_history.append(float(last_rewards.mean()))

        if is_main and (step % 10 == 0 or args.smoke):
            if last_rewards is not None:
                roll_mean = sum(reward_history) / len(reward_history) if reward_history else 0.0
                print(
                    f"step {step} loss {last_loss:.4f} grad_norm {grad_norm:.4f} "
                    f"reward {float(last_rewards.mean()):.3f} std {float(last_rewards.std()):.3f} "
                    f"reward_avg{len(reward_history)} {roll_mean:.3f} "
                    f"(eff_batch={world * accum})"
                )
            else:
                print(f"step {step} all-skipped (eff_batch={world * accum})")

        # 周期保存：崩溃不丢进度，可从最近 checkpoint 评测/续训（step 0 不保存，无意义）
        if is_main and args.save_steps > 0 and step > 0 and step % args.save_steps == 0:
            ckpt_dir = os.path.join(args.output_dir, f"lora-step{step}")
            save_checkpoint(peft, opt, step, ckpt_dir)
            print(f"saved checkpoint at step {step} to {ckpt_dir}")
        step += 1

    if is_main:
        if step == 0:
            print("warning: 未发生梯度步（所有样本组内奖励方差为 0）")
        lora_dir = os.path.join(args.output_dir, "lora")
        save_checkpoint(peft, opt, step, lora_dir)
        print("saved LoRA to", lora_dir)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
