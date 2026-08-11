# finetuning/train_asr_nlu.py
"""ASR + ASR+NLU 联合 LoRA SFT 训练。

只训 NLU 会让 LoRA 扰动 thinker 导致 ASR 崩；用同一批 TTS 两用数据(音频+文本+意图)
派生两种样本联合训练同一个 LoRA, 让它同时服务 ASR 和 ASR+NLU：
  ASR     : system="转写语音",             target="language X<asr_text>文本"
  ASR+NLU : system="转写语音并提取用户意图", target="language X<asr_text>文本\n意图JSON"
LoRA 打在 thinker 文本解码器(复用 grpo_core.apply_lora), 只算 LLM loss, 不动 ctc/rnnt。

数据 jsonl 每行 {"messages":[{system},{user:audio_path},{assistant}]}，
assistant 格式 "language X<asr_text>文本\n意图JSON"。每条派生 ASR + ASR+NLU 两个样本。

用法：见 train_asr_nlu.sh。
"""
import argparse
import os
import random
from typing import Dict, List

import torch
from datasets import Dataset, load_dataset
from filelock import FileLock
from transformers import Trainer, TrainingArguments

from qwen_asr.joint import Qwen3ASRJointModel
from qwen_asr.joint.defaults import DEFAULT_ATTN_IMPLEMENTATION
from finetuning.train import DataCollatorForJointTraining
from finetuning.grpo_core import apply_lora, assert_only_text_decoder_trainable

ASR_PROMPT = "转写语音"
ASR_NLU_PROMPT = "转写语音并提取用户意图"

def expand_two_way(example: Dict) -> List[Dict]:
    """一条 ASR+NLU 标注 -> ASR 样本 + ASR+NLU 样本。"""
    msgs = example["messages"]
    audio_path = next(m["content"][0]["path"] for m in msgs if m["role"] == "user")
    assistant = next(m["content"] for m in msgs if m["role"] == "assistant")
    text_part = assistant.rsplit("\n", 1)[0]  # language X<asr_text>文本
    return [
        {"audio": audio_path, "prompt": ASR_PROMPT, "target": text_part},
        {"audio": audio_path, "prompt": ASR_NLU_PROMPT, "target": assistant},
    ]

def parse_args():
    p = argparse.ArgumentParser("Qwen3-ASR ASR+NLU 联合 LoRA SFT")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--train_file", type=str, required=True, help="两用 jsonl(messages 格式)")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--eval_ratio", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_acc", type=int, default=4)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--log_steps", type=int, default=20)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--logging_dir", type=str, default="./logs_asr_nlu")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--resume_from", type=str, default="")
    return p.parse_args()

def main():
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main = local_rank == 0

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    joint = Qwen3ASRJointModel.from_pretrained(
        args.ckpt,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map=None,
        load_heads=False,
        attn_implementation=DEFAULT_ATTN_IMPLEMENTATION,
    )
    joint.train_tasks = ("llm",)  # 只算 LLM loss，不动 ctc/rnnt
    processor = joint.processor

    if hasattr(joint.qwen_model, "gradient_checkpointing_enable"):
        try:
            joint.qwen_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            if is_main:
                print("当前 transformers 不支持非重入 checkpoint。")

    resume_from = (args.resume_from or "").strip()
    if resume_from:
        from peft import PeftModel
        peft = PeftModel.from_pretrained(joint, resume_from, is_trainable=True)
    else:
        peft = apply_lora(joint, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
    assert_only_text_decoder_trainable(peft)
    if hasattr(peft, "enable_input_require_grads"):
        peft.enable_input_require_grads()

    if is_main:
        trainable = sum(p.numel() for p in peft.parameters() if p.requires_grad)
        total = sum(p.numel() for p in peft.parameters())
        print(f"ASR+NLU LoRA 可训练参数：{trainable:,} / {total:,}")

    os.makedirs(args.output_dir, exist_ok=True)
    lock_path = os.path.join(args.output_dir, ".dataset_cache.lock")
    with FileLock(lock_path):
        raw = load_dataset("json", data_files={"train": args.train_file})["train"]
        examples: List[Dict] = []
        for ex in raw:
            examples.extend(expand_two_way(ex))
    rng = random.Random(args.seed)
    rng.shuffle(examples)
    n_eval = int(len(examples) * args.eval_ratio)
    eval_ds = Dataset.from_list(examples[:n_eval])
    train_ds = Dataset.from_list(examples[n_eval:])
    if is_main:
        print(f"派生样本：{len(examples)}（ASR + ASR+NLU 各 {len(raw)}），训练 {len(train_ds)}，验证 {len(eval_ds)}")

    collator = DataCollatorForJointTraining(processor, {}, None, need_aux=False)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=args.log_steps,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_safetensors=True,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        bf16=use_bf16,
        fp16=not use_bf16,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to="tensorboard",
        logging_dir=args.logging_dir,
    )
    trainer = Trainer(
        model=peft,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=processor.tokenizer,
    )
    trainer_resume = (
        resume_from
        if resume_from and os.path.exists(os.path.join(resume_from, "trainer_state.json"))
        else None
    )
    trainer.train(resume_from_checkpoint=trainer_resume)
    trainer.save_model(args.output_dir)
    if is_main:
        print(f"ASR+NLU LoRA 已保存到：{args.output_dir}")

if __name__ == "__main__":
    main()
