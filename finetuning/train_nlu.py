# finetuning/train_nlu.py
"""NLU（用户意图提取）LoRA SFT 训练。

基线 joint checkpoint 冻结，LoRA 打在 thinker 文本解码器（复用 grpo_core.apply_lora），
纯文本 SFT：user 语句 -> assistant 意图 JSON。不走 audio_tower / ctc / rnnt；
joint.forward 检测到 input_features 为空时走纯文本 thinker 前向分支。

数据：jsonl 每行 {"messages": [{system}, {user}, {assistant}]}，按 eval_ratio 随机切 train/eval。

用法：见 train_nlu.sh。
"""
import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from filelock import FileLock
from transformers import Trainer, TrainingArguments

from qwen_asr.joint import Qwen3ASRJointModel
from qwen_asr.joint.defaults import DEFAULT_ATTN_IMPLEMENTATION
from qwen_asr.tools.nlu import NLU_SYSTEM_PROMPT, build_nlu_prompt, nlu_messages
from finetuning.grpo_core import apply_lora, assert_only_text_decoder_trainable


@dataclass
class NluCollator:
    """纯文本 NLU collator：messages -> input_ids + labels（mask prompt，只对 assistant+eos 算 loss）。"""

    processor: Any
    max_len: int = 512

    def __post_init__(self):
        self.tokenizer = self.processor.tokenizer
        self.eos = self.tokenizer.eos_token or ""

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prefix_texts, full_texts = [], []
        for item in features:
            msgs = item["messages"]
            system = next((m["content"] for m in msgs if m["role"] == "system"), NLU_SYSTEM_PROMPT)
            user = next((m["content"] for m in msgs if m["role"] == "user"), "")
            assistant = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
            prefix = build_nlu_prompt(self.processor, nlu_messages(system, user), add_generation_prompt=True)
            prefix_texts.append(prefix)
            full_texts.append(prefix + assistant + self.eos)

        old_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"
        try:
            full_tok = self.tokenizer(
                full_texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_len,
            )
            prefix_tok = self.tokenizer(
                prefix_texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_len,
            )
        finally:
            self.tokenizer.padding_side = old_side

        labels = full_tok["input_ids"].clone()
        for idx, length in enumerate(prefix_tok["attention_mask"].sum(dim=1).tolist()):
            labels[idx, :length] = -100
        pad_id = self.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        full_tok["labels"] = labels
        return full_tok


def parse_args():
    p = argparse.ArgumentParser("Qwen3-ASR NLU LoRA SFT")
    p.add_argument("--ckpt", type=str, required=True, help="基线 joint checkpoint 目录（含 joint_config.json）")
    p.add_argument("--train_file", type=str, required=True, help="NLU jsonl，每行 {messages:[system,user,assistant]}")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--eval_ratio", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--grad_acc", type=int, default=4)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--log_steps", type=int, default=20)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--logging_dir", type=str, default="./logs_nlu")
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
        print(f"NLU LoRA 可训练参数：{trainable:,} / {total:,}")

    os.makedirs(args.output_dir, exist_ok=True)
    lock_path = os.path.join(args.output_dir, ".dataset_cache.lock")
    with FileLock(lock_path):
        ds = load_dataset("json", data_files={"train": args.train_file})["train"]
        split = ds.train_test_split(test_size=args.eval_ratio, seed=args.seed)
    train_ds, eval_ds = split["train"], split["test"]
    if is_main:
        print(f"训练样本：{len(train_ds)}，验证样本：{len(eval_ds)}")

    collator = NluCollator(processor, max_len=args.max_len)
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
    trainer.train(resume_from_checkpoint=resume_from or None)
    trainer.save_model(args.output_dir)
    if is_main:
        print(f"NLU LoRA 已保存到：{args.output_dir}")


if __name__ == "__main__":
    main()
