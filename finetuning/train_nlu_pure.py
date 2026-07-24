# finetuning/train_nlu_pure.py
"""NLU（用户意图提取）纯 Qwen3 LLM 训练（不经 Qwen3ASRJointModel）。

直接用 AutoModelForCausalLM 加载原始 Qwen3 checkpoint，
纯文本 SFT：user 语句 -> assistant 意图/Action。

数据：jsonl 每行 {"messages": [{system}, {user}, {assistant}]}。
用法：见 train_nlu_pure.sh。
"""
import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from filelock import FileLock
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel

from qwen_asr.tools.nlu import NLU_SYSTEM_PROMPT, NLU_CHAT_TEMPLATE, build_nlu_prompt, nlu_messages


@dataclass
class NluCollator:
    tokenizer: Any

    def __post_init__(self):
        self.eos = self.tokenizer.eos_token or ""

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prefix_texts, full_texts = [], []
        for item in features:
            msgs = item["messages"]
            system = next((m["content"] for m in msgs if m["role"] == "system"), NLU_SYSTEM_PROMPT)
            user = next((m["content"] for m in msgs if m["role"] == "user"), "")
            assistant = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
            prefix = self.tokenizer.apply_chat_template(
                nlu_messages(system, user),
                add_generation_prompt=True,
                enable_thinking=False,
                tokenize=False,
            )
            prefix_texts.append(prefix)
            full_texts.append(prefix + assistant + self.eos)

        old_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"
        try:
            full_tok = self.tokenizer(full_texts, return_tensors="pt", padding=True)
            prefix_tok = self.tokenizer(prefix_texts, return_tensors="pt", padding=True)
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
    p = argparse.ArgumentParser("纯 Qwen3 NLU SFT")
    p.add_argument("--ckpt", type=str, required=True, help="Qwen3 模型目录")
    p.add_argument("--train_file", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--eval_ratio", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_acc", type=int, default=16)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--log_steps", type=int, default=20)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--logging_dir", type=str, default="./logs_nlu_pure")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--resume_from", type=str, default="")
    p.add_argument("--full_ft", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main = local_rank == 0

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map=None,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            if is_main:
                print("当前 transformers 不支持非重入 checkpoint。")

    resume_from = (args.resume_from or "").strip()
    if args.full_ft:
        for p in model.parameters():
            p.requires_grad = True
    elif resume_from:
        model = PeftModel.from_pretrained(model, resume_from, is_trainable=True)
    else:
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
    if not args.full_ft and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    if is_main:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"{'全参' if args.full_ft else 'LoRA'} 可训练参数：{trainable:,} / {total:,}")

    os.makedirs(args.output_dir, exist_ok=True)
    lock_path = os.path.join(args.output_dir, ".dataset_cache.lock")
    with FileLock(lock_path):
        ds = load_dataset("json", data_files={"train": args.train_file})["train"]
        split = ds.train_test_split(test_size=args.eval_ratio, seed=args.seed)
    train_ds, eval_ds = split["train"], split["test"]
    if is_main:
        print(f"训练样本：{len(train_ds)}，验证样本：{len(eval_ds)}")

    collator = NluCollator(tokenizer)
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
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train(resume_from_checkpoint=resume_from or None)
    trainer.save_model(args.output_dir)
    if is_main:
        tokenizer.save_pretrained(args.output_dir)
        print(f"NLU 模型已保存到：{args.output_dir}")


if __name__ == "__main__":
    main()
