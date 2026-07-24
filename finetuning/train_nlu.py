# finetuning/train_nlu.py
"""纯文本 NLU / Agent SFT，支持 joint checkpoint 和纯 Qwen3 LLM。"""
import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from filelock import FileLock
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from finetuning.grpo_core import apply_lora, assert_only_text_decoder_trainable
from qwen_asr.joint import Qwen3ASRJointModel
from qwen_asr.joint.defaults import DEFAULT_ATTN_IMPLEMENTATION
from qwen_asr.tools.nlu import NLU_SYSTEM_PROMPT, build_nlu_prompt, nlu_messages


@dataclass
class NluCollator:
    tokenizer: Any
    backend: str
    processor: Any = None
    max_len: int = 512

    def __post_init__(self):
        self.eos = self.tokenizer.eos_token or ""

    def render(self, messages: List[Dict[str, str]]) -> str:
        if self.backend == "joint":
            return build_nlu_prompt(self.processor, messages, add_generation_prompt=True)
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=False,
        )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prefix_texts, full_texts = [], []
        for item in features:
            msgs = item["messages"]
            system = next((m["content"] for m in msgs if m["role"] == "system"), NLU_SYSTEM_PROMPT)
            user = next((m["content"] for m in msgs if m["role"] == "user"), "")
            assistant = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
            prefix = self.render(nlu_messages(system, user))
            prefix_texts.append(prefix)
            full_texts.append(prefix + assistant + self.eos)

        old_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"
        kwargs = {
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
            "max_length": self.max_len,
        }
        try:
            full_tok = self.tokenizer(full_texts, **kwargs)
            prefix_tok = self.tokenizer(prefix_texts, **kwargs)
        finally:
            self.tokenizer.padding_side = old_side

        labels = full_tok["input_ids"].clone()
        for idx, length in enumerate(prefix_tok["attention_mask"].sum(dim=1).tolist()):
            labels[idx, :length] = -100
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        full_tok["labels"] = labels
        return full_tok


def parse_args():
    p = argparse.ArgumentParser("NLU / Agent 纯文本 SFT")
    p.add_argument("--backend", choices=["joint", "llm"], default="joint")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--train_file", required=True)
    p.add_argument("--output_dir", required=True)
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
    p.add_argument("--logging_dir", default="./logs_nlu")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--resume_from", default="")
    p.add_argument("--full_ft", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    is_main = int(os.environ.get("LOCAL_RANK", "0")) == 0
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    if args.backend == "joint":
        base = Qwen3ASRJointModel.from_pretrained(
            args.ckpt,
            dtype=dtype,
            device_map=None,
            load_heads=False,
            attn_implementation=DEFAULT_ATTN_IMPLEMENTATION,
        )
        processor = base.processor
        tokenizer = processor.tokenizer
        checkpoint_model = base.qwen_model
    else:
        processor = None
        tokenizer = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            args.ckpt,
            dtype=dtype,
            device_map=None,
            trust_remote_code=True,
            attn_implementation=DEFAULT_ATTN_IMPLEMENTATION,
        )
        checkpoint_model = base

    if hasattr(checkpoint_model, "gradient_checkpointing_enable"):
        try:
            checkpoint_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            if is_main:
                print("当前 transformers 不支持非重入 checkpoint。")

    resume_from = args.resume_from.strip()
    trainer_resume = resume_from or None
    if args.full_ft:
        if args.backend == "joint":
            from finetuning.train import set_trainable

            set_trainable(base, ("llm",))
        else:
            for param in base.parameters():
                param.requires_grad_(True)
        model = base
    elif resume_from:
        model = PeftModel.from_pretrained(base, resume_from, is_trainable=True)
        if args.backend == "joint":
            assert_only_text_decoder_trainable(model)
        if not os.path.exists(os.path.join(resume_from, "trainer_state.json")):
            trainer_resume = None
    elif args.backend == "joint":
        model = apply_lora(base, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
    else:
        model = get_peft_model(
            base,
            LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                task_type="CAUSAL_LM",
            ),
        )

    if not args.full_ft and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    if is_main:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        mode = "全参" if args.full_ft else "LoRA"
        print(f"{args.backend} {mode} 可训练参数：{trainable:,} / {total:,}")

    os.makedirs(args.output_dir, exist_ok=True)
    with FileLock(os.path.join(args.output_dir, ".dataset_cache.lock")):
        ds = load_dataset("json", data_files={"train": args.train_file})["train"]
        split = ds.train_test_split(test_size=args.eval_ratio, seed=args.seed)
    train_ds, eval_ds = split["train"], split["test"]
    if is_main:
        print(f"训练样本：{len(train_ds)}，验证样本：{len(eval_ds)}")

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
    collator = NluCollator(tokenizer, args.backend, processor, args.max_len)
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "data_collator": collator,
        "tokenizer": tokenizer,
    }
    if args.backend == "joint" and args.full_ft:
        from finetuning.train import JointTrainer

        trainer = JointTrainer(
            lr_by_group={name: args.lr for name in ("llm", "proj", "encoder", "ctc", "rnnt")},
            save_heads=(),
            head_source=args.ckpt,
            **trainer_kwargs,
        )
    else:
        trainer = Trainer(**trainer_kwargs)

    trainer.train(resume_from_checkpoint=trainer_resume)
    trainer.save_model(args.output_dir)
    if is_main:
        if args.backend == "llm":
            tokenizer.save_pretrained(args.output_dir)
        print(f"NLU 模型已保存到：{args.output_dir}")


if __name__ == "__main__":
    main()
