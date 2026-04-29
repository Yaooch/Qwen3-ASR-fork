# coding=utf-8
import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import editdistance
import librosa
import torch
from datasets import load_dataset
from qwen_asr import Qwen3ASRModel
from qwen_asr.joint import Qwen3ASRJointModel
from safetensors.torch import load_model as load_safetensors_model
from transformers import GenerationConfig, Trainer, TrainerCallback, TrainingArguments
from transformers.modeling_utils import load_sharded_checkpoint


def patch_outer_forward(model):
    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return
    if not hasattr(model, "thinker") or not hasattr(model.thinker, "forward"):
        raise RuntimeError("Cannot patch forward: model has no `.thinker.forward`.")

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_features=None,
        feature_attention_mask=None,
        labels=None,
        **kwargs,
    ):
        return self.thinker.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels,
            **kwargs,
        )

    cls.forward = forward
    cls._forward_patched = True


_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    if not output_dir or not os.path.isdir(output_dir):
        return None

    best_step, best_path = None, None
    for name in os.listdir(output_dir):
        m = _CKPT_RE.match(name)
        if not m:
            continue
        step = int(m.group(1))
        path = os.path.join(output_dir, name)
        if os.path.isdir(path) and (best_step is None or step > best_step):
            best_step, best_path = step, path
    return best_path


def load_audio(path: str, sr: int = 16000, max_duration: float = 12.0):
    try:
        wav, _ = librosa.load(path, sr=sr, mono=True)
        duration = len(wav) / sr
        if duration > max_duration:
            print(f"音频过长，跳过：{path}")
            return None
        return wav
    except Exception as e:
        print(f"音频读取失败，跳过：{path}，{e}")
        return None


def build_prefix_messages(prompt: str, audio_array):
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]


def make_preprocess_fn_prefix_only(processor):
    def _preprocess(ex: Dict[str, Any]) -> Dict[str, Any]:
        prompt = ex.get("prompt", "")
        prefix_msgs = build_prefix_messages(prompt, None)
        prefix_text = processor.apply_chat_template(
            [prefix_msgs],
            add_generation_prompt=True,
            tokenize=False,
        )[0]
        return {
            "prompt": prompt,
            "audio": ex["audio"],
            "target": ex["text"],
            "prefix_text": prefix_text,
        }

    return _preprocess


@dataclass
class DataCollatorForJointTraining:
    processor: Any
    vocab: dict
    sp_model: Any
    sampling_rate: int = 16000

    def __init__(self, processor, vocab, sp_model, sampling_rate=16000):
        self.processor = processor
        self.vocab = vocab
        self.sp_model = sp_model
        self.sampling_rate = sampling_rate
        self.cjk_pattern = re.compile(r"([\u4e00-\u9fff])")
        self._id_to_token = {v: k for k, v in vocab.items()}

    def __call__(self, features: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        valid_features = []
        for f in features:
            wav = load_audio(f["audio"], sr=self.sampling_rate)
            if wav is not None:
                f["_audio"] = wav
                valid_features.append(f)

        if not valid_features:
            return None

        prefix_texts = [f["prefix_text"] for f in valid_features]
        targets = [f["target"] for f in valid_features]
        audios = [f["_audio"] for f in valid_features]

        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [pfx + tgt + eos for pfx, tgt in zip(prefix_texts, targets)]

        full_inputs = self.processor(
            text=full_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        prefix_inputs = self.processor(
            text=prefix_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()
        labels = full_inputs["input_ids"].clone()
        for i, pl in enumerate(prefix_lens):
            labels[i, :pl] = -100

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        full_inputs["labels"] = labels

        target_ids_list = [self._text_to_aux_ids(t) for t in targets]
        max_target_len = max((len(t) for t in target_ids_list), default=0)
        batch_size = len(valid_features)

        target_ids = torch.zeros(batch_size, max_target_len, dtype=torch.long)
        target_lengths = torch.zeros(batch_size, dtype=torch.long)
        for i, ids in enumerate(target_ids_list):
            if ids:
                target_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            target_lengths[i] = len(ids)

        # 字段名沿用 ctc_target_*，RNNT 分支也复用这份标签。
        full_inputs["ctc_target_ids"] = target_ids
        full_inputs["ctc_target_lengths"] = target_lengths
        full_inputs["texts"] = targets
        return full_inputs

    def _text_to_aux_ids(self, text: str) -> List[int]:
        if "<asr_text>" in text:
            text = text.split("<asr_text>")[-1]
        text = text.strip()
        if not text:
            return []

        ids = []
        pieces = [w for w in self.cjk_pattern.split(text) if w.strip()]
        for item in pieces:
            if self.cjk_pattern.fullmatch(item):
                ids.append(self.vocab.get(item, self.vocab.get("<unk>", 1)))
                continue

            for piece in self.sp_model.encode_as_pieces(item.upper()):
                if piece in self.vocab:
                    ids.append(self.vocab[piece])
                else:
                    ids.append(self.vocab.get(piece.replace("▁", ""), self.vocab.get("<unk>", 1)))
        return ids


class JointTrainer(Trainer):
    def __init__(self, ctc_lr=1e-3, qwen_lr=2e-5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ctc_lr = ctc_lr
        self.qwen_lr = qwen_lr

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        decay_parameters = self.get_decay_parameter_names(self.model)
        optimizer_grouped_parameters = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            is_aux_head = (
                name.startswith("ctc.")
                or name.startswith("rnnt.")
                or name.startswith("module.ctc.")
                or name.startswith("module.rnnt.")
            )
            lr = self.ctc_lr if is_aux_head else self.qwen_lr
            weight_decay = self.args.weight_decay if name in decay_parameters else 0.0

            for group in optimizer_grouped_parameters:
                if group["lr"] == lr and group["weight_decay"] == weight_decay:
                    group["params"].append(param)
                    break
            else:
                optimizer_grouped_parameters.append(
                    {"params": [param], "lr": lr, "weight_decay": weight_decay}
                )

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        optimizer_kwargs.pop("lr", None)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        if self.args.process_index == 0:
            print(f"学习率：Qwen={self.qwen_lr}，Aux={self.ctc_lr}")
        return self.optimizer

    def _prepare_inputs(self, inputs):
        if inputs is None:
            return None
        inputs = super()._prepare_inputs(inputs)
        model_dtype = getattr(self.model, "dtype", None)
        if model_dtype is not None:
            for k, v in list(inputs.items()):
                if torch.is_tensor(v) and v.is_floating_point():
                    inputs[k] = v.to(dtype=model_dtype)
        return inputs

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        base_model = self.model.module if hasattr(self.model, "module") else self.model
        aux_loss_type = getattr(base_model, "aux_loss_type", "ctc")
        aux_name = aux_loss_type.upper()

        self.model.eval()
        dataloader = self.get_eval_dataloader(eval_dataset)
        total_edits, total_chars = 0, 0
        debug_samples, max_debug = 0, 5

        with torch.no_grad():
            for batch in dataloader:
                inputs = self._prepare_inputs(batch)
                predictions = base_model.decode_feats(
                    inputs["input_features"],
                    inputs.get("feature_attention_mask", None),
                )
                texts = inputs.get("texts", [])

                for pred_text, ref_text in zip(predictions, texts):
                    ref_clean = ref_text.split("<asr_text>")[-1].strip() if "<asr_text>" in ref_text else ref_text.strip()
                    ref_pieces = self.data_collator.sp_model.encode_as_pieces(ref_clean.upper())
                    ref_processed = "".join(ref_pieces).replace("▁", " ").strip().lower()

                    if self.args.process_index == 0 and debug_samples < max_debug:
                        print(f"[验证样例{debug_samples}] {aux_name}: {pred_text}")
                        print(f"[验证样例{debug_samples}] 参考: {ref_processed}")
                        debug_samples += 1

                    total_edits += editdistance.eval(pred_text, ref_processed)
                    total_chars += len(ref_processed)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            metrics_tensor = torch.tensor(
                [total_edits, total_chars],
                dtype=torch.float64,
                device=self.args.device,
            )
            torch.distributed.all_reduce(metrics_tensor, op=torch.distributed.ReduceOp.SUM)
            total_edits, total_chars = metrics_tensor[0].item(), metrics_tensor[1].item()

        cer = total_edits / total_chars if total_chars > 0 else 0.0
        if self.args.process_index == 0:
            print(f"{aux_name} 验证 CER：{cer:.4f}")

        metric_name = f"{metric_key_prefix}_{aux_loss_type}_cer"
        metrics[metric_name] = cer
        self.log({metric_name: cer})
        return metrics

    def _load_from_checkpoint(self, resume_from_checkpoint: str, model=None):
        if model is None:
            model = self.model
        base_model = model.module if hasattr(model, "module") else model

        if self.args.process_index == 0:
            print(f"恢复训练：{resume_from_checkpoint}")

        self._load_qwen_base_weights(base_model.qwen_model, resume_from_checkpoint)
        self._load_aux_weights(base_model, resume_from_checkpoint)

        if hasattr(base_model.qwen_model, "tie_weights"):
            base_model.qwen_model.tie_weights()

        if self.args.process_index == 0:
            print("模型恢复完成")

    @staticmethod
    def _load_qwen_base_weights(qwen_model, ckpt_dir: str):
        index_file = os.path.join(ckpt_dir, "model.safetensors.index.json")
        safetensors_file = os.path.join(ckpt_dir, "model.safetensors")
        bin_file = os.path.join(ckpt_dir, "pytorch_model.bin")

        if os.path.exists(index_file):
            return load_sharded_checkpoint(qwen_model, ckpt_dir, strict=False)
        if os.path.exists(safetensors_file):
            return load_safetensors_model(qwen_model, safetensors_file, strict=False)
        if os.path.exists(bin_file):
            return qwen_model.load_state_dict(torch.load(bin_file, map_location="cpu"), strict=False)
        raise FileNotFoundError(f"没有在 checkpoint 中找到底座权重文件：{ckpt_dir}")

    def _load_aux_weights(self, base_model, ckpt_dir: str):
        if getattr(base_model, "aux_loss_type", "ctc") == "rnnt":
            aux_name, module = "rnnt_head.pt", base_model.rnnt
        else:
            aux_name, module = "ctc_head.pt", base_model.ctc

        aux_path = os.path.join(ckpt_dir, aux_name)
        if not os.path.exists(aux_path):
            if self.args.process_index == 0:
                print(f"未找到辅助头权重：{aux_path}")
            return

        missing, unexpected = module.load_state_dict(torch.load(aux_path, map_location="cpu"), strict=False)
        if self.args.process_index == 0:
            print(f"已加载辅助头：{aux_path}")
            if missing:
                print(f"缺少参数：{missing}")
            if unexpected:
                print(f"多余参数：{unexpected}")

    def save_model(self, output_dir=None, _internal_call=False):
        if self.args.process_index != 0:
            return
        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        base_model = self.model.module if hasattr(self.model, "module") else self.model
        base_model.qwen_model.save_pretrained(output_dir, safe_serialization=True)
        base_model.save_aux(output_dir)

        old_joint = os.path.join(output_dir, "pytorch_model.bin")
        if os.path.exists(old_joint):
            os.remove(old_joint)
            print(f"已删除旧权重文件：{old_joint}")


def copy_required_hf_files_for_qwen_asr(src_dir: str, dst_dir: str):
    os.makedirs(dst_dir, exist_ok=True)
    required = [
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "chat_template.json",
        "merges.txt",
        "vocab.json",
    ]
    for fn in required:
        src = os.path.join(src_dir, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, fn))


class MakeEveryCheckpointInferableCallback(TrainerCallback):
    def __init__(self, base_model_path: str):
        self.base_model_path = base_model_path

    def on_save(self, args: TrainingArguments, state, control, **kwargs):
        if args.process_index != 0:
            return control
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(ckpt_dir):
            ckpt_dir = kwargs.get("checkpoint", ckpt_dir)
        copy_required_hf_files_for_qwen_asr(self.base_model_path, ckpt_dir)
        return control


def enable_aux_only_training(joint_model: Qwen3ASRJointModel) -> None:
    """Warm up only the CTC/RNNT head with frozen Qwen features."""
    joint_model.ctc_only = True
    joint_model.ctc_weight = 1.0

    for param in joint_model.qwen_model.parameters():
        param.requires_grad = False
    for param in joint_model.aux_head.parameters():
        param.requires_grad = True


def parse_args():
    p = argparse.ArgumentParser("Qwen3-ASR Joint Finetuning")

    p.add_argument("--model_path", type=str, default="Qwen/Qwen3-ASR-1.7B")
    p.add_argument("--train_file", type=str, default="train.jsonl")
    p.add_argument("--eval_file", type=str, default="")
    p.add_argument("--output_dir", type=str, default="./qwen3-asr-joint-out")

    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--ctc_weight", type=float, default=0.3, help="辅助 loss 权重")
    p.add_argument("--aux_loss_type", type=str, default="ctc", choices=["ctc", "rnnt"])
    p.add_argument("--aux_only", type=int, default=0, help="只训练 CTC/RNNT 辅助头，冻结 Qwen 并跳过 LLM forward")
    p.add_argument("--aux_encoder_batch_size", type=int, default=1, help="CTC/RNNT 辅助头 audio encoder micro-batch，1 最稳")
    p.add_argument("--aux_streaming_train", type=int, default=0, help="训练 aux loss 时使用流式窗口：当前块 + 随机左上下文 + 右上下文")
    p.add_argument("--aux_stream_chunk_frames", type=int, default=64, help="流式 aux 训练当前块长度，单位为 feature frames，64 约等于 640ms")
    p.add_argument("--aux_stream_left_context_frames", type=int, default=64, help="流式 aux 训练最大左上下文，训练时会在 [0, max] 随机采样")
    p.add_argument("--aux_stream_right_context_frames", type=int, default=7, help="流式 aux 训练右上下文，7 约等于 70ms")
    p.add_argument("--aux_stream_random_left", type=int, default=1, help="是否随机采样左上下文长度；0 表示总是使用最大左上下文")
    p.add_argument("--aux_stream_window_batch_size", type=int, default=4, help="流式 aux 训练时一次送入 encoder 的窗口数量")
    p.add_argument("--vocab_path", type=str, default="")
    p.add_argument("--sp_model_path", type=str, default=None)

    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_acc", type=int, default=4)
    p.add_argument("--qwen_lr", type=float, default=2e-5)
    p.add_argument("--ctc_lr", type=float, default=1e-3, help="CTC/RNNT 辅助头学习率")
    p.add_argument("--epochs", type=float, default=10)
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--lr_scheduler_type", type=str, default="linear")
    p.add_argument("--warmup_ratio", type=float, default=0.02)

    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", type=int, default=1)
    p.add_argument("--persistent_workers", type=int, default=1)
    p.add_argument("--prefetch_factor", type=int, default=2)

    p.add_argument("--save_strategy", type=str, default="steps")
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=5)

    p.add_argument("--resume_from", type=str, default="")
    p.add_argument("--resume", type=int, default=0)
    return p.parse_args()


def load_vocab(vocab_path: str):
    vocab = {}
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                vocab[parts[0]] = int(parts[1])
    return vocab


def main():
    args_cli = parse_args()
    if not args_cli.train_file:
        raise ValueError("TRAIN_FILE is required.")
    if not args_cli.vocab_path:
        raise ValueError("--vocab_path is required.")
    if not args_cli.sp_model_path:
        raise ValueError("--sp_model_path is required.")

    import sentencepiece as spm
    from accelerate import PartialState

    print(f"加载 SentencePiece：{args_cli.sp_model_path}")
    sp_model = spm.SentencePieceProcessor()
    sp_model.load(args_cli.sp_model_path)

    print(f"加载词表：{args_cli.vocab_path}")
    vocab = load_vocab(args_cli.vocab_path)
    assert vocab.get("<blank>") == 0, "vocab must have <blank>: 0"
    assert vocab.get("<unk>") == 1, "vocab must have <unk>: 1"
    print(f"词表大小：{len(vocab)}")

    os.makedirs(args_cli.output_dir, exist_ok=True)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank == 0:
        with open(os.path.join(args_cli.output_dir, "ctc_vocab.json"), "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        with open(os.path.join(args_cli.output_dir, "sp_model_config.json"), "w", encoding="utf-8") as f:
            json.dump({"sp_model_path": args_cli.sp_model_path}, f)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    asr_wrapper = Qwen3ASRModel.from_pretrained(
        args_cli.model_path,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map=None,
    )
    qwen_model = asr_wrapper.model
    processor = asr_wrapper.processor

    patch_outer_forward(qwen_model)
    qwen_model.generation_config = GenerationConfig.from_model_config(qwen_model.config)
    if hasattr(qwen_model, "gradient_checkpointing_enable"):
        # Reentrant checkpointing can make DDP mark the same parameter ready
        # multiple times when the audio tower is called repeatedly for
        # aux_encoder_batch_size micro-batches inside one forward pass.
        try:
            qwen_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            print("当前 transformers 不支持非重入 checkpoint，建议关闭 gradient checkpoint。")

    joint_model = Qwen3ASRJointModel(
        qwen_model=qwen_model,
        vocab_size=len(vocab),
        vocab=vocab,
        ctc_weight=args_cli.ctc_weight,
        blank_id=0,
        ctc_only=(args_cli.aux_only == 1),
        aux_loss_type=args_cli.aux_loss_type,
        aux_encoder_batch_size=args_cli.aux_encoder_batch_size,
        aux_streaming_train=(args_cli.aux_streaming_train == 1),
        aux_stream_chunk_frames=args_cli.aux_stream_chunk_frames,
        aux_stream_left_context_frames=args_cli.aux_stream_left_context_frames,
        aux_stream_right_context_frames=args_cli.aux_stream_right_context_frames,
        aux_stream_random_left=(args_cli.aux_stream_random_left == 1),
        aux_stream_window_batch_size=args_cli.aux_stream_window_batch_size,
    )

    if args_cli.aux_only == 1:
        enable_aux_only_training(joint_model)
        if local_rank == 0:
            trainable = sum(p.numel() for p in joint_model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in joint_model.parameters())
            print("Aux-only：只训练辅助头")
            print(f"可训练参数：{trainable:,} / {total:,}")

    with PartialState().main_process_first():
        raw_ds = load_dataset(
            "json",
            data_files={
                "train": args_cli.train_file,
                **({"validation": args_cli.eval_file} if args_cli.eval_file else {}),
            },
        )
        ds = raw_ds.map(
            make_preprocess_fn_prefix_only(processor),
            num_proc=16,
            desc="Preprocessing dataset",
        )

    keep = {"prompt", "audio", "target", "prefix_text"}
    for split in ds.keys():
        drop = [c for c in ds[split].column_names if c not in keep]
        if drop:
            ds[split] = ds[split].remove_columns(drop)

    collator = DataCollatorForJointTraining(
        processor=processor,
        vocab=vocab,
        sp_model=sp_model,
        sampling_rate=args_cli.sr,
    )

    training_args = TrainingArguments(
        output_dir=args_cli.output_dir,
        per_device_train_batch_size=args_cli.batch_size,
        gradient_accumulation_steps=args_cli.grad_acc,
        learning_rate=args_cli.qwen_lr,
        num_train_epochs=args_cli.epochs,
        logging_steps=args_cli.log_steps,
        lr_scheduler_type=args_cli.lr_scheduler_type,
        warmup_ratio=args_cli.warmup_ratio,
        dataloader_num_workers=args_cli.num_workers,
        dataloader_pin_memory=(args_cli.pin_memory == 1),
        dataloader_persistent_workers=(args_cli.persistent_workers == 1),
        dataloader_prefetch_factor=args_cli.prefetch_factor if args_cli.num_workers > 0 else None,
        save_strategy=args_cli.save_strategy,
        save_steps=args_cli.save_steps,
        save_total_limit=args_cli.save_total_limit,
        save_safetensors=True,
        eval_strategy="steps",
        eval_steps=args_cli.save_steps,
        do_eval=bool(args_cli.eval_file),
        bf16=use_bf16,
        fp16=not use_bf16,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to="tensorboard",
        logging_dir="./logs_joint",
    )

    trainer = JointTrainer(
        ctc_lr=args_cli.ctc_lr,
        qwen_lr=args_cli.qwen_lr,
        model=joint_model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation", None),
        data_collator=collator,
        tokenizer=processor.tokenizer,
        callbacks=[MakeEveryCheckpointInferableCallback(base_model_path=args_cli.model_path)],
    )

    resume_from = (args_cli.resume_from or "").strip()
    if not resume_from and args_cli.resume == 1:
        resume_from = find_latest_checkpoint(training_args.output_dir) or ""

    if resume_from:
        if trainer.args.process_index == 0:
            print(f"继续训练：{resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()


if __name__ == "__main__":
    main()
