# coding=utf-8
import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import editdistance
import librosa
import torch
from datasets import load_dataset
from qwen_asr import Qwen3ASRModel
from qwen_asr.joint import Qwen3ASRJointModel
from qwen_asr.joint.defaults import DEFAULT_PROMPT, JOINT_CONFIG, TRAIN_SP_MODEL_PATH, TRAIN_VOCAB_PATH
from safetensors.torch import load_model as load_safetensors_model
from transformers import GenerationConfig, Trainer, TrainerCallback, TrainingArguments
from transformers.modeling_utils import load_sharded_checkpoint


DEFAULT_ATTN_IMPLEMENTATION = "flash_attention_2"


TASKS = {"llm", "proj", "encoder", "ctc", "rnnt"}
PROJ_MODULES = ("proj1", "act", "proj2")
PROJ_PARAM_KEYS = tuple(f".{name}." for name in PROJ_MODULES)


def read_joint_cfg(path: str) -> Dict:
    cfg_path = os.path.join(path or "", JOINT_CONFIG)
    if not os.path.exists(cfg_path):
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_ctc_cfg(args, source_cfg: Dict) -> None:
    if args.ctc_adapter == "auto":
        args.ctc_adapter = source_cfg.get("ctc_adapter", "mlp")


def ctc_train_cfg(args) -> Dict:
    return {"adapter_type": args.ctc_adapter}


def sync_audio_attention(qwen_model) -> None:
    audio_tower = qwen_model.thinker.audio_tower
    audio_tower.config._attn_implementation = DEFAULT_ATTN_IMPLEMENTATION
    qwen_model.config.thinker_config.audio_config._attn_implementation = DEFAULT_ATTN_IMPLEMENTATION
    qwen_model.thinker.config.audio_config._attn_implementation = DEFAULT_ATTN_IMPLEMENTATION


def csv_set(value: str, allowed: set, name: str) -> tuple:
    out = []
    for part in (value or "").split(","):
        item = part.strip().lower()
        if not item:
            continue
        if item not in allowed:
            raise ValueError(f"不支持的 {name}: {item}")
        if item not in out:
            out.append(item)
    if not out:
        raise ValueError(f"{name} 不能为空")
    return tuple(out)


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


def latest_checkpoint(output_dir: str) -> Optional[str]:
    if not output_dir or not os.path.isdir(output_dir):
        return None
    best = None
    for name in os.listdir(output_dir):
        m = _CKPT_RE.match(name)
        path = os.path.join(output_dir, name)
        if m and os.path.isdir(path) and (best is None or int(m.group(1)) > best[0]):
            best = (int(m.group(1)), path)
    return best[1] if best else None


def load_audio(path: str, sr: int = 16000, max_duration: float = 30.0):
    try:
        wav, _ = librosa.load(path, sr=sr, mono=True)
        if len(wav) / sr > max_duration:
            print(f"音频过长，跳过：{path}")
            return None
        return wav
    except Exception as exc:
        print(f"音频读取失败，跳过：{path}，{exc}")
        return None


def build_prefix_messages(prompt: str, audio_array):
    return [
        {"role": "system", "content": prompt or DEFAULT_PROMPT},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]


def make_preprocess_fn(processor):
    def _preprocess(ex: Dict[str, Any]) -> Dict[str, Any]:
        prompt = ex.get("prompt") or DEFAULT_PROMPT
        target = ex.get("target") or ex.get("text") or ""
        prefix_text = processor.apply_chat_template(
            [build_prefix_messages(prompt, None)],
            add_generation_prompt=True,
            tokenize=False,
        )[0]
        return {
            "audio": ex["audio"],
            "prompt": prompt,
            "target": target,
            "prefix_text": prefix_text,
        }

    return _preprocess


@dataclass
class DataCollatorForJointTraining:
    processor: Any
    vocab: dict
    sp_model: Any
    sampling_rate: int = 16000

    def __post_init__(self):
        self.cjk = re.compile(r"([\u4e00-\u9fff])")

    def __call__(self, features: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        rows = []
        for item in features:
            wav = load_audio(item["audio"], sr=self.sampling_rate)
            if wav is not None:
                item = dict(item)
                item["_audio"] = wav
                rows.append(item)
        if not rows:
            return None

        prefix_texts = [x["prefix_text"] for x in rows]
        targets = [x["target"] for x in rows]
        audios = [x["_audio"] for x in rows]
        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [prefix + target + eos for prefix, target in zip(prefix_texts, targets)]

        full = self.processor(text=full_texts, audio=audios, return_tensors="pt", padding=True, truncation=False)
        prefix = self.processor(text=prefix_texts, audio=audios, return_tensors="pt", padding=True, truncation=False)
        labels = full["input_ids"].clone()
        for idx, length in enumerate(prefix["attention_mask"].sum(dim=1).tolist()):
            labels[idx, :length] = -100
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        full["labels"] = labels

        ids_list = [self.text_to_ids(text) for text in targets]
        max_len = max((len(ids) for ids in ids_list), default=0)
        target_ids = torch.zeros(len(rows), max_len, dtype=torch.long)
        target_lens = torch.zeros(len(rows), dtype=torch.long)
        for idx, ids in enumerate(ids_list):
            if ids:
                target_ids[idx, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            target_lens[idx] = len(ids)
        full["ctc_target_ids"] = target_ids
        full["ctc_target_lengths"] = target_lens
        full["texts"] = targets
        return full

    def text_to_ids(self, text: str) -> List[int]:
        if "<asr_text>" in text:
            text = text.split("<asr_text>")[-1]
        ids = []
        for part in [x for x in self.cjk.split(text.strip()) if x.strip()]:
            if self.cjk.fullmatch(part):
                ids.append(self.vocab.get(part, self.vocab.get("<unk>", 1)))
                continue
            for piece in self.sp_model.encode_as_pieces(part.upper()):
                ids.append(self.vocab.get(piece, self.vocab.get(piece.replace("▁", ""), self.vocab.get("<unk>", 1))))
        return ids


def module_set_trainable(module, enabled: bool) -> None:
    if module is None or not hasattr(module, "parameters"):
        return
    for param in module.parameters():
        param.requires_grad = enabled


def set_proj_trainable(audio_tower, enabled: bool) -> None:
    for name in PROJ_MODULES:
        module_set_trainable(getattr(audio_tower, name, None), enabled)


def set_trainable(model: Qwen3ASRJointModel, tasks: Iterable[str]) -> None:
    tasks = set(tasks)
    module_set_trainable(model, False)
    audio_tower = model.qwen_model.thinker.audio_tower

    if "llm" in tasks:
        module_set_trainable(model.qwen_model, True)
        module_set_trainable(audio_tower, False)
    if "encoder" in tasks:
        module_set_trainable(audio_tower, True)
        if "proj" not in tasks:
            set_proj_trainable(audio_tower, False)
    if "proj" in tasks:
        set_proj_trainable(audio_tower, True)
    if "ctc" in tasks:
        module_set_trainable(model.ctc, True)
    if "rnnt" in tasks:
        module_set_trainable(model.rnnt, True)


def group_name(param_name: str) -> str:
    if param_name.startswith(("ctc.", "module.ctc.")):
        return "ctc"
    if param_name.startswith(("rnnt.", "module.rnnt.")):
        return "rnnt"
    if ".audio_tower." in param_name:
        return "proj" if any(x in param_name for x in PROJ_PARAM_KEYS) else "encoder"
    return "llm"


class JointTrainer(Trainer):
    def __init__(self, lr_by_group: Dict[str, float], save_heads=(), head_source="", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr_by_group = lr_by_group
        self.save_heads = tuple(save_heads)
        self.head_source = head_source
        self._loss_sums = {}
        self._loss_count = 0

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        decay_names = self.get_decay_parameter_names(self.model)
        groups = {}
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            group = group_name(name)
            key = (group, self.lr_by_group[group], self.args.weight_decay if name in decay_names else 0.0)
            groups.setdefault(key, []).append(param)

        params = [
            {"params": values, "name": name, "lr": lr, "weight_decay": wd}
            for (name, lr, wd), values in groups.items()
        ]
        opt_cls, opt_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        opt_kwargs.pop("lr", None)
        self.optimizer = opt_cls(params, **opt_kwargs)
        if self.args.process_index == 0:
            print("学习率：" + "，".join(f"{g['name']}={g['lr']}" for g in params))
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        if isinstance(outputs, dict) and model.training:
            for key in ("loss", "llm_loss", "ctc_loss", "rnnt_loss"):
                value = loss if key == "loss" else outputs.get(key)
                if torch.is_tensor(value):
                    self._loss_sums[key] = self._loss_sums.get(key, 0.0) + float(value.detach().float().item())
            self._loss_count += 1
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        if self._loss_count > 0 and "loss" in logs:
            merged = {k: v / self._loss_count for k, v in self._loss_sums.items()}
            merged.update({k: v for k, v in logs.items() if k not in merged and k != "learning_rate"})
            for group in self.optimizer.param_groups if self.optimizer else []:
                merged[f"{group.get('name')}_lr"] = group["lr"]
            logs = merged
            self._loss_sums = {}
            self._loss_count = 0
        return super().log(logs, *args, **kwargs)

    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        dtype = getattr(self.model, "dtype", None)
        if dtype is not None:
            for key, value in list(inputs.items()):
                if torch.is_tensor(value) and value.is_floating_point():
                    inputs[key] = value.to(dtype=dtype)
        return inputs

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        base = self.model.module if hasattr(self.model, "module") else self.model
        heads = [x for x in ("ctc", "rnnt") if x in base.heads]
        if not heads:
            return metrics

        dataloader = self.get_eval_dataloader(eval_dataset)
        self.model.eval()
        for head in heads:
            edits, chars, shown = 0, 0, 0
            with torch.no_grad():
                for batch in dataloader:
                    inputs = self._prepare_inputs(batch)
                    preds = base.decode_feats(
                        inputs["input_features"],
                        inputs.get("feature_attention_mask"),
                        head=head,
                    )
                    for pred, ref in zip(preds, inputs.get("texts", [])):
                        ref = ref.split("<asr_text>")[-1].strip() if "<asr_text>" in ref else ref.strip()
                        ref = "".join(self.data_collator.sp_model.encode_as_pieces(ref.upper())).replace("▁", " ").strip().lower()
                        if self.args.process_index == 0 and shown < 3:
                            print(f"[验证样例{shown}] {head.upper()}: {pred}")
                            print(f"[验证样例{shown}] 参考: {ref}")
                            shown += 1
                        edits += editdistance.eval(pred, ref)
                        chars += len(ref)
            cer = edits / chars if chars else 0.0
            key = f"{metric_key_prefix}_{head}_cer"
            metrics[key] = cer
            if self.args.process_index == 0:
                print(f"{head.upper()} 验证 CER：{cer:.4f}")
        return metrics

    def _load_from_checkpoint(self, resume_from_checkpoint: str, model=None):
        model = model or self.model
        base = model.module if hasattr(model, "module") else model
        if self.args.process_index == 0:
            print(f"恢复训练：{resume_from_checkpoint}")
        self._load_qwen(base.qwen_model, resume_from_checkpoint)
        load_heads(base, resume_from_checkpoint, is_main=self.args.process_index == 0, strict=False)

    @staticmethod
    def _load_qwen(qwen_model, ckpt_dir: str):
        if os.path.exists(os.path.join(ckpt_dir, "model.safetensors.index.json")):
            return load_sharded_checkpoint(qwen_model, ckpt_dir, strict=False)
        if os.path.exists(os.path.join(ckpt_dir, "model.safetensors")):
            return load_safetensors_model(qwen_model, os.path.join(ckpt_dir, "model.safetensors"), strict=False)
        if os.path.exists(os.path.join(ckpt_dir, "pytorch_model.bin")):
            return qwen_model.load_state_dict(torch.load(os.path.join(ckpt_dir, "pytorch_model.bin"), map_location="cpu"), strict=False)
        raise FileNotFoundError(f"没有在 checkpoint 中找到底座权重文件：{ckpt_dir}")

    def save_model(self, output_dir=None, _internal_call=False):
        if self.args.process_index != 0:
            return
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        base = self.model.module if hasattr(self.model, "module") else self.model
        copy_hf_files(self.head_source, output_dir)
        sync_audio_window_config(base.qwen_model)
        base.qwen_model.save_pretrained(output_dir, safe_serialization=True)
        base.save_aux(output_dir, heads=self.save_heads, copy_heads_from=self.head_source)
        old = os.path.join(output_dir, "pytorch_model.bin")
        if os.path.exists(old):
            os.remove(old)


def copy_hf_files(src_dir: str, dst_dir: str):
    os.makedirs(dst_dir, exist_ok=True)
    for name in (
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
    ):
        src = os.path.join(src_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, name))


class InferableCheckpointCallback(TrainerCallback):
    def __init__(self, base_model_path: str, save_heads=()):
        self.base_model_path = base_model_path
        self.save_heads = tuple(save_heads)

    def on_save(self, args: TrainingArguments, state, control, **kwargs):
        if args.process_index != 0:
            return control
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        copy_hf_files(self.base_model_path, ckpt_dir)
        model = kwargs.get("model")
        if model is not None:
            base = model.module if hasattr(model, "module") else model
            sync_audio_window_config(base.qwen_model)
            base.save_aux(ckpt_dir, heads=self.save_heads, copy_heads_from=self.base_model_path)
        return control


def sync_audio_window_config(qwen_model, n_window: Optional[int] = None, n_window_infer: Optional[int] = None):
    audio_tower = qwen_model.thinker.audio_tower
    audio_config = qwen_model.config.thinker_config.audio_config
    if n_window is not None:
        audio_tower.n_window = n_window
        audio_tower.config.n_window = n_window
        audio_config.n_window = n_window
    if n_window_infer is not None:
        audio_tower.n_window_infer = n_window_infer
        audio_tower.config.n_window_infer = n_window_infer
        audio_config.n_window_infer = n_window_infer
    return audio_tower


def load_heads(model: Qwen3ASRJointModel, path: str, heads=None, is_main: bool = True, strict: bool = True):
    for name in model.heads if heads is None else heads:
        head_path = os.path.join(path, f"{name}_head.pt")
        if not os.path.exists(head_path):
            if strict:
                raise FileNotFoundError(f"未找到 {name.upper()} 头：{head_path}")
            continue
        missing, unexpected = getattr(model, name).load_state_dict(torch.load(head_path, map_location="cpu"), strict=False)
        if is_main:
            print(f"已加载 {name.upper()} 头：{head_path}")
            if missing:
                print(f"缺少参数：{missing}")
            if unexpected:
                print(f"多余参数：{unexpected}")


def existing_heads(path: str) -> tuple:
    return tuple(
        name
        for name in ("ctc", "rnnt")
        if os.path.exists(os.path.join(path, f"{name}_head.pt"))
    )


def parse_args():
    p = argparse.ArgumentParser("Qwen3-ASR 训练")
    p.add_argument("--model_path", type=str, default="Qwen/Qwen3-ASR-1.7B")
    p.add_argument("--train_file", type=str, required=True)
    p.add_argument("--eval_file", type=str, default="")
    p.add_argument("--output_dir", type=str, default="./qwen3-asr-joint-out")
    p.add_argument("--train", type=str, default="llm,ctc", help="逗号组合：llm,proj,encoder,ctc,rnnt")
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--vocab_path", type=str, default=TRAIN_VOCAB_PATH)
    p.add_argument("--sp_model_path", type=str, default=TRAIN_SP_MODEL_PATH)

    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_acc", type=int, default=4)
    p.add_argument("--epochs", type=float, default=10)
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--logging_dir", type=str, default="./logs_joint")
    p.add_argument("--lr_scheduler_type", type=str, default="linear")
    p.add_argument("--warmup_ratio", type=float, default=0.02)
    p.add_argument("--lr_llm", type=float, default=2e-5)
    p.add_argument("--lr_proj", type=float, default=2e-5)
    p.add_argument("--lr_encoder", type=float, default=1e-5)
    p.add_argument("--lr_ctc", type=float, default=1e-3)
    p.add_argument("--lr_rnnt", type=float, default=1e-3)
    p.add_argument("--w_llm", type=float, default=1.0)
    p.add_argument("--w_ctc", type=float, default=1.0)
    p.add_argument("--w_rnnt", type=float, default=1.0)

    p.add_argument("--ctc_adapter", type=str, default="auto", help="auto/mlp/moe，auto 会继承源 checkpoint")

    p.add_argument("--audio_n_window", type=int, default=0)
    p.add_argument("--audio_n_window_infer", type=int, default=0)
    p.add_argument("--stream_train", type=int, default=0, help="1 表示 CTC/RNNT 按流式窗口特征训练")

    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", type=int, default=1)
    p.add_argument("--persistent_workers", type=int, default=1)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--save_strategy", type=str, default="steps")
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=5)
    p.add_argument("--resume_from", type=str, default="")
    p.add_argument("--resume", type=int, default=0)
    args = p.parse_args()
    source_cfg = read_joint_cfg(args.model_path)
    resolve_ctc_cfg(args, source_cfg)
    args.tasks = csv_set(args.train, TASKS, "train")
    if "proj" in args.tasks and "llm" not in args.tasks:
        raise ValueError("proj 只作用于 LLM 路径，请和 llm 一起训练。")
    args.loss_tasks = tuple(x for x in args.tasks if x not in ("encoder", "proj"))
    if not args.loss_tasks:
        raise ValueError("--train 至少需要包含 llm/ctc/rnnt 之一；encoder/proj 只能配合这些 loss 一起训练。")
    save_heads = [*existing_heads(args.model_path)]
    active_heads = []
    for name in args.loss_tasks:
        if name in ("ctc", "rnnt"):
            if name not in save_heads:
                save_heads.append(name)
            active_heads.append(name)
    args.heads = tuple(save_heads)
    args.active_heads = tuple(active_heads)
    return args


def load_vocab(path: str):
    vocab = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                vocab[parts[0]] = int(parts[1])
    return vocab


def main():
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main = local_rank == 0

    import sentencepiece as spm
    from accelerate import PartialState

    sp_model = spm.SentencePieceProcessor()
    sp_model.load(args.sp_model_path)
    vocab = load_vocab(args.vocab_path)
    assert vocab.get("<blank>") == 0, "vocab must have <blank>: 0"
    assert vocab.get("<unk>") == 1, "vocab must have <unk>: 1"

    os.makedirs(args.output_dir, exist_ok=True)
    if is_main:
        print(f"训练任务：{','.join(args.tasks)}")
        print(f"CTC adapter：{args.ctc_adapter}")
        print(f"流式训练：{'开启' if args.stream_train == 1 else '关闭'}")
        print(f"词表大小：{len(vocab)}")
        with open(os.path.join(args.output_dir, "ctc_vocab.json"), "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        with open(os.path.join(args.output_dir, "sp_model_config.json"), "w", encoding="utf-8") as f:
            json.dump({"sp_model_path": args.sp_model_path}, f)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    wrapper = Qwen3ASRModel.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map=None,
        attn_implementation=DEFAULT_ATTN_IMPLEMENTATION,
    )
    qwen_model = wrapper.model
    sync_audio_attention(qwen_model)
    processor = wrapper.processor
    audio_tower = sync_audio_window_config(
        qwen_model,
        n_window=args.audio_n_window if args.audio_n_window > 0 else None,
        n_window_infer=args.audio_n_window_infer if args.audio_n_window_infer > 0 else None,
    )
    if audio_tower.n_window_infer < audio_tower.n_window * 2:
        raise ValueError("audio_n_window_infer 必须 >= audio_n_window * 2")
    if is_main:
        print(f"Audio 窗口：n_window={audio_tower.n_window}，n_window_infer={audio_tower.n_window_infer}")

    patch_outer_forward(qwen_model)
    qwen_model.generation_config = GenerationConfig.from_model_config(qwen_model.config)
    if hasattr(qwen_model, "gradient_checkpointing_enable"):
        try:
            qwen_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            if is_main:
                print("当前 transformers 不支持非重入 checkpoint。")

    model = Qwen3ASRJointModel(
        qwen_model=qwen_model,
        vocab_size=len(vocab),
        vocab=vocab,
        blank_id=0,
        heads=args.active_heads,
        train_tasks=args.loss_tasks,
        loss_weights={"llm": args.w_llm, "ctc": args.w_ctc, "rnnt": args.w_rnnt},
        ctc_config=ctc_train_cfg(args),
        stream_train=(args.stream_train == 1),
    )
    model.processor = processor
    if args.active_heads:
        load_heads(model, args.model_path, heads=args.active_heads, is_main=is_main, strict=False)
    set_trainable(model, args.tasks)
    if is_main:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"可训练参数：{trainable:,} / {total:,}")

    with PartialState().main_process_first():
        raw_ds = load_dataset(
            "json",
            data_files={"train": args.train_file, **({"validation": args.eval_file} if args.eval_file else {})},
        )
        ds = raw_ds.map(make_preprocess_fn(processor), num_proc=16, desc="Preprocessing dataset")

    keep = {"prompt", "audio", "target", "prefix_text"}
    for split in ds.keys():
        drop = [col for col in ds[split].column_names if col not in keep]
        if drop:
            ds[split] = ds[split].remove_columns(drop)

    collator = DataCollatorForJointTraining(processor, vocab, sp_model, sampling_rate=args.sr)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr_llm,
        num_train_epochs=args.epochs,
        logging_steps=args.log_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=(args.pin_memory == 1),
        dataloader_persistent_workers=(args.persistent_workers == 1),
        dataloader_prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_safetensors=True,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        do_eval=bool(args.eval_file),
        bf16=use_bf16,
        fp16=not use_bf16,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to="tensorboard",
        logging_dir=args.logging_dir,
    )
    trainer = JointTrainer(
        lr_by_group={
            "llm": args.lr_llm,
            "proj": args.lr_proj,
            "encoder": args.lr_encoder,
            "ctc": args.lr_ctc,
            "rnnt": args.lr_rnnt,
        },
        save_heads=args.heads,
        head_source=args.model_path,
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation", None),
        data_collator=collator,
        tokenizer=processor.tokenizer,
        callbacks=[InferableCheckpointCallback(base_model_path=args.model_path, save_heads=args.heads)],
    )

    resume_from = (args.resume_from or "").strip()
    if not resume_from and args.resume == 1:
        resume_from = latest_checkpoint(training_args.output_dir) or ""
    trainer.train(resume_from_checkpoint=resume_from or None)
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
