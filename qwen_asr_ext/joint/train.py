# coding=utf-8
import argparse
import json
import os
import re
import shutil
import statistics
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import editdistance
import librosa
import torch
import torch.distributed as dist
from datasets import load_dataset
from filelock import FileLock
from qwen_asr import Qwen3ASRModel
from qwen_asr_ext.glclap.train import (
    ENGLISH_WORD_DF,
    TEXT_MODEL as GLCLAP_TEXT_MODEL,
    load_word_df,
    parse_text,
    sample_subtext,
)
from qwen_asr_ext.joint import Qwen3ASRJointModel
from qwen_asr_ext.joint.defaults import (
    DEFAULT_ATTN_IMPLEMENTATION, DEFAULT_PROMPT, TRAIN_MASK_CURRENT_FRAMES, TRAIN_MASK_LEFT_FRAMES,
    TRAIN_MASK_RIGHT_FRAMES, TRAIN_SP_MODEL_PATH, TRAIN_VOCAB_PATH,
)
from qwen_asr_ext.joint.encoder import conv_len, encode_offline, encode_train_mask, feature_lens
from qwen_asr_ext.joint.model import names, read_cfg
from safetensors.torch import load_model as load_safetensors_model
from transformers import BertTokenizer, Trainer, TrainingArguments
from transformers.modeling_utils import load_sharded_checkpoint


TASKS = {"llm", "proj", "encoder", "ctc", "rnnt", "glclap"}
PROJ_MODULES = ("proj1", "act", "proj2")
PROJ_PARAM_KEYS = tuple(f".{name}." for name in PROJ_MODULES)
MAX_TRAIN_AUDIO_SECONDS = 30.0


_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def latest_checkpoint(output_dir: str) -> Optional[str]:
    if not output_dir or not os.path.isdir(output_dir):
        return None
    best = None
    for name in os.listdir(output_dir):
        m = _CKPT_RE.match(name)
        path = os.path.join(output_dir, name)
        complete = (
            os.path.isfile(os.path.join(path, "trainer_state.json"))
            and any(
                os.path.isfile(os.path.join(path, model_file))
                for model_file in (
                    "model.safetensors",
                    "model.safetensors.index.json",
                    "pytorch_model.bin",
                )
            )
        )
        if (
            m
            and os.path.isdir(path)
            and complete
            and (best is None or int(m.group(1)) > best[0])
        ):
            best = (int(m.group(1)), path)
    return best[1] if best else None


@dataclass
class DataCollatorForJointTraining:
    processor: Any
    vocab: dict
    sp_model: Any
    sampling_rate: int = 16000
    stream_train: bool = False
    need_llm: bool = True
    need_aux: bool = True
    need_glclap: bool = False
    glclap_tokenizer: Any = None
    glclap_word_df: Optional[dict] = None
    glclap_median_df: float = 1.0
    max_subtext_units: int = 8
    max_audio_seconds: float = MAX_TRAIN_AUDIO_SECONDS

    def __post_init__(self):
        self.cjk = re.compile(r"([\u4e00-\u9fff])")

    def __call__(self, features: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        rows = []
        for item in features:
            try:
                wav, _ = librosa.load(item["audio"], sr=self.sampling_rate, mono=True)
            except Exception as exc:
                print(f"音频读取失败，跳过：{item['audio']}，{exc}")
                continue
            if len(wav) / self.sampling_rate > self.max_audio_seconds:
                print(
                    f"音频超过{self.max_audio_seconds:g}秒，跳过：{item['audio']}"
                )
                continue
            rows.append((item, wav))
        if not rows:
            return None

        audios = [wav for _, wav in rows]
        targets = [(item.get("target") or item.get("text") or "") for item, _ in rows]
        if self.need_llm:
            prefix_texts = [
                self.processor.apply_chat_template(
                    [[
                        {"role": "system", "content": item.get("prompt") or DEFAULT_PROMPT},
                        {"role": "user", "content": [{"type": "audio", "audio": None}]},
                    ]],
                    add_generation_prompt=True,
                    tokenize=False,
                )[0]
                for item, _ in rows
            ]
            eos = self.processor.tokenizer.eos_token or ""
            full_texts = [prefix + target + eos for prefix, target in zip(prefix_texts, targets)]
            full = self.processor(text=full_texts, audio=audios, return_tensors="pt", padding=True, truncation=False)
            audio_token = self.processor.audio_token
            if self.stream_train:
                audio_lens = [
                    max(1, int(conv_len(int(length))))
                    for length in full["feature_attention_mask"].sum(dim=1).tolist()
                ]
                full.update(self.processor.tokenizer(
                    [text.replace(audio_token, audio_token * length, 1) for text, length in zip(full_texts, audio_lens)],
                    return_tensors="pt", padding=True, truncation=False,
                ))
            else:
                audio_id = self.processor.tokenizer.convert_tokens_to_ids(audio_token)
                audio_lens = (full["input_ids"] == audio_id).sum(dim=1).tolist()
            prefix = self.processor.tokenizer(
                [text.replace(audio_token, audio_token * length, 1) for text, length in zip(prefix_texts, audio_lens)],
                return_tensors="pt", padding=True, truncation=False,
            )

            labels = full["input_ids"].clone()
            for idx, length in enumerate(prefix["attention_mask"].sum(dim=1).tolist()):
                labels[idx, :length] = -100
            pad_id = self.processor.tokenizer.pad_token_id
            if pad_id is not None:
                labels[labels == pad_id] = -100
            full["labels"] = labels
        else:
            full = self.processor.feature_extractor(
                audios, sampling_rate=self.sampling_rate, return_tensors="pt",
                padding=True, truncation=False, return_attention_mask=True,
            )
            full["feature_attention_mask"] = full.pop("attention_mask")

        if self.need_glclap:
            parsed = [parse_text(text) for text in targets]
            glclap_texts = [text for _, text in parsed]
            subtexts = [
                sample_subtext(
                    text,
                    language,
                    self.max_subtext_units,
                    self.glclap_word_df,
                    self.glclap_median_df,
                )
                for language, text in parsed
            ]
            text_tokens = self.glclap_tokenizer(
                glclap_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            subtext_tokens = self.glclap_tokenizer(
                subtexts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            full["glclap_text_input_ids"] = text_tokens["input_ids"]
            full["glclap_text_attention_mask"] = text_tokens["attention_mask"]
            full["glclap_subtext_input_ids"] = subtext_tokens["input_ids"]
            full["glclap_subtext_attention_mask"] = subtext_tokens["attention_mask"]

        if not self.need_aux:
            return full

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


def set_trainable(model: Qwen3ASRJointModel, tasks: Iterable[str]) -> None:
    def set_module(module, enabled: bool) -> None:
        if module is not None and hasattr(module, "parameters"):
            for param in module.parameters():
                param.requires_grad = enabled

    tasks = set(tasks)
    audio_tower = model.qwen_model.thinker.audio_tower
    set_module(model, False)
    if "llm" in tasks:
        set_module(model.qwen_model, True)
        set_module(audio_tower, False)
    if "encoder" in tasks:
        set_module(audio_tower, True)
        if "proj" not in tasks:
            for name in PROJ_MODULES:
                set_module(getattr(audio_tower, name, None), False)
    if "proj" in tasks:
        for name in PROJ_MODULES:
            set_module(getattr(audio_tower, name, None), True)
    set_module(model.ctc, "ctc" in tasks)
    set_module(model.rnnt, "rnnt" in tasks)
    set_module(model.glclap, "glclap" in tasks)


def weights_from_grad_norms(
    medians: Dict[str, float],
    current: Dict[str, float],
    target_ratios: Dict[str, float],
) -> Dict[str, float]:
    """把辅助任务的目标梯度比例换算成loss权重。"""
    if medians.get("llm", 0.0) <= 0:
        raise ValueError("LLM梯度范数必须大于0。")
    weights = dict(current)
    llm_weight = weights.get("llm", 1.0)
    for name, ratio in target_ratios.items():
        if medians.get(name, 0.0) <= 0:
            raise ValueError(f"{name}梯度范数必须大于0。")
        value = ratio * llm_weight * medians["llm"] / medians[name]
        weights[name] = min(100.0, max(1e-4, value))
    return weights


def calibrate_loss_weights(
    model: Qwen3ASRJointModel,
    trainer,
    steps: int,
    target_ratios: Dict[str, float],
    output_dir: str,
) -> None:
    """按共享 Encoder 梯度范数校准辅助任务权重。"""
    if steps <= 0:
        return
    raw_keys = {"llm": "llm_loss", "ctc": "ctc_loss", "glclap": "glclap_loss"}
    tasks = ["llm", *[name for name in ("ctc", "glclap") if name in target_ratios]]
    layers = model.qwen_model.thinker.audio_tower.layers[-2:]
    probe = [param for layer in layers for param in layer.parameters() if param.requires_grad]
    if not probe:
        raise ValueError("loss校准要求阶段二放开Audio Encoder。")

    norms = {name: [] for name in tasks}
    iterator = iter(trainer.get_train_dataloader())
    model.train()
    for _ in range(steps):
        try:
            inputs = next(iterator)
        except StopIteration:
            iterator = iter(trainer.get_train_dataloader())
            inputs = next(iterator)
        inputs = trainer._prepare_inputs(inputs)
        with trainer.accelerator.autocast():
            outputs = model(**inputs)
        missing = [name for name in tasks if raw_keys[name] not in outputs]
        if missing:
            raise ValueError(f"loss校准缺少任务输出：{missing}")
        for name in tasks:
            grads = torch.autograd.grad(
                outputs[raw_keys[name]],
                probe,
                retain_graph=True,
                allow_unused=True,
            )
            squared = sum(
                grad.detach().float().square().sum()
                for grad in grads
                if grad is not None
            )
            if not torch.is_tensor(squared):
                squared = torch.zeros((), device=next(model.parameters()).device)
            if dist.is_initialized():
                dist.all_reduce(squared, op=dist.ReduceOp.SUM)
                squared /= dist.get_world_size()
            norms[name].append(float(squared.sqrt().cpu()))
        del outputs

    medians = {name: statistics.median(values) for name, values in norms.items()}
    model.loss_weights.update(
        weights_from_grad_norms(medians, model.loss_weights, target_ratios)
    )

    result = {
        "steps": steps,
        "gradient_norm_median": medians,
        "target_gradient_ratio": target_ratios,
        "loss_weights": model.loss_weights,
    }
    if trainer.args.process_index == 0:
        path = os.path.join(output_dir, "loss_calibration.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print("Loss校准：" + json.dumps(result, ensure_ascii=False))


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

        def group_name(param_name: str) -> str:
            if param_name.startswith(("ctc.", "module.ctc.")):
                return "ctc"
            if param_name.startswith(("rnnt.", "module.rnnt.")):
                return "rnnt"
            if param_name.startswith(("glclap.text_encoder.", "module.glclap.text_encoder.")):
                return "glclap_text"
            if param_name.startswith(("glclap.", "module.glclap.")):
                return "glclap_proj"
            if ".audio_tower." in param_name:
                return "proj" if any(x in param_name for x in PROJ_PARAM_KEYS) else "encoder"
            return "llm"

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
            base = model.module if hasattr(model, "module") else model
            for name in ("llm", "ctc", "rnnt", "glclap"):
                raw = outputs.get(f"{name}_loss")
                if torch.is_tensor(raw):
                    weight = float(base.loss_weights.get(name, 1.0))
                    outputs[f"{name}_weight"] = raw.new_tensor(weight)
                    outputs[f"weighted_{name}_loss"] = weight * raw
            for key in (
                "loss", "llm_loss", "ctc_loss", "rnnt_loss", "glclap_loss",
                "glclap_global_loss", "glclap_local_loss",
                "llm_weight", "ctc_weight", "rnnt_weight", "glclap_weight",
                "weighted_llm_loss", "weighted_ctc_loss", "weighted_rnnt_loss",
                "weighted_glclap_loss",
            ):
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
        logs = {
            key: (
                float(f"{value:.3e}")
                if key.endswith("_lr")
                else round(value, 3)
            )
            if isinstance(value, float)
            else value
            for key, value in logs.items()
        }
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
        stats = {head: {"edits": 0, "chars": 0, "shown": 0} for head in heads}
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                inputs = self._prepare_inputs(batch)
                if "input_features" not in inputs:
                    return metrics
                ref_param = next(base.qwen_model.parameters())
                feats = inputs["input_features"].to(device=ref_param.device, dtype=ref_param.dtype)
                mask = inputs.get("feature_attention_mask")
                mask = mask.to(device=ref_param.device) if mask is not None else None
                lens_in = feature_lens(feats, mask)
                tower = base.qwen_model.thinker.audio_tower
                if base.stream_train:
                    hs, _, lens = encode_train_mask(
                        tower, feats, lens_in,
                        TRAIN_MASK_LEFT_FRAMES, TRAIN_MASK_CURRENT_FRAMES, TRAIN_MASK_RIGHT_FRAMES,
                        False,
                    )
                else:
                    hs, _, lens = encode_offline(tower, feats, lens_in, False)

                refs = []
                for ref in inputs.get("texts", []):
                    ref = ref.split("<asr_text>")[-1].strip() if "<asr_text>" in ref else ref.strip()
                    refs.append("".join(self.data_collator.sp_model.encode_as_pieces(ref.upper())).replace("▁", " ").strip().lower())
                for head in heads:
                    stat = stats[head]
                    for pred, ref in zip(base.decode_aux(head, hs, lens), refs):
                        if self.args.process_index == 0 and stat["shown"] < 3:
                            print(f"[验证样例{stat['shown']}] {head.upper()}: {pred}")
                            print(f"[验证样例{stat['shown']}] 参考: {ref}")
                            stat["shown"] += 1
                        stat["edits"] += editdistance.eval(pred, ref)
                        stat["chars"] += len(ref)

        for head, stat in stats.items():
            cer = stat["edits"] / stat["chars"] if stat["chars"] else 0.0
            key = f"{metric_key_prefix}_{head}_cer"
            metrics[key] = cer
            self.log({key: cer})
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
        cfg = read_cfg(resume_from_checkpoint)
        if cfg.get("loss_weights"):
            base.loss_weights.update(cfg["loss_weights"])

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
        for name in (
            "config.json", "generation_config.json", "preprocessor_config.json",
            "processor_config.json", "tokenizer_config.json", "tokenizer.json",
            "special_tokens_map.json", "chat_template.json", "merges.txt", "vocab.json",
        ):
            src = os.path.join(self.head_source, name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(output_dir, name))
        gen = getattr(base.qwen_model, "generation_config", None)
        if gen is not None and not getattr(gen, "do_sample", False):
            gen.temperature = None
        base.qwen_model.save_pretrained(output_dir, safe_serialization=False)
        base.save_aux(output_dir, heads=self.save_heads, copy_heads_from=self.head_source)


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


def parse_args():
    p = argparse.ArgumentParser("Qwen3-ASR 训练")
    p.add_argument("--model_path", type=str, default="Qwen/Qwen3-ASR-1.7B")
    p.add_argument("--train_file", type=str, required=True)
    p.add_argument("--eval_file", type=str, default="")
    p.add_argument("--output_dir", type=str, default="./qwen3-asr-joint-out")
    p.add_argument(
        "--train",
        type=str,
        default="llm,ctc",
        help="逗号组合：llm,proj,encoder,ctc,rnnt,glclap",
    )
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
    p.add_argument("--lr_glclap_text", type=float, default=1e-5)
    p.add_argument("--lr_glclap_proj", type=float, default=1e-3)
    p.add_argument("--w_llm", type=float, default=1.0)
    p.add_argument("--w_ctc", type=float, default=1.0)
    p.add_argument("--w_rnnt", type=float, default=1.0)
    p.add_argument("--w_glclap", type=float, default=1.0)
    p.add_argument("--glclap_text_model", type=str, default=GLCLAP_TEXT_MODEL)
    p.add_argument("--glclap_embed_dim", type=int, default=512)
    p.add_argument("--english_word_df", type=str, default=ENGLISH_WORD_DF)
    p.add_argument("--max_subtext_units", type=int, default=8)
    p.add_argument("--loss_calibration_steps", type=int, default=0)
    p.add_argument("--ctc_grad_ratio", type=float, default=0.25)
    p.add_argument("--glclap_grad_ratio", type=float, default=0.25)

    p.add_argument("--ctc_adapter", type=str, default="auto", help="auto/mlp/moe，auto 会继承源 checkpoint")

    p.add_argument("--stream_train", type=int, default=0, help="1 表示所有音频任务使用 train-mask Encoder")

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
    source_cfg = read_cfg(args.model_path)
    if args.ctc_adapter == "auto":
        args.ctc_adapter = source_cfg.get("ctc_adapter", "mlp")
    args.tasks = names(args.train, TASKS, "train")
    if not args.tasks:
        raise ValueError("train 不能为空")
    if "proj" in args.tasks and "llm" not in args.tasks:
        raise ValueError("proj 只作用于 LLM 路径，请和 llm 一起训练。")
    args.loss_tasks = tuple(x for x in args.tasks if x not in ("encoder", "proj"))
    if not args.loss_tasks:
        raise ValueError("--train 至少需要包含 llm/ctc/rnnt/glclap 之一；encoder/proj 只能配合这些 loss 一起训练。")
    if args.loss_calibration_steps > 0:
        required = {"llm", "encoder"}
        if not required.issubset(args.tasks):
            raise ValueError("loss校准要求 --train 同时包含 llm 和 encoder。")
        if not ({"ctc", "glclap"} & set(args.loss_tasks)):
            raise ValueError("loss校准至少需要一个CTC或GLCLAP辅助任务。")
        if args.ctc_grad_ratio <= 0 or args.glclap_grad_ratio <= 0:
            raise ValueError("loss校准的目标梯度比例必须大于0。")
    save_heads = [
        name
        for name in ("ctc", "rnnt", "glclap")
        if os.path.exists(os.path.join(args.model_path, f"{name}_head.pt"))
    ]
    active_heads = []
    for name in args.loss_tasks:
        if name in ("ctc", "rnnt", "glclap"):
            if name not in save_heads:
                save_heads.append(name)
            active_heads.append(name)
    args.heads = tuple(save_heads)
    args.active_heads = tuple(active_heads)
    return args


def main():
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main = local_rank == 0

    import sentencepiece as spm

    sp_model = spm.SentencePieceProcessor()
    sp_model.load(args.sp_model_path)
    vocab = {}
    with open(args.vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                vocab[parts[0]] = int(parts[1])
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
    processor = wrapper.processor

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
        loss_weights={
            "llm": args.w_llm,
            "ctc": args.w_ctc,
            "rnnt": args.w_rnnt,
            "glclap": args.w_glclap,
        },
        ctc_config={"adapter_type": args.ctc_adapter},
        glclap_config={
            "text_model": args.glclap_text_model,
            "embed_dim": args.glclap_embed_dim,
        } if "glclap" in args.heads else None,
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

    lock_path = os.path.join(args.output_dir, ".dataset_cache.lock")
    with FileLock(lock_path):
        ds = load_dataset(
            "json",
            data_files={"train": args.train_file, **({"validation": args.eval_file} if args.eval_file else {})},
        )
    keep = {"prompt", "audio", "target", "text"}
    for split in ds.keys():
        drop = [col for col in ds[split].column_names if col not in keep]
        if drop:
            ds[split] = ds[split].remove_columns(drop)

    glclap_tokenizer = None
    glclap_word_df = None
    glclap_median_df = 1.0
    if "glclap" in args.loss_tasks:
        glclap_tokenizer = BertTokenizer.from_pretrained(
            args.glclap_text_model, local_files_only=True
        )
        glclap_word_df, glclap_median_df = load_word_df(args.english_word_df)

    collator = DataCollatorForJointTraining(
        processor,
        vocab,
        sp_model,
        need_aux=bool(args.active_heads),
        sampling_rate=args.sr,
        stream_train=(args.stream_train == 1),
        need_llm=("llm" in args.loss_tasks),
        need_glclap=("glclap" in args.loss_tasks),
        glclap_tokenizer=glclap_tokenizer,
        glclap_word_df=glclap_word_df,
        glclap_median_df=glclap_median_df,
        max_subtext_units=args.max_subtext_units,
    )
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
        dataloader_drop_last=True,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_safetensors=False,
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
            "glclap_text": args.lr_glclap_text,
            "glclap_proj": args.lr_glclap_proj,
        },
        save_heads=args.heads,
        head_source=args.model_path,
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation", None),
        data_collator=collator,
        tokenizer=processor.tokenizer,
    )

    resume_from = (args.resume_from or "").strip()
    if not resume_from and args.resume == 1:
        resume_from = latest_checkpoint(training_args.output_dir) or ""
    if args.loss_calibration_steps > 0 and not resume_from:
        ratios = {}
        if "ctc" in args.loss_tasks:
            ratios["ctc"] = args.ctc_grad_ratio
        if "glclap" in args.loss_tasks:
            ratios["glclap"] = args.glclap_grad_ratio
        calibrate_loss_weights(
            model,
            trainer,
            args.loss_calibration_steps,
            ratios,
            training_args.output_dir,
        )
        if dist.is_initialized():
            dist.barrier()
    trainer.train(resume_from_checkpoint=resume_from or None)
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
