# coding: utf-8
"""独立训练 GLCLAP 音频-热词检索模型。"""
import argparse
import json
import os
import random
import re
import time
import unicodedata
from typing import Dict, Iterator, List

import librosa
import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from transformers import AutoFeatureExtractor, BertTokenizer, get_cosine_schedule_with_warmup

from qwen_asr_ext.glclap.model import GLCLAPModel
from qwen_asr_ext.joint.defaults import (
    TRAIN_MASK_CURRENT_FRAMES,
    TRAIN_MASK_LEFT_FRAMES,
    TRAIN_MASK_RIGHT_FRAMES,
)

TRAIN_JSONL = "/cfs/data/private/WangYaoChi/train_data/all/train_700w_shuffled.jsonl"
EVAL_JSONL = "/cfs/data/private/WangYaoChi/train_data/all/eval_shuffled.jsonl"
AUDIO_MODEL = "/cfs/data/private/WangYaoChi/model/glclap/data2vec-audio-large"
TEXT_MODEL = "/cfs/data/private/WangYaoChi/model/glclap/bert-base-multilingual-uncased"
OUTPUT_DIR = "/cfs/data/private/WangYaoChi/model/glclap/retrieval_v4"
ENGLISH_WORD_DF = "/cfs/data/private/WangYaoChi/train_data/all/english_word_df.json"
ASR_TAG = "<asr_text>"
HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
UNIT_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]|[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")
ENGLISH_LENGTH_WEIGHTS = {1: 4, 2: 3, 3: 2, 4: 1}
ENGLISH_STOPWORDS = frozenset(
    """
    a an the and or but if then else of to in on at by for from with without
    is am are was were be been being do does did have has had this that these
    those it its i you he she we they me him her us them my your his our their
    as not no yes can could will would shall should may might must s t d ll m
    re ve
    """.split()
)

def parse_text(raw: str):
    """解析 language X<asr_text>文本，不改写 ground truth。"""
    raw = str(raw or "").strip()
    if ASR_TAG not in raw:
        return "", raw
    meta, text = raw.split(ASR_TAG, 1)
    language = meta.strip()
    if language.lower().startswith("language "):
        language = language[9:].strip()
    return language, text.strip()

def normalize_word(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower()

def load_word_df(path: str):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    frequencies = data.get("document_frequency", data)
    frequencies = {normalize_word(str(word)): int(count) for word, count in frequencies.items()}
    eligible = [
        count for word, count in frequencies.items()
        if count >= 5 and word not in ENGLISH_STOPWORDS
    ]
    if not eligible:
        raise ValueError(f"英文词频文件没有 df>=5 的有效内容词：{path}")
    return frequencies, float(np.median(eligible))

def sample_subtext(
    text: str,
    language: str = "",
    max_units: int = 8,
    word_df=None,
    median_df: float = 1.0,
    rng=random,
) -> str:
    """抽取 V4 连续 span。"""
    matches = list(UNIT_RE.finditer(text))
    english = language.lower().startswith("english")
    chinese = bool(HAN_RE.search(text)) and not english
    min_length, max_length = (2, 8) if chinese else (1, 4)
    max_length = min(max_length, max_units, len(matches))
    if max_length < min_length:
        return text.strip()

    lengths = list(range(min_length, max_length + 1))
    weights = (
        [3 if length in (2, 3) else 1 for length in lengths]
        if chinese else [ENGLISH_LENGTH_WEIGHTS[length] for length in lengths]
    )
    length = rng.choices(lengths, weights=weights, k=1)[0]
    if not chinese and length == 1:
        frequencies = word_df or {}
        candidates = []
        candidate_weights = []
        for index, match in enumerate(matches):
            word = normalize_word(match.group())
            frequency = frequencies.get(word, 0)
            if word not in ENGLISH_STOPWORDS and frequency >= 5:
                candidates.append(index)
                candidate_weights.append(
                    min(4.0, max(0.5, (median_df / frequency) ** 0.5))
                )
        if candidates and rng.random() < 0.8:
            start = rng.choices(candidates, weights=candidate_weights, k=1)[0]
        else:
            start = rng.randrange(len(matches))
    else:
        start = rng.randint(0, len(matches) - length)

    value = text[matches[start].start():matches[start + length - 1].end()].strip()
    return value

def iter_jsonl_shard(path: str, part: int, parts: int) -> Iterator[Dict]:
    """按字节区间读取 JSONL，避免每个 rank 扫描完整大文件。"""
    size = os.path.getsize(path)
    start = size * part // parts
    end = size * (part + 1) // parts
    with open(path, "rb") as f:
        if start:
            f.seek(start - 1)
            if f.read(1) != b"\n":
                f.readline()
        while f.tell() < end:
            line = f.readline()
            if not line:
                break
            try:
                row = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(row, dict):
                yield row

class AudioTextDataset(IterableDataset):
    """多卡/多 worker 字节分片的流式音频数据集。"""

    def __init__(
        self,
        path: str,
        rank: int,
        world_size: int,
        repeat: bool,
        min_duration: float,
        max_duration: float,
    ):
        super().__init__()
        self.path = path
        self.rank = rank
        self.world_size = world_size
        self.repeat = repeat
        self.min_samples = int(min_duration * 16000)
        self.max_samples = int(max_duration * 16000)

    def __iter__(self):
        worker = get_worker_info()
        workers = worker.num_workers if worker else 1
        worker_id = worker.id if worker else 0
        part = self.rank * workers + worker_id
        parts = self.world_size * workers

        while True:
            valid = 0
            for row in iter_jsonl_shard(self.path, part, parts):
                language, text = parse_text(row.get("text", ""))
                audio = row.get("audio")
                if not audio or not text:
                    continue
                try:
                    wav, _ = librosa.load(audio, sr=16000, mono=True)
                except Exception:
                    continue
                if not self.min_samples <= wav.shape[0] <= self.max_samples:
                    continue
                valid += 1
                yield {
                    "audio": np.asarray(wav, dtype=np.float32),
                    "text": text,
                    "language": language,
                }
            if not self.repeat:
                return
            if not valid:
                raise RuntimeError(f"数据分片没有有效样本：{self.path} part={part}/{parts}")

class GLCLAPCollator:
    def __init__(
        self,
        audio_model: str,
        text_model: str,
        max_text_length: int,
        max_subtext_units: int,
        english_word_df: str,
        audio_backend: str = "data2vec",
    ):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            audio_model, local_files_only=True
        )
        # 当前 transformers 对该 BERT fast tokenizer 有误报，slow tokenizer 行为稳定。
        self.tokenizer = BertTokenizer.from_pretrained(text_model, local_files_only=True)
        self.audio_backend = audio_backend
        self.max_text_length = max_text_length
        self.max_subtext_units = max_subtext_units
        self.word_df, self.median_df = load_word_df(english_word_df)

    def __call__(self, rows: List[Dict]) -> Dict[str, torch.Tensor]:
        audio = self.feature_extractor(
            [x["audio"] for x in rows],
            sampling_rate=16000,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        texts = [x["text"] for x in rows]
        sampled = [
            sample_subtext(
                x["text"],
                x["language"],
                self.max_subtext_units,
                self.word_df,
                self.median_df,
            )
            for x in rows
        ]
        subtexts = sampled
        text = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        subtext = self.tokenizer(
            subtexts,
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        batch = {
            "text_input_ids": text["input_ids"],
            "text_attention_mask": text["attention_mask"],
            "subtext_input_ids": subtext["input_ids"],
            "subtext_attention_mask": subtext["attention_mask"],
        }
        if self.audio_backend == "qwen":
            mask = audio.get("feature_attention_mask", audio.get("attention_mask"))
            batch.update({
                "input_features": audio["input_features"],
                "feature_attention_mask": mask,
            })
        else:
            batch.update({
                "input_values": audio["input_values"],
                "audio_attention_mask": audio["attention_mask"],
            })
        return batch

def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)

def move_batch(batch: Dict[str, torch.Tensor], device: torch.device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}

def recall(logits: torch.Tensor, k: int) -> float:
    labels = torch.arange(logits.shape[0], device=logits.device)
    topk = logits.topk(min(k, logits.shape[1]), dim=1).indices
    return float((topk == labels.unsqueeze(1)).any(dim=1).float().mean())

@torch.no_grad()
def evaluate(model, loader, batches: int, device: torch.device, dtype: torch.dtype):
    model.eval()
    iterator = iter(loader)
    total = {"loss": 0.0, "global_r1": 0.0, "local_r1": 0.0, "local_r5": 0.0}
    for _ in range(batches):
        batch = next(iterator)
        batch = move_batch(batch, device)
        with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            out = model(**batch, gather=True)
        total["loss"] += float(out["loss"])
        total["global_r1"] += recall(out["global_logits"], 1)
        total["local_r1"] += recall(out["local_logits"], 1)
        total["local_r5"] += recall(out["local_logits"], 5)
    model.train()
    return {key: value / batches for key, value in total.items()}

def save_checkpoint(model, optimizer, scheduler, args, step: int, name: str = "") -> str:
    output = os.path.join(args.output_dir, name or f"checkpoint-{step}")
    os.makedirs(output, exist_ok=True)
    trainable = {
        key: value.detach().cpu().contiguous()
        for key, value in model.named_parameters()
        if value.requires_grad
    }
    save_file(trainable, os.path.join(output, "model.safetensors"))
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        os.path.join(output, "trainer_state.pt"),
    )
    with open(os.path.join(output, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"step": step, **vars(args)}, f, ensure_ascii=False, indent=2)
    return output

def init_from_checkpoint(model, path: str) -> None:
    """只加载已有可训练权重，用于改变解冻范围后的下一训练阶段。"""
    weights = load_file(os.path.join(path, "model.safetensors"))
    unexpected = model.load_state_dict(weights, strict=False).unexpected_keys
    if unexpected:
        raise RuntimeError(f"checkpoint 含未知参数：{unexpected}")

def load_checkpoint(model, optimizer, scheduler, path: str) -> int:
    weights = load_file(os.path.join(path, "model.safetensors"))
    result = model.load_state_dict(weights, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"checkpoint 含未知参数：{result.unexpected_keys}")
    params = dict(model.named_parameters())
    missing = [
        key for key in result.missing_keys
        if key in params and params[key].requires_grad
    ]
    if missing:
        raise RuntimeError(f"checkpoint 缺少当前可训练参数：{missing}")
    state = torch.load(os.path.join(path, "trainer_state.pt"), map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    return int(state["step"])

def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
    return rank, local_rank, world_size, torch.device(f"cuda:{local_rank}")

def make_loader(path, args, collator, rank, world_size, repeat=True):
    dataset = AudioTextDataset(
        path,
        rank,
        world_size,
        repeat,
        args.min_duration,
        args.max_duration,
    )
    kwargs = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "collate_fn": collator,
        "drop_last": True,
        "pin_memory": True,
        "worker_init_fn": seed_worker,
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)

def build_optimizer(model, args):
    projection, encoder = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad or name == "logit_scale":
            continue
        if name.startswith(("audio_projection.", "text_projection.")):
            projection.append(param)
        else:
            encoder.append(param)
    groups = [
        {"params": projection, "lr": args.lr_projection, "weight_decay": args.weight_decay},
        {"params": [model.logit_scale], "lr": args.lr_projection, "weight_decay": 0.0},
    ]
    if encoder:
        groups.append({"params": encoder, "lr": args.lr_encoder, "weight_decay": args.weight_decay})
    return AdamW(groups)

def parse_args():
    parser = argparse.ArgumentParser(description="训练 GLCLAP 音频-热词检索模型")
    parser.add_argument("--train_jsonl", default=TRAIN_JSONL)
    parser.add_argument("--eval_jsonl", default=EVAL_JSONL)
    parser.add_argument("--audio_model", default=AUDIO_MODEL)
    parser.add_argument("--audio_backend", choices=["data2vec", "qwen"], default="data2vec")
    parser.add_argument("--stream_left_frames", type=int, default=TRAIN_MASK_LEFT_FRAMES)
    parser.add_argument("--stream_current_frames", type=int, default=TRAIN_MASK_CURRENT_FRAMES)
    parser.add_argument("--stream_right_frames", type=int, default=TRAIN_MASK_RIGHT_FRAMES)
    parser.add_argument("--text_model", default=TEXT_MODEL)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--english_word_df", default=ENGLISH_WORD_DF)
    parser.add_argument("--resume_from", default="")
    parser.add_argument("--init_from", default="")
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32, help="每卡 batch")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--eval_batches", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--lr_projection", type=float, default=1e-3)
    parser.add_argument("--lr_encoder", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--unfreeze_audio_layers", type=int, default=0, help="-1 表示全部解冻")
    parser.add_argument("--unfreeze_text_layers", type=int, default=0, help="-1 表示全部解冻")
    parser.add_argument("--max_text_length", type=int, default=128)
    parser.add_argument("--max_subtext_units", type=int, default=8)
    parser.add_argument("--min_duration", type=float, default=0.2)
    parser.add_argument("--max_duration", type=float, default=30.0)
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    args.encoder_mode = "train_mask" if args.audio_backend == "qwen" else "data2vec"
    if args.resume_from and args.init_from:
        raise ValueError("--resume_from 与 --init_from 不能同时使用。")
    rank, local_rank, world_size, device = setup_distributed()
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    collator = GLCLAPCollator(
        args.audio_model,
        args.text_model,
        args.max_text_length,
        args.max_subtext_units,
        args.english_word_df,
        args.audio_backend,
    )
    train_loader = make_loader(args.train_jsonl, args, collator, rank, world_size)
    eval_loader = make_loader(args.eval_jsonl, args, collator, rank, world_size)

    raw_model = GLCLAPModel(
        args.audio_model,
        args.text_model,
        args.embed_dim,
        args.unfreeze_audio_layers,
        args.unfreeze_text_layers,
        not args.no_gradient_checkpointing,
        args.audio_backend,
        args.stream_left_frames,
        args.stream_current_frames,
        args.stream_right_frames,
    ).to(device)
    if args.init_from:
        init_from_checkpoint(raw_model, args.init_from)
    optimizer = build_optimizer(raw_model, args)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )
    start_step = 0
    if args.resume_from:
        start_step = load_checkpoint(raw_model, optimizer, scheduler, args.resume_from)

    model = raw_model
    if world_size > 1:
        model = DDP(raw_model, device_ids=[local_rank], broadcast_buffers=False)

    if rank == 0:
        total = sum(p.numel() for p in raw_model.parameters())
        trainable = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
        print(
            f"GLCLAP 启动：world_size={world_size} per_gpu_batch={args.batch_size} "
            f"global_batch={world_size * args.batch_size} 参数={total:,} 可训练={trainable:,}"
        )

    iterator = iter(train_loader)
    running = 0.0
    last_time = time.time()
    for step in range(start_step + 1, args.max_steps + 1):
        batch = next(iterator)
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            out = model(**batch, gather=True)
            loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        running += float(loss.detach())

        if step % args.log_every == 0:
            if rank == 0:
                elapsed = time.time() - last_time
                print(
                    f"step={step} loss={running / args.log_every:.4f} "
                    f"global={float(out['global_loss'].detach()):.4f} "
                    f"local={float(out['local_loss'].detach()):.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e} step_s={elapsed / args.log_every:.3f}"
                )
                running = 0.0
                last_time = time.time()

        if args.eval_every > 0 and step % args.eval_every == 0:
            metrics = evaluate(model, eval_loader, args.eval_batches, device, dtype)
            if rank == 0:
                print(
                    f"eval step={step} loss={metrics['loss']:.4f} "
                    f"global_r1={metrics['global_r1']:.4f} "
                    f"local_r1={metrics['local_r1']:.4f} local_r5={metrics['local_r5']:.4f}"
                )

        if args.save_every > 0 and step % args.save_every == 0:
            if world_size > 1:
                dist.barrier()
            if rank == 0:
                path = save_checkpoint(raw_model, optimizer, scheduler, args, step)
                print(f"保存 checkpoint：{path}")
            if world_size > 1:
                dist.barrier()

    if world_size > 1:
        dist.barrier()
    if rank == 0:
        path = save_checkpoint(raw_model, optimizer, scheduler, args, args.max_steps, "final")
        print(f"训练完成：{path}")
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
