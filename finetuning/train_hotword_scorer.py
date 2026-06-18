# coding=utf-8
import argparse
import json
import os
from typing import Any, Dict, List, Optional

import librosa
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from qwen_asr.joint import Qwen3ASRJointModel
from qwen_asr.joint.defaults import DEFAULT_ATTN_IMPLEMENTATION
from qwen_asr.joint.encoder import encode_stream
from qwen_asr.joint.hotword_scorer import (
    HotwordScorer,
    batch_tokenize_hotwords,
    extract_hotwords,
    extract_ref,
    hotword_label,
)


class JsonlDataset(Dataset):
    def __init__(self, path: str):
        self.rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


class HotwordCollator:
    def __init__(self, tokenizer, sr: int, max_audio_sec: float, max_hotword_len: int):
        self.tokenizer = tokenizer
        self.sr = sr
        self.max_audio_sec = max_audio_sec
        self.max_hotword_len = max_hotword_len

    def __call__(self, features: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        rows, hotwords_batch, labels_batch = [], [], []
        for item in features:
            audio = item.get("audio")
            ref = extract_ref(item.get("target") or item.get("text") or "")
            hotwords = extract_hotwords(item.get("prompt") or "")
            if not audio or not ref or not hotwords:
                continue
            try:
                wav, _ = librosa.load(audio, sr=self.sr, mono=True)
            except Exception as exc:
                print(f"音频读取失败，跳过：{audio}，{exc}")
                continue
            if self.max_audio_sec > 0 and len(wav) / self.sr > self.max_audio_sec:
                print(f"音频过长，跳过：{audio}")
                continue
            rows.append((item, wav.astype("float32", copy=False)))
            hotwords_batch.append(hotwords)
            labels_batch.append([hotword_label(ref, word) for word in hotwords])
        if not rows:
            return None

        ids, token_mask, valid = batch_tokenize_hotwords(
            self.tokenizer,
            hotwords_batch,
            max_len=self.max_hotword_len,
        )
        labels = torch.zeros(valid.shape, dtype=torch.float32)
        for b, labels_row in enumerate(labels_batch):
            labels[b, :len(labels_row)] = torch.tensor(labels_row, dtype=torch.float32)
        return {
            "audios": [wav for _, wav in rows],
            "hotwords": hotwords_batch,
            "hotword_ids": ids,
            "hotword_token_mask": token_mask,
            "hotword_valid_mask": valid,
            "labels": labels,
        }


def parse_args():
    p = argparse.ArgumentParser("训练 Encoder 热词打分器")
    p.add_argument("--model_path", required=True)
    p.add_argument("--train_file", required=True)
    p.add_argument("--eval_file", default="")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--pos_weight", type=float, default=6.0)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--max_audio_sec", type=float, default=30.0)
    p.add_argument("--max_hotword_len", type=int, default=24)
    p.add_argument("--scorer_dim", type=int, default=384)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--ffn_mult", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--chunk_hotwords", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--log_steps", type=int, default=20)
    p.add_argument("--logging_dir", default="")
    return p.parse_args()


def stream_hs(model: Qwen3ASRJointModel, audios: List, device: torch.device):
    ref = next(model.qwen_model.parameters())
    chunks, _ = encode_stream(
        model.qwen_model.thinker.audio_tower,
        model.processor.feature_extractor,
        audios,
        ref,
        need_llm=False,
    )
    seqs = [torch.cat(x, dim=0) for x in chunks]
    lens = torch.tensor([x.shape[0] for x in seqs], dtype=torch.long, device=device)
    hs = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)
    return hs, lens


def batch_loss(scorer, model, batch, device, pos_weight: float, chunk_hotwords: int):
    ids = batch["hotword_ids"].to(device)
    token_mask = batch["hotword_token_mask"].to(device)
    valid = batch["hotword_valid_mask"].to(device)
    labels = batch["labels"].to(device)
    with torch.no_grad():
        hs, lens = stream_hs(model, batch["audios"], device)
        embed = model.qwen_model.thinker.get_input_embeddings()
        hotword_embeds = embed(ids).detach()
    logits = scorer(hs, lens, hotword_embeds, token_mask, valid, chunk_size=chunk_hotwords)
    active = valid & torch.isfinite(logits)
    if not active.any():
        return None, logits.detach(), labels, valid
    weight = torch.tensor(float(pos_weight), device=device)
    loss = F.binary_cross_entropy_with_logits(logits[active], labels[active], pos_weight=weight)
    return loss, logits.detach(), labels.detach(), valid.detach()


def new_stats() -> Dict[str, float]:
    return {"tp": 0, "fp": 0, "fn": 0, "valid": 0, "pred": 0, "items": 0, "loss_sum": 0.0, "loss_count": 0}


def add_loss(stats: Dict[str, float], loss: torch.Tensor) -> None:
    stats["loss_sum"] += float(loss.detach().float().item())
    stats["loss_count"] += 1


def update_stats(stats: Dict[str, float], logits, labels, valid, threshold: float):
    probs = torch.sigmoid(logits)
    pred = (probs >= threshold) & valid
    gold = (labels > 0.5) & valid
    stats["tp"] += int((pred & gold).sum().item())
    stats["fp"] += int((pred & ~gold).sum().item())
    stats["fn"] += int((~pred & gold).sum().item())
    stats["valid"] += int(valid.sum().item())
    stats["pred"] += int(pred.sum().item())
    stats["items"] += int(valid.shape[0])


def finish_stats(stats: Dict[str, float]) -> Dict[str, float]:
    tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    out = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_pred": stats["pred"] / stats["items"] if stats["items"] else 0.0,
        "valid_pairs": stats["valid"],
    }
    if stats.get("loss_count", 0):
        out["loss"] = stats["loss_sum"] / stats["loss_count"]
    return out


def reduce_stats(stats: Dict[str, float], device: torch.device) -> Dict[str, float]:
    keys = ("tp", "fp", "fn", "valid", "pred", "items", "loss_sum", "loss_count")
    values = [float(stats.get(key, 0.0)) for key in keys]
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return {key: tensor[idx].item() for idx, key in enumerate(keys)}


def all_ranks_have(value: bool, device: torch.device) -> bool:
    if not (dist.is_available() and dist.is_initialized()):
        return value
    flag = torch.tensor(1 if value else 0, dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def setup_dist():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size


def cleanup_dist():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def run_eval(scorer, model, loader, device, args):
    scorer.eval()
    stats = new_stats()
    with torch.no_grad():
        for batch in loader:
            if not all_ranks_have(batch is not None, device):
                continue
            loss, logits, labels, valid = batch_loss(
                scorer, model, batch, device, args.pos_weight, args.chunk_hotwords
            )
            if loss is not None:
                add_loss(stats, loss)
                update_stats(stats, logits, labels, valid, args.threshold)
    return finish_stats(reduce_stats(stats, device))


def save_scorer(scorer, output_dir: str, name: str, metrics: Dict[str, float], args) -> str:
    path = os.path.join(output_dir, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    base = scorer.module if hasattr(scorer, "module") else scorer
    base.save(path, extra={"metrics": metrics, "threshold": args.threshold})
    return path


def make_writer(logging_dir: str):
    if not logging_dir:
        return None
    os.makedirs(logging_dir, exist_ok=True)
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError("使用 --logging_dir 需要安装 tensorboard。") from exc
    return SummaryWriter(logging_dir)


def log_metrics(writer, prefix: str, metrics: Dict[str, float], step: int) -> None:
    if writer is None:
        return
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            writer.add_scalar(f"{prefix}/{key}", float(value), step)


def main():
    args = parse_args()
    rank, local_rank, world_size = setup_dist()
    is_main = rank == 0
    os.makedirs(args.output_dir, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("训练热词 scorer 需要 CUDA。")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    use_bf16 = torch.cuda.get_device_capability(local_rank)[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    model = Qwen3ASRJointModel.from_pretrained(
        args.model_path,
        dtype=dtype,
        device_map=None,
        load_heads=False,
        attn_implementation=DEFAULT_ATTN_IMPLEMENTATION,
    ).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    embed = model.qwen_model.thinker.get_input_embeddings()
    embed_dim = int(embed.weight.shape[1])
    if args.scorer_dim % args.num_heads != 0:
        raise ValueError(f"scorer_dim={args.scorer_dim} 不能被 num_heads={args.num_heads} 整除")
    scorer = HotwordScorer(
        encoder_dim=model.encoder_output_size,
        embed_dim=embed_dim,
        scorer_dim=args.scorer_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ffn_mult=args.ffn_mult,
        dropout=args.dropout,
        max_hotword_len=args.max_hotword_len,
    ).to(device)
    if world_size > 1:
        scorer = DDP(scorer, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    collator = HotwordCollator(model.processor.tokenizer, args.sr, args.max_audio_sec, args.max_hotword_len)
    train_dataset = JsonlDataset(args.train_file)
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if world_size > 1 else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collator,
    )
    eval_loader = None
    if args.eval_file:
        eval_dataset = JsonlDataset(args.eval_file)
        eval_sampler = DistributedSampler(eval_dataset, shuffle=False) if world_size > 1 else None
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=eval_sampler,
            num_workers=args.num_workers,
            collate_fn=collator,
        )

    if is_main:
        print("热词 scorer 训练配置")
        print(f"模型：{args.model_path}")
        print(f"输出：{args.output_dir}")
        print(f"训练样本：{len(train_loader.dataset)}")
        print(f"GPU 数：{world_size}")
        print(f"每卡 batch：{args.batch_size}")
        print(f"有效 batch：{args.batch_size * world_size}")
        print(f"热词 token 上限：{args.max_hotword_len}")
        print(f"scorer 维度：{args.scorer_dim}")
        print(f"scorer 层数：{args.num_layers}")
        print(f"阈值：{args.threshold}")
        if args.logging_dir:
            print(f"TensorBoard：{args.logging_dir}")
    writer = make_writer(args.logging_dir if is_main else "")

    best_f1 = -1.0
    global_step = 0
    total_batches = len(train_loader)
    total_steps = total_batches * args.epochs
    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        scorer.train()
        stats = new_stats()
        for batch_idx, batch in enumerate(train_loader, 1):
            if not all_ranks_have(batch is not None, device):
                continue
            loss, logits, labels, valid = batch_loss(
                scorer, model, batch, device, args.pos_weight, args.chunk_hotwords
            )
            if not all_ranks_have(loss is not None, device):
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            global_step += 1
            add_loss(stats, loss)
            update_stats(stats, logits, labels, valid, args.threshold)
            if is_main and global_step % args.log_steps == 0:
                metrics = finish_stats(stats)
                print(
                    f"epoch {epoch}/{args.epochs} batch {batch_idx}/{total_batches} "
                    f"global_step {global_step}/{total_steps} loss={metrics.get('loss', 0.0):.4f} "
                    f"p={metrics['precision']:.4f} r={metrics['recall']:.4f} "
                    f"f1={metrics['f1']:.4f} avg_pred={metrics['avg_pred']:.2f}",
                    flush=True,
                )
                log_metrics(writer, "train_step", metrics, global_step)

        train_metrics = finish_stats(reduce_stats(stats, device))
        if is_main:
            print(f"epoch {epoch} train: {json.dumps(train_metrics, ensure_ascii=False)}", flush=True)
            log_metrics(writer, "train_epoch", train_metrics, epoch)
        metrics = train_metrics
        if eval_loader is not None:
            metrics = run_eval(scorer, model, eval_loader, device, args)
            if is_main:
                print(f"epoch {epoch} eval: {json.dumps(metrics, ensure_ascii=False)}", flush=True)
                log_metrics(writer, "eval", metrics, epoch)
        if is_main:
            save_scorer(scorer, args.output_dir, "hotword_scorer_last.pt", metrics, args)
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                save_scorer(scorer, args.output_dir, "hotword_scorer_best.pt", metrics, args)
    if writer is not None:
        writer.close()
    cleanup_dist()


if __name__ == "__main__":
    main()
