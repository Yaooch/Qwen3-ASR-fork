# coding: utf-8
"""使用音频查询评测 GLCLAP 热词检索。"""
import argparse
import json
import os
import time
from typing import Dict, List, Tuple

import librosa
import torch
from safetensors.torch import load_file
from transformers import AutoFeatureExtractor, BertTokenizer

from qwen_asr_ext.glclap.benchmark import RetrievalMetrics, latency_stats, load_hotword_benchmark
from qwen_asr_ext.glclap.model import GLCLAPModel


def load_model(checkpoint: str, device: torch.device) -> Tuple[GLCLAPModel, Dict]:
    with open(os.path.join(checkpoint, "config.json"), encoding="utf-8") as f:
        config = json.load(f)
    model = GLCLAPModel(
        config["audio_model"],
        config["text_model"],
        config["embed_dim"],
        config["unfreeze_audio_layers"],
        config["unfreeze_text_layers"],
        gradient_checkpointing=False,
        audio_backend=config.get("audio_backend", "data2vec"),
        stream_left_frames=config.get("stream_left_frames", 24),
        stream_current_frames=config.get("stream_current_frames", 8),
        stream_right_frames=config.get("stream_right_frames", 0),
    )
    weights = load_file(os.path.join(checkpoint, "model.safetensors"))
    result = model.load_state_dict(weights, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"checkpoint 含未知参数：{result.unexpected_keys}")
    return model.to(device).eval(), config


@torch.no_grad()
def encode_candidates(
    model: GLCLAPModel,
    tokenizer: BertTokenizer,
    candidates: List[str],
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    embeddings = []
    for start in range(0, len(candidates), batch_size):
        tokens = tokenizer(
            candidates[start:start + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        embeddings.append(model.encode_text(tokens["input_ids"], tokens["attention_mask"]))
    return torch.cat(embeddings)


@torch.no_grad()
def retrieve_features(
    model: GLCLAPModel,
    hidden: torch.Tensor,
    audio_mask: torch.Tensor,
    candidates: torch.Tensor,
    top_k: int,
):
    _, audio_local, audio_mask = model.encode_audio_features(hidden, audio_mask)
    similarity = torch.einsum("kd,btd->bkt", candidates, audio_local)
    similarity = similarity.masked_fill(
        ~audio_mask[:, None, :], torch.finfo(similarity.dtype).min
    ).amax(dim=-1)
    return similarity.topk(top_k, dim=-1)


def parse_args():
    parser = argparse.ArgumentParser(description="使用音频查询评测 GLCLAP 热词检索")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--text_batch_size", type=int, default=256)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--max_utts", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    return parser.parse_args()


def make_audio(feature_extractor, waveforms, backend: str, device: torch.device):
    audio = feature_extractor(
        waveforms,
        sampling_rate=16000,
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
    ).to(device)
    if backend == "qwen":
        return {
            "input_features": audio["input_features"],
            "feature_attention_mask": audio.get(
                "feature_attention_mask", audio.get("attention_mask")
            ),
        }
    return {
        "input_values": audio["input_values"],
        "attention_mask": audio["attention_mask"],
    }


def main():
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    candidates, transcripts, records = load_hotword_benchmark(
        args.data_dir, "wav.scp", args.max_utts
    )

    model, config = load_model(args.checkpoint, device)
    backend = config.get("audio_backend", "data2vec")
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        config["audio_model"], local_files_only=True
    )
    tokenizer = BertTokenizer.from_pretrained(config["text_model"], local_files_only=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    candidate_start = time.perf_counter()
    with torch.autocast("cuda", dtype=dtype, enabled=device.type == "cuda" and dtype != torch.float32):
        candidate_embeddings = encode_candidates(
            model,
            tokenizer,
            candidates,
            args.text_batch_size,
            config["max_text_length"],
            device,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    candidate_encode_ms = (time.perf_counter() - candidate_start) * 1000

    top_k = min(args.top_k, len(candidates))
    warmup_wav = librosa.load(records[0][1], sr=16000, mono=True)[0]
    warmup_audio = make_audio(feature_extractor, [warmup_wav], backend, device)
    with torch.autocast("cuda", dtype=dtype, enabled=device.type == "cuda" and dtype != torch.float32):
        warmup_hidden, warmup_mask = model.extract_audio_features(**warmup_audio)
        retrieve_features(
            model, warmup_hidden, warmup_mask, candidate_embeddings, top_k
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    metrics = RetrievalMetrics(top_k, len(records), progress_every=100)
    encoder_ms = []
    for start in range(0, len(records), args.batch_size):
        batch = records[start:start + args.batch_size]
        waveforms = [librosa.load(path, sr=16000, mono=True)[0] for _, path, _ in batch]
        audio = make_audio(feature_extractor, waveforms, backend, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        encoder_start = time.perf_counter()
        with torch.autocast(
            "cuda", dtype=dtype, enabled=device.type == "cuda" and dtype != torch.float32
        ):
            hidden, audio_mask = model.extract_audio_features(**audio)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        encoder_ms.append((time.perf_counter() - encoder_start) * 1000 / len(batch))

        retrieve_start = time.perf_counter()
        with torch.autocast(
            "cuda", dtype=dtype, enabled=device.type == "cuda" and dtype != torch.float32
        ):
            scores, indices = retrieve_features(
                model, hidden, audio_mask, candidate_embeddings, top_k
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        item_ms = (time.perf_counter() - retrieve_start) * 1000 / len(batch)

        scores = scores.float().cpu().tolist()
        indices = indices.cpu().tolist()
        for (utt_id, _, target), row_scores, row_indices in zip(batch, scores, indices):
            metrics.add(
                utt_id,
                target,
                transcripts.get(utt_id, ""),
                [candidates[index] for index in row_indices],
                item_ms,
                scores=row_scores,
            )

    stats = latency_stats(metrics.latencies)
    encoder_stats = latency_stats(encoder_ms)
    metrics.write(args.output)
    print(f"候选编码：{candidate_encode_ms:.2f} ms（离线一次）")
    print(
        f"Encoder 参考 batch={args.batch_size} mean={encoder_stats['mean']:.2f} ms "
        f"p50={encoder_stats['p50']:.2f} ms p95={encoder_stats['p95']:.2f} ms"
    )
    print(
        f"Encoder 后检索 batch={args.batch_size} mean={stats['mean']:.2f} ms "
        f"p50={stats['p50']:.2f} ms p95={stats['p95']:.2f} ms max={stats['max']:.2f} ms"
    )
    print(
        f"结果 checkpoint={args.checkpoint} utterances={len(records)} "
        f"candidates={len(candidates)} top1_recall={metrics.top1_recall:.4f} "
        f"top{top_k}_recall={metrics.topk_recall:.4f}"
    )


if __name__ == "__main__":
    main()
