# coding: utf-8
"""在 STOP1/STOP2 上评测 GLCLAP 热词检索。"""
import argparse
import json
import os
from typing import Dict, List, Tuple

import librosa
import torch
from safetensors.torch import load_file
from transformers import AutoFeatureExtractor, BertTokenizer

from qwen_asr.joint.glclap import GLCLAPModel


def normalize_text(text: str) -> str:
    return " ".join(text.strip().upper().split())


def read_key_value(path: str) -> Dict[str, str]:
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            fields = line.strip().split(maxsplit=1)
            if len(fields) == 2:
                rows[fields[0]] = fields[1].strip()
    return rows


def read_candidates(path: str) -> List[str]:
    candidates = []
    seen = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            text = normalize_text(line)
            if text and text not in seen:
                candidates.append(text)
                seen.add(text)
    return candidates


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


def parse_args():
    parser = argparse.ArgumentParser(description="在 STOP1/STOP2 上评测 GLCLAP 热词检索")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--text_batch_size", type=int, default=256)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_utts", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    wavs = read_key_value(os.path.join(args.data_dir, "wav.scp"))
    targets = {
        key: normalize_text(value)
        for key, value in read_key_value(os.path.join(args.data_dir, "utt_hotword.txt")).items()
    }
    transcripts = read_key_value(os.path.join(args.data_dir, "text"))
    candidates = read_candidates(os.path.join(args.data_dir, "hotword.txt"))
    candidate_to_index = {text: index for index, text in enumerate(candidates)}
    records = [(key, path, targets[key]) for key, path in wavs.items() if key in targets]
    if args.max_utts > 0:
        records = records[:args.max_utts]
    missing = sorted({target for _, _, target in records} - candidate_to_index.keys())
    if missing:
        raise ValueError(f"目标热词不在候选词库中：{missing[:10]}")

    model, config = load_model(args.checkpoint, device)
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        config["audio_model"], local_files_only=True
    )
    tokenizer = BertTokenizer.from_pretrained(config["text_model"], local_files_only=True)
    with torch.autocast("cuda", dtype=dtype, enabled=device.type == "cuda" and dtype != torch.float32):
        candidate_embeddings = encode_candidates(
            model,
            tokenizer,
            candidates,
            args.text_batch_size,
            config["max_text_length"],
            device,
        )

    top_k = min(args.top_k, len(candidates))
    top1_hits = 0
    topk_hits = 0
    details = []
    for start in range(0, len(records), args.batch_size):
        batch = records[start:start + args.batch_size]
        waveforms = [librosa.load(path, sr=16000, mono=True)[0] for _, path, _ in batch]
        audio = feature_extractor(
            waveforms,
            sampling_rate=16000,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad(), torch.autocast(
            "cuda", dtype=dtype, enabled=device.type == "cuda" and dtype != torch.float32
        ):
            _, audio_local, audio_mask = model.encode_audio(
                audio["input_values"], audio["attention_mask"]
            )
            similarity = torch.einsum("kd,btd->bkt", candidate_embeddings, audio_local)
            similarity = similarity.masked_fill(
                ~audio_mask[:, None, :], torch.finfo(similarity.dtype).min
            ).amax(dim=-1)
            scores, indices = similarity.topk(top_k, dim=-1)

        scores = scores.float().cpu().tolist()
        indices = indices.cpu().tolist()
        for (utt_id, _, target), row_scores, row_indices in zip(batch, scores, indices):
            retrieved = [candidates[index] for index in row_indices]
            hit_top1 = retrieved[0] == target
            hit_topk = target in retrieved
            top1_hits += hit_top1
            topk_hits += hit_topk
            details.append({
                "utt_id": utt_id,
                "target": target,
                "text": transcripts.get(utt_id, ""),
                "hit_top1": hit_top1,
                f"hit_top{top_k}": hit_topk,
                "retrieved": [
                    {"text": text, "score": round(score, 6)}
                    for text, score in zip(retrieved, row_scores)
                ],
            })
        done = min(start + args.batch_size, len(records))
        if done % 100 == 0 or done == len(records):
            print(
                f"进度 {done}/{len(records)} top1_recall={top1_hits / done:.4f} "
                f"top{top_k}_recall={topk_hits / done:.4f}",
                flush=True,
            )

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            for row in details:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"明细：{args.output}")
    print(
        f"结果 checkpoint={args.checkpoint} utterances={len(records)} "
        f"candidates={len(candidates)} top1_recall={top1_hits / len(records):.4f} "
        f"top{top_k}_recall={topk_hits / len(records):.4f}"
    )


if __name__ == "__main__":
    main()
