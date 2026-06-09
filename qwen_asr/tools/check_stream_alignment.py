import argparse
import json
from pathlib import Path
from typing import Dict, List

import librosa
import torch
import torch.nn.functional as F

from qwen_asr.joint import Qwen3ASRJointModel
from qwen_asr.joint.defaults import STREAM_CNN_LEFT_FRAMES
from qwen_asr.joint.stream import StreamingFeatureState


def parse_args():
    p = argparse.ArgumentParser("比较流式训练与真实流式推理的 Encoder、CTC 输出")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--input_scp", required=True)
    p.add_argument("--text", default="")
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    return p.parse_args()


def read_rows(path: str) -> List[tuple[str, str]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                rows.append((parts[0], parts[1]))
    return rows


def enc_len(length: int) -> int:
    for _ in range(3):
        length = (length + 1) // 2
    return length


def stream_mel(feature_extractor, wav) -> tuple[torch.Tensor, List[int]]:
    """复现线上增量 Mel，只返回每轮新增帧。"""
    state = StreamingFeatureState(feature_extractor)
    chunk_samples = max(1, int(round(0.64 * state.sampling_rate)))
    pieces = []
    lengths = []
    for start in range(0, len(wav), chunk_samples):
        segment, left_frames = state.append(wav[start:start + chunk_samples])
        batch = feature_extractor(
            [segment],
            sampling_rate=state.sampling_rate,
            return_tensors="pt",
            padding=True,
            truncation=False,
            return_attention_mask=True,
        )
        valid = int(batch["attention_mask"][0].sum().item())
        piece = batch["input_features"][0, :, left_frames:valid]
        if piece.shape[1] > 0:
            pieces.append(piece)
            lengths.append(int(piece.shape[1]))
    return torch.cat(pieces, dim=1), lengths


def stream_cnn(audio_tower, mel: torch.Tensor, chunk_lens: List[int]) -> torch.Tensor:
    """固定 Mel，仅复现线上 8 帧 overlap CNN。"""
    pieces = []
    tail = None
    offset = 0
    for length in chunk_lens:
        new_mel = mel[:, offset:offset + length]
        cnn_input = new_mel if tail is None else torch.cat([tail, new_mel], dim=1)
        drop_prefix = 0 if tail is None else enc_len(STREAM_CNN_LEFT_FRAMES)
        pieces.append(audio_tower._conv_subsample_chunk(cnn_input)[drop_prefix:])
        tail = cnn_input[:, -STREAM_CNN_LEFT_FRAMES:]
        offset += length
    return torch.cat(pieces, dim=0)


def cache_encoder_from_cnn(audio_tower, cnn_states: torch.Tensor, chunk_out: int) -> torch.Tensor:
    """固定 CNN 输出，仅复现线上 Transformer KV cache。"""
    caches = [None] * len(audio_tower.layers)
    cache_size = 7 * chunk_out
    pieces = []
    for start in range(0, cnn_states.shape[0], chunk_out):
        hidden_states = cnn_states[start:start + chunk_out]
        positional_embedding = audio_tower.positional_embedding.positional_embedding[start:start + hidden_states.shape[0]]
        hidden_states = hidden_states + positional_embedding.to(hidden_states.device, dtype=hidden_states.dtype)
        new_caches = []
        for layer, cache in zip(audio_tower.layers, caches):
            outputs, layer_caches = layer.forward_batch_chunk(
                [hidden_states],
                kv_caches=[cache],
                cache_size=cache_size,
                detach_cache=True,
            )
            hidden_states = outputs[0]
            new_caches.append(layer_caches[0])
        caches = new_caches
        pieces.append(audio_tower.ln_post(hidden_states))
    return torch.cat(pieces, dim=0)


def diff_stats(left: torch.Tensor, right: torch.Tensor) -> Dict[str, float]:
    length = min(left.shape[0], right.shape[0])
    if length == 0:
        return {"mean_abs_diff": 0.0, "max_abs_diff": 0.0, "cosine": 1.0}
    left = left[:length].float()
    right = right[:length].float()
    diff = (left - right).abs()
    return {
        "mean_abs_diff": float(diff.mean().item()),
        "max_abs_diff": float(diff.max().item()),
        "cosine": float(F.cosine_similarity(left.flatten(), right.flatten(), dim=0).item()),
    }


def prefixed_stats(prefix: str, left: torch.Tensor, right: torch.Tensor) -> Dict:
    return {f"{prefix}_{key}": value for key, value in diff_stats(left, right).items()}


@torch.inference_mode()
def compare_one(model, wav_path: str) -> Dict:
    wav, _ = librosa.load(wav_path, sr=16000, mono=True)
    wav = wav.astype("float32", copy=False)
    extractor = model.processor.feature_extractor
    batch = extractor(
        [wav],
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
        truncation=False,
        return_attention_mask=True,
    )
    full_mel = batch["input_features"][0, :, : int(batch["attention_mask"][0].sum().item())]
    infer_mel, mel_chunk_lens = stream_mel(extractor, wav)

    ref = next(model.qwen_model.parameters())
    input_features = batch["input_features"].to(device=ref.device, dtype=ref.dtype)
    feature_mask = batch["attention_mask"].to(device=ref.device)
    hs_train, train_lens, _ = model._stream_train_mask(input_features, feature_mask)
    hs_train = hs_train[0, : int(train_lens[0].item())]

    audio_tower = model.qwen_model.thinker.audio_tower
    cnn_full = audio_tower._conv_subsample_chunk(full_mel.to(device=ref.device, dtype=ref.dtype))
    cnn_stream = stream_cnn(audio_tower, full_mel.to(device=ref.device, dtype=ref.dtype), mel_chunk_lens)
    chunk_out = enc_len(model._sec_to_feature_count(0.64, min_value=1))
    hs_cache = cache_encoder_from_cnn(audio_tower, cnn_full, chunk_out)

    chunks_list, _ = model._encode_stream_waveforms([wav], need_llm=False)
    hs_infer = torch.cat(chunks_list[0], dim=0)

    head_dtype = next(model.ctc.parameters()).dtype
    hs_train_ctc = hs_train.unsqueeze(0).to(head_dtype)
    hs_infer_ctc = hs_infer.unsqueeze(0).to(head_dtype)
    train_lens = torch.tensor([hs_train.shape[0]], dtype=torch.long, device=hs_train.device)
    infer_lens = torch.tensor([hs_infer.shape[0]], dtype=torch.long, device=hs_infer.device)
    ctc_train = model.ctc.log_softmax(hs_train_ctc, train_lens)[0]
    ctc_infer = model.ctc.log_softmax(hs_infer_ctc, infer_lens)[0]
    ctc_len = min(ctc_train.shape[0], ctc_infer.shape[0])
    top1_match = (ctc_train[:ctc_len].argmax(dim=1) == ctc_infer[:ctc_len].argmax(dim=1)).float()
    train_text = model._decode_head("ctc", hs_train_ctc, train_lens)[0]
    infer_text = model._decode_head("ctc", hs_infer_ctc, infer_lens)[0]

    return {
        "mel_train_len": int(full_mel.shape[1]),
        "mel_infer_len": int(infer_mel.shape[1]),
        "mel_len_equal": full_mel.shape[1] == infer_mel.shape[1],
        **prefixed_stats("mel", full_mel.transpose(0, 1), infer_mel.transpose(0, 1)),
        **prefixed_stats("cnn", cnn_full, cnn_stream),
        **prefixed_stats("cache", hs_train, hs_cache),
        "encoder_train_len": int(hs_train.shape[0]),
        "encoder_infer_len": int(hs_infer.shape[0]),
        "encoder_len_equal": hs_train.shape[0] == hs_infer.shape[0],
        **prefixed_stats("encoder", hs_train, hs_infer),
        **prefixed_stats("ctc", ctc_train, ctc_infer),
        "ctc_top1_match_rate": float(top1_match.mean().item()) if ctc_len else 1.0,
        "ctc_train_text": train_text,
        "ctc_infer_text": infer_text,
        "ctc_text_equal": train_text == infer_text,
    }


def mean(rows: List[Dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows) if rows else 0.0


def summarize(rows: List[Dict]) -> Dict:
    keys = (
        "mel_len_equal",
        "mel_mean_abs_diff",
        "cnn_mean_abs_diff",
        "cache_mean_abs_diff",
        "encoder_len_equal",
        "encoder_mean_abs_diff",
        "encoder_cosine",
        "ctc_mean_abs_diff",
        "ctc_top1_match_rate",
        "ctc_text_equal",
    )
    summary = {"samples": len(rows)}
    for key in keys:
        name = key.replace("_equal", "_equal_rate") if key.endswith("_equal") else key
        summary[name] = mean(rows, key)
    return summary


def main():
    args = parse_args()
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("当前环境不可用 CUDA。")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = Qwen3ASRJointModel.from_pretrained(
        args.ckpt,
        dtype=dtype,
        device_map=None,
        attn_implementation="flash_attention_2",
    ).to(args.device)
    model.eval()
    if model.ctc is None:
        raise RuntimeError("checkpoint 没有 CTC 头。")

    rows = read_rows(args.input_scp)
    if args.limit > 0:
        rows = rows[:args.limit]
    refs = dict(read_rows(args.text)) if args.text else {}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    results = []
    with output.open("w", encoding="utf-8") as f:
        for idx, (utt_id, wav_path) in enumerate(rows, 1):
            row = {"utt_id": utt_id, "wav": wav_path, "ref": refs.get(utt_id, "")}
            try:
                row.update(compare_one(model, wav_path))
                results.append(row)
                print(
                    f"[{idx}/{len(rows)}] {utt_id} "
                    f"cnn={row['cnn_mean_abs_diff']:.6f} "
                    f"enc={row['encoder_mean_abs_diff']:.6f} "
                    f"ctc_top1={row['ctc_top1_match_rate']:.4f} "
                    f"text_equal={row['ctc_text_equal']}",
                    flush=True,
                )
            except Exception as exc:
                row["error"] = str(exc)
                print(f"[{idx}/{len(rows)}] {utt_id} 失败：{exc}", flush=True)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = summarize(results)
    summary["errors"] = len(rows) - len(results)
    summary_path = output.with_suffix(output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"明细：{output}")
    print(f"汇总：{summary_path}")


if __name__ == "__main__":
    main()
