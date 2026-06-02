import argparse
import json
from pathlib import Path
from typing import Dict, List

import librosa
import torch
import torch.nn.functional as F

from qwen_asr.joint import Qwen3ASRJointModel
from qwen_asr.joint.defaults import STREAM_CHUNK_SEC, STREAM_CNN_LEFT_FRAMES, STREAM_LEFT_CHUNKS
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


def read_text(path: str) -> Dict[str, str]:
    return dict(read_rows(path)) if path else {}


def enc_len(length: int) -> int:
    for _ in range(3):
        length = (length + 1) // 2
    return length


def stream_mel(feature_extractor, wav) -> tuple[torch.Tensor, List[int]]:
    state = StreamingFeatureState(feature_extractor)
    sr = state.sampling_rate
    chunk_samples = max(1, int(round(STREAM_CHUNK_SEC * sr)))
    pieces = []
    lengths = []
    for start in range(0, len(wav), chunk_samples):
        end = min(len(wav), start + chunk_samples)
        final = end >= len(wav)
        segment, segment_start = state.prepare(wav[start:end])
        batch = feature_extractor(
            [segment],
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
            truncation=False,
            return_attention_mask=True,
        )
        valid = int(batch["attention_mask"][0].sum().item())
        piece = state.finish(batch["input_features"][0], valid, segment_start, final)
        state.set_tail(segment)
        if piece.shape[1] > 0:
            pieces.append(piece)
            lengths.append(int(piece.shape[1]))
    return torch.cat(pieces, dim=1), lengths


def overlap_cnn(audio_tower, full_mel: torch.Tensor, mel_chunk_lens: List[int]) -> torch.Tensor:
    """固定离线 Mel，仅复现推理侧 overlap CNN 和裁剪。"""
    pieces = []
    feat_tail = None
    feat_offset = 0
    cnn_left = max(0, int(STREAM_CNN_LEFT_FRAMES))
    for length in mel_chunk_lens:
        new_feat = full_mel[:, feat_offset:feat_offset + length]
        feat = new_feat if feat_tail is None else torch.cat([feat_tail, new_feat], dim=1)
        left = 0 if feat_tail is None else feat_tail.shape[1]
        feat_start = max(0, feat_offset - left)
        feat_end = feat_offset + new_feat.shape[1]
        keep_start = enc_len(feat_offset) - enc_len(feat_start)
        keep_end = enc_len(feat_end) - enc_len(feat_start)
        hidden_states = audio_tower._conv_subsample_chunk(feat)
        pieces.append(hidden_states[keep_start:keep_end])
        feat_offset = feat_end
        feat_tail = feat[:, -cnn_left:] if feat.shape[1] > cnn_left else feat
    return torch.cat(pieces, dim=0)


def aligned_overlap_cnn(
    audio_tower,
    full_mel: torch.Tensor,
    mel_chunk_lens: List[int],
    wait_right: bool,
) -> torch.Tensor:
    """模拟修正：CNN 起点对齐 stride 网格，可选择等待完整右上下文。"""
    pieces = []
    feat_offset = 0
    emitted = 0
    for idx, length in enumerate(mel_chunk_lens):
        feat_offset += length
        final = idx == len(mel_chunk_lens) - 1
        emit_end = enc_len(feat_offset) if final or not wait_right else feat_offset // 8
        if emit_end <= emitted:
            continue
        feat_start = max(0, (emitted - 1) * 8)
        hidden_states = audio_tower._conv_subsample_chunk(full_mel[:, feat_start:feat_offset])
        local_offset = feat_start // 8
        pieces.append(hidden_states[emitted - local_offset:emit_end - local_offset])
        emitted = emit_end
    return torch.cat(pieces, dim=0)


def cache_encoder_from_cnn(audio_tower, cnn_states: torch.Tensor, chunk_out: int) -> torch.Tensor:
    """固定整条 CNN 输出，仅复现推理侧 Transformer KV cache。"""
    caches = [None] * len(audio_tower.layers)
    cache_size = max(0, int(STREAM_LEFT_CHUNKS)) * chunk_out
    pieces = []
    for start in range(0, cnn_states.shape[0], chunk_out):
        hidden_states = cnn_states[start:start + chunk_out]
        positional_embedding = audio_tower.positional_embedding.positional_embedding[start:start + hidden_states.shape[0]]
        hidden_states = hidden_states + positional_embedding.to(hidden_states.device, dtype=hidden_states.dtype)
        new_caches = []
        for encoder_layer, cache in zip(audio_tower.layers, caches):
            hidden_states, cache = encoder_layer.forward_chunk(
                hidden_states,
                kv_cache=cache,
                cache_size=cache_size,
                detach_cache=True,
            )
            new_caches.append(cache)
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


def mel_stats(full_mel: torch.Tensor, infer_mel: torch.Tensor) -> Dict:
    full_mel = full_mel.transpose(0, 1)
    infer_mel = infer_mel.transpose(0, 1)
    stats = diff_stats(full_mel, infer_mel)
    return {
        "mel_train_len": int(full_mel.shape[0]),
        "mel_infer_len": int(infer_mel.shape[0]),
        "mel_len_equal": full_mel.shape[0] == infer_mel.shape[0],
        **{f"mel_{key}": value for key, value in stats.items()},
    }


@torch.inference_mode()
def compare_one(model, wav_path: str) -> Dict:
    wav, _ = librosa.load(wav_path, sr=16000, mono=True)
    wav = wav.astype("float32", copy=False)
    feature_extractor = model.processor.feature_extractor
    batch = feature_extractor(
        [wav],
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
        truncation=False,
        return_attention_mask=True,
    )
    full_mel = batch["input_features"][0, :, : int(batch["attention_mask"][0].sum().item())]
    infer_mel, mel_chunk_lens = stream_mel(feature_extractor, wav)

    ref = next(model.qwen_model.parameters())
    input_features = batch["input_features"].to(device=ref.device, dtype=ref.dtype)
    feature_mask = batch["attention_mask"].to(device=ref.device)
    hs_train, train_lens, _ = model._stream_train_mask(input_features, feature_mask)
    train_len = int(train_lens[0].item())
    hs_train = hs_train[0, :train_len]

    audio_tower = model.qwen_model.thinker.audio_tower
    full_mel_device = full_mel.to(device=ref.device, dtype=ref.dtype)
    cnn_full = audio_tower._conv_subsample_chunk(full_mel_device)
    cnn_overlap = overlap_cnn(audio_tower, full_mel_device, mel_chunk_lens)
    cnn_aligned = aligned_overlap_cnn(audio_tower, full_mel_device, mel_chunk_lens, wait_right=False)
    cnn_aligned_wait = aligned_overlap_cnn(audio_tower, full_mel_device, mel_chunk_lens, wait_right=True)
    infer_mel_device = infer_mel.to(device=ref.device, dtype=ref.dtype)
    cnn_aligned_wait_infer_mel = aligned_overlap_cnn(audio_tower, infer_mel_device, mel_chunk_lens, wait_right=True)
    chunk_out = enc_len(model._sec_to_feature_count(STREAM_CHUNK_SEC, min_value=1))
    hs_cache_from_full_cnn = cache_encoder_from_cnn(audio_tower, cnn_full, chunk_out)
    hs_cache_from_overlap_cnn = cache_encoder_from_cnn(audio_tower, cnn_overlap, chunk_out)
    hs_cache_from_aligned_wait_cnn = cache_encoder_from_cnn(audio_tower, cnn_aligned_wait, chunk_out)
    hs_cache_from_aligned_wait_infer_mel = cache_encoder_from_cnn(audio_tower, cnn_aligned_wait_infer_mel, chunk_out)

    chunks_list, _ = model._stream_batch_feats([wav], need_llm=False)
    hs_infer = torch.cat(chunks_list[0], dim=0)

    head_dtype = next(model.ctc.parameters()).dtype
    hs_train_ctc = hs_train.unsqueeze(0).to(head_dtype)
    hs_infer_ctc = hs_infer.unsqueeze(0).to(head_dtype)
    train_lens = torch.tensor([hs_train.shape[0]], dtype=torch.long, device=hs_train.device)
    infer_lens = torch.tensor([hs_infer.shape[0]], dtype=torch.long, device=hs_infer.device)
    ctc_train = model.ctc.log_softmax(hs_train_ctc, train_lens)[0]
    ctc_infer = model.ctc.log_softmax(hs_infer_ctc, infer_lens)[0]
    hs_fixed_ctc = hs_cache_from_aligned_wait_infer_mel.unsqueeze(0).to(head_dtype)
    fixed_lens = torch.tensor([hs_cache_from_aligned_wait_infer_mel.shape[0]], dtype=torch.long, device=hs_train.device)
    ctc_fixed = model.ctc.log_softmax(hs_fixed_ctc, fixed_lens)[0]

    ctc_len = min(ctc_train.shape[0], ctc_infer.shape[0])
    top1_match = (ctc_train[:ctc_len].argmax(dim=1) == ctc_infer[:ctc_len].argmax(dim=1)).float()
    fixed_len = min(ctc_train.shape[0], ctc_fixed.shape[0])
    fixed_top1_match = (ctc_train[:fixed_len].argmax(dim=1) == ctc_fixed[:fixed_len].argmax(dim=1)).float()
    train_text = model._decode_head("ctc", hs_train_ctc, train_lens)[0]
    infer_text = model._decode_head("ctc", hs_infer_ctc, infer_lens)[0]
    fixed_text = model._decode_head("ctc", hs_fixed_ctc, fixed_lens)[0]

    return {
        **mel_stats(full_mel, infer_mel),
        **{f"cnn_overlap_{key}": value for key, value in diff_stats(cnn_full, cnn_overlap).items()},
        **{f"cnn_aligned_{key}": value for key, value in diff_stats(cnn_full, cnn_aligned).items()},
        **{f"cnn_aligned_wait_{key}": value for key, value in diff_stats(cnn_full, cnn_aligned_wait).items()},
        **{f"cache_from_full_cnn_{key}": value for key, value in diff_stats(hs_train, hs_cache_from_full_cnn).items()},
        **{f"cache_from_overlap_cnn_{key}": value for key, value in diff_stats(hs_train, hs_cache_from_overlap_cnn).items()},
        **{f"cache_from_aligned_wait_cnn_{key}": value for key, value in diff_stats(hs_train, hs_cache_from_aligned_wait_cnn).items()},
        **{f"cache_from_aligned_wait_infer_mel_{key}": value for key, value in diff_stats(hs_train, hs_cache_from_aligned_wait_infer_mel).items()},
        "encoder_train_len": int(hs_train.shape[0]),
        "encoder_infer_len": int(hs_infer.shape[0]),
        "encoder_len_equal": hs_train.shape[0] == hs_infer.shape[0],
        **{f"encoder_{key}": value for key, value in diff_stats(hs_train, hs_infer).items()},
        **{f"ctc_{key}": value for key, value in diff_stats(ctc_train, ctc_infer).items()},
        "ctc_top1_match_rate": float(top1_match.mean().item()) if ctc_len else 1.0,
        "ctc_fixed_top1_match_rate": float(fixed_top1_match.mean().item()) if fixed_len else 1.0,
        "ctc_train_text": train_text,
        "ctc_infer_text": infer_text,
        "ctc_fixed_text": fixed_text,
        "ctc_text_equal": train_text == infer_text,
        "ctc_fixed_text_equal": train_text == fixed_text,
    }


def mean(rows: List[Dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows) if rows else 0.0


def summarize(rows: List[Dict]) -> Dict:
    return {
        "samples": len(rows),
        "mel_len_equal_rate": mean(rows, "mel_len_equal"),
        "mel_mean_abs_diff": mean(rows, "mel_mean_abs_diff"),
        "cnn_overlap_mean_abs_diff": mean(rows, "cnn_overlap_mean_abs_diff"),
        "cnn_aligned_mean_abs_diff": mean(rows, "cnn_aligned_mean_abs_diff"),
        "cnn_aligned_wait_mean_abs_diff": mean(rows, "cnn_aligned_wait_mean_abs_diff"),
        "cache_from_full_cnn_mean_abs_diff": mean(rows, "cache_from_full_cnn_mean_abs_diff"),
        "cache_from_overlap_cnn_mean_abs_diff": mean(rows, "cache_from_overlap_cnn_mean_abs_diff"),
        "cache_from_aligned_wait_cnn_mean_abs_diff": mean(rows, "cache_from_aligned_wait_cnn_mean_abs_diff"),
        "cache_from_aligned_wait_infer_mel_mean_abs_diff": mean(rows, "cache_from_aligned_wait_infer_mel_mean_abs_diff"),
        "encoder_len_equal_rate": mean(rows, "encoder_len_equal"),
        "encoder_mean_abs_diff": mean(rows, "encoder_mean_abs_diff"),
        "encoder_cosine": mean(rows, "encoder_cosine"),
        "ctc_mean_abs_diff": mean(rows, "ctc_mean_abs_diff"),
        "ctc_top1_match_rate": mean(rows, "ctc_top1_match_rate"),
        "ctc_fixed_top1_match_rate": mean(rows, "ctc_fixed_top1_match_rate"),
        "ctc_text_equal_rate": mean(rows, "ctc_text_equal"),
        "ctc_fixed_text_equal_rate": mean(rows, "ctc_fixed_text_equal"),
    }


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
    refs = read_text(args.text)
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
