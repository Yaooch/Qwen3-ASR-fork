import argparse
import json
from typing import List

import librosa
import torch

from qwen_joint.joint_model import Qwen3ASRJointModel


def get_dtype(name: str):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def load_records(path: str, limit: int) -> List[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if len(records) >= limit:
                break
    return records


def build_prefix_messages(prompt, audio_array):
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]


def make_prefix_texts(processor, records):
    texts = []
    for rec in records:
        prefix_msgs = build_prefix_messages(rec.get("prompt", ""), None)
        prefix_text = processor.apply_chat_template(
            [prefix_msgs],
            add_generation_prompt=True,
            tokenize=False,
        )[0]
        texts.append(prefix_text)
    return texts


def strip_ref(text: str) -> str:
    return text.split("<asr_text>")[-1].strip() if "<asr_text>" in text else text.strip()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("Reproduce training-time RNNT eval decoding from an eval jsonl.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--rnnt_max_symbols_per_step", type=int, default=5)
    parser.add_argument("--rnnt_decode_strategy", choices=["legacy", "cached"], default="legacy")
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=0,
        help="If >0, decode train_style records in chunks to compare batch-size effects.",
    )
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    model = Qwen3ASRJointModel.from_pretrained(
        args.ckpt,
        dtype=get_dtype(args.dtype),
        device_map=None,
    )
    model = model.to(device)
    model.eval()

    records = load_records(args.eval_file, args.num_samples)
    audios = [librosa.load(rec["audio"], sr=16000, mono=True)[0] for rec in records]
    audio_paths = [rec["audio"] for rec in records]
    refs = [strip_ref(rec["text"]) for rec in records]

    prefix_texts = make_prefix_texts(model.processor, records)
    eos = model.processor.tokenizer.eos_token or ""
    full_texts = [prefix + rec["text"] + eos for prefix, rec in zip(prefix_texts, records)]

    train_style = model.processor(
        text=full_texts,
        audio=audios,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    train_features = train_style["input_features"].to(device)
    train_mask = train_style.get("feature_attention_mask", None)
    if train_mask is not None:
        train_mask = train_mask.to(device)

    if args.chunk_size and args.chunk_size > 0:
        train_preds = []
        for start in range(0, len(records), args.chunk_size):
            end = start + args.chunk_size
            train_preds.extend(
                model.decode_aux_features(
                    train_features[start:end],
                    train_mask[start:end] if train_mask is not None else None,
                    max_symbols_per_step=args.rnnt_max_symbols_per_step,
                    rnnt_decode_strategy=args.rnnt_decode_strategy,
                )
            )
    else:
        train_preds = model.decode_aux_features(
            train_features,
            train_mask,
            max_symbols_per_step=args.rnnt_max_symbols_per_step,
            rnnt_decode_strategy=args.rnnt_decode_strategy,
        )

    infer_preds = model.transcribe_rnnt(
        audio_paths,
        max_symbols_per_step=args.rnnt_max_symbols_per_step,
        rnnt_decode_strategy=args.rnnt_decode_strategy,
    )
    if isinstance(infer_preds, str):
        infer_preds = [infer_preds]

    train_lens = train_mask.sum(dim=1).tolist() if train_mask is not None else ["None"] * len(records)

    print("=" * 100)
    print(f"ckpt: {args.ckpt}")
    print(f"eval_file: {args.eval_file}")
    print(f"device: {device}, dtype: {args.dtype}")
    print(f"rnnt_max_symbols_per_step: {args.rnnt_max_symbols_per_step}")
    print(f"rnnt_decode_strategy: {args.rnnt_decode_strategy}")
    print(f"chunk_size: {args.chunk_size}")
    print("=" * 100)

    for idx, (rec, ref, train_pred, infer_pred, feat_len) in enumerate(
        zip(records, refs, train_preds, infer_preds, train_lens)
    ):
        print(f"\n[{idx}] audio: {rec['audio']}")
        print(f"feature_len(train_style): {feat_len}")
        print(f"ref:         {ref}")
        print(f"train_style: {train_pred}")
        print(f"infer_style: {infer_pred}")
        print(f"same:        {train_pred == infer_pred}")


if __name__ == "__main__":
    main()
