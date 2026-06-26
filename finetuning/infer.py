import argparse
import json
import multiprocessing as mp
import os
from datetime import datetime
from typing import Dict, List

import torch

from qwen_asr import Qwen3ASRModel
from qwen_asr.joint import HotwordRetriever, Qwen3ASRJointModel
from qwen_asr.joint.defaults import DEFAULT_ATTN_IMPLEMENTATION, DEFAULT_PROMPT, JOINT_CONFIG
from qwen_asr.joint.model import names


def log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-ASR 批量推理。")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--input_scp", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", default="llm", help="逗号组合：llm,ctc,rnnt；不再支持 joint")
    parser.add_argument("--gpu_ids", default="0")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--language", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--hotword_file", default=None)
    parser.add_argument("--hotword_topk", type=int, default=10)
    parser.add_argument("--keep_origin_llm", type=int, choices=[0, 1], default=1)
    parser.add_argument("--hotword_pinyin_style", choices=["normal", "tone3"], default="normal")
    parser.add_argument("--hotword_retriever", choices=["pinyin", "asr_hotword"], default="pinyin")
    parser.add_argument("--encoder_mode", choices=["offline", "stream", "train_mask"], default="offline")
    parser.add_argument("--stream", action="store_const", const="stream", dest="encoder_mode")
    parser.add_argument("--no_stream", action="store_const", const="offline", dest="encoder_mode")
    parser.add_argument("--lora", default=None, help="加载 RL 训出的 LoRA 目录（仅 joint checkpoint 分支）")
    args = parser.parse_args()
    args.modes = names(args.mode, {"llm", "ctc", "rnnt"}, "mode")
    if not args.modes:
        raise ValueError("mode 不能为空")
    return args


def load_scp(path: str, default_language: str = None) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"scp 第 {line_no} 行格式错误：{line}")
            items.append({
                "utt_id": parts[0],
                "audio": parts[1],
                "language": parts[2] if len(parts) >= 3 else default_language,
            })
    return items


def batches(items: List[Dict], size: int):
    for start in range(0, len(items), size):
        yield items[start: start + size]


def make_hotword(args):
    if not args.hotword_file:
        return None
    if "llm" not in args.modes or not ({"ctc", "rnnt"} & set(args.modes)):
        raise ValueError("热词 prompt 需要同时跑 llm 和 ctc/rnnt，例如 --mode llm,ctc")
    if args.hotword_retriever == "asr_hotword":
        from qwen_asr.joint.asr_hotword import AsrHotwordRetriever
        return AsrHotwordRetriever.from_file(args.hotword_file)
    return HotwordRetriever.from_file(args.hotword_file, pinyin_style=args.hotword_pinyin_style)


def worker(rank: int, gpu_id: int, world_size: int, args, tmp_path: str):
    shard = load_scp(args.input_scp, default_language=args.language)[rank::world_size]
    if not torch.cuda.is_available():
        raise RuntimeError("当前环境不可用 CUDA。")
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    log(f"进程{rank}启动：GPU {gpu_id}，样本 {len(shard)}")

    rows = []
    if os.path.exists(os.path.join(args.ckpt, JOINT_CONFIG)):
        # ---- joint checkpoint 分支 ----
        model = Qwen3ASRJointModel.from_pretrained(
            args.ckpt,
            dtype=dtype,
            device_map=None,
            attn_implementation=DEFAULT_ATTN_IMPLEMENTATION,
        ).to(device)
        if args.lora:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.lora)
        model.eval()
        missing = [name for name in args.modes if name in ("ctc", "rnnt") and name not in model.heads]
        if missing:
            raise RuntimeError(f"checkpoint 没有这些头：{','.join(missing)}")
        hotword = make_hotword(args)

        with torch.no_grad():
            for idx, batch in enumerate(batches(shard, args.batch_size), 1):
                if idx == 1 or idx % 10 == 0:
                    log(f"进程{rank}推理 batch {idx}")
                outs = model.transcribe(
                    [x["audio"] for x in batch],
                    modes=args.modes,
                    language=[x.get("language") for x in batch],
                    prompt=args.prompt or DEFAULT_PROMPT,
                    hotword_retriever=hotword,
                    hotword_topk=args.hotword_topk,
                    keep_origin_llm=bool(args.keep_origin_llm),
                    encoder_mode=args.encoder_mode,
                )
                if not isinstance(outs, list):
                    outs = [outs]
                for item, out in zip(batch, outs):
                    out.update({
                        "utt_id": item["utt_id"],
                        "audio": item["audio"],
                        "input_language": item.get("language"),
                        "language": out.get("language") or item.get("language") or "unknown",
                        "mode": ",".join(args.modes),
                        "encoder_mode": args.encoder_mode,
                        "stream": args.encoder_mode == "stream",
                    })
                    rows.append(out)
    else:
        # ---- 原始 Qwen checkpoint 分支（仅 LLM） ----
        if args.modes != ("llm",):
            raise RuntimeError("原始 Qwen checkpoint 只支持 --mode llm。")
        if args.encoder_mode != "offline":
            raise RuntimeError("原始 Qwen checkpoint 只支持 --encoder_mode offline。")
        model = Qwen3ASRModel.from_pretrained(
            args.ckpt,
            dtype=dtype,
            device_map=None,
            attn_implementation=DEFAULT_ATTN_IMPLEMENTATION,
            max_inference_batch_size=args.batch_size,
        )
        model.model = model.model.to(device)
        model.model.eval()
        model.device = torch.device(device)

        with torch.no_grad():
            for idx, batch in enumerate(batches(shard, args.batch_size), 1):
                if idx == 1 or idx % 10 == 0:
                    log(f"进程{rank}推理 batch {idx}")
                audios = [x["audio"] for x in batch]
                languages = [x.get("language") for x in batch]
                outs = model.transcribe(audios, language=languages, context=args.prompt or DEFAULT_PROMPT)
                if not isinstance(outs, list):
                    outs = [outs]
                for item, out in zip(batch, outs):
                    if out is None:
                        text, lang = "", None
                    elif isinstance(out, str):
                        text, lang = out, None
                    elif isinstance(out, dict):
                        text, lang = out.get("text") or "", out.get("language")
                    else:
                        t = getattr(out, "text", None)
                        text, lang = (str(t), getattr(out, "language", None)) if t is not None else (str(out), None)
                    rows.append({
                        "utt_id": item["utt_id"],
                        "audio": item["audio"],
                        "input_language": item.get("language"),
                        "language": lang or item.get("language") or "unknown",
                        "mode": ",".join(args.modes),
                        "text": text,
                        "llm_text": text,
                        "encoder_mode": args.encoder_mode,
                        "stream": False,
                    })

    with open(tmp_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log(f"进程{rank}完成")

def merge(tmp_files: List[str], output_dir: str, encoder_mode: str):
    os.makedirs(output_dir, exist_ok=True)
    details = os.path.join(output_dir, "details")
    os.makedirs(details, exist_ok=True)
    rows = []
    for path in tmp_files:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    rows.sort(key=lambda x: x["utt_id"])

    # 根据结果中实际存在的字段确定输出规格
    specs = []
    for name, field in (
        ("llm", "llm_text"),
        ("hotword_llm", "hotword_llm_text"),
        ("ctc", "ctc_text"),
        ("rnnt", "rnnt_text"),
    ):
        if any(row.get(field) for row in rows):
            specs.append((name, f"results_{name}.txt", field))
    if not specs:
        specs = [("text", "results_text.txt", "text")]

    paths = {}
    files = []
    for name, filename, field in specs:
        path = os.path.join(output_dir, filename)
        paths[name] = path
        files.append((field, open(path, "w", encoding="utf-8")))

    detail_path = os.path.join(details, "results_detail.jsonl")
    try:
        with open(detail_path, "w", encoding="utf-8") as detail:
            for row in rows:
                language = row.get("language") or row.get("input_language") or "unknown"
                for field, f in files:
                    f.write(f'{row["utt_id"]}\t{row.get(field) or row.get("text") or ""}\t{language}\n')
                detail.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        for _, f in files:
            f.close()

    with open(os.path.join(details, "encoder_mode.txt"), "w", encoding="utf-8") as f:
        f.write(encoder_mode + "\n")
    for path in tmp_files:
        if os.path.exists(path):
            os.remove(path)
    return paths, detail_path


def main():
    args = parse_args()
    items = load_scp(args.input_scp, default_language=args.language)
    if not items:
        raise ValueError(f"scp 文件中没有可用样本：{args.input_scp}")
    ids = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip()]
    if not ids:
        raise ValueError("gpu_ids 不能为空")
    os.makedirs(args.output_dir, exist_ok=True)
    tmp_files = [os.path.join(args.output_dir, f"tmp_rank{rank}.jsonl") for rank in range(len(ids))]

    print("推理配置")
    print(f"模型：{args.ckpt}")
    print(f"模式：{','.join(args.modes)}")
    print(f"Encoder 模式：{args.encoder_mode}")
    print(f"输出：{args.output_dir}")
    print(f"GPU：{ids}")
    print(f"样本数：{len(items)}")

    if len(ids) == 1:
        worker(0, ids[0], len(ids), args, tmp_files[0])
    else:
        ctx = mp.get_context("spawn")
        procs = [ctx.Process(target=worker, args=(rank, gpu, len(ids), args, tmp_files[rank])) for rank, gpu in enumerate(ids)]
        for proc in procs:
            proc.start()
        for proc in procs:
            proc.join()
            if proc.exitcode != 0:
                raise RuntimeError(f"子进程失败，exitcode={proc.exitcode}")

    paths, detail_path = merge(tmp_files, args.output_dir, args.encoder_mode)
    print("推理完成")
    for name, path in paths.items():
        print(f"结果[{name}]：{path}")
    print(f"明细：{detail_path}")


if __name__ == "__main__":
    main()
