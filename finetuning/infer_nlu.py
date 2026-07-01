# finetuning/infer_nlu.py
"""NLU（用户意图提取）批量推理 + 评测。

加载基线 joint checkpoint + NLU LoRA，文本输入（user 语句）-> 意图 JSON。
多卡数据并行（spawn），结构与 infer.py 一致。

输入 jsonl 每行支持：
  {"messages": [{system}, {user}, {assistant}]}  # assistant 可选；--eval 时作为 ref
  {"text": "..."} 或 {"system": "...", "user": "...", "assistant": "..."}

用法：见 infer_nlu.sh。
"""
import argparse
import json
import multiprocessing as mp
import os
from datetime import datetime
from typing import Dict, List

import torch

from qwen_asr.joint import Qwen3ASRJointModel
from qwen_asr.joint.defaults import DEFAULT_ATTN_IMPLEMENTATION
from qwen_asr.tools.nlu import (
    NLU_SYSTEM_PROMPT,
    build_nlu_prompt,
    intent_metrics,
    nlu_messages,
    parse_intent,
)


def log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="Qwen3-ASR NLU 批量推理 / 评测。")
    p.add_argument("--ckpt", required=True, help="基线 joint checkpoint 目录")
    p.add_argument("--lora", default=None, help="NLU LoRA 目录；不传则只跑基线")
    p.add_argument("--input_file", required=True, help="输入 jsonl")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--gpu_ids", default="0")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--system", default=None, help="覆盖 system prompt；默认用数据里的或 NLU_SYSTEM_PROMPT")
    p.add_argument("--eval", action="store_true", help="输入含 assistant ref，输出意图指标")
    return p.parse_args()


def load_items(path: str, default_system: str) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "messages" in r:
                msgs = r["messages"]
                system = next((m["content"] for m in msgs if m["role"] == "system"), default_system)
                user = next((m["content"] for m in msgs if m["role"] == "user"), "")
                assistant = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
            else:
                system = r.get("system") or default_system
                user = r.get("text") or r.get("user") or ""
                assistant = r.get("assistant")
            items.append({"utt_id": str(line_no), "system": system, "user": user, "ref": assistant})
    return items


def batches(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def worker(rank, gpu_id, world_size, args, items, tmp_path):
    shard = items[rank::world_size]
    if not torch.cuda.is_available():
        raise RuntimeError("当前环境不可用 CUDA。")
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    log(f"进程{rank}启动：GPU {gpu_id}，样本 {len(shard)}")

    model = Qwen3ASRJointModel.from_pretrained(
        args.ckpt, dtype=dtype, device_map=None, load_heads=False,
        attn_implementation=DEFAULT_ATTN_IMPLEMENTATION,
    ).to(device)
    if args.lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora)
    model.eval()
    processor = model.processor
    default_system = args.system or NLU_SYSTEM_PROMPT

    rows = []
    with torch.no_grad():
        for idx, batch in enumerate(batches(shard, args.batch_size), 1):
            if idx == 1 or idx % 20 == 0:
                log(f"进程{rank}推理 batch {idx}")
            prompts = [
                build_nlu_prompt(
                    processor,
                    nlu_messages(b["system"] or default_system, b["user"]),
                    add_generation_prompt=True,
                )
                for b in batch
            ]
            inputs = processor(text=prompts, audio=None, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}
            gen = model.qwen_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=args.max_new_tokens,
            )
            seq = gen.sequences if hasattr(gen, "sequences") else gen
            decoded = processor.batch_decode(
                seq[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for b, text in zip(batch, decoded):
                rows.append({
                    "utt_id": b["utt_id"],
                    "user": b["user"],
                    "pred_text": text,
                    "intent": parse_intent(text),
                    "ref": b["ref"],
                })

    with open(tmp_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log(f"进程{rank}完成")


def merge(tmp_files, output_dir, do_eval):
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for path in tmp_files:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                rows.extend(json.loads(line) for line in f if line.strip())
    rows.sort(key=lambda x: x["utt_id"])

    detail_path = os.path.join(output_dir, "results_detail.jsonl")
    with open(detail_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if do_eval:
        preds = [r["intent"] for r in rows]
        refs = [parse_intent(r["ref"]) or {} for r in rows]
        metrics = intent_metrics(preds, refs)
        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print("NLU 评测指标：")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
    print(f"明细：{detail_path}")


def main():
    args = parse_args()
    default_system = args.system or NLU_SYSTEM_PROMPT
    items = load_items(args.input_file, default_system)
    if not items:
        raise ValueError(f"输入文件无可用样本：{args.input_file}")
    ids = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip()]
    if not ids:
        raise ValueError("gpu_ids 不能为空")
    os.makedirs(args.output_dir, exist_ok=True)
    tmp_files = [os.path.join(args.output_dir, f"tmp_rank{rank}.jsonl") for rank in range(len(ids))]

    print("NLU 推理配置")
    print(f"基线：{args.ckpt}")
    print(f"LoRA：{args.lora or '(无，基线)'}")
    print(f"输入：{args.input_file}")
    print(f"输出：{args.output_dir}")
    print(f"GPU：{ids}")
    print(f"样本数：{len(items)}")
    print(f"评测：{'是' if args.eval else '否'}")

    if len(ids) == 1:
        worker(0, ids[0], 1, args, items, tmp_files[0])
    else:
        ctx = mp.get_context("spawn")
        procs = [
            ctx.Process(target=worker, args=(rank, gpu, len(ids), args, items, tmp_files[rank]))
            for rank, gpu in enumerate(ids)
        ]
        for proc in procs:
            proc.start()
        for proc in procs:
            proc.join()
            if proc.exitcode != 0:
                raise RuntimeError(f"子进程失败，exitcode={proc.exitcode}")

    merge(tmp_files, args.output_dir, args.eval)
    for path in tmp_files:
        if os.path.exists(path):
            os.remove(path)


if __name__ == "__main__":
    main()
