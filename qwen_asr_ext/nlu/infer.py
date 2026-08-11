# finetuning/infer_nlu.py
"""纯文本 NLU / Agent 批量推理与评测，支持 joint checkpoint 和纯 Qwen3 LLM。"""
import argparse
import json
import multiprocessing as mp
import os
from collections import Counter
from datetime import datetime
from typing import Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen_asr_ext.joint import Qwen3ASRJointModel
from qwen_asr_ext.joint.defaults import DEFAULT_ATTN_IMPLEMENTATION
from qwen_asr_ext.nlu.common import (
    NLU_SYSTEM_PROMPT,
    agent_metrics,
    build_nlu_prompt,
    intent_metrics,
    nlu_messages,
    parse_agent,
    parse_intent,
)


def log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="NLU / Agent 批量推理与评测")
    p.add_argument("--backend", choices=["joint", "llm"], default="joint")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--lora", default=None)
    p.add_argument("--input_file", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--gpu_ids", default="0")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--system", default=None)
    p.add_argument("--task", choices=["nlu", "agent"], default=None)
    p.add_argument("--eval", action="store_true")
    return p.parse_args()


def load_items(path: str, default_system: str) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "messages" in row:
                msgs = row["messages"]
                system = next((m["content"] for m in msgs if m["role"] == "system"), default_system)
                user = next((m["content"] for m in msgs if m["role"] == "user"), "")
                assistant = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
            else:
                system = row.get("system") or default_system
                user = row.get("text") or row.get("user") or ""
                assistant = row.get("assistant")
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

    if args.backend == "joint":
        model = Qwen3ASRJointModel.from_pretrained(
            args.ckpt,
            dtype=dtype,
            device_map=None,
            load_heads=False,
            attn_implementation=DEFAULT_ATTN_IMPLEMENTATION,
        ).to(device)
        if args.lora:
            model = PeftModel.from_pretrained(model, args.lora)
        processor = model.processor
        tokenizer = processor.tokenizer
        suppress_tokens = None
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.ckpt,
            dtype=dtype,
            device_map=None,
            trust_remote_code=True,
            attn_implementation=DEFAULT_ATTN_IMPLEMENTATION,
        ).to(device)
        if args.lora:
            model = PeftModel.from_pretrained(model, args.lora)
        processor = None
        tokenizer.padding_side = "left"
        suppress_tokens = [151667] if 151667 in tokenizer.get_added_vocab().values() else None
    model.eval()

    rows = []
    default_system = args.system or NLU_SYSTEM_PROMPT
    with torch.no_grad():
        for idx, batch in enumerate(batches(shard, args.batch_size), 1):
            if idx == 1 or idx % 20 == 0:
                log(f"进程{rank}推理 batch {idx}")
            messages = [nlu_messages(b["system"] or default_system, b["user"]) for b in batch]
            if args.backend == "joint":
                prompts = [
                    build_nlu_prompt(processor, value, add_generation_prompt=True)
                    for value in messages
                ]
                inputs = processor(text=prompts, audio=None, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}
                generated = model.qwen_model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=args.max_new_tokens,
                )
                sequences = generated.sequences if hasattr(generated, "sequences") else generated
                decoder = processor
            else:
                prompts = [
                    tokenizer.apply_chat_template(
                        value,
                        add_generation_prompt=True,
                        enable_thinking=False,
                        tokenize=False,
                    )
                    for value in messages
                ]
                inputs = tokenizer(prompts, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}
                sequences = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=args.max_new_tokens,
                    suppress_tokens=suppress_tokens,
                    do_sample=False,
                )
                decoder = tokenizer
            decoded = decoder.batch_decode(
                sequences[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for item, text in zip(batch, decoded):
                rows.append({
                    "utt_id": item["utt_id"],
                    "user": item["user"],
                    "pred_text": text,
                    "ref": item["ref"],
                })

    with open(tmp_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log(f"进程{rank}完成")


def write_badcase(rows, path: str, task: str) -> int:
    if task == "agent":
        main_key, param_key = "action", "params"
    else:
        main_key, param_key = "name", "arguments"
    bad = []
    reasons = Counter()
    invalid_ref = 0
    for row in rows:
        pred, ref = row.get("pred_parsed"), row.get("ref_parsed")
        if ref is None:
            invalid_ref += 1
            continue
        if pred == ref:
            continue
        bad.append(row)
        if pred is None:
            reasons["格式无效"] += 1
        if pred and pred.get(main_key) != ref.get(main_key):
            reasons[f"{main_key}错"] += 1
        if pred and pred.get(main_key) == ref.get(main_key):
            ref_keys = set((ref.get(param_key) or {}).keys())
            pred_keys = set((pred.get(param_key) or {}).keys())
            if ref_keys - pred_keys:
                reasons["缺参数"] += 1
            if pred_keys - ref_keys:
                reasons["字段名错/多参数"] += 1
            if any(
                (ref.get(param_key) or {})[key] != (pred.get(param_key) or {})[key]
                for key in ref_keys & pred_keys
            ):
                reasons["值错"] += 1

    lines = [f"===== BADCASE 报告 (task={task}, {len(bad)}/{len(rows)} 条错) =====\n"]
    if invalid_ref:
        lines.append(f"无效参考标注（未计入模型 badcase）: {invalid_ref} 条")
    lines.append("错因汇总(同条可多类):")
    for name, count in reasons.most_common():
        lines.append(f"  {name}: {count}")
    lines.append("\n===== 逐条 =====\n")
    for idx, row in enumerate(bad, 1):
        pred, ref = row.get("pred_parsed"), row.get("ref_parsed")
        lines.append(f"[{idx:04d}] user: {(row.get('user') or '')[-200:]}")
        lines.append(f"  REF : {(row.get('ref') or '')[:200]}")
        lines.append(f"  PRED: {(row.get('pred_text') or '')[:200]}")
        if pred is None:
            lines.append(f"  解析: PRED 格式无效; REF {main_key}={ref.get(main_key)}")
        else:
            ref_value, pred_value = ref.get(main_key), pred.get(main_key)
            lines.append(
                f"  [{'OK' if ref_value == pred_value else 'XX'}] "
                f"{main_key}: ref={ref_value} | pred={pred_value}"
            )
            ref_params = ref.get(param_key) or {}
            pred_params = pred.get(param_key) or {}
            keys = list(ref_params) + [key for key in pred_params if key not in ref_params]
            for key in keys:
                ref_value = ref_params.get(key, "〈缺〉")
                pred_value = pred_params.get(key, "〈缺〉")
                tag = " (多余)" if ref_value == "〈缺〉" else (" (缺)" if pred_value == "〈缺〉" else "")
                lines.append(
                    f"       [{'OK' if ref_value == pred_value else 'XX'}] "
                    f"{key}: ref={ref_value} | pred={pred_value}{tag}"
                )
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return len(bad)


def merge(tmp_files, output_dir, do_eval, task):
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for path in tmp_files:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                rows.extend(json.loads(line) for line in f if line.strip())
    rows.sort(key=lambda row: int(row["utt_id"]))

    parser = parse_agent if task == "agent" else parse_intent
    for row in rows:
        row["pred_parsed"] = parser(row.get("pred_text", ""))
        row["ref_parsed"] = parser(row["ref"]) if row.get("ref") else None

    detail_path = os.path.join(output_dir, "results_detail.jsonl")
    with open(detail_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if do_eval:
        preds = [row["pred_parsed"] for row in rows]
        refs = [row["ref_parsed"] or {} for row in rows]
        metrics = agent_metrics(preds, refs) if task == "agent" else intent_metrics(preds, refs)
        if task == "agent":
            metrics["exact_str_rate"] = (
                sum(
                    1
                    for row in rows
                    if (row.get("pred_text") or "").strip() == (row.get("ref") or "").strip()
                )
                / len(rows)
                if rows
                else 0.0
            )
        with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"{task.upper()} 评测指标：")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")

    badcase_path = os.path.join(output_dir, "badcase.txt")
    count = write_badcase(rows, badcase_path, task)
    print(f"badcase：{badcase_path} ({count}/{len(rows)} 条错)")
    print(f"明细：{detail_path}")


def main():
    args = parse_args()
    if args.task is None:
        args.task = "nlu" if args.backend == "joint" else "agent"
    items = load_items(args.input_file, args.system or NLU_SYSTEM_PROMPT)
    if not items:
        raise ValueError(f"输入文件无可用样本：{args.input_file}")
    gpu_ids = [int(value.strip()) for value in args.gpu_ids.split(",") if value.strip()]
    if not gpu_ids:
        raise ValueError("gpu_ids 不能为空")

    os.makedirs(args.output_dir, exist_ok=True)
    tmp_files = [
        os.path.join(args.output_dir, f"tmp_rank{rank}.jsonl")
        for rank in range(len(gpu_ids))
    ]
    print(f"{args.task.upper()} 推理配置")
    print(f"后端：{args.backend}")
    print(f"基线：{args.ckpt}")
    print(f"LoRA：{args.lora or '(无，基线)'}")
    print(f"输入：{args.input_file}")
    print(f"输出：{args.output_dir}")
    print(f"GPU：{gpu_ids}")
    print(f"样本数：{len(items)}")
    print(f"评测：{'是' if args.eval else '否'}")

    if len(gpu_ids) == 1:
        worker(0, gpu_ids[0], 1, args, items, tmp_files[0])
    else:
        ctx = mp.get_context("spawn")
        processes = [
            ctx.Process(
                target=worker,
                args=(rank, gpu_id, len(gpu_ids), args, items, tmp_files[rank]),
            )
            for rank, gpu_id in enumerate(gpu_ids)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(f"子进程失败，exitcode={process.exitcode}")

    merge(tmp_files, args.output_dir, args.eval, args.task)
    for path in tmp_files:
        if os.path.exists(path):
            os.remove(path)


if __name__ == "__main__":
    main()
