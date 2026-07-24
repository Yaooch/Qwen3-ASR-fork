# finetuning/infer_nlu_pure.py
"""纯 Qwen3 LLM NLU/Agent 批量推理 + 评测（不经 Qwen3ASRJointModel）。

直接用 AutoModelForCausalLM 加载原始 Qwen3 checkpoint + LoRA。
数据格式与 infer_nlu.py 一致。
"""
import argparse
import json
import multiprocessing as mp
import os
from datetime import datetime
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from qwen_asr.tools.nlu import (
    NLU_SYSTEM_PROMPT,
    agent_metrics,
    intent_metrics,
    nlu_messages,
    parse_agent,
    parse_intent,
)


def log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="纯 Qwen3 NLU/Agent 批量推理 / 评测。")
    p.add_argument("--ckpt", required=True, help="Qwen3 模型目录")
    p.add_argument("--lora", default=None, help="LoRA 目录")
    p.add_argument("--input_file", required=True, help="输入 jsonl")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--gpu_ids", default="0")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--system", default=None)
    p.add_argument("--task", choices=["nlu", "agent"], default="agent")
    p.add_argument("--eval", action="store_true")
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
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    log(f"进程{rank}启动：GPU {gpu_id}，样本 {len(shard)}")

    tokenizer = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt, dtype=dtype, device_map=None, trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).to(device)
    if args.lora:
        model = PeftModel.from_pretrained(model, args.lora)
    model.eval()
    default_system = args.system or NLU_SYSTEM_PROMPT
    tokenizer.padding_side = "left"

    # 查  thinking token id，推理时 suppress 阻止模型进入 thinking
    _added = tokenizer.get_added_vocab()
    _think_id = next((v for v in _added.values() if v == 151667), None)
    _suppress = [_think_id] if _think_id is not None else None

    rows = []
    with torch.no_grad():
        for idx, batch in enumerate(batches(shard, args.batch_size), 1):
            if idx == 1 or idx % 20 == 0:
                log(f"进程{rank}推理 batch {idx}")
            prompts = [
                tokenizer.apply_chat_template(
                    nlu_messages(b["system"] or default_system, b["user"]),
                    add_generation_prompt=True,
                    enable_thinking=False,
                    tokenize=False,
                )
                for b in batch
            ]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}
            gen = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=args.max_new_tokens,
                suppress_tokens=_suppress,
                do_sample=False,
            )
            decoded = tokenizer.batch_decode(
                gen[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for b, text in zip(batch, decoded):
                rows.append({
                    "utt_id": b["utt_id"],
                    "user": b["user"],
                    "pred_text": text,
                    "ref": b["ref"],
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
    from collections import Counter
    n = len(rows)
    bad = []
    reasons = Counter()
    invalid_ref = 0
    for r in rows:
        pp, rp = r.get("pred_parsed"), r.get("ref_parsed")
        if rp is None:
            invalid_ref += 1
            continue
        if pp == rp:
            continue
        bad.append(r)
        if pp is None:
            reasons["格式无效"] += 1
        if rp and pp and pp.get(main_key) != rp.get(main_key):
            reasons[f"{main_key}错"] += 1
        if rp and pp and pp.get(main_key) == rp.get(main_key):
            rpk = set((rp.get(param_key) or {}).keys())
            ppk = set((pp.get(param_key) or {}).keys())
            if rpk - ppk:
                reasons["缺参数"] += 1
            if ppk - rpk:
                reasons["字段名错/多参数"] += 1
            common = rpk & ppk
            if any((rp.get(param_key) or {})[k] != (pp.get(param_key) or {})[k] for k in common):
                reasons["值错"] += 1

    lines = [f"===== BADCASE 报告 (task={task}, {len(bad)}/{n} 条错) =====\n"]
    if invalid_ref:
        lines.append(f"无效参考标注（未计入模型 badcase）: {invalid_ref} 条")
    lines.append("错因汇总(同条可多类):")
    for k, v in reasons.most_common():
        lines.append(f"  {k}: {v}")
    lines.append("\n===== 逐条 =====\n")
    for i, r in enumerate(bad, 1):
        pp, rp = r.get("pred_parsed"), r.get("ref_parsed")
        lines.append(f"[{i:04d}] user: {(r.get('user') or '')[-200:]}")
        lines.append(f"  REF : {(r.get('ref') or '')[:200]}")
        lines.append(f"  PRED: {(r.get('pred_text') or '')[:200]}")
        if rp is None:
            lines.append("  解析: REF 格式无效, 无法 diff")
        elif pp is None:
            lines.append(f"  解析: PRED 格式无效; REF {main_key}={rp.get(main_key)}")
        else:
            rv, pv = rp.get(main_key), pp.get(main_key)
            lines.append(f"  [{'OK' if rv == pv else 'XX'}] {main_key}: ref={rv} | pred={pv}")
            rpk = rp.get(param_key) or {}
            ppk = pp.get(param_key) or {}
            all_keys = list(rpk.keys()) + [k for k in ppk if k not in rpk]
            for k in all_keys:
                rv2 = rpk.get(k, "〈缺〉")
                pv2 = ppk.get(k, "〈缺〉")
                tag = " (多余)" if rv2 == "〈缺〉" else (" (缺)" if pv2 == "〈缺〉" else "")
                lines.append(f"       [{'OK' if rv2 == pv2 else 'XX'}] {k}: ref={rv2} | pred={pv2}{tag}")
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
    rows.sort(key=lambda x: x["utt_id"])

    parser = parse_agent if task == "agent" else parse_intent
    for r in rows:
        r["pred_parsed"] = parser(r.get("pred_text", ""))
        r["ref_parsed"] = parser(r["ref"]) if r.get("ref") else None

    detail_path = os.path.join(output_dir, "results_detail.jsonl")
    with open(detail_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if do_eval:
        preds = [r["pred_parsed"] for r in rows]
        refs = [r["ref_parsed"] or {} for r in rows]
        if task == "agent":
            metrics = agent_metrics(preds, refs)
            n = len(rows)
            metrics["exact_str_rate"] = (
                sum(1 for r in rows if (r.get("pred_text") or "").strip() == (r.get("ref") or "").strip()) / n
                if n else 0.0
            )
        else:
            metrics = intent_metrics(preds, refs)
        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"{task.upper()} 评测指标：")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    badcase_path = os.path.join(output_dir, "badcase.txt")
    n_bad = write_badcase(rows, badcase_path, task)
    print(f"badcase：{badcase_path} ({n_bad}/{len(rows)} 条错)")
    print(f"明细：{os.path.join(output_dir, 'results_detail.jsonl')}")


def main():
    args = parse_args()
    gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
    items = load_items(args.input_file, args.system or NLU_SYSTEM_PROMPT)
    print(f"任务：{args.task}")
    print(f"输入：{args.input_file}")
    print(f"输出：{args.output_dir}")
    print(f"GPU：{gpu_ids}")
    print(f"样本数：{len(items)}")
    print(f"评测：{'是' if args.eval else '否'}")

    os.makedirs(args.output_dir, exist_ok=True)
    tmp_files = [os.path.join(args.output_dir, f"_tmp_rank{r}.jsonl") for r in gpu_ids]

    ctx = mp.get_context("spawn")
    procs = []
    for rank, gpu_id in enumerate(gpu_ids):
        p = ctx.Process(target=worker, args=(rank, gpu_id, len(gpu_ids), args, items, tmp_files[rank]))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    merge(tmp_files, args.output_dir, args.eval, args.task)
    for f in tmp_files:
        if os.path.exists(f):
            os.remove(f)


if __name__ == "__main__":
    main()
