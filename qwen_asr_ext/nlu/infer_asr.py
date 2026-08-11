# finetuning/infer_asr_nlu.py
"""ASR+NLU 批量推理 + 评测。

加载基线 joint + ASR+NLU LoRA，音频 -> "文本\n意图"，评测文本 CER + 意图指标
(name_acc / args_exact / json_valid)。

输入 jsonl 每行 {"messages":[{system},{user:audio_path},{assistant="language X<asr_text>文本\n意图JSON"}]}。
用法：见 infer_asr_nlu.sh。
"""
import argparse
import json
import multiprocessing as mp
import os
from typing import Dict, List

import editdistance
import torch

from qwen_asr_ext.joint.infer import batches, load_joint_model, log
from qwen_asr_ext.nlu.common import intent_metrics, parse_intent

ASR_NLU_PROMPT = "转写语音并提取用户意图"

def parse_args():
    p = argparse.ArgumentParser(description="Qwen3-ASR ASR+NLU 批量推理 / 评测。")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--lora", default=None)
    p.add_argument("--input_file", required=True, help="jsonl, 每行 messages 格式(带 audio path + assistant ref)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--gpu_ids", default="0")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--encoder_mode", choices=["offline", "stream", "train_mask"], default="offline")
    p.add_argument("--language", default="Chinese", help="force_language, 与训练 target 的 language X 前缀对齐")
    p.add_argument("--prompt", default=ASR_NLU_PROMPT)
    p.add_argument("--max_new_tokens", type=int, default=256)
    return p.parse_args()

def load_items(path: str) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            msgs = r["messages"]
            audio = next(m["content"][0]["path"] for m in msgs if m["role"] == "user")
            assistant = next(m["content"] for m in msgs if m["role"] == "assistant")
            items.append({"utt_id": str(line_no), "audio": audio, "ref": assistant})
    return items

def split_text_intent(s: str):
    """从 'language X<asr_text>文本\n意图JSON' 或 '文本\n意图JSON' 拆出 (文本, 意图dict|None)。"""
    if not s:
        return "", None
    if "\n" not in s:
        # 无意图, 整串是文本
        return s.split("<asr_text>", 1)[-1], None
    text_part, _, intent_str = s.rpartition("\n")
    text_part = text_part.split("<asr_text>", 1)[-1]
    intent = parse_intent(intent_str) if intent_str.strip().startswith("{") else None
    return text_part, intent

def worker(rank, gpu_id, world_size, args, items, tmp_path):
    shard = items[rank::world_size]
    if not torch.cuda.is_available():
        raise RuntimeError("当前环境不可用 CUDA。")
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    log(f"进程{rank}启动：GPU {gpu_id}，样本 {len(shard)}")

    model = load_joint_model(args.ckpt, dtype, device, args.lora, load_heads=False)

    rows = []
    with torch.no_grad():
        for idx, batch in enumerate(batches(shard, args.batch_size), 1):
            if idx == 1 or idx % 10 == 0:
                log(f"进程{rank}推理 batch {idx}")
            outs = model.transcribe(
                [x["audio"] for x in batch],
                modes="llm",
                language=[args.language] * len(batch),
                prompt=args.prompt,
                encoder_mode=args.encoder_mode,
                max_new_tokens=args.max_new_tokens,
            )
            if not isinstance(outs, list):
                outs = [outs]
            for b, out in zip(batch, outs):
                pred_raw = out.get("llm_text", "") if isinstance(out, dict) else str(out)
                pred_text, pred_intent = split_text_intent(pred_raw)
                ref_text, ref_intent = split_text_intent(b["ref"])
                rows.append({
                    "utt_id": b["utt_id"],
                    "audio": b["audio"],
                    "pred_text": pred_text,
                    "pred_intent": pred_intent,
                    "pred_raw": pred_raw,
                    "ref_text": ref_text,
                    "ref_intent": ref_intent,
                })

    with open(tmp_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log(f"进程{rank}完成")

def merge(tmp_files, output_dir):
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

    edits, chars = 0, 0
    for r in rows:
        edits += editdistance.eval(r["pred_text"], r["ref_text"])
        chars += len(r["ref_text"])
    cer = edits / chars if chars else 0.0
    preds = [r["pred_intent"] for r in rows]
    refs = [r["ref_intent"] or {} for r in rows]
    metrics = intent_metrics(preds, refs)
    metrics["text_cer"] = cer
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("ASR+NLU 评测指标：")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"明细：{detail_path}")

def main():
    args = parse_args()
    items = load_items(args.input_file)
    if not items:
        raise ValueError(f"输入文件无可用样本：{args.input_file}")
    ids = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip()]
    if not ids:
        raise ValueError("gpu_ids 不能为空")
    os.makedirs(args.output_dir, exist_ok=True)
    tmp_files = [os.path.join(args.output_dir, f"tmp_rank{r}.jsonl") for r in range(len(ids))]

    print("ASR+NLU 推理配置")
    print(f"基线：{args.ckpt}")
    print(f"LoRA：{args.lora or '(无，基线)'}")
    print(f"输入：{args.input_file}")
    print(f"输出：{args.output_dir}")
    print(f"GPU：{ids}")
    print(f"样本数：{len(items)}")
    print(f"encoder_mode：{args.encoder_mode}  language：{args.language}")

    if len(ids) == 1:
        worker(0, ids[0], 1, args, items, tmp_files[0])
    else:
        ctx = mp.get_context("spawn")
        procs = [
            ctx.Process(target=worker, args=(r, g, len(ids), args, items, tmp_files[r]))
            for r, g in enumerate(ids)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"子进程失败，exitcode={p.exitcode}")

    merge(tmp_files, args.output_dir)
    for path in tmp_files:
        if os.path.exists(path):
            os.remove(path)

if __name__ == "__main__":
    main()
