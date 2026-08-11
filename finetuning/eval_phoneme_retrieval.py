# coding: utf-8
"""在 STOP1/STOP2 上评测文本音素热词检索。"""
import argparse
import json
import os
import time

from finetuning.eval_glclap import latency_stats, normalize_text, read_candidates, read_key_value
from qwen_asr.joint.hotword import HotwordRetriever


def parse_args():
    parser = argparse.ArgumentParser(description="在 STOP1/STOP2 上评测文本音素热词检索")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--fast_threshold", type=float, default=0.55)
    parser.add_argument("--recall_threshold", type=float, default=0.65)
    parser.add_argument("--max_utts", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    queries = read_key_value(os.path.join(args.data_dir, "text"))
    targets = {
        key: normalize_text(value)
        for key, value in read_key_value(os.path.join(args.data_dir, "utt_hotword.txt")).items()
    }
    candidates = read_candidates(os.path.join(args.data_dir, "hotword.txt"))
    records = [(key, queries[key], targets[key]) for key in queries.keys() & targets.keys()]
    records.sort(key=lambda row: row[0])
    if args.max_utts > 0:
        records = records[:args.max_utts]

    index_start = time.perf_counter()
    retriever = HotwordRetriever(
        candidates,
        fast_threshold=args.fast_threshold,
        recall_threshold=args.recall_threshold,
    )
    index_ms = (time.perf_counter() - index_start) * 1000

    top1_hits = 0
    topk_hits = 0
    returned = 0
    empty = 0
    latencies = []
    details = []
    for index, (utt_id, query, target) in enumerate(records, 1):
        retrieve_start = time.perf_counter()
        words = retriever.retrieve(query, topk=args.top_k)
        retrieve_ms = (time.perf_counter() - retrieve_start) * 1000
        words = [normalize_text(word) for word in words]
        hit_top1 = bool(words) and words[0] == target
        hit_topk = target in words
        top1_hits += hit_top1
        topk_hits += hit_topk
        returned += len(words)
        empty += not words
        latencies.append(retrieve_ms)
        details.append({
            "utt_id": utt_id,
            "target": target,
            "text": query,
            "hit_top1": hit_top1,
            f"hit_top{args.top_k}": hit_topk,
            "retrieve_ms": round(retrieve_ms, 3),
            "returned_count": len(words),
            "retrieved": words,
        })
        if index % 500 == 0 or index == len(records):
            print(
                f"进度 {index}/{len(records)} top1_recall={top1_hits / index:.4f} "
                f"top{args.top_k}_recall={topk_hits / index:.4f}",
                flush=True,
            )

    stats = latency_stats(latencies)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            for row in details:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"明细：{args.output}")
    print(f"候选建库：{index_ms:.2f} ms（离线一次）")
    print(
        f"在线延迟 mean={stats['mean']:.2f} ms p50={stats['p50']:.2f} ms "
        f"p95={stats['p95']:.2f} ms max={stats['max']:.2f} ms"
    )
    print(
        f"结果 utterances={len(records)} candidates={len(candidates)} "
        f"top1_recall={top1_hits / len(records):.4f} "
        f"top{args.top_k}_recall={topk_hits / len(records):.4f} "
        f"mean_returned={returned / len(records):.2f} empty_rate={empty / len(records):.4f}"
    )


if __name__ == "__main__":
    main()
