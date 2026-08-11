# coding: utf-8
"""使用文本查询评测音素热词检索基线。"""
import argparse
import time

from qwen_asr_ext.glclap.benchmark import RetrievalMetrics, latency_stats, load_hotword_benchmark
from qwen_asr_ext.hotword import HotwordRetriever


def parse_args():
    parser = argparse.ArgumentParser(description="使用文本查询评测音素热词检索基线")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--fast_threshold", type=float, default=0.55)
    parser.add_argument("--recall_threshold", type=float, default=0.65)
    parser.add_argument("--max_utts", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    candidates, _, records = load_hotword_benchmark(
        args.data_dir, "text", args.max_utts, sort_records=True
    )

    index_start = time.perf_counter()
    retriever = HotwordRetriever(
        candidates,
        fast_threshold=args.fast_threshold,
        recall_threshold=args.recall_threshold,
    )
    index_ms = (time.perf_counter() - index_start) * 1000

    metrics = RetrievalMetrics(args.top_k, len(records), progress_every=500)
    for utt_id, query, target in records:
        retrieve_start = time.perf_counter()
        words = retriever.retrieve(query, topk=args.top_k)
        retrieve_ms = (time.perf_counter() - retrieve_start) * 1000
        metrics.add(utt_id, target, query, words, retrieve_ms)

    stats = latency_stats(metrics.latencies)
    metrics.write(args.output)
    print(f"候选建库：{index_ms:.2f} ms（离线一次）")
    print(
        f"在线延迟 mean={stats['mean']:.2f} ms p50={stats['p50']:.2f} ms "
        f"p95={stats['p95']:.2f} ms max={stats['max']:.2f} ms"
    )
    print(
        f"结果 utterances={len(records)} candidates={len(candidates)} "
        f"top1_recall={metrics.top1_recall:.4f} "
        f"top{args.top_k}_recall={metrics.topk_recall:.4f} "
        f"mean_returned={metrics.mean_returned:.2f} empty_rate={metrics.empty_rate:.4f}"
    )


if __name__ == "__main__":
    main()
