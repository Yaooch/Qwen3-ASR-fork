# coding: utf-8
"""固定候选词表热词检索 benchmark 的公共数据和指标逻辑。"""
import json
import os
from typing import Dict, List


def normalize_text(text: str) -> str:
    return " ".join(text.strip().upper().split())


def read_key_value(path: str) -> Dict[str, str]:
    rows = {}
    with open(path, encoding="utf-8") as file:
        for line in file:
            fields = line.strip().split(maxsplit=1)
            if len(fields) == 2:
                rows[fields[0]] = fields[1].strip()
    return rows


def read_candidates(path: str) -> List[str]:
    candidates = []
    seen = set()
    with open(path, encoding="utf-8") as file:
        for line in file:
            text = normalize_text(line)
            if text and text not in seen:
                candidates.append(text)
                seen.add(text)
    return candidates


def load_hotword_benchmark(data_dir: str, source_name: str, max_utts: int = 0, sort_records=False):
    sources = read_key_value(os.path.join(data_dir, source_name))
    targets = {
        key: [normalize_text(word) for word in value.replace("，", ",").split(",") if word.strip()]
        for key, value in read_key_value(os.path.join(data_dir, "utt_hotword.txt")).items()
    }
    transcripts = read_key_value(os.path.join(data_dir, "text"))
    candidates = read_candidates(os.path.join(data_dir, "hotword.txt"))
    expand_ids = any(len(words) > 1 for words in targets.values())
    records = []
    for key, source in sources.items():
        for index, target in enumerate(targets.get(key, [])):
            record_id = f"{key}__hot{index}" if expand_ids else key
            records.append((record_id, source, target))
            transcripts[record_id] = transcripts.get(key, "")
    if sort_records:
        records.sort(key=lambda row: row[0])
    if max_utts > 0:
        records = records[:max_utts]
    missing = sorted({target for _, _, target in records} - set(candidates))
    if missing:
        raise ValueError(f"目标热词不在候选词库中：{missing[:10]}")
    return candidates, transcripts, records


def latency_stats(values: List[float]) -> Dict[str, float]:
    ordered = sorted(values)

    def percentile(q: float) -> float:
        return ordered[round((len(ordered) - 1) * q)]

    return {
        "mean": sum(ordered) / len(ordered),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": ordered[-1],
    }


class RetrievalMetrics:
    """统一命中统计、detail schema、进度和延迟汇总。"""

    def __init__(self, top_k: int, total: int, progress_every: int):
        self.top_k = top_k
        self.total = total
        self.progress_every = progress_every
        self.top1_hits = 0
        self.topk_hits = 0
        self.returned = 0
        self.empty = 0
        self.latencies = []
        self.details = []

    def add(self, utt_id, target, text, retrieved, retrieve_ms, scores=None):
        words = [normalize_text(word) for word in retrieved]
        hit_top1 = bool(words) and words[0] == target
        hit_topk = target in words
        self.top1_hits += hit_top1
        self.topk_hits += hit_topk
        self.returned += len(words)
        self.empty += not words
        self.latencies.append(retrieve_ms)
        payload = words
        if scores is not None:
            payload = [
                {"text": word, "score": round(score, 6)}
                for word, score in zip(words, scores)
            ]
        self.details.append({
            "utt_id": utt_id,
            "target": target,
            "text": text,
            "hit_top1": hit_top1,
            f"hit_top{self.top_k}": hit_topk,
            "retrieve_ms": round(retrieve_ms, 3),
            "returned_count": len(words),
            "retrieved": payload,
        })
        done = len(self.details)
        if done % self.progress_every == 0 or done == self.total:
            print(
                f"进度 {done}/{self.total} top1_recall={self.top1_hits / done:.4f} "
                f"top{self.top_k}_recall={self.topk_hits / done:.4f}",
                flush=True,
            )

    @property
    def top1_recall(self):
        return self.top1_hits / self.total

    @property
    def topk_recall(self):
        return self.topk_hits / self.total

    @property
    def mean_returned(self):
        return self.returned / self.total

    @property
    def empty_rate(self):
        return self.empty / self.total

    def write(self, output: str):
        if not output:
            return
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        with open(output, "w", encoding="utf-8") as file:
            for row in self.details:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"明细：{output}")
