#!/usr/bin/env python3
# coding=utf-8
import argparse
import os
import re
import sys
import time
from typing import List, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from qwen_asr.joint.hotword import HotwordRetriever


def split_words(text: str) -> List[str]:
    return [x.strip() for x in re.split(r"[,，]", text or "") if x.strip()]


def read_rows(path: str) -> Tuple[List[Tuple[str, List[str]]], int]:
    rows = []
    total = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            total += 1
            parts = line.split("\t", 1)
            utt_id = parts[0]
            words = split_words(parts[1] if len(parts) > 1 else "")
            if words:
                rows.append((utt_id, words))
    return rows, total


def query_tokens(retriever: HotwordRetriever, query: str):
    return (
        retriever._tokens(query, retriever._style),
        retriever._tokens(query, retriever._tone3_style),
        retriever._tokens(query, retriever._initials_style),
        retriever._tokens(query, retriever._finals_style),
    )


def score_full_library(retriever: HotwordRetriever, query: str):
    q_py, q_tone, q_initials, q_finals = query_tokens(retriever, query)
    best_score = -1.0
    best_word = ""
    for entry in retriever._entries:
        score = retriever._window_score(entry, q_py, q_tone, q_initials, q_finals)
        if score > best_score:
            best_score = score
            best_word = entry.word
    return best_word, best_score


def main():
    parser = argparse.ArgumentParser("实体词对全库热词拼音打分耗时")
    parser.add_argument("--hotword_file", required=True)
    parser.add_argument("--target_hotword_file", required=True)
    parser.add_argument("--pinyin_style", choices=["normal", "tone3"], default="normal")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    t0 = time.perf_counter()
    retriever = HotwordRetriever.from_file(args.hotword_file, pinyin_style=args.pinyin_style)
    load_sec = time.perf_counter() - t0
    rows, total_rows = read_rows(args.target_hotword_file)
    query_count = sum(len(words) for _utt, words in rows)
    if not rows or query_count == 0:
        raise RuntimeError("target_hotword_file 中没有可用热词标签")

    checksum = 0.0
    total_ms = 0.0
    repeat = max(1, int(args.repeat))
    for _ in range(repeat):
        for _utt_id, words in rows:
            t1 = time.perf_counter()
            for word in words:
                _best_word, best_score = score_full_library(retriever, word)
                checksum += best_score
            total_ms += (time.perf_counter() - t1) * 1000.0

    audio_ops = len(rows) * repeat
    query_ops = query_count * repeat
    print(f"热词数：{len(retriever._entries)}")
    print(f"target 文件行数：{total_rows}")
    print(f"有效音频条数：{len(rows)}")
    print(f"跳过无热词音频：{total_rows - len(rows)}")
    print(f"实体 query 数：{query_count}")
    print(f"repeat：{repeat}")
    print(f"热词库加载耗时：{load_sec:.3f} s（不计入平均耗时）")
    print(f"全库直接打分平均耗时：{total_ms / audio_ops:.3f} ms/有效音频")
    print(f"全库直接打分平均耗时：{total_ms / (total_rows * repeat):.3f} ms/target文件行")
    print(f"全库直接打分平均耗时：{total_ms / query_ops:.3f} ms/query")
    print(f"单轮全量打分耗时：{total_ms / repeat / 1000.0:.3f} s")
    print(f"checksum：{checksum:.4f}")


if __name__ == "__main__":
    main()
