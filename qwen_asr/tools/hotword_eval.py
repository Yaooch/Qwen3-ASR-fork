#!/usr/bin/env python3
import argparse
import json
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, List


TAG_RE = re.compile(r"<\|.*?\|>|<[^>]+>")


@dataclass
class Counts:
    samples: int = 0
    samples_with_target: int = 0
    target_total: int = 0
    retrieved_total: int = 0
    retrieved_true: int = 0
    retrieved_final_hit: int = 0
    retrieved_false: int = 0
    top1: int = 0
    top3: int = 0
    top5: int = 0
    top10: int = 0
    base_hit: int = 0
    final_hit: int = 0
    corrected: int = 0
    regressed: int = 0
    false_final: int = 0


def parse_args():
    p = argparse.ArgumentParser("热词 prompt 评估")
    p.add_argument("--ref_path", "--ref_dir", required=True)
    p.add_argument("--detail_path", required=True)
    p.add_argument("--target_hotword_file", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--badcase_path", default="")
    p.add_argument("--topk_badcases", type=int, default=100)
    return p.parse_args()


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    return TAG_RE.sub("", text).lower()


def compact(text: str) -> str:
    return re.sub(r"\s+", "", norm(text))


def iter_files(path: str) -> Iterable[str]:
    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            full = os.path.join(path, name)
            if os.path.isfile(full):
                yield full
    else:
        yield path


def read_refs(path: str) -> Dict[str, str]:
    refs = {}
    for file in iter_files(path):
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    refs[parts[0]] = parts[1]
    return refs


def split_words(text: str) -> List[str]:
    return [x.strip() for x in re.split(r"[,，]", text or "") if x.strip()]


def read_target_hotwords(path: str) -> Dict[str, List[str]]:
    rows = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if parts and parts[0]:
                rows[parts[0]] = split_words(parts[1] if len(parts) > 1 else "")
    return rows


def read_details(path: str) -> Dict[str, dict]:
    rows = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            utt_id = obj.get("utt_id")
            if utt_id:
                rows[utt_id] = obj
    return rows


def contains(text: str, word: str) -> bool:
    return compact(word) in compact(text)


def percent(num: int, den: int) -> str:
    value = num / den if den else 0.0
    return f"{value * 100:.2f}% ({num}/{den})"


def ratio(num: int, den: int) -> str:
    value = num / den if den else 0.0
    return f"{value * 100:.2f}%"


def evaluate(args):
    refs = read_refs(args.ref_path)
    details = read_details(args.detail_path)
    target_map = read_target_hotwords(args.target_hotword_file)
    counts = Counts()
    badcases = []
    missing = 0

    for utt_id, ref_text in refs.items():
        obj = details.get(utt_id)
        if obj is None:
            missing += 1
            continue

        counts.samples += 1
        base_text = obj.get("llm_text") or ""
        final_text = obj.get("hotword_llm_text") or obj.get("text") or ""
        retrieved = obj.get("hotwords") or []
        target_words = target_map.get(utt_id, [])
        target_set = set(target_words)
        false_retrieved = [w for w in retrieved if w not in target_set]
        false_final = [w for w in false_retrieved if contains(final_text, w) and not contains(ref_text, w)]

        counts.target_total += len(target_words)
        counts.retrieved_total += len(retrieved)
        counts.retrieved_true += sum(1 for w in retrieved if w in target_set)
        counts.retrieved_false += len(false_retrieved)
        counts.false_final += len(false_final)
        counts.samples_with_target += int(bool(target_words))

        target_rows = []
        for word in target_words:
            rank = retrieved.index(word) + 1 if word in retrieved else None
            base_hit = contains(base_text, word)
            final_hit = contains(final_text, word)
            counts.top1 += int(rank is not None and rank <= 1)
            counts.top3 += int(rank is not None and rank <= 3)
            counts.top5 += int(rank is not None and rank <= 5)
            counts.top10 += int(rank is not None and rank <= 10)
            counts.base_hit += int(base_hit)
            counts.final_hit += int(final_hit)
            counts.retrieved_final_hit += int(rank is not None and final_hit)
            counts.corrected += int((not base_hit) and final_hit)
            counts.regressed += int(base_hit and not final_hit)
            target_rows.append({"word": word, "rank": rank, "base_hit": base_hit, "final_hit": final_hit})

        if false_final or any(not row["final_hit"] or row["base_hit"] != row["final_hit"] for row in target_rows):
            badcases.append({
                "utt_id": utt_id,
                "ref": ref_text,
                "llm": base_text,
                "hotword_llm": final_text,
                "target": target_words,
                "retrieved": retrieved,
                "false_retrieved": false_retrieved,
                "false_final": false_final,
                "targets": target_rows,
            })

    return counts, badcases, missing


def write_summary(path: str, counts: Counts, missing: int):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    precision = ratio(counts.retrieved_true, counts.retrieved_total)
    with open(path, "w", encoding="utf-8") as f:
        print("热词 prompt 评估", file=f)
        print(f"样本数：{counts.samples}", file=f)
        print(f"缺失数：{missing}", file=f)
        print(f"含目标热词样本数：{counts.samples_with_target}", file=f)
        print(f"目标热词数：{counts.target_total}", file=f)
        print("", file=f)
        print("召回：", file=f)
        print(f"R@1：{percent(counts.top1, counts.target_total)}", file=f)
        print(f"R@3：{percent(counts.top3, counts.target_total)}", file=f)
        print(f"R@5：{percent(counts.top5, counts.target_total)}", file=f)
        print(f"R@10：{percent(counts.top10, counts.target_total)}", file=f)
        print(f"召回准确率：{precision}", file=f)
        print(f"误召回数：{counts.retrieved_false}", file=f)
        print("", file=f)
        print("识别：", file=f)
        print(f"默认 LLM 热词识别率：{percent(counts.base_hit, counts.target_total)}", file=f)
        print(f"热词 LLM 热词识别率：{percent(counts.final_hit, counts.target_total)}", file=f)
        print(f"正确召回热词 LLM 识别率：{percent(counts.retrieved_final_hit, counts.retrieved_true)}", file=f)
        print(f"新增修对数：{counts.corrected}", file=f)
        print(f"改坏数：{counts.regressed}", file=f)
        print(f"误注入识别热词数：{counts.false_final}", file=f)


def write_badcases(path: str, rows: List[dict], topk: int):
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows[:topk]:
            print(f"utt_id: {row['utt_id']}", file=f)
            print("target: " + ",".join(row["target"]), file=f)
            print("retrieved: " + ",".join(row["retrieved"]), file=f)
            print("false_retrieved: " + ",".join(row["false_retrieved"]), file=f)
            print("false_final: " + ",".join(row["false_final"]), file=f)
            print("ref: " + row["ref"], file=f)
            print("llm: " + row["llm"], file=f)
            print("hotword_llm: " + row["hotword_llm"], file=f)
            print("", file=f)


def main():
    args = parse_args()
    counts, badcases, missing = evaluate(args)
    write_summary(args.output_path, counts, missing)
    write_badcases(args.badcase_path, badcases, args.topk_badcases)
    print(f"热词评测完成：{args.output_path}")


if __name__ == "__main__":
    main()
