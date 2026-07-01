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
    p.add_argument("--topk_badcases", type=int, default=0,
                   help="badcase 输出条数上限，0 或负数=全部输出（默认全部）")
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
    retrieve_ms: List[float] = []
    badcases = []
    missing = 0

    for utt_id, ref_text in refs.items():
        obj = details.get(utt_id)
        if obj is None:
            missing += 1
            continue

        counts.samples += 1
        ms = obj.get("hotword_retrieve_ms")
        if ms is not None:
            retrieve_ms.append(float(ms))
        base_text = obj.get("llm_text") or ""
        ctc_text = obj.get("ctc_text") or ""
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

        # badcase 判定：目标热词未召回、或召回但LLM没识别对、或改坏、或误注入干扰词；
        # 仅误召回但LLM正确未注入不算 badcase（LLM 抗住检索噪声是好行为）
        if false_final or any(
            (t["rank"] is None) or (not t["final_hit"]) for t in target_rows
        ):
            badcases.append({
                "utt_id": utt_id,
                "ref": ref_text,
                "ctc": ctc_text,
                "llm": base_text,
                "hotword_llm": final_text,
                "target": target_words,
                "retrieved": retrieved,
                "false_retrieved": false_retrieved,
                "false_final": false_final,
                "targets": target_rows,
            })

    return counts, badcases, missing, retrieve_ms


def write_summary(path: str, counts: Counts, missing: int, retrieve_ms: List[float]):
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
        print("", file=f)
        print("检索耗时：", file=f)
        if retrieve_ms:
            sv = sorted(retrieve_ms)
            n = len(sv)
            p50 = sv[min(n - 1, int(n * 0.5))]
            p95 = sv[min(n - 1, int(n * 0.95))]
            print(f"样本数：{n}", file=f)
            print(f"mean：{sum(sv) / n:.3f}ms", file=f)
            print(f"p50：{p50:.3f}ms", file=f)
            print(f"p95：{p95:.3f}ms", file=f)
            print(f"max：{sv[-1]:.3f}ms", file=f)
            print(f"总耗时：{sum(sv):.1f}ms", file=f)
        else:
            print("无检索耗时记录（detail 缺 hotword_retrieve_ms 字段）", file=f)


BADCASE_GROUPS = [
    ("regressed", "改坏（默认LLM对、加热词后错）"),
    ("retrieved_miss", "召回但LLM未修对"),
    ("missed_recall", "检索未召回"),
    ("false_final", "误注入干扰词"),
]


def badcase_groups(row: dict) -> Dict[str, List[str]]:
    """按目标热词分 3 类（+误注入）：
    - regressed 改坏: 召回了、默认LLM对、加 prompt 后错（prompt 帮倒忙）
    - retrieved_miss 召回但LLM未修对: 召回了、默认也错、最终仍错（LLM 注入失败）
    - missed_recall 检索未召回: 没召回（检索器漏，含最终碰巧对的）
    - false_final 误注入: 干扰词被写进输出（不在目标热词维度）
    """
    groups = {key: [] for key, _ in BADCASE_GROUPS}
    groups["false_final"] = row["false_final"]
    for item in row["targets"]:
        word = item["word"]
        if item["rank"] is None:
            groups["missed_recall"].append(word)   # 未召回 → 检索问题（不论最终对错）
        elif not item["final_hit"]:
            if item["base_hit"]:
                groups["regressed"].append(word)   # 召回+base对+final错 → 改坏
            else:
                groups["retrieved_miss"].append(word)  # 召回+base错+final错 → LLM未修对
        # else: 召回且最终对 → good
    return groups


def print_badcase(f, row: dict):
    print(f"utt_id: {row['utt_id']}", file=f)
    print("target: " + ",".join(row["target"]), file=f)
    print("retrieved: " + ",".join(row["retrieved"]), file=f)
    print(f"{'ref':<5}: {row['ref']}", file=f)
    print(f"{'ctc':<5}: {row['ctc']}", file=f)
    print(f"{'llm':<5}: {row['llm']}", file=f)
    print(f"{'final':<5}: {row['hotword_llm']}", file=f)
    print("", file=f)


def write_badcases(path: str, rows: List[dict], topk: int):
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    buckets = {key: [] for key, _ in BADCASE_GROUPS}
    if topk > 0:
        rows = rows[:topk]
    for row in rows:
        for key, words in badcase_groups(row).items():
            if words:
                buckets[key].append(row)

    with open(path, "w", encoding="utf-8") as f:
        for key, title in BADCASE_GROUPS:
            print(f"===== {title} ({len(buckets[key])}) =====", file=f)
            if not buckets[key]:
                print("无", file=f)
                print("", file=f)
                continue
            for row in buckets[key]:
                print_badcase(f, row)


def main():
    args = parse_args()
    counts, badcases, missing, retrieve_ms = evaluate(args)
    write_summary(args.output_path, counts, missing, retrieve_ms)
    write_badcases(args.badcase_path, badcases, args.topk_badcases)
    print(f"热词评测完成：{args.output_path}")


if __name__ == "__main__":
    main()
