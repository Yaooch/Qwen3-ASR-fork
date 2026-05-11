#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""热词专项评测：检索、识别、纠错和非热词带偏分析。"""

import argparse
import json
import os
import re
import sys
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


TAG_RE = re.compile(r"<[^>]*>")


@dataclass
class EditResult:
    n: int = 0
    cor: int = 0
    sub: int = 0
    dele: int = 0
    ins: int = 0

    @property
    def err(self) -> int:
        return self.sub + self.dele + self.ins

    @property
    def cer(self) -> float:
        return self.err / self.n if self.n else 0.0

    def add(self, other: "EditResult") -> None:
        self.n += other.n
        self.cor += other.cor
        self.sub += other.sub
        self.dele += other.dele
        self.ins += other.ins


@dataclass
class Counts:
    samples: int = 0
    samples_with_target: int = 0
    target_instances: int = 0
    aux_hit_instances: int = 0
    baseline_hit_instances: int = 0
    retrieval_hit_instances: int = 0
    final_hit_instances: int = 0
    retrieved_then_final_instances: int = 0
    correction_candidates: int = 0
    correction_retrieved: int = 0
    correction_success: int = 0
    correction_success_after_retrieval: int = 0
    correction_regressions: int = 0
    retrieved_total: int = 0
    retrieved_true: int = 0
    retrieved_false: int = 0
    false_final_hotwords: int = 0
    no_target_samples: int = 0
    no_target_retrieved_samples: int = 0
    no_target_false_final_samples: int = 0
    nonhot_degraded_samples: int = 0
    top1_hits: int = 0
    top3_hits: int = 0
    top5_hits: int = 0
    top10_hits: int = 0
    baseline_nonhot: EditResult = None
    final_nonhot: EditResult = None

    def __post_init__(self) -> None:
        if self.baseline_nonhot is None:
            self.baseline_nonhot = EditResult()
        if self.final_nonhot is None:
            self.final_nonhot = EditResult()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate hotword retrieval and recognition from results_detail.jsonl.")
    parser.add_argument(
        "--ref_path",
        "--ref_dir",
        required=True,
        help="参考文本文件或目录。每行：utt_id<TAB>text[<TAB>domain]。",
    )
    parser.add_argument(
        "--detail_path",
        required=True,
        help="注入热词后的 results_detail.jsonl。",
    )
    parser.add_argument(
        "--baseline_detail_path",
        default=None,
        help="可选：不注入热词的 baseline results_detail.jsonl，用于公平评估非热词带偏。",
    )
    parser.add_argument(
        "--hotword_file",
        required=True,
        help="热词文件，每行一个热词。",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="汇总报告输出路径。",
    )
    parser.add_argument(
        "--detail_output_path",
        default=None,
        help="可选：样本级明细 jsonl。",
    )
    parser.add_argument(
        "--badcase_path",
        default=None,
        help="可选：人工查看 badcase txt。",
    )
    parser.add_argument(
        "--final_field",
        default="text",
        help="注入热词后的最终识别文本字段，默认 text。",
    )
    parser.add_argument(
        "--baseline_field",
        default="text",
        help="baseline detail 中的最终识别文本字段，默认 text。",
    )
    parser.add_argument(
        "--aux_field",
        default="aux_text",
        help="粗识别文本字段，默认 aux_text。",
    )
    parser.add_argument(
        "--hotwords_field",
        default="hotwords",
        help="召回热词字段，默认 hotwords。",
    )
    parser.add_argument(
        "--case_sensitive",
        action="store_true",
        help="英文大小写敏感。",
    )
    parser.add_argument(
        "--topk_badcases",
        type=int,
        default=100,
        help="每类最多输出多少条 badcase。",
    )
    return parser.parse_args()


def normalize(text: str, case_sensitive: bool = False) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = TAG_RE.sub("", text)
    if not case_sensitive:
        text = text.lower()
    return text


def compact(text: str, case_sensitive: bool = False) -> str:
    text = normalize(text, case_sensitive=case_sensitive)
    chars = []
    for ch in text:
        cat = unicodedata.category(ch)
        if ch.isspace() or cat.startswith("P") or cat.startswith("S"):
            continue
        chars.append(ch)
    return "".join(chars)


def iter_files(path: str) -> Iterable[str]:
    if os.path.isfile(path):
        yield path
        return
    if not os.path.isdir(path):
        return
    for name in sorted(os.listdir(path)):
        sub = os.path.join(path, name)
        if os.path.isfile(sub):
            yield sub


def read_refs(path: str) -> Dict[str, str]:
    refs: Dict[str, str] = {}
    paths = list(iter_files(path))
    if not paths:
        raise FileNotFoundError(f"未找到参考文件：{path}")

    for cur in paths:
        with open(cur, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                tab_parts = line.split("\t")
                ws_parts = line.split(maxsplit=1)
                if len(tab_parts) >= 2:
                    parts = tab_parts
                elif len(ws_parts) >= 2:
                    parts = ws_parts
                else:
                    print(f"跳过非法参考行：{cur}:{line_no}", file=sys.stderr)
                    continue
                utt_id = parts[0]
                text = parts[1] if len(parts) == 2 else "\t".join(parts[1:-1]).strip()
                refs[utt_id] = text
    return refs


def read_hotwords(path: str) -> List[str]:
    words = []
    seen = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip()
            if not word:
                continue
            if word in seen:
                continue
            seen.add(word)
            words.append(word)
    return words


def read_details(path: str) -> Dict[str, dict]:
    rows: Dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            utt_id = obj.get("utt_id")
            if not utt_id:
                print(f"跳过无 utt_id 明细行：{path}:{line_no}", file=sys.stderr)
                continue
            rows[str(utt_id)] = obj
    return rows


def count_occurrences(text: str, word: str) -> int:
    if not word:
        return 0
    count = 0
    start = 0
    while True:
        idx = text.find(word, start)
        if idx < 0:
            return count
        count += 1
        start = idx + len(word)


def target_instances(ref_text: str, hotwords: Sequence[str], case_sensitive: bool) -> List[str]:
    ref = compact(ref_text, case_sensitive=case_sensitive)
    out: List[str] = []
    for word in hotwords:
        norm_word = compact(word, case_sensitive=case_sensitive)
        for _ in range(count_occurrences(ref, norm_word)):
            out.append(word)
    return out


def contains_word(text: str, word: str, case_sensitive: bool) -> bool:
    return compact(word, case_sensitive=case_sensitive) in compact(text, case_sensitive=case_sensitive)


def mask_hotwords(text: str, hotwords: Sequence[str], case_sensitive: bool) -> str:
    out = compact(text, case_sensitive=case_sensitive)
    norm_words = sorted(
        {compact(w, case_sensitive=case_sensitive) for w in hotwords if compact(w, case_sensitive=case_sensitive)},
        key=len,
        reverse=True,
    )
    for word in norm_words:
        out = out.replace(word, "")
    return out


def hotword_mask(text: str, hotwords: Sequence[str], case_sensitive: bool) -> Tuple[str, List[bool]]:
    ref = compact(text, case_sensitive=case_sensitive)
    mask = [False] * len(ref)
    norm_words = sorted(
        {compact(w, case_sensitive=case_sensitive) for w in hotwords if compact(w, case_sensitive=case_sensitive)},
        key=len,
        reverse=True,
    )
    for word in norm_words:
        start = 0
        while True:
            idx = ref.find(word, start)
            if idx < 0:
                break
            for pos in range(idx, idx + len(word)):
                mask[pos] = True
            start = idx + len(word)
    return ref, mask


def edit_distance_counts(
    ref: Sequence[str],
    hyp: Sequence[str],
    exclude_ref_mask: Optional[Sequence[bool]] = None,
) -> EditResult:
    n = len(ref)
    m = len(hyp)
    if exclude_ref_mask is None:
        exclude_ref_mask = [False] * n
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bt = [[""] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        bt[i][0] = "del"
    for j in range(1, m + 1):
        dp[0][j] = j
        bt[0][j] = "ins"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                best = (dp[i - 1][j - 1], "cor")
            else:
                best = (dp[i - 1][j - 1] + 1, "sub")
            best = min(best, (dp[i - 1][j] + 1, "del"), (dp[i][j - 1] + 1, "ins"), key=lambda x: x[0])
            dp[i][j], bt[i][j] = best

    result = EditResult(n=sum(1 for x in exclude_ref_mask if not x))
    i, j = n, m
    while i > 0 or j > 0:
        op = bt[i][j]
        if op == "cor":
            if not exclude_ref_mask[i - 1]:
                result.cor += 1
            i -= 1
            j -= 1
        elif op == "sub":
            if not exclude_ref_mask[i - 1]:
                result.sub += 1
            i -= 1
            j -= 1
        elif op == "del":
            if not exclude_ref_mask[i - 1]:
                result.dele += 1
            i -= 1
        elif op == "ins":
            near_excluded = (i > 0 and exclude_ref_mask[i - 1]) or (i < n and exclude_ref_mask[i])
            if not near_excluded:
                result.ins += 1
            j -= 1
        else:
            break
    return result


def field_text(obj: dict, field: str) -> str:
    return str(obj.get(field) or "")


def field_hotwords(obj: dict, field: str) -> List[str]:
    val = obj.get(field) or []
    if isinstance(val, list):
        return [str(x) for x in val if str(x)]
    if isinstance(val, str):
        return [x.strip() for x in re.split(r"[,，\s]+", val) if x.strip()]
    return []


def percent(num: int, den: int) -> str:
    return f"{(num / den * 100.0):.2f}%" if den else "0.00%"


def ratio_text(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def evaluate(args: argparse.Namespace) -> Tuple[Counts, List[dict], Dict[str, List[dict]], int, int]:
    refs = read_refs(args.ref_path)
    details = read_details(args.detail_path)
    baseline_details = read_details(args.baseline_detail_path) if args.baseline_detail_path else {}
    hotwords = read_hotwords(args.hotword_file)
    counts = Counts()
    detail_rows: List[dict] = []
    badcases = {
        "召回失败": [],
        "召回成功但识别失败": [],
        "误召回": [],
        "误注入识别": [],
        "非热词退化": [],
    }
    missing = 0
    missing_baseline = 0

    for utt_id, ref_text in refs.items():
        obj = details.get(utt_id)
        if obj is None:
            missing += 1
            continue
        base_obj = baseline_details.get(utt_id) if baseline_details else None
        if baseline_details and base_obj is None:
            missing_baseline += 1

        counts.samples += 1
        aux_text = field_text(obj, args.aux_field)
        final_text = field_text(obj, args.final_field)
        baseline_text = field_text(base_obj, args.baseline_field) if base_obj is not None else aux_text
        retrieved = field_hotwords(obj, args.hotwords_field)
        prompt = obj.get("prompt")
        targets = target_instances(ref_text, hotwords, args.case_sensitive)
        target_set = set(targets)
        retrieved_set = set(retrieved)

        if targets:
            counts.samples_with_target += 1
        else:
            counts.no_target_samples += 1
            if retrieved:
                counts.no_target_retrieved_samples += 1

        counts.target_instances += len(targets)
        counts.retrieved_total += len(retrieved)
        true_retrieved = [w for w in retrieved if w in target_set]
        false_retrieved = [w for w in retrieved if w not in target_set]
        counts.retrieved_true += len(true_retrieved)
        counts.retrieved_false += len(false_retrieved)

        false_final = [
            w for w in false_retrieved
            if contains_word(final_text, w, args.case_sensitive) and not contains_word(ref_text, w, args.case_sensitive)
        ]
        counts.false_final_hotwords += len(false_final)
        if not targets and false_final:
            counts.no_target_false_final_samples += 1

        target_rows = []
        retrieval_hits = 0
        final_hits = 0
        aux_hits = 0
        corrected = 0
        for word in targets:
            aux_hit = contains_word(aux_text, word, args.case_sensitive)
            baseline_hit = contains_word(baseline_text, word, args.case_sensitive)
            final_hit = contains_word(final_text, word, args.case_sensitive)
            rank = retrieved.index(word) + 1 if word in retrieved else None
            retrieval_hit = rank is not None

            counts.aux_hit_instances += int(aux_hit)
            counts.baseline_hit_instances += int(baseline_hit)
            counts.final_hit_instances += int(final_hit)
            counts.retrieval_hit_instances += int(retrieval_hit)
            counts.retrieved_then_final_instances += int(retrieval_hit and final_hit)
            counts.top1_hits += int(rank is not None and rank <= 1)
            counts.top3_hits += int(rank is not None and rank <= 3)
            counts.top5_hits += int(rank is not None and rank <= 5)
            counts.top10_hits += int(rank is not None and rank <= 10)

            if not baseline_hit:
                counts.correction_candidates += 1
                counts.correction_retrieved += int(retrieval_hit)
                counts.correction_success += int(final_hit)
                counts.correction_success_after_retrieval += int(retrieval_hit and final_hit)
            counts.correction_regressions += int(baseline_hit and not final_hit)
            if not baseline_hit and retrieval_hit and final_hit:
                corrected += 1

            aux_hits += int(aux_hit)
            retrieval_hits += int(retrieval_hit)
            final_hits += int(final_hit)
            target_rows.append({
                "word": word,
                "aux_hit": aux_hit,
                "baseline_hit": baseline_hit,
                "retrieval_rank": rank,
                "retrieval_hit": retrieval_hit,
                "final_hit": final_hit,
                "corrected": bool((not baseline_hit) and retrieval_hit and final_hit),
                "regressed": bool(baseline_hit and not final_hit),
            })

        ref_compact, ref_hotword_mask = hotword_mask(ref_text, target_set, args.case_sensitive)
        ref_nonhot = "".join(ch for ch, masked in zip(ref_compact, ref_hotword_mask) if not masked)
        baseline_nonhot = mask_hotwords(baseline_text, target_set, args.case_sensitive)
        final_nonhot = mask_hotwords(final_text, target_set, args.case_sensitive)
        baseline_edit = edit_distance_counts(
            list(ref_compact),
            list(compact(baseline_text, args.case_sensitive)),
            exclude_ref_mask=ref_hotword_mask,
        )
        final_edit = edit_distance_counts(
            list(ref_compact),
            list(compact(final_text, args.case_sensitive)),
            exclude_ref_mask=ref_hotword_mask,
        )
        counts.baseline_nonhot.add(baseline_edit)
        counts.final_nonhot.add(final_edit)
        nonhot_degraded = final_edit.cer > baseline_edit.cer
        counts.nonhot_degraded_samples += int(nonhot_degraded)

        row = {
            "utt_id": utt_id,
            "ref": ref_text,
            "aux_text": aux_text,
            "baseline_text": baseline_text,
            "final_text": final_text,
            "ref_norm": compact(ref_text, args.case_sensitive),
            "aux_norm": compact(aux_text, args.case_sensitive),
            "baseline_norm": compact(baseline_text, args.case_sensitive),
            "final_norm": compact(final_text, args.case_sensitive),
            "retrieved": retrieved,
            "target_hotwords": targets,
            "targets": target_rows,
            "target_count": len(targets),
            "aux_hit_count": aux_hits,
            "retrieval_hit_count": retrieval_hits,
            "final_hit_count": final_hits,
            "corrected_count": corrected,
            "false_retrieved": false_retrieved,
            "false_final_hotwords": false_final,
            "ref_nonhot": ref_nonhot,
            "baseline_nonhot": baseline_nonhot,
            "final_nonhot": final_nonhot,
            "baseline_nonhot_cer": baseline_edit.cer,
            "final_nonhot_cer": final_edit.cer,
            "nonhot_degraded": nonhot_degraded,
            "prompt": prompt,
        }
        detail_rows.append(row)

        if targets and retrieval_hits < len(targets):
            badcases["召回失败"].append(row)
        if retrieval_hits > 0 and final_hits < retrieval_hits:
            badcases["召回成功但识别失败"].append(row)
        if false_retrieved:
            badcases["误召回"].append(row)
        if false_final:
            badcases["误注入识别"].append(row)
        if nonhot_degraded:
            badcases["非热词退化"].append(row)

    return counts, detail_rows, badcases, missing, missing_baseline


def write_summary(path: str, counts: Counts, missing: int, missing_baseline: int, has_baseline: bool) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    retrieval_precision = counts.retrieved_true / counts.retrieved_total if counts.retrieved_total else 0.0
    recognized_after_retrieval = (
        counts.retrieved_then_final_instances / counts.retrieval_hit_instances
        if counts.retrieval_hit_instances else 0.0
    )
    correction_after_retrieval = (
        counts.correction_success_after_retrieval / counts.correction_retrieved
        if counts.correction_retrieved else 0.0
    )
    with open(path, "w", encoding="utf-8") as f:
        print("热词专项评测", file=f)
        print("", file=f)
        print(f"样本数：{counts.samples}", file=f)
        print(f"缺失样本数：{missing}", file=f)
        if has_baseline:
            print(f"baseline 缺失样本数：{missing_baseline}", file=f)
        print(f"含热词样本数：{counts.samples_with_target}", file=f)
        print(f"无热词样本数：{counts.no_target_samples}", file=f)
        print(f"目标热词实例数：{counts.target_instances}", file=f)
        print("", file=f)
        print("检索：", file=f)
        print(f"R@1：{percent(counts.top1_hits, counts.target_instances)}", file=f)
        print(f"R@3：{percent(counts.top3_hits, counts.target_instances)}", file=f)
        print(f"R@5：{percent(counts.top5_hits, counts.target_instances)}", file=f)
        print(f"R@10：{percent(counts.top10_hits, counts.target_instances)}", file=f)
        print(f"召回率：{percent(counts.retrieval_hit_instances, counts.target_instances)}", file=f)
        print(f"召回准确率：{ratio_text(retrieval_precision)}", file=f)
        print(f"召回热词总数：{counts.retrieved_total}", file=f)
        print(f"正确召回数：{counts.retrieved_true}", file=f)
        print(f"误召回数：{counts.retrieved_false}", file=f)
        print("", file=f)
        print("识别：", file=f)
        if has_baseline:
            print(f"baseline LLM 热词识别率：{percent(counts.baseline_hit_instances, counts.target_instances)}", file=f)
        else:
            print(f"对比文本热词识别率：{percent(counts.baseline_hit_instances, counts.target_instances)}", file=f)
        print(f"最终热词识别率：{percent(counts.final_hit_instances, counts.target_instances)}", file=f)
        print(f"召回后识别率：{ratio_text(recognized_after_retrieval)}", file=f)
        print("", file=f)
        print("纠错：", file=f)
        base_name = "baseline LLM" if has_baseline else "对比文本"
        print(f"{base_name} 未识别热词数：{counts.correction_candidates}", file=f)
        print(f"{base_name} 未识别热词召回率：{percent(counts.correction_retrieved, counts.correction_candidates)}", file=f)
        print(f"相对 {base_name} 新增修对数：{counts.correction_success}", file=f)
        print(f"相对 {base_name} 整体纠错成功率：{percent(counts.correction_success, counts.correction_candidates)}", file=f)
        print(f"相对 {base_name} 召回后纠错成功率：{ratio_text(correction_after_retrieval)}", file=f)
        print(f"相对 {base_name} 热词改坏数：{counts.correction_regressions}", file=f)
        print(f"相对 {base_name} 热词净增数：{counts.correction_success - counts.correction_regressions}", file=f)
        print("", file=f)
        print("风险：", file=f)
        print(f"无热词样本误召回样本数：{counts.no_target_retrieved_samples}", file=f)
        print(f"误注入识别热词数：{counts.false_final_hotwords}", file=f)
        print(f"无热词样本误注入识别样本数：{counts.no_target_false_final_samples}", file=f)
        print(f"非热词退化样本数：{counts.nonhot_degraded_samples}", file=f)
        print(f"{base_name}非热词 CER：{ratio_text(counts.baseline_nonhot.cer)}", file=f)
        print(f"最终非热词 CER：{ratio_text(counts.final_nonhot.cer)}", file=f)


def write_detail(path: Optional[str], rows: Sequence[dict]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_badcases(path: Optional[str], badcases: Dict[str, List[dict]], topk: int) -> None:
    if not path:
        return

    def show(label: str, value) -> None:
        text = str(value)
        text = text.replace("\n", "\n" + " " * 24)
        print(f"{label:<22}: {text}", file=f)

    def yn(value: bool) -> str:
        return "Y" if value else "N"

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for name, rows in badcases.items():
            print(f"=============== {name} count={len(rows)} ===============", file=f)
            for row in rows[:topk]:
                target_detail = []
                for target in row["targets"]:
                    rank = target["retrieval_rank"] if target["retrieval_rank"] is not None else "-"
                    target_detail.append(
                        f"{target['word']}(aux={yn(target['aux_hit'])},base={yn(target['baseline_hit'])},"
                        f"rank={rank},final={yn(target['final_hit'])},fix={yn(target['corrected'])},"
                        f"bad={yn(target.get('regressed', False))})"
                    )
                show("utt_id", row["utt_id"])
                show("target", ",".join(row["target_hotwords"]))
                show("target_detail", " ".join(target_detail))
                show("retrieved", ",".join(row["retrieved"]))
                show("false_retrieved", ",".join(row["false_retrieved"]))
                show("false_final_hotwords", ",".join(row["false_final_hotwords"]))
                show("baseline_nonhot_cer", f"{row['baseline_nonhot_cer']:.4f}")
                show("final_nonhot_cer", f"{row['final_nonhot_cer']:.4f}")
                show("ref", row["ref_norm"])
                show("aux", row["aux_norm"])
                show("baseline", row["baseline_norm"])
                show("final", row["final_norm"])
                if row.get("prompt"):
                    show("prompt", row["prompt"])
                print("", file=f)
            print("", file=f)


def main() -> None:
    args = parse_args()
    counts, rows, badcases, missing, missing_baseline = evaluate(args)
    write_summary(
        args.output_path,
        counts,
        missing,
        missing_baseline,
        has_baseline=bool(args.baseline_detail_path),
    )
    write_detail(args.detail_output_path, rows)
    write_badcases(args.badcase_path, badcases, args.topk_badcases)
    print(f"热词评测完成：{args.output_path}")


if __name__ == "__main__":
    main()
