#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute pinyin-level similarity for ASR hypotheses.

This script is intentionally separate from inference: it consumes an existing
reference file plus either results.txt or results_detail.jsonl.
"""

import argparse
import json
import os
import re
import sys
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


CHINESE_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
ASCII_WORD_RE = re.compile(r"[a-zA-Z0-9]+")


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
    def per(self) -> float:
        return self.err / self.n if self.n else 0.0

    @property
    def similarity(self) -> float:
        return max(0.0, 1.0 - self.per)

    def add(self, other: "EditResult") -> None:
        self.n += other.n
        self.cor += other.cor
        self.sub += other.sub
        self.dele += other.dele
        self.ins += other.ins


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute pinyin edit similarity between reference text and ASR hypotheses."
    )
    parser.add_argument(
        "--ref_path",
        "--ref_dir",
        required=True,
        help="Reference file. Each line: utt_id<TAB>text[<TAB>domain].",
    )
    parser.add_argument(
        "--result_path",
        default=None,
        help="results.txt from infer.py. Each line: utt_id<TAB>text<TAB>language.",
    )
    parser.add_argument(
        "--detail_path",
        default=None,
        help="Optional results_detail.jsonl. Use with --hyp_field to evaluate aux_stream_text etc.",
    )
    parser.add_argument(
        "--hyp_field",
        default="text",
        help="Field to read from detail jsonl. Common values: text, aux_stream_text, llm_refined_text.",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Summary output path.",
    )
    parser.add_argument(
        "--detail_output_path",
        default=None,
        help="Optional per-utterance jsonl output path.",
    )
    parser.add_argument(
        "--badcase_path",
        default=None,
        help="Optional human-readable badcase txt sorted by PER descending.",
    )
    parser.add_argument(
        "--style",
        choices=["normal", "tone3"],
        default="normal",
        help="Pinyin style. normal ignores tone; tone3 keeps tone numbers, e.g. hao3.",
    )
    parser.add_argument(
        "--keep_non_chinese",
        action="store_true",
        help="Keep ASCII words/digits as tokens. Pure non-Chinese text keeps ASCII tokens automatically.",
    )
    parser.add_argument(
        "--topk_badcases",
        type=int,
        default=100,
        help="How many high-PER examples to write to --badcase_path.",
    )
    parser.add_argument(
        "--case_sensitive",
        action="store_true",
        help="Keep English case for non-Chinese tokens when --keep_non_chinese is set.",
    )
    return parser.parse_args()


def require_pypinyin(style_name: str):
    try:
        from pypinyin import Style, lazy_pinyin
    except ImportError as exc:
        raise SystemExit(
            "缺少依赖 pypinyin。请先安装：\n"
            "  pip install pypinyin\n"
            "或在当前环境中运行：\n"
            "  /root/miniconda3/envs/qwen3-asr/bin/pip install pypinyin"
        ) from exc

    if style_name == "normal":
        style = Style.NORMAL
    elif style_name == "tone3":
        style = Style.TONE3
    else:
        raise ValueError(f"Unsupported pinyin style: {style_name}")
    return lazy_pinyin, style


def normalize_text(text: str, case_sensitive: bool = False) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = strip_tags(text)
    if not case_sensitive:
        text = text.lower()
    return text


def strip_tags(text: str) -> str:
    chars = []
    in_tag = False
    for ch in text:
        if ch == "<":
            in_tag = True
            continue
        if in_tag:
            if ch == ">":
                in_tag = False
            continue
        chars.append(ch)
    return "".join(chars)


def is_chinese_char(ch: str) -> bool:
    return bool(CHINESE_RE.fullmatch(ch))


def text_to_tokens(
    text: str,
    lazy_pinyin,
    pinyin_style,
    keep_non_chinese: bool,
    case_sensitive: bool,
) -> List[str]:
    text = normalize_text(text, case_sensitive=case_sensitive)
    tokens: List[str] = []
    ascii_buf: List[str] = []
    keep_ascii = keep_non_chinese or not any(is_chinese_char(ch) for ch in text)

    def flush_ascii() -> None:
        if not ascii_buf:
            return
        chunk = "".join(ascii_buf)
        if keep_ascii and ASCII_WORD_RE.fullmatch(chunk):
            tokens.append(chunk)
        ascii_buf.clear()

    for ch in text:
        if is_chinese_char(ch):
            flush_ascii()
            py = lazy_pinyin(ch, style=pinyin_style, errors="ignore")
            if py:
                tokens.append(py[0])
            continue

        cat = unicodedata.category(ch)
        if ch.isspace() or cat.startswith("P") or cat.startswith("S"):
            flush_ascii()
            continue

        if ch.isascii() and ch.isalnum():
            ascii_buf.append(ch)
        else:
            flush_ascii()

    flush_ascii()
    return tokens


def edit_distance_counts(ref: List[str], hyp: List[str]) -> EditResult:
    n = len(ref)
    m = len(hyp)
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

            cand_del = (dp[i - 1][j] + 1, "del")
            cand_ins = (dp[i][j - 1] + 1, "ins")
            best = min(best, cand_del, cand_ins, key=lambda x: x[0])
            dp[i][j], bt[i][j] = best

    result = EditResult(n=n)
    i, j = n, m
    while i > 0 or j > 0:
        op = bt[i][j]
        if op == "cor":
            result.cor += 1
            i -= 1
            j -= 1
        elif op == "sub":
            result.sub += 1
            i -= 1
            j -= 1
        elif op == "del":
            result.dele += 1
            i -= 1
        elif op == "ins":
            result.ins += 1
            j -= 1
        else:
            break

    return result


def read_refs(ref_path: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    refs: Dict[str, str] = {}
    domains: Dict[str, str] = {}

    paths = list(iter_ref_files(ref_path))
    if not paths:
        raise FileNotFoundError(f"未找到参考文件：{ref_path}")

    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
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
                    print(f"跳过非法参考行：{path}:{line_no}", file=sys.stderr)
                    continue
                utt_id = parts[0]
                if len(parts) == 2:
                    text = parts[1]
                    domain = "default"
                else:
                    text = "\t".join(parts[1:-1]).strip()
                    domain = parts[-1].strip() or "default"
                refs[utt_id] = text
                domains[utt_id] = domain
    return refs, domains


def iter_ref_files(ref_path: str) -> Iterable[str]:
    if os.path.isfile(ref_path):
        yield ref_path
        return
    if not os.path.isdir(ref_path):
        return
    for name in sorted(os.listdir(ref_path)):
        path = os.path.join(ref_path, name)
        if os.path.isfile(path):
            yield path


def read_hyps_from_results(result_path: str) -> Dict[str, str]:
    hyps: Dict[str, str] = {}
    with open(result_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                print(f"跳过非法识别行：{result_path}:{line_no}", file=sys.stderr)
                continue
            hyps[parts[0]] = parts[1]
    return hyps


def read_hyps_from_detail(detail_path: str, hyp_field: str) -> Dict[str, str]:
    hyps: Dict[str, str] = {}
    with open(detail_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            utt_id = obj.get("utt_id")
            if not utt_id:
                print(f"跳过无 utt_id 明细行：{detail_path}:{line_no}", file=sys.stderr)
                continue
            hyps[utt_id] = str(obj.get(hyp_field) or "")
    return hyps


def percent(x: float) -> str:
    return f"{x * 100.0:.2f}%"


def main() -> None:
    args = parse_args()
    lazy_pinyin, pinyin_style = require_pypinyin(args.style)

    refs, domains = read_refs(args.ref_path)
    if args.detail_path:
        hyps = read_hyps_from_detail(args.detail_path, args.hyp_field)
        hyp_source = f"{args.detail_path}:{args.hyp_field}"
    elif args.result_path:
        hyps = read_hyps_from_results(args.result_path)
        hyp_source = args.result_path
    else:
        raise SystemExit("必须提供 --result_path 或 --detail_path")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    if args.detail_output_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.detail_output_path)), exist_ok=True)
    if args.badcase_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.badcase_path)), exist_ok=True)

    overall = EditResult()
    domain_stats: Dict[str, EditResult] = {}
    domain_exact: Dict[str, int] = {}
    domain_pairs: Dict[str, int] = {}
    exact_match = 0
    pair_count = 0
    detail_rows = []
    missing = 0

    for utt_id, ref_text in refs.items():
        if utt_id not in hyps:
            missing += 1
            continue
        hyp_text = hyps[utt_id]
        ref_tokens = text_to_tokens(
            ref_text,
            lazy_pinyin=lazy_pinyin,
            pinyin_style=pinyin_style,
            keep_non_chinese=args.keep_non_chinese,
            case_sensitive=args.case_sensitive,
        )
        hyp_tokens = text_to_tokens(
            hyp_text,
            lazy_pinyin=lazy_pinyin,
            pinyin_style=pinyin_style,
            keep_non_chinese=args.keep_non_chinese,
            case_sensitive=args.case_sensitive,
        )
        result = edit_distance_counts(ref_tokens, hyp_tokens)
        domain = domains.get(utt_id, "default")

        pair_count += 1
        is_exact = int(ref_tokens == hyp_tokens)
        exact_match += is_exact
        overall.add(result)
        domain_stats.setdefault(domain, EditResult()).add(result)
        domain_exact[domain] = domain_exact.get(domain, 0) + is_exact
        domain_pairs[domain] = domain_pairs.get(domain, 0) + 1

        detail_rows.append(
            {
                "utt_id": utt_id,
                "domain": domain,
                "ref_text": ref_text,
                "hyp_text": hyp_text,
                "ref_pinyin": ref_tokens,
                "hyp_pinyin": hyp_tokens,
                "sar": float(is_exact),
                "PER": result.per,
                "N": result.n,
                "C": result.cor,
                "S": result.sub,
                "D": result.dele,
                "I": result.ins,
            }
        )

    with open(args.output_path, "w", encoding="utf-8") as f:
        print(f"参考文件: {args.ref_path}", file=f)
        print(f"识别结果: {hyp_source}", file=f)
        print(f"拼音风格: {args.style}", file=f)
        print(f"保留非中文: {int(args.keep_non_chinese)}", file=f)
        print(f"匹配条数: {pair_count}", file=f)
        print(f"缺失条数: {missing}", file=f)
        print("", file=f)

        for domain, stat in sorted(domain_stats.items()):
            domain_count = domain_pairs.get(domain, 0)
            domain_sar = domain_exact.get(domain, 0) / domain_count if domain_count else 0.0
            print(
                f"Domain -> {domain}, "
                f"sar: {percent(domain_sar)}, "
                f"PER: {percent(stat.per)}, "
                f"cnt: {domain_count}, "
                f"N={stat.n} C={stat.cor} S={stat.sub} D={stat.dele} I={stat.ins}",
                file=f,
            )

        sar = exact_match / pair_count if pair_count else 0.0
        print(
            f"Overall -> OVERALL, "
            f"sar: {percent(sar)}, "
            f"PER: {percent(overall.per)}, "
            f"cnt: {pair_count}, "
            f"N={overall.n} C={overall.cor} S={overall.sub} D={overall.dele} I={overall.ins}",
            file=f,
        )

    if args.detail_output_path:
        with open(args.detail_output_path, "w", encoding="utf-8") as f:
            for row in detail_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.badcase_path:
        rows = sorted(detail_rows, key=lambda x: (-x["PER"], -x["N"]))
        with open(args.badcase_path, "w", encoding="utf-8") as f:
            for row in rows[: args.topk_badcases]:
                print(f"utt: {row['utt_id']}", file=f)
                print(f"domain: {row['domain']}", file=f)
                print(
                    f"sar: {percent(row['sar'])}, "
                    f"PER: {percent(row['PER'])}, "
                    f"N={row['N']} C={row['C']} S={row['S']} D={row['D']} I={row['I']}",
                    file=f,
                )
                print(f"ref: {row['ref_text']}", file=f)
                print(f"hyp: {row['hyp_text']}", file=f)
                print("ref_pinyin: " + " ".join(row["ref_pinyin"]), file=f)
                print("hyp_pinyin: " + " ".join(row["hyp_pinyin"]), file=f)
                print("", file=f)

    print(f"拼音汇总：{args.output_path}")
    if args.detail_output_path:
        print(f"拼音明细：{args.detail_output_path}")
    if args.badcase_path:
        print(f"拼音 badcase：{args.badcase_path}")


if __name__ == "__main__":
    main()
