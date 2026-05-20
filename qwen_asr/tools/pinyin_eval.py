import argparse
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, List


TAG_RE = re.compile(r"<\|.*?\|>|<[^>]+>")
@dataclass
class EditStat:
    sub: int = 0
    ins: int = 0
    dele: int = 0
    ref: int = 0

    @property
    def err(self):
        return self.sub + self.ins + self.dele

    @property
    def per(self):
        return self.err / self.ref if self.ref else 0.0

    def add(self, other):
        self.sub += other.sub
        self.ins += other.ins
        self.dele += other.dele
        self.ref += other.ref


def parse_args():
    p = argparse.ArgumentParser("拼音相似度评估")
    p.add_argument("--ref_path", "--ref_dir", required=True)
    p.add_argument("--result_path", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--badcase_path", default="")
    p.add_argument("--style", choices=["normal", "tone3"], default="tone3")
    p.add_argument("--topk_badcases", type=int, default=100)
    return p.parse_args()


def require_pypinyin(style_name: str):
    try:
        from pypinyin import Style, lazy_pinyin
    except ImportError as exc:
        raise RuntimeError("缺少依赖 pypinyin，请先安装。") from exc
    return lazy_pinyin, Style.TONE3 if style_name == "tone3" else Style.NORMAL


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    return TAG_RE.sub("", text).lower().strip()


def is_chinese(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def tokens(text: str, lazy_pinyin, style) -> List[str]:
    text = norm(text)
    out = []
    buf = []

    def flush():
        if buf:
            out.append("".join(buf))
            buf.clear()

    for ch in text:
        if is_chinese(ch):
            flush()
            items = lazy_pinyin(ch, style=style, errors="ignore", strict=False)
            if items:
                out.append(items[0].lower())
        elif ch.isascii() and ch.isalnum():
            buf.append(ch)
        else:
            flush()
    flush()
    return out


def edit(ref: List[str], hyp: List[str]) -> EditStat:
    n, m = len(ref), len(hyp)
    dp = [[(0, EditStat()) for _ in range(m + 1)] for _ in range(n + 1)]
    for i in range(1, n + 1):
        stat = EditStat(dele=i, ref=i)
        dp[i][0] = (i, stat)
    for j in range(1, m + 1):
        stat = EditStat(ins=j, ref=0)
        dp[0][j] = (j, stat)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            choices = []
            cost, prev = dp[i - 1][j - 1]
            stat = EditStat(prev.sub, prev.ins, prev.dele, i)
            if ref[i - 1] != hyp[j - 1]:
                stat.sub += 1
                cost += 1
            choices.append((cost, stat))

            cost, prev = dp[i - 1][j]
            choices.append((cost + 1, EditStat(prev.sub, prev.ins, prev.dele + 1, i)))

            cost, prev = dp[i][j - 1]
            choices.append((cost + 1, EditStat(prev.sub, prev.ins + 1, prev.dele, i)))
            dp[i][j] = min(choices, key=lambda x: x[0])
    stat = dp[n][m][1]
    stat.ref = n
    return stat


def iter_files(path: str) -> Iterable[str]:
    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            full = os.path.join(path, name)
            if os.path.isfile(full):
                yield full
    else:
        yield path


def parse_line(line: str):
    line = line.rstrip("\n")
    if not line:
        return None, None
    if "\t" in line:
        parts = line.split("\t")
        return parts[0], parts[1] if len(parts) > 1 else ""
    parts = line.split(maxsplit=1)
    return parts[0], parts[1] if len(parts) > 1 else ""


def read_refs(path: str) -> Dict[str, str]:
    refs = {}
    for file in iter_files(path):
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                utt_id, text = parse_line(line)
                if utt_id:
                    refs[utt_id] = text
    return refs


def read_results(path: str) -> Dict[str, str]:
    hyps = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            utt_id, text = parse_line(line)
            if utt_id:
                hyps[utt_id] = text
    return hyps


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def main():
    args = parse_args()
    lazy_pinyin, style = require_pypinyin(args.style)
    refs = read_refs(args.ref_path)
    hyps = read_results(args.result_path)
    total = EditStat()
    exact = 0
    count = 0
    missing = 0
    rows = []

    for utt_id, ref_text in refs.items():
        if utt_id not in hyps:
            missing += 1
            continue
        ref_tokens = tokens(ref_text, lazy_pinyin, style)
        hyp_tokens = tokens(hyps[utt_id], lazy_pinyin, style)
        stat = edit(ref_tokens, hyp_tokens)
        total.add(stat)
        count += 1
        exact += int(ref_tokens == hyp_tokens)
        rows.append({
            "utt_id": utt_id,
            "per": stat.per,
            "ref": ref_text,
            "hyp": hyps[utt_id],
            "ref_pinyin": ref_tokens,
            "hyp_pinyin": hyp_tokens,
        })

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    sar = exact / count if count else 0.0
    with open(args.output_path, "w", encoding="utf-8") as f:
        print("拼音评估", file=f)
        print(f"样本数：{count}", file=f)
        print(f"缺失数：{missing}", file=f)
        print(
            f"Overall -> SAR: {percent(sar)} PER: {percent(total.per)} "
            f"SUB: {total.sub} INS: {total.ins} DEL: {total.dele} REF: {total.ref}",
            file=f,
        )

    if args.badcase_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.badcase_path)), exist_ok=True)
        with open(args.badcase_path, "w", encoding="utf-8") as f:
            for row in sorted(rows, key=lambda x: x["per"], reverse=True)[:args.topk_badcases]:
                print(f"utt_id: {row['utt_id']}", file=f)
                print(f"PER: {percent(row['per'])}", file=f)
                print(f"ref: {row['ref']}", file=f)
                print(f"hyp: {row['hyp']}", file=f)
                print("ref_pinyin: " + " ".join(row["ref_pinyin"]), file=f)
                print("hyp_pinyin: " + " ".join(row["hyp_pinyin"]), file=f)
                print("", file=f)

    print(f"拼音汇总：{args.output_path}")


if __name__ == "__main__":
    main()
