#!/usr/bin/env python3
import argparse
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, List


TAG_RE = re.compile(r"<\|.*?\|>|<[^>]+>")


@dataclass
class Stat:
    sub: int = 0
    ins: int = 0
    dele: int = 0
    ref: int = 0

    @property
    def err(self):
        return self.sub + self.ins + self.dele

    @property
    def wer(self):
        return self.err / self.ref if self.ref else 0.0


def parse_args():
    p = argparse.ArgumentParser("文本 badcase 生成")
    p.add_argument("--ref_path", "--ref_dir", required=True)
    p.add_argument("--result_path", required=True)
    p.add_argument("--badcase_path", required=True)
    p.add_argument("--topk_badcases", type=int, default=100)
    return p.parse_args()


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    return TAG_RE.sub("", text).lower().strip()


def tokens(text: str) -> List[str]:
    out, buf = [], []

    def flush():
        if buf:
            out.append("".join(buf))
            buf.clear()

    for ch in norm(text):
        if "\u4e00" <= ch <= "\u9fff":
            flush()
            out.append(ch)
        elif ch.isascii() and ch.isalnum():
            buf.append(ch)
        else:
            flush()
    flush()
    return out


def edit(ref: List[str], hyp: List[str]) -> Stat:
    n, m = len(ref), len(hyp)
    dp = [[(0, Stat()) for _ in range(m + 1)] for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = (i, Stat(dele=i, ref=i))
    for j in range(1, m + 1):
        dp[0][j] = (j, Stat(ins=j, ref=0))
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost, prev = dp[i - 1][j - 1]
            sub = int(ref[i - 1] != hyp[j - 1])
            choices = [
                (cost + sub, Stat(prev.sub + sub, prev.ins, prev.dele, i)),
                (dp[i - 1][j][0] + 1, Stat(dp[i - 1][j][1].sub, dp[i - 1][j][1].ins, dp[i - 1][j][1].dele + 1, i)),
                (dp[i][j - 1][0] + 1, Stat(dp[i][j - 1][1].sub, dp[i][j - 1][1].ins + 1, dp[i][j - 1][1].dele, i)),
            ]
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


def read_texts(path: str) -> Dict[str, str]:
    rows = {}
    for file in iter_files(path):
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                utt_id, text = parse_line(line)
                if utt_id:
                    rows[utt_id] = text
    return rows


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def main():
    args = parse_args()
    refs = read_texts(args.ref_path)
    hyps = read_texts(args.result_path)
    rows = []
    for utt_id, ref in refs.items():
        if utt_id not in hyps:
            continue
        hyp = hyps[utt_id]
        stat = edit(tokens(ref), tokens(hyp))
        if stat.err:
            rows.append({"utt_id": utt_id, "wer": stat.wer, "ref": ref, "hyp": hyp})

    rows.sort(key=lambda x: x["wer"], reverse=True)
    if args.topk_badcases > 0:
        rows = rows[:args.topk_badcases]

    os.makedirs(os.path.dirname(os.path.abspath(args.badcase_path)), exist_ok=True)
    with open(args.badcase_path, "w", encoding="utf-8") as f:
        for row in rows:
            print(f"utt_id: {row['utt_id']}", file=f)
            print(f"WER: {percent(row['wer'])}", file=f)
            print(f"ref: {row['ref']}", file=f)
            print(f"hyp: {row['hyp']}", file=f)
            print("", file=f)
    print(f"文本Badcase：{args.badcase_path}")


if __name__ == "__main__":
    main()
