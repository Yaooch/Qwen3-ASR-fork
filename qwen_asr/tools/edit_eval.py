"""文本评测共用的编辑距离与文本文件读取工具。"""
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
    def rate(self):
        return self.err / self.ref if self.ref else 0.0

    per = rate
    wer = rate

    def add(self, other):
        self.sub += other.sub
        self.ins += other.ins
        self.dele += other.dele
        self.ref += other.ref


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    return TAG_RE.sub("", text).lower().strip()


def edit(ref: List[str], hyp: List[str]) -> EditStat:
    n, m = len(ref), len(hyp)
    dp = [[(0, EditStat()) for _ in range(m + 1)] for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = (i, EditStat(dele=i, ref=i))
    for j in range(1, m + 1):
        dp[0][j] = (j, EditStat(ins=j))
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost, prev = dp[i - 1][j - 1]
            sub = int(ref[i - 1] != hyp[j - 1])
            choices = [(cost + sub, EditStat(prev.sub + sub, prev.ins, prev.dele, i))]

            cost, prev = dp[i - 1][j]
            choices.append((cost + 1, EditStat(prev.sub, prev.ins, prev.dele + 1, i)))

            cost, prev = dp[i][j - 1]
            choices.append((cost + 1, EditStat(prev.sub, prev.ins + 1, prev.dele, i)))
            dp[i][j] = min(choices, key=lambda item: item[0])
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
