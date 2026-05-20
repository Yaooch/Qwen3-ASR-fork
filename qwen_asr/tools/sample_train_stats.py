#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List


FILES = {
    "chuan": "chuan/sichuan_train_data.jsonl",
    "yue": "yue/yue_train_data.jsonl",
    "mandarin": "mandarin/mandarin_train_data_pure.jsonl",
    "mix": "mix/mix_train_data_final.jsonl",
    "english_pure": "english/pure_english_fixed.jsonl",
    "english_mixed": "english/chinese_mixed.jsonl",
}
LETTER_WORDS = ["HUD", "AUTO", "WLAN", "QQ", "AI", "AR", "USB", "FM", "AM", "APP", "GPS", "CD", "2D", "3D"]
CAR_WORDS = ["导航", "播放", "音乐", "座椅", "空调", "车窗", "天窗", "后视镜", "氛围灯", "蓝牙", "HUD", "AUTO", "WLAN", "QQ", "AI", "AR"]
YUE_CHARS = set("嘅咗唔佢哋咁噉嗰呢冇啲喺嚟睇攞畀晒系啱噶啦")
POI_WORDS = ["导航到", "导航去", "我要去", "我想去"]
MEDIA_WORDS = ["我要听", "我想听", "播放", "放一首", "来一首"]


def parse_args():
    p = argparse.ArgumentParser("训练集抽样统计")
    p.add_argument("--base_dir", default="/cfs/data/private/WangYaoChi/train_data/all")
    p.add_argument("--samples", type=int, default=50000)
    p.add_argument("--seed", type=int, default=20260515)
    return p.parse_args()


def content(text: str) -> str:
    return text.split("<asr_text>", 1)[1] if "<asr_text>" in text else text


def lang(text: str) -> str:
    m = re.search(r"language ([^<]+)<asr_text>", text or "")
    return m.group(1) if m else ""


def sample_lines(path: Path, n: int, rng: random.Random) -> List[str]:
    size = path.stat().st_size
    out, seen = [], set()
    with path.open("rb") as f:
        tries = 0
        while len(out) < n and tries < n * 10:
            tries += 1
            pos = rng.randrange(max(size, 1))
            if pos in seen:
                continue
            seen.add(pos)
            f.seek(pos)
            if pos:
                f.readline()
            line = f.readline()
            if not line:
                continue
            try:
                out.append(line.decode("utf-8"))
            except UnicodeDecodeError:
                continue
    return out


def has_ascii(text: str) -> bool:
    return bool(re.search(r"[A-Za-z0-9]", text))


def stat(name: str, path: Path, n: int, rng: random.Random) -> Dict:
    rows = sample_lines(path, n, rng)
    s = {
        "name": name,
        "path": str(path),
        "sampled": 0,
        "bad_json": 0,
        "empty": 0,
        "bad_prefix": 0,
        "ascii": 0,
        "upper": 0,
        "lower": 0,
        "zh_ascii": 0,
        "car": 0,
        "poi": 0,
        "media": 0,
        "yue_chars": 0,
        "dup_text": 0,
        "langs": {},
        "letters": {w: 0 for w in LETTER_WORDS},
        "lens": [],
        "ascii_examples": [],
    }
    seen = set()
    for line in rows:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            s["bad_json"] += 1
            continue
        text = obj.get("text", "")
        c = content(text).strip()
        l = lang(text)
        s["sampled"] += 1
        s["langs"][l] = s["langs"].get(l, 0) + 1
        s["empty"] += int(not c)
        s["bad_prefix"] += int(text.count("<asr_text>") != 1 or not text.startswith("language "))
        s["lens"].append(len(c))
        s["ascii"] += int(has_ascii(c))
        s["upper"] += int(bool(re.search(r"[A-Z]", c)))
        s["lower"] += int(bool(re.search(r"[a-z]", c)))
        s["zh_ascii"] += int(has_ascii(c) and bool(re.search(r"[\u4e00-\u9fff]", c)))
        s["car"] += int(any(w.lower() in c.lower() for w in CAR_WORDS))
        s["poi"] += int(any(w in c for w in POI_WORDS))
        s["media"] += int(any(w in c for w in MEDIA_WORDS))
        s["yue_chars"] += int(any(ch in c for ch in YUE_CHARS))
        s["dup_text"] += int(c in seen)
        seen.add(c)
        for w in LETTER_WORDS:
            s["letters"][w] += int(bool(re.search(re.escape(w), c, re.I)))
        if has_ascii(c) and len(s["ascii_examples"]) < 5:
            s["ascii_examples"].append(c)
    return s


def pct(num: int, den: int) -> str:
    return f"{num / den * 100:.2f}%" if den else "0.00%"


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    base = Path(args.base_dir)
    for name, rel in FILES.items():
        path = base / rel
        if not path.exists():
            print(f"{name}\t缺失：{path}")
            continue
        s = stat(name, path, args.samples, rng)
        n = s["sampled"]
        lens = sorted(s["lens"])
        p50 = lens[len(lens) // 2] if lens else 0
        p95 = lens[int(len(lens) * 0.95)] if lens else 0
        letters = {k: v for k, v in s["letters"].items() if v}
        print(f"\n### {name}")
        print(f"path: {path}")
        print(f"sampled: {n} bad_json: {s['bad_json']} langs: {s['langs']}")
        print(f"empty: {s['empty']} ({pct(s['empty'], n)}) bad_prefix: {s['bad_prefix']} ({pct(s['bad_prefix'], n)}) dup_text: {s['dup_text']} ({pct(s['dup_text'], n)})")
        print(f"len_p50: {p50} len_p95: {p95}")
        print(f"ascii: {s['ascii']} ({pct(s['ascii'], n)}) upper: {s['upper']} ({pct(s['upper'], n)}) lower: {s['lower']} ({pct(s['lower'], n)}) zh_ascii: {s['zh_ascii']} ({pct(s['zh_ascii'], n)})")
        print(f"car: {s['car']} ({pct(s['car'], n)}) poi: {s['poi']} ({pct(s['poi'], n)}) media: {s['media']} ({pct(s['media'], n)}) yue_chars: {s['yue_chars']} ({pct(s['yue_chars'], n)})")
        print(f"letters: {letters}")
        print("ascii_examples:")
        for item in s["ascii_examples"]:
            print(f"  {item}")


if __name__ == "__main__":
    main()
