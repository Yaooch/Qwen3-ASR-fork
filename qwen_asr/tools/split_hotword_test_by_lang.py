#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


LANGS = {
    "zh": {"tag": "ZH", "name": "中文"},
    "en": {"tag": "EN", "name": "英文"},
}


def parse_args():
    p = argparse.ArgumentParser("按语言拆分热词测试集")
    p.add_argument("--input_dir", default="/cfs/data/private/WangYaoChi/open_datasets/ContextASR/hotword_test")
    p.add_argument("--zh_dir", default="")
    p.add_argument("--en_dir", default="")
    return p.parse_args()


def rows(path: Path) -> List[Tuple[str, str]]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            out.append((parts[0], parts[1] if len(parts) > 1 else ""))
    return out


def lang_of(utt_id: str, value: str = "") -> str:
    if re.search(r"(^|_)ZH($|[._-])", utt_id):
        return "zh"
    if re.search(r"(^|_)EN($|[._-])", utt_id):
        return "en"
    low = value.lower()
    if "/mandarin/" in low or "\\mandarin\\" in low:
        return "zh"
    if "/english/" in low or "\\english\\" in low:
        return "en"
    raise ValueError(f"无法判断语言：{utt_id}")


def split_words(text: str) -> List[str]:
    return [x.strip() for x in re.split(r"[,，]", text or "") if x.strip()]


def uniq(xs: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def out_dirs(args) -> Dict[str, Path]:
    src = Path(args.input_dir)
    return {
        "zh": Path(args.zh_dir) if args.zh_dir else src / "Mandarin",
        "en": Path(args.en_dir) if args.en_dir else src / "English",
    }


def check_ids(name: str, data: List[Tuple[str, str]], known: set):
    unknown = [utt_id for utt_id, _ in data if utt_id not in known]
    if unknown:
        raise ValueError(f"{name} 存在 {len(unknown)} 个 wav.scp 中没有的 utt_id，例如：{unknown[0]}")


def write_split(out_dir: Path, lang: str, wavs, texts, utt_words):
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = [utt_id for utt_id, value in wavs if lang_of(utt_id, value) == lang]
    id_set = set(ids)

    hotwords = []
    with (out_dir / "wav.scp").open("w", encoding="utf-8") as wav_f, \
            (out_dir / "text").open("w", encoding="utf-8") as text_f, \
            (out_dir / "utt_hotword.txt").open("w", encoding="utf-8") as utt_f:
        for utt_id, value in wavs:
            if utt_id in id_set:
                wav_f.write(f"{utt_id}\t{value}\n")
        for utt_id, value in texts:
            if utt_id in id_set:
                text_f.write(f"{utt_id}\t{value}\n")
        for utt_id, value in utt_words:
            if utt_id in id_set:
                utt_f.write(f"{utt_id}\t{value}\n")
                hotwords.extend(split_words(value))

    with (out_dir / "hotword.txt").open("w", encoding="utf-8") as hot_f:
        for word in uniq(hotwords):
            hot_f.write(word + "\n")

    return len(ids), len(uniq(hotwords))


def main():
    args = parse_args()
    src = Path(args.input_dir)
    wavs = rows(src / "wav.scp")
    texts = rows(src / "text")
    utt_words = rows(src / "utt_hotword.txt")
    known = {utt_id for utt_id, _ in wavs}
    check_ids("text", texts, known)
    check_ids("utt_hotword.txt", utt_words, known)

    dirs = out_dirs(args)
    for lang, meta in LANGS.items():
        n, hot_n = write_split(dirs[lang], lang, wavs, texts, utt_words)
        print(f"{meta['name']}：{n} 条，热词 {hot_n} 个，输出 {dirs[lang]}")


if __name__ == "__main__":
    main()
