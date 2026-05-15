import argparse
import json
import os
import random
import re
from pathlib import Path


LANGS = {
    "Mandarin": {
        "config": "ContextASR-Dialogue_Mandarin.jsonl",
        "train": "/cfs/data/private/WangYaoChi/train_data/all/train_contextasr_dialogue_cut_mandarin.jsonl",
        "tag": "ZH",
    },
    "English": {
        "config": "ContextASR-Dialogue_English.jsonl",
        "train": "/cfs/data/private/WangYaoChi/train_data/all/train_contextasr_dialogue_cut_english.jsonl",
        "tag": "EN",
    },
}


def parse_args():
    p = argparse.ArgumentParser("抽取 ContextASR Dialogue 热词测试集和验证集")
    p.add_argument("--base_dir", default="/cfs/data/private/WangYaoChi/open_datasets/ContextASR")
    p.add_argument("--cut_audio_root", default=None)
    p.add_argument("--test_dir", default="/cfs/data/private/WangYaoChi/open_datasets/ContextASR/hotword_test_dialogue")
    p.add_argument("--val_mandarin", default="/cfs/data/private/WangYaoChi/train_data/all/val_contextasr_dialogue_cut_mandarin.jsonl")
    p.add_argument("--val_english", default="/cfs/data/private/WangYaoChi/train_data/all/val_contextasr_dialogue_cut_english.jsonl")
    p.add_argument("--test_per_lang", type=int, default=1000)
    p.add_argument("--val_per_lang", type=int, default=500)
    p.add_argument("--seed", type=int, default=20260514)
    return p.parse_args()


def norm_id(x):
    s = re.sub(r"[^0-9A-Za-z._-]+", "_", str(x)).strip("_")
    return s or "sample"


def uniq(xs):
    seen = set()
    out = []
    for x in xs:
        if x is None:
            continue
        x = str(x).strip()
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def rows(path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def hit_terms(text, terms):
    low = text.lower()
    hits = []
    for term in terms:
        term = str(term).strip()
        if term and term.lower() in low:
            hits.append(term)
    return uniq(hits)


def dialogue_items(lang, cfg, audio_dir):
    tag = LANGS[lang]["tag"]
    items = []
    for row in rows(cfg):
        uid = norm_id(row.get("uniq_id") or "")
        terms = row.get("entity_list") or []
        for idx, turn in enumerate(row.get("dialogue") or []):
            text = (turn.get("text") or "").strip()
            if not text:
                continue
            hits = hit_terms(text, terms)
            if not hits:
                continue
            wav = audio_dir / f"{uid}_{idx:03d}.wav"
            if not wav.exists():
                continue
            items.append({
                "utt": f"{uid}_{idx:03d}_{tag}",
                "audio": str(wav),
                "text": text,
                "hotwords": hits,
            })
    return items


def write_test(test_dir, selected):
    test_dir.mkdir(parents=True, exist_ok=True)
    hotwords = []
    with (test_dir / "wav.scp").open("w", encoding="utf-8") as wav_f, \
            (test_dir / "text").open("w", encoding="utf-8") as text_f, \
            (test_dir / "utt_hotword.txt").open("w", encoding="utf-8") as utt_f:
        for item in selected:
            wav_f.write(f"{item['utt']}\t{item['audio']}\n")
            text_f.write(f"{item['utt']}\t{item['text']}\n")
            utt_f.write(f"{item['utt']}\t{','.join(item['hotwords'])}\n")
            hotwords.extend(item["hotwords"])

    with (test_dir / "hotword.txt").open("w", encoding="utf-8") as hot_f:
        for word in uniq(hotwords):
            hot_f.write(word + "\n")


def write_val(train_path, out_path, n, exclude_audio, rng):
    rows_all = list(rows(train_path))
    candidates = [x for x in rows_all if x.get("audio") not in exclude_audio]
    if len(candidates) < n:
        raise ValueError(f"{train_path} 剩余样本不足：需要 {n}，实际 {len(candidates)}")
    selected = rng.sample(candidates, n)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for item in selected:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    os.replace(tmp, out_path)
    return selected


def main():
    args = parse_args()
    base = Path(args.base_dir)
    cut_audio_root = Path(args.cut_audio_root) if args.cut_audio_root else base / "audio" / "ContextASR-Dialogue-cut"
    rng = random.Random(args.seed)

    selected_test = []
    test_audio = set()
    by_lang = {}
    for lang, meta in LANGS.items():
        items = dialogue_items(lang, base / meta["config"], cut_audio_root / lang)
        if len(items) < args.test_per_lang:
            raise ValueError(f"{lang} 有真实热词的切片不足：需要 {args.test_per_lang}，实际 {len(items)}")
        picks = rng.sample(items, args.test_per_lang)
        selected_test.extend(picks)
        test_audio.update(x["audio"] for x in picks)
        by_lang[lang] = len(picks)
        print(f"{lang} 可选热词测试切片 {len(items)} 条，抽取 {len(picks)} 条")

    rng.shuffle(selected_test)
    write_test(Path(args.test_dir), selected_test)

    val_outs = {
        "Mandarin": Path(args.val_mandarin),
        "English": Path(args.val_english),
    }
    for lang, meta in LANGS.items():
        selected = write_val(
            train_path=Path(meta["train"]),
            out_path=val_outs[lang],
            n=args.val_per_lang,
            exclude_audio=test_audio,
            rng=rng,
        )
        print(f"{lang} 验证集抽取 {len(selected)} 条：{val_outs[lang]}")

    print(f"测试集目录：{args.test_dir}")
    print(f"测试集总数：{len(selected_test)}，其中 {by_lang}")


if __name__ == "__main__":
    main()
