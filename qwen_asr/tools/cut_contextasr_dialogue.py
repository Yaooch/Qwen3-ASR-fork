import argparse
import json
import os
import random
import re
import wave
from pathlib import Path


PROMPT_HEAD = "转写语音，专属名词优先按列表原文输出。\n专属名词："
LANG_PREFIX = {
    "Mandarin": "Chinese",
    "English": "English",
}


def parse_args():
    p = argparse.ArgumentParser("切分 ContextASR Dialogue 并生成训练 jsonl")
    p.add_argument("--base_dir", default="/cfs/data/private/WangYaoChi/open_datasets/ContextASR")
    p.add_argument("--out_audio_root", default=None)
    p.add_argument("--out_mandarin", default="/cfs/data/private/WangYaoChi/train_data/all/train_contextasr_dialogue_cut_mandarin.jsonl")
    p.add_argument("--out_english", default="/cfs/data/private/WangYaoChi/train_data/all/train_contextasr_dialogue_cut_english.jsonl")
    p.add_argument("--noise_scope", choices=["language", "all"], default="language")
    p.add_argument("--seed", type=int, default=20260514)
    p.add_argument("--overwrite_audio", action="store_true")
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
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield i, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{i} JSON 解析失败：{exc}") from exc


def configs(base):
    return {
        "Mandarin": base / "ContextASR-Dialogue_Mandarin.jsonl",
        "English": base / "ContextASR-Dialogue_English.jsonl",
    }


def hotword_pools(cfgs):
    pools = {lang: [] for lang in cfgs}
    all_terms = []
    for lang, path in cfgs.items():
        for _, row in rows(path):
            terms = row.get("entity_list") or []
            pools[lang].extend(terms)
            all_terms.extend(terms)
    return {lang: uniq(terms) for lang, terms in pools.items()}, uniq(all_terms)


def cut_wav(src, dst_items, overwrite):
    with wave.open(str(src), "rb") as r:
        params = r.getparams()
        sr = r.getframerate()
        total = r.getnframes()
        data = r.readframes(total)

    frame_bytes = params.sampwidth * params.nchannels
    made = []
    for idx, turn, dst in dst_items:
        start = float(turn.get("start") or 0.0)
        end = float(turn.get("end") or 0.0)
        s = max(0, min(total, int(round(start * sr))))
        e = max(0, min(total, int(round(end * sr))))
        if e <= s:
            made.append((idx, False, "空片段"))
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        if overwrite or not (dst.exists() and dst.stat().st_size > 44):
            with wave.open(str(dst), "wb") as w:
                w.setnchannels(params.nchannels)
                w.setsampwidth(params.sampwidth)
                w.setframerate(sr)
                w.writeframes(data[s * frame_bytes:e * frame_bytes])
        made.append((idx, True, str(dst)))
    return made


def prompt_terms(base_terms, pool, rng):
    base_terms = uniq(base_terms)
    base_set = set(base_terms)
    candidates = [x for x in pool if x not in base_set]
    n_noise = rng.randint(1, 5)
    noise = rng.sample(candidates, min(n_noise, len(candidates))) if candidates else []
    terms = uniq(base_terms + noise)
    rng.shuffle(terms)
    return terms


def run_lang(lang, cfg, out_audio_dir, out_jsonl, pool, base_dir, rng, overwrite):
    tmp = out_jsonl.with_suffix(out_jsonl.suffix + ".tmp")
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_audio_dir.mkdir(parents=True, exist_ok=True)

    n_rows = 0
    n_seg = 0
    n_skip = 0
    with tmp.open("w", encoding="utf-8") as f:
        for _, row in rows(cfg):
            n_rows += 1
            uid = norm_id(row.get("uniq_id") or f"{n_rows:08d}")
            rel_audio = row.get("audio") or ""
            src = Path(rel_audio) if str(rel_audio).startswith("/") else base_dir / rel_audio
            dialogue = row.get("dialogue") or []
            indexed = [(i, d) for i, d in enumerate(dialogue) if (d.get("text") or "").strip()]

            if not src.exists():
                n_skip += len(indexed)
                print(f"{lang} 缺少音频，跳过：{src}")
                continue

            dst_items = [
                (i, d, out_audio_dir / f"{uid}_{i:03d}.wav")
                for i, d in indexed
            ]
            try:
                made = cut_wav(src, dst_items, overwrite)
            except Exception as exc:
                n_skip += len(dst_items)
                print(f"{lang} 切分失败，跳过：{src}，{exc}")
                continue

            base_terms = row.get("entity_list") or []
            for idx, ok, audio_path in made:
                if not ok:
                    n_skip += 1
                    continue
                text = (dialogue[idx].get("text") or "").strip()
                terms = prompt_terms(base_terms, pool, rng)
                item = {
                    "audio": audio_path,
                    "text": f"language {LANG_PREFIX.get(lang, lang)}<asr_text>{text}",
                    "prompt": PROMPT_HEAD + "[" + "，".join(terms) + "]",
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                n_seg += 1

            if n_rows % 500 == 0:
                print(f"{lang} 已处理 {n_rows} 条原始数据，生成 {n_seg} 条切片")

    os.replace(tmp, out_jsonl)
    print(f"{lang} 完成：原始 {n_rows} 条，切片 {n_seg} 条，跳过 {n_skip} 条")
    print(f"{lang} jsonl：{out_jsonl}")
    print(f"{lang} 音频目录：{out_audio_dir}")
    return n_rows, n_seg, n_skip


def main():
    args = parse_args()
    base = Path(args.base_dir)
    out_audio_root = Path(args.out_audio_root) if args.out_audio_root else base / "audio" / "ContextASR-Dialogue-cut"
    cfgs = configs(base)
    lang_pools, all_pool = hotword_pools(cfgs)
    rng = random.Random(args.seed)

    outs = {
        "Mandarin": Path(args.out_mandarin),
        "English": Path(args.out_english),
    }
    totals = {}
    for lang, cfg in cfgs.items():
        pool = all_pool if args.noise_scope == "all" else lang_pools[lang]
        totals[lang] = run_lang(
            lang=lang,
            cfg=cfg,
            out_audio_dir=out_audio_root / lang,
            out_jsonl=outs[lang],
            pool=pool,
            base_dir=base,
            rng=rng,
            overwrite=args.overwrite_audio,
        )

    print("全部完成")
    for lang, (n_rows, n_seg, n_skip) in totals.items():
        print(f"{lang}: 原始 {n_rows} 条，切片 {n_seg} 条，跳过 {n_skip} 条")


if __name__ == "__main__":
    main()
