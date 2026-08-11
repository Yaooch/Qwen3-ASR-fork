#!/usr/bin/env python3
import argparse
import os
from typing import List

if __package__:
    from .edit_eval import edit, norm, percent, read_texts
else:
    from edit_eval import edit, norm, percent, read_texts

def parse_args():
    p = argparse.ArgumentParser("文本 badcase 生成")
    p.add_argument("--ref_path", "--ref_dir", required=True)
    p.add_argument("--result_path", required=True)
    p.add_argument("--badcase_path", required=True)
    p.add_argument("--topk_badcases", type=int, default=100)
    return p.parse_args()

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
