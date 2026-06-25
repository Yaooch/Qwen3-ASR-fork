#!/usr/bin/env python3
# coding: utf-8
"""热词 label 拼音扰动：在固定音频的前提下，把热词改成相近音的另一个字，
用来模拟“说话人发音不标准、与要召回的词拼音略有偏差”的场景。

背景与 caveat 见 docs/notes；简述：
- 检索器是拼音/音素级模糊匹配（NEAR_INITIALS/NEAR_FINALS）。
- 音频不变，把热词 label 翻成相近音字，制造“音频拼音 ↔ 热词拼音”偏差。
- 方向与真实口音相反（音频正、热词偏），对检索器对称，对 LLM 解码器是
  “bias 压过音频”的代理指标，非真实口音的完美复刻。

只翻转 target 热词（出现在 utt_hotword 里的），每个至多 1 个字；
hotword.txt / utt_hotword.txt / text 三处同步替换，wav.scp 原样复制。
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from pypinyin import Style, lazy_pinyin

# 复用检索器内置混淆表，保证测的就是系统声称能容忍的偏差
from qwen_asr.joint.hotword import NEAR_INITIALS, NEAR_FINALS


def build_index(chars: str) -> Dict[Tuple[str, str], List[Tuple[str, str]]]:
    """char 集合 -> {(initial, final): [(full_pinyin, char), ...]} 索引。
    用 (initial, final) 做键，避免零声母音节（万/王/余 等）下
    “ini+fin 重构全拼”与 NORMAL 拼写（wan/wang/yu）对不上的问题。"""
    idx: Dict[Tuple[str, str], List[Tuple[str, str]]] = defaultdict(list)
    for ch in set(chars):
        if not ch.strip():
            continue
        full, ini, fin = char_pinyin(ch)
        if not full:
            continue
        idx[(ini, fin)].append((full, ch))
    for k in idx:
        idx[k].sort()
    return dict(idx)


def char_pinyin(ch: str) -> Tuple[str, str, str]:
    """单字 -> (full, initial, final)，toneless。"""
    full = lazy_pinyin(ch, style=Style.NORMAL)
    ini = lazy_pinyin(ch, style=Style.INITIALS)
    fin = lazy_pinyin(ch, style=Style.FINALS)
    full = full[0] if full else ""
    ini = ini[0] if ini else ""
    fin = fin[0] if fin else ""
    return full, ini, fin


def candidate_replacements(
    ch: str,
    idx: Dict[Tuple[str, str], List[Tuple[str, str]]],
) -> List[Tuple[str, str, str, str]]:
    """对单字枚举 (confusion_type, old_full, new_full, new_char) 候选。
    候选字的全拼直接取自索引，不做 ini+fin 重构。已去重并稳定排序。"""
    full, ini, fin = char_pinyin(ch)
    seen = set()
    out: List[Tuple[str, str, str, str]] = []

    def add(ctype: str, key: Tuple[str, str]) -> None:
        for nfull, cand in idx.get(key, []):
            if cand != ch and nfull not in seen:
                seen.add(nfull)
                out.append((ctype, full, nfull, cand))

    # 翻声母：同韵母，声母换成近音
    if ini and ini in NEAR_INITIALS:
        for ni in sorted(NEAR_INITIALS[ini]):
            add(f"initial:{ini}->{ni}", (ni, fin))
    # 翻韵母：同声母，韵母换成近音
    if fin and fin in NEAR_FINALS:
        for nf in sorted(NEAR_FINALS[fin]):
            add(f"final:{fin}->{nf}", (ini, nf))

    out.sort()
    return out


def flip_hotword(
    word: str,
    idx: Dict[Tuple[str, str], List[Tuple[str, str]]],
    existing: set,
) -> Optional[Tuple[str, int, str, str, str, str, str]]:
    """尝试翻转 word 中 1 个字。返回 (new_word, pos, old_ch, new_ch, ctype, old_py, new_py) 或 None。"""
    for i, ch in enumerate(word):
        for ctype, old_py, new_py, new_ch in candidate_replacements(ch, idx):
            new_word = word[:i] + new_ch + word[i + 1:]
            if new_word == word:
                continue
            if new_word in existing:
                continue  # 与已有热词撞名，跳过
            return new_word, i, ch, new_ch, ctype, old_py, new_py
    return None


def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def write_lines(path: str, lines: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def split_words(text: str) -> List[str]:
    return [x.strip() for x in text.replace("，", ",").split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser("热词 label 拼音扰动")
    ap.add_argument("--baseurl", required=True, help="原始测试集目录")
    ap.add_argument("--output_dir", required=True, help="扰动后输出目录")
    args = ap.parse_args()

    src = args.baseurl
    dst = args.output_dir
    os.makedirs(dst, exist_ok=True)

    hotwords = [w for w in read_lines(os.path.join(src, "hotword.txt")) if w.strip()]
    utt_lines = read_lines(os.path.join(src, "utt_hotword.txt"))
    text_lines = read_lines(os.path.join(src, "text"))

    existing = set(hotwords)

    # 收集 target 热词
    targets: set = set()
    for line in utt_lines:
        parts = line.split("\t", 1)
        if len(parts) == 2:
            for w in split_words(parts[1]):
                targets.add(w)

    # 字池：用数据集自身所有字做反查，替换字都是真实姓名用字
    all_chars = "".join(hotwords) + "".join(utt_lines) + "".join(text_lines)
    idx = build_index(all_chars)

    # 计算每个 target 的扰动映射
    mapping: Dict[str, Tuple[str, int, str, str, str, str, str]] = {}
    log_rows: List[str] = []
    skipped_collision = 0
    skipped_no_candidate = 0
    for w in sorted(targets):
        if w not in existing:
            continue  # target 不在候选池，跳过
        res = flip_hotword(w, idx, existing)
        if res is None:
            skipped_no_candidate += 1
            continue
        new_w, pos, old_ch, new_ch, ctype, old_py, new_py = res
        # 变体不能撞到其它已扰动变体
        used = {mapping[k][0] for k in mapping}
        if new_w in used or new_w in existing:
            skipped_collision += 1
            continue
        mapping[w] = res
        log_rows.append(
            f"{w}\t{new_w}\t{pos}\t{old_ch}\t{new_ch}\t{ctype}\t{old_py}\t{new_py}"
        )

    # 写 hotword.txt：target 翻变体，非 target 原样
    new_hotwords = []
    for w in hotwords:
        if w in mapping:
            new_hotwords.append(mapping[w][0])
        else:
            new_hotwords.append(w)
    write_lines(os.path.join(dst, "hotword.txt"), new_hotwords)

    # 写 utt_hotword.txt
    new_utt = []
    for line in utt_lines:
        parts = line.split("\t", 1)
        if len(parts) != 2:
            new_utt.append(line)
            continue
        utt_id, words_field = parts
        words = split_words(words_field)
        new_words = [mapping[w][0] if w in mapping else w for w in words]
        new_utt.append(f"{utt_id}\t{','.join(new_words)}")
    write_lines(os.path.join(dst, "utt_hotword.txt"), new_utt)

    # 写 text：按本 utt 的 target 替换子串
    new_text = []
    for line in text_lines:
        parts = line.split("\t", 1)
        if len(parts) != 2:
            new_text.append(line)
            continue
        utt_id, transcript = parts
        # 用该 utt 的 target 做精确替换，避免跨热词级联
        for w in words_targets_for(utt_lines, utt_id):
            if w in mapping:
                transcript = transcript.replace(w, mapping[w][0])
        new_text.append(f"{utt_id}\t{transcript}")
    write_lines(os.path.join(dst, "text"), new_text)

    # 复制 wav.scp
    shutil.copyfile(os.path.join(src, "wav.scp"), os.path.join(dst, "wav.scp"))

    # mapping 日志
    log_path = os.path.join(dst, "perturb_mapping.tsv")
    header = "old_hotword\tnew_hotword\tchar_idx\told_char\tnew_char\tconfusion\told_pinyin\tnew_pinyin"
    write_lines(log_path, [header] + log_rows)

    n_targets = len([w for w in targets if w in existing])
    print(f"输出目录: {dst}")
    print(f"target 热词总数: {n_targets}")
    print(f"成功扰动: {len(mapping)}")
    print(f"跳过(无可用替换字): {skipped_no_candidate}")
    print(f"跳过(变体撞名): {skipped_collision}")
    print(f"未扰动(对照): {n_targets - len(mapping)}")
    print(f"映射日志: {log_path}")
    return 0


def words_targets_for(utt_lines: List[str], utt_id: str) -> List[str]:
    for line in utt_lines:
        parts = line.split("\t", 1)
        if len(parts) == 2 and parts[0] == utt_id:
            return split_words(parts[1])
    return []


if __name__ == "__main__":
    sys.exit(main())
