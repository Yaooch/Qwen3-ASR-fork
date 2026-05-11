#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 ContextASR-Speech 转成 finetuning/train.py 可读的 jsonl。"""

import argparse
import json
import os
import random
from typing import Dict, List, Sequence, Tuple


DEFAULT_ROOT = "/cfs/data/private/WangYaoChi/open_datasets/ContextASR"
DEFAULT_FILES = {
    "Mandarin": "ContextASR-Speech_Mandarin.jsonl",
    "English": "ContextASR-Speech_English.jsonl",
}
OUTPUT_LANGUAGE = {
    "Mandarin": "Chinese",
    "English": "English",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ContextASR jsonl to Qwen3-ASR training jsonl.")
    parser.add_argument("--root", default=DEFAULT_ROOT, help="ContextASR 数据根目录。")
    parser.add_argument(
        "--languages",
        default="Mandarin",
        help="逗号分隔语言：Mandarin,English,all。默认只转中文。",
    )
    parser.add_argument("--output_train", required=True, help="训练 jsonl 输出路径。")
    parser.add_argument("--output_eval", default="", help="可选验证 jsonl 输出路径。")
    parser.add_argument("--output_test", default="", help="可选测试 jsonl 输出路径。")
    parser.add_argument("--eval_ratio", type=float, default=0.02, help="验证集比例，仅设置 output_eval 时生效。")
    parser.add_argument("--eval_per_language", type=int, default=0, help="每个语言固定抽多少条验证集，优先于 eval_ratio。")
    parser.add_argument("--test_per_language", type=int, default=0, help="每个语言固定抽多少条测试集。")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_duration", type=float, default=80.0, help="大于该秒数的样本跳过；<=0 表示不过滤。")
    parser.add_argument("--min_entities", type=int, default=1, help="实体数少于该值的样本跳过。")
    parser.add_argument("--max_entities", type=int, default=8, help="每条最多注入多少真实实体；<=0 表示不限制。")
    parser.add_argument("--random_negative_min", type=int, default=0, help="每条最少随机注入多少个非本句实体。")
    parser.add_argument("--random_negative_max", type=int, default=0, help="每条最多随机注入多少个非本句实体。")
    parser.add_argument(
        "--negative_scope",
        choices=["language", "global"],
        default="language",
        help="随机干扰热词来源：同语言实体池或全局实体池。",
    )
    parser.add_argument("--shuffle_entities", action="store_true", help="打乱注入热词顺序。")
    parser.add_argument(
        "--prompt_template",
        default=(
            "转写语音，专属名词优先按列表原文输出。\n"
            "专属名词：[{hotwords}]"
        ),
        help="Prompt 模板，使用 {hotwords} 放置实体列表。",
    )
    parser.add_argument("--keep_meta", action="store_true", help="保留 uniq_id/language/domain/entity_list/duration 等调试字段。")
    return parser.parse_args()


def langs(text: str) -> List[str]:
    items = [x.strip() for x in text.split(",") if x.strip()]
    if any(x.lower() == "all" for x in items):
        return list(DEFAULT_FILES)
    bad = [x for x in items if x not in DEFAULT_FILES]
    if bad:
        raise ValueError(f"不支持的语言：{bad}，可选：{','.join(DEFAULT_FILES)}")
    return items


def uniq(items: Sequence[str]) -> List[str]:
    out = []
    seen = set()
    for item in items:
        word = str(item).strip()
        if not word or word in seen:
            continue
        seen.add(word)
        out.append(word)
    return out


def abs_audio(root: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(root, path))


def sample_negatives(
    text: str,
    positives: Sequence[str],
    pool: Sequence[str],
    args: argparse.Namespace,
    rng: random.Random,
) -> List[str]:
    max_n = max(0, args.random_negative_max)
    min_n = max(0, args.random_negative_min)
    if max_n <= 0:
        return []
    if min_n > max_n:
        min_n, max_n = max_n, min_n

    positive_set = set(positives)
    if not pool:
        return []
    n = rng.randint(min_n, max_n)
    negatives = []
    seen = set()
    max_try = min(max(len(pool), n * 80), 800)
    for _ in range(max_try):
        word = rng.choice(pool)
        if word in seen or word in positive_set or word in text:
            continue
        seen.add(word)
        negatives.append(word)
        if len(negatives) >= n:
            break
    return negatives


def row(
    obj: Dict,
    root: str,
    args: argparse.Namespace,
    rng: random.Random,
    negative_pool: Sequence[str],
) -> Dict:
    text = str(obj.get("text") or "")
    language = str(obj.get("language") or "")
    output_language = OUTPUT_LANGUAGE.get(language, language or "None")
    entities = uniq(obj.get("entity_list") or [])
    if args.max_entities > 0:
        entities = entities[: args.max_entities]
    negatives = sample_negatives(text, entities, negative_pool, args, rng)

    hotword_entities = entities + negatives
    if args.shuffle_entities:
        rng.shuffle(hotword_entities)

    hotwords = "，".join(hotword_entities)
    out = {
        "audio": abs_audio(root, str(obj.get("audio") or "")),
        "text": f"language {output_language}<asr_text>{text}",
        "prompt": args.prompt_template.format(hotwords=hotwords),
    }
    if args.keep_meta:
        out.update(
            {
                "uniq_id": obj.get("uniq_id"),
                "language": obj.get("language"),
                "domain_label": obj.get("domain_label"),
                "entity_list": hotword_entities,
                "positive_entity_list": entities,
                "negative_entity_list": negatives,
                "duration": obj.get("duration"),
            }
        )
    return out


def load(root: str, language: str, args: argparse.Namespace) -> List[Dict]:
    path = os.path.join(root, DEFAULT_FILES[language])
    objs = []
    skipped_duration = 0
    skipped_entity = 0
    skipped_empty = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = str(obj.get("text") or "").strip()
            audio = str(obj.get("audio") or "").strip()
            entities = uniq(obj.get("entity_list") or [])
            duration = float(obj.get("duration") or 0.0)
            if not text or not audio:
                skipped_empty += 1
                continue
            if args.max_duration > 0 and duration > args.max_duration:
                skipped_duration += 1
                continue
            if len(entities) < args.min_entities:
                skipped_entity += 1
                continue
            objs.append(obj)

    print(
        f"{language}: 保留 {len(objs)} 条，"
        f"跳过过长 {skipped_duration} 条，跳过少实体 {skipped_entity} 条，跳过空字段 {skipped_empty} 条"
    )
    return objs


def entity_pool(objs: Sequence[Dict]) -> List[str]:
    pool = []
    for obj in objs:
        pool.extend(obj.get("entity_list") or [])
    return uniq(pool)


def split_language(
    objs: List[Dict],
    args: argparse.Namespace,
    rng: random.Random,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    items = list(objs)
    rng.shuffle(items)
    test_n = min(max(0, args.test_per_language), len(items)) if args.output_test else 0
    test_rows = items[:test_n]
    remain = items[test_n:]

    if args.output_eval:
        if args.eval_per_language > 0:
            eval_n = min(args.eval_per_language, len(remain))
        else:
            eval_n = int(round(len(remain) * args.eval_ratio))
            eval_n = max(1, eval_n) if remain and args.eval_ratio > 0 else 0
    else:
        eval_n = 0

    eval_rows = remain[:eval_n]
    train_rows = remain[eval_n:]
    return train_rows, eval_rows, test_rows


def write(path: str, rows: Sequence[Dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"写出：{path}，{len(rows)} 条")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    selected_langs = langs(args.languages)
    objs_by_lang = {language: load(args.root, language, args) for language in selected_langs}
    pools_by_lang = {language: entity_pool(objs) for language, objs in objs_by_lang.items()}
    global_pool = uniq(word for pool in pools_by_lang.values() for word in pool)
    print(f"全局实体池：{len(global_pool)} 个")
    for language, pool in pools_by_lang.items():
        print(f"{language} 实体池：{len(pool)} 个")

    train_objs: List[Dict] = []
    eval_objs: List[Dict] = []
    test_objs: List[Dict] = []
    for language in selected_langs:
        cur_train, cur_eval, cur_test = split_language(objs_by_lang[language], args, rng)
        print(f"{language}: train={len(cur_train)}, eval={len(cur_eval)}, test={len(cur_test)}")
        train_objs.extend(cur_train)
        eval_objs.extend(cur_eval)
        test_objs.extend(cur_test)

    def pool_for(obj: Dict) -> Sequence[str]:
        if args.negative_scope == "global":
            return global_pool
        return pools_by_lang.get(str(obj.get("language") or ""), global_pool)

    train_rows = [row(obj, args.root, args, rng, pool_for(obj)) for obj in train_objs]
    eval_rows = [row(obj, args.root, args, rng, pool_for(obj)) for obj in eval_objs]
    test_rows = [row(obj, args.root, args, rng, pool_for(obj)) for obj in test_objs]
    rng.shuffle(train_rows)
    rng.shuffle(eval_rows)
    rng.shuffle(test_rows)

    write(args.output_train, train_rows)
    if args.output_eval:
        write(args.output_eval, eval_rows)
    if args.output_test:
        write(args.output_test, test_rows)


if __name__ == "__main__":
    main()
