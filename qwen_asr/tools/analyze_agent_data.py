#!/usr/bin/env python3
"""全量检查 Agent JSONL 数据并生成可筛选的静态 HTML 报告。

只根据 messages、候选 API 和参数 Schema 做确定性检查，不运行模型 Decode，
也不读取测试集。
"""

import argparse
import hashlib
import html as html_lib
import json
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple


BARE_ACTION_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
QUESTION_RE = re.compile(r"(?:^|\n)Question:\s*(.*)")
CANDIDATE_RE = re.compile(r"should be one of \[([^\]]*)\]", re.I)
API_RE = re.compile(r"(?m)^([A-Za-z_][A-Za-z0-9_]*): Call this tool")
SCHEMA_RE = re.compile(r"Parameters(?:\s+of\s+[^:\n]+)?:\s*")
ISSUE_NAMES = {
    "duplicate": "完全重复",
    "conflict": "标签冲突",
    "action": "Action 不在候选",
    "param": "参数不在 Schema",
    "format": "格式错误",
    "structure": "数据结构错误",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Agent JSONL 数据质检")
    parser.add_argument("--input", required=True, help="待检查的 JSONL 文件")
    parser.add_argument("--output", required=True, help="输出 HTML 路径")
    parser.add_argument("--sample-limit", type=int, default=100, help="普通类别最多展示多少项")
    parser.add_argument("--seed", type=int, default=42, help="代表样本抽样种子")
    return parser.parse_args()


def extract_messages(obj: Dict) -> Tuple[str, str, str, bool]:
    messages = obj.get("messages") if isinstance(obj, dict) else None
    if not isinstance(messages, list):
        return "", "", "", False
    contents = {}
    valid = True
    for role in ("system", "user", "assistant"):
        values = [
            item.get("content")
            for item in messages
            if isinstance(item, dict) and item.get("role") == role
        ]
        if not values or not isinstance(values[-1], str):
            valid = False
            contents[role] = ""
        else:
            contents[role] = values[-1]
    return contents["system"], contents["user"], contents["assistant"], valid


def extract_question(user: str) -> str:
    markdown = re.search(r"-\s*用户问题\s*\n\s*(.+?)\s*$", user, re.S)
    if markdown:
        return markdown.group(1).strip()
    matches = QUESTION_RE.findall(user)
    return matches[-1].strip() if matches else user[-200:].strip()


def extract_candidates(user: str) -> Optional[List[str]]:
    matches = CANDIDATE_RE.findall(user)
    if matches:
        return [
            item.strip().strip("'\"")
            for item in matches[-1].split(",")
            if item.strip()
        ]
    apis = API_RE.findall(user)
    return list(dict.fromkeys(apis)) if apis else None


def parse_agent(text: str) -> Tuple[Optional[Dict], Optional[str]]:
    """解析标准 Action 输出；无参数裸 Action 合法。"""
    cleaned = (text or "").strip()
    if not cleaned:
        return None, "空答案"
    if BARE_ACTION_RE.fullmatch(cleaned):
        return {"action": cleaned, "params": {}, "format": "bare-action"}, None
    if "&&" not in cleaned:
        return None, "缺少 &&"
    action, action_input = cleaned.split("&&", 1)
    action = action.strip()
    if not BARE_ACTION_RE.fullmatch(action):
        return None, "Action 格式异常"
    params = {}
    action_input = action_input.strip()
    if not action_input or action_input.lower() == "none":
        return {"action": action, "params": params, "format": "action&&"}, None
    for part in action_input.split("@"):
        if "=&" not in part:
            return None, "参数片段缺少 =&"
        name, value = part.split("=&", 1)
        name = name.strip()
        if not BARE_ACTION_RE.fullmatch(name):
            return None, "参数名格式异常"
        if name in params:
            return None, "重复参数名"
        params[name] = value.strip()
    return {"action": action, "params": params, "format": "action&&"}, None


def extract_schema(user: str, action: str) -> Optional[set]:
    """提取 Action 参数名，兼容 Parameters 与 Parameters of xxx。"""
    start = user.find("\n\n" + action + ": Call this tool")
    if start < 0:
        start = user.find(action + ": Call this tool")
    if start < 0:
        return None
    next_api = API_RE.search(user, start + len(action) + 1)
    end = next_api.start() if next_api else len(user)
    match = SCHEMA_RE.search(user, start, end)
    if not match:
        return None
    array_start = user.find("[", match.end(), end)
    if array_start < 0:
        return None
    try:
        schema, _ = json.JSONDecoder().raw_decode(user[array_start:end])
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(schema, list):
        return None
    return {
        item["name"]
        for item in schema
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }


def semantic_key(answer: str) -> str:
    """生成参数顺序无关的标签键；解析失败时退回原始字符串。"""
    parsed, _ = parse_agent(answer)
    if parsed is None:
        return "RAW\0" + (answer or "").strip()
    params = json.dumps(parsed["params"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "PARSED\0" + parsed["action"] + "\0" + params


def add_sample(samples, seen, category, line_no, limit, rng):
    seen[category] += 1
    rows = samples[category]
    if len(rows) < limit:
        rows.append(line_no)
        return
    index = rng.randrange(seen[category])
    if index < limit:
        rows[index] = line_no


def scan(path: Path, sample_limit: int, seed: int):
    stats = Counter()
    for name in (
        "json_invalid",
        "message_structure_invalid",
        "candidate_unparsed",
        "schema_unparsed",
        "action_not_candidate",
        "parameter_not_schema",
        "invalid_format",
        "format_action&&",
        "format_bare-action",
    ):
        stats[name] = 0
    rng = random.Random(seed)
    samples = {name: [] for name in ("action", "param", "format", "structure")}
    seen = Counter()
    groups = {}
    started = time.time()

    with path.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            stats["total"] += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                stats["json_invalid"] += 1
                add_sample(samples, seen, "structure", line_no, sample_limit, rng)
                continue
            _, user, answer, valid = extract_messages(obj)
            if not valid:
                stats["message_structure_invalid"] += 1
                add_sample(samples, seen, "structure", line_no, sample_limit, rng)
                continue

            digest = hashlib.sha256(user.encode("utf-8")).hexdigest()
            label_key = semantic_key(answer)
            group = groups.get(digest)
            if group is None:
                groups[digest] = {
                    "count": 1,
                    "first_line": line_no,
                    "variants": {label_key: {"answer": answer, "lines": [line_no]}},
                }
            else:
                group["count"] += 1
                variant = group["variants"].get(label_key)
                if variant is None:
                    group["variants"][label_key] = {"answer": answer, "lines": [line_no]}
                else:
                    variant["lines"].append(line_no)

            candidates = extract_candidates(user)
            if candidates is None:
                stats["candidate_unparsed"] += 1
            parsed, _ = parse_agent(answer)
            if parsed is None:
                stats["invalid_format"] += 1
                add_sample(samples, seen, "format", line_no, sample_limit, rng)
            else:
                stats["format_" + parsed["format"]] += 1
                action = parsed["action"]
                if candidates is not None and action not in candidates:
                    stats["action_not_candidate"] += 1
                    add_sample(samples, seen, "action", line_no, sample_limit, rng)
                if parsed["params"] and candidates is not None and action in candidates:
                    schema = extract_schema(user, action)
                    if schema is None:
                        stats["schema_unparsed"] += 1
                    elif set(parsed["params"]) - schema:
                        stats["parameter_not_schema"] += 1
                        add_sample(samples, seen, "param", line_no, sample_limit, rng)
            if line_no % 25000 == 0:
                print(f"扫描 {line_no:,} 条，耗时 {time.time() - started:.1f}s", flush=True)

    duplicate_groups = [group for group in groups.values() if group["count"] > 1]
    conflict_groups = [group for group in groups.values() if len(group["variants"]) > 1]
    stats["unique_inputs"] = len(groups)
    stats["duplicate_groups"] = len(duplicate_groups)
    stats["duplicate_rows_in_groups"] = sum(group["count"] for group in duplicate_groups)
    stats["duplicate_extra"] = sum(group["count"] - 1 for group in duplicate_groups)
    stats["max_repeat"] = max((group["count"] for group in groups.values()), default=0)
    stats["conflict_groups"] = len(conflict_groups)
    stats["conflict_rows"] = sum(group["count"] for group in conflict_groups)
    duplicate_groups.sort(key=lambda group: (-group["count"], group["first_line"]))
    conflict_groups.sort(key=lambda group: (-group["count"], group["first_line"]))
    return stats, samples, duplicate_groups[:sample_limit], conflict_groups


def load_records(path: Path, selected: set) -> Dict[int, Dict]:
    records = {}
    with path.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if line_no not in selected:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                records[line_no] = {
                    "line": line_no,
                    "question": "JSON 无法解析",
                    "user": line.rstrip(),
                    "answer": "",
                    "candidates": None,
                    "structure_reason": "JSON 无法解析",
                }
                continue
            system, user, answer, valid = extract_messages(obj)
            parsed, format_reason = parse_agent(answer)
            record = {
                "line": line_no,
                "question": extract_question(user),
                "user": user,
                "system": system,
                "answer": answer,
                "candidates": extract_candidates(user),
                "format_reason": format_reason,
            }
            if not valid:
                record["structure_reason"] = "缺少 system/user/assistant 或 content 不是字符串"
            if parsed:
                action = parsed["action"]
                record.update(action=action, params=parsed["params"], answer_format=parsed["format"])
                candidates = record["candidates"]
                if candidates is not None and action in candidates:
                    schema = extract_schema(user, action)
                    if schema is not None:
                        record["schema_params"] = sorted(schema)
                        record["extra_params"] = sorted(set(parsed["params"]) - schema)
            records[line_no] = record
    return records


def build_issues(path, samples, duplicate_groups, conflict_groups):
    selected = set().union(*samples.values())
    selected.update(group["first_line"] for group in duplicate_groups)
    for group in conflict_groups:
        for variant in group["variants"].values():
            selected.update(variant["lines"])
    records = load_records(path, selected)
    issues = []
    for group in duplicate_groups:
        card = dict(records[group["first_line"]])
        card.update(category="duplicate", count=group["count"])
        issues.append(card)
    for group in conflict_groups:
        variants = []
        for variant in sorted(group["variants"].values(), key=lambda item: item["lines"][0]):
            for line_no in variant["lines"]:
                row = records[line_no]
                variants.append({"line": line_no, "answer": row["answer"], "user": row["user"]})
        first = records[group["first_line"]]
        issues.append({
            "category": "conflict",
            "line": group["first_line"],
            "question": first["question"],
            "user": first["user"],
            "candidates": first["candidates"],
            "conflict_sample_count": group["count"],
            "variant_count": len(group["variants"]),
            "variants": variants,
        })
    for category in ("action", "param", "format", "structure"):
        for line_no in samples[category]:
            card = dict(records[line_no])
            card["category"] = category
            if category == "action":
                card["action_issue"] = f"{card.get('action')} 不在候选列表"
            elif category == "param":
                card["missing_schema_params"] = card.get("extra_params", [])
            issues.append(card)
    return issues


CSS = r"""
:root{--bg:#f5f7fb;--panel:#fff;--text:#18212f;--muted:#667085;--line:#dfe4ec;--blue:#315efb}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif}header{padding:28px max(24px,calc((100% - 1480px)/2));background:#18212f;color:white}h1{margin:0 0 6px;font-size:25px}.sub{color:#c9d2df;overflow-wrap:anywhere}main{max-width:1480px;margin:auto;padding:22px}.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}.stat{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}.stat strong{display:block;font-size:21px}.stat span{color:var(--muted)}.note{margin:14px 0;padding:12px 15px;background:#eef3ff;border-left:4px solid var(--blue);border-radius:6px}.controls{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0 10px;position:sticky;top:0;background:var(--bg);padding:10px 0;z-index:2}button,.search{border:1px solid var(--line);background:white;border-radius:8px;padding:8px 12px}button{cursor:pointer}button.active{background:var(--blue);border-color:var(--blue);color:white}.search{flex:1;min-width:260px}.meta{color:var(--muted);margin-bottom:10px}.card{background:white;border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:10px 0;box-shadow:0 2px 8px #18212f0a}.card h3{font-size:16px;margin:0 0 10px}.tag{display:inline-block;margin-right:8px;padding:2px 8px;border-radius:99px;background:#eef2ff;color:#2949b6;font-size:12px}.row{display:grid;grid-template-columns:125px minmax(0,1fr);gap:10px;padding:5px 0;border-top:1px dashed #edf0f4}.label{color:var(--muted)}.code,details .code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;white-space:pre-wrap;overflow-wrap:anywhere;background:#f7f8fa;border-radius:6px;padding:7px}details{margin-top:8px}summary{cursor:pointer;color:var(--blue)}.variant{margin:10px 0;padding:10px;border-left:3px solid #f0a020;background:#fffbf2;border-radius:4px}.empty{text-align:center;padding:40px;color:var(--muted)}@media(max-width:650px){main{padding:12px}.row{grid-template-columns:1fr}.controls{position:static}}
"""

SCRIPT = r"""
const names=__NAMES__;let cat='all';const val=v=>v==null?'—':Array.isArray(v)?v.join(', '):typeof v==='object'?JSON.stringify(v,null,2):String(v);function add(p,c,t,g='div'){const e=document.createElement(g);if(c)e.className=c;e.textContent=val(t);p.appendChild(e);return e}function row(c,l,v,code=false){const r=add(c,'row','');add(r,'label',l);add(r,code?'code':'',v)}function fold(c,l,v){const d=add(c,'','','details');add(d,'',l,'summary');add(d,'code',v)}function showStats(){const s=DATA.stats,n=s.total||1,a=[['训练总量',s.total.toLocaleString()],['唯一完整 Prompt',s.unique_inputs.toLocaleString()],['重复样本',`${s.duplicate_rows_in_groups.toLocaleString()}条 / ${s.duplicate_groups.toLocaleString()}组`],['额外重复行',s.duplicate_extra.toLocaleString()],['标签冲突',`${s.conflict_rows.toLocaleString()}条 / ${s.conflict_groups.toLocaleString()}组`],['Action 不在候选',`${s.action_not_candidate.toLocaleString()}条 (${(100*s.action_not_candidate/n).toFixed(2)}%)`],['参数不在 Schema',`${s.parameter_not_schema.toLocaleString()}条 (${(100*s.parameter_not_schema/n).toFixed(2)}%)`],['格式错误',`${s.invalid_format.toLocaleString()}条 (${(100*s.invalid_format/n).toFixed(2)}%)`],['合法裸 Action',s['format_bare-action'].toLocaleString()],['候选列表未解析',s.candidate_unparsed.toLocaleString()],['数据结构错误',(s.json_invalid+s.message_structure_invalid).toLocaleString()],['最大重复次数',s.max_repeat.toLocaleString()]],b=document.getElementById('stats');a.forEach(([k,v])=>{const c=add(b,'stat','');add(c,'',v,'strong');add(c,'',k,'span')})}function card(x){const c=document.createElement('article');c.className='card';const h=add(c,'','', 'h3');add(h,'tag',names[x.category]||x.category,'span');h.appendChild(document.createTextNode(x.question||'（无问题文本）'));if(x.line)row(c,'训练行号',x.line);if(x.count)row(c,'重复次数',x.count);if(x.conflict_sample_count)row(c,'冲突组样本数',x.conflict_sample_count);if(x.variant_count)row(c,'不同语义标签数',x.variant_count);if(x.candidates!==undefined)row(c,'候选工具',x.candidates||'未提取到');if(x.action!==undefined)row(c,'标注 Action',x.action);if(x.params!==undefined)row(c,'标注参数',x.params,true);if(x.action_issue)row(c,'问题',x.action_issue);if(x.missing_schema_params)row(c,'Schema 外字段',x.missing_schema_params);if(x.format_reason)row(c,'格式问题',x.format_reason);if(x.structure_reason)row(c,'结构问题',x.structure_reason);if(x.answer!==undefined)row(c,'标注答案',x.answer,true);if(x.variants)x.variants.forEach((v,i)=>{const box=add(c,'variant','');row(box,`冲突样本 ${i+1} 行号`,v.line);row(box,'标注答案',v.answer,true);fold(box,'展开该行 User Prompt',v.user)});if(x.user!==undefined)fold(c,'展开原始 User Prompt',x.user);if(x.system)fold(c,'展开 System Prompt',x.system);return c}function render(){const q=document.getElementById('search').value.trim().toLowerCase(),a=DATA.issues.filter(x=>(cat==='all'||x.category===cat)&&(!q||JSON.stringify(x).toLowerCase().includes(q))),l=document.getElementById('list');l.replaceChildren();document.getElementById('meta').textContent=`显示 ${a.length} 个代表项（普通类别最多 ${DATA.sample_limit} 个；冲突组全部展示）`;if(!a.length){add(l,'empty','没有匹配样本');return}a.forEach(x=>l.appendChild(card(x)))}document.querySelectorAll('button[data-cat]').forEach(b=>b.onclick=()=>{document.querySelectorAll('button[data-cat]').forEach(x=>x.classList.remove('active'));b.classList.add('active');cat=b.dataset.cat;render()});document.getElementById('search').oninput=render;showStats();render();
"""


def render_html(source: Path, output: Path, stats: Counter, issues: List[Dict], sample_limit: int):
    payload = {
        "source": str(source),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "sample_limit": sample_limit,
        "stats": dict(stats),
        "issues": issues,
    }
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    data_json = data_json.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    names_json = json.dumps(ISSUE_NAMES, ensure_ascii=False)
    structure_count = stats["json_invalid"] + stats["message_structure_invalid"]
    buttons = [
        ("all", "全部"),
        ("duplicate", f"完全重复（{stats['duplicate_rows_in_groups']:,}条/{stats['duplicate_groups']:,}组）"),
        ("conflict", f"标签冲突（{stats['conflict_rows']:,}条/{stats['conflict_groups']:,}组）"),
        ("action", f"Action 不在候选（{stats['action_not_candidate']:,}条）"),
        ("param", f"参数不在 Schema（{stats['parameter_not_schema']:,}条）"),
        ("format", f"格式错误（{stats['invalid_format']:,}条）"),
        ("structure", f"结构错误（{structure_count:,}条）"),
    ]
    button_html = "".join(
        f'<button class="{"active" if key == "all" else ""}" data-cat="{key}">{label}</button>'
        for key, label in buttons
    )
    generated_at = payload["generated_at"]
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Voyah Agent 训练数据质检报告</title><style>{CSS}</style></head><body><header><h1>Voyah Agent 训练数据质检报告</h1><div class="sub">源文件：{html_lib.escape(str(source))}</div><div class="sub">生成时间：{generated_at}；未运行模型 Decode，也未与测试集比对</div></header><main><section class="stats" id="stats"></section><div class="note">口径：<b>phone_reject</b> 这类无参数裸 Action 视为合法；参数顺序不影响标签冲突判断；同一完整 User Prompt 对应不同 Action/参数才算标签冲突。各普通类别最多展示 {sample_limit} 个代表项，标签冲突组全部展示。类别之间可能重叠。</div><div class="controls">{button_html}<input id="search" class="search" placeholder="搜索问题、Action、参数或行号…"></div><div class="meta" id="meta"></div><section id="list"></section></main><script>const DATA={data_json};{SCRIPT.replace('__NAMES__', names_json)}</script></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")


def main():
    args = parse_args()
    source = Path(args.input)
    output = Path(args.output)
    if args.sample_limit <= 0:
        raise ValueError("--sample-limit 必须大于 0")
    if not source.is_file():
        raise FileNotFoundError(source)
    stats, samples, duplicate_groups, conflict_groups = scan(source, args.sample_limit, args.seed)
    issues = build_issues(source, samples, duplicate_groups, conflict_groups)
    render_html(source, output, stats, issues, args.sample_limit)
    print(json.dumps(dict(stats), ensure_ascii=False, indent=2))
    print(f"报告：{output}")


if __name__ == "__main__":
    main()
