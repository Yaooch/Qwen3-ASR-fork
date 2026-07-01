# qwen_asr/tools/nlu.py
"""NLU(用户意图提取)工具:prompt 构造、意图解析、评测指标。

NLU 是纯文本任务(user 语句 -> assistant 意图 JSON),不走 ASR 的 audio chat_template
(其 user 槽只渲染 audio placeholder,会丢弃 user 文本)。这里用标准 Qwen chat format
独立构造 prompt,system prompt 做任务路由,与 ASR 路径互不影响。

三能力(ASR / NLU / ASR+NLU)通过 system prompt 区分:
  ASR      : "转写语音"            输入音频 -> text
  NLU      : "提取用户意图"         输入文本 -> 意图 JSON
  ASR+NLU  : "转写语音并提取用户意图" 输入音频 -> 文本\n意图
本期只实现 NLU(纯文本),但 prompt 构造以 system prompt 驱动,为后期统一留接口。
"""
import json
import re
from typing import Dict, List, Optional

NLU_SYSTEM_PROMPT = "提取用户意图"

# 标准 Qwen chat format:system + user 文本 + assistant。与 ASR 的 audio chat_template 并存。
NLU_CHAT_TEMPLATE = (
    "{%- for message in messages -%}"
    "{%- if message['role'] == 'system' -%}"
    "{{- '<|im_start|>system\\n' + message['content'] + '<|im_end|>\\n' -}}"
    "{%- elif message['role'] == 'user' -%}"
    "{{- '<|im_start|>user\\n' + message['content'] + '<|im_end|>\\n' -}}"
    "{%- elif message['role'] == 'assistant' -%}"
    "{{- '<|im_start|>assistant\\n' + message['content'] + '<|im_end|>\\n' -}}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}"
    "{{- '<|im_start|>assistant\\n' -}}"
    "{%- endif -%}"
)


def nlu_messages(system: str, user: str, assistant: Optional[str] = None) -> List[Dict[str, str]]:
    """组装 NLU 对话消息列表。system 为空时用默认意图提取 prompt。"""
    msgs = [
        {"role": "system", "content": system or NLU_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    if assistant is not None:
        msgs.append({"role": "assistant", "content": assistant})
    return msgs


def build_nlu_prompt(processor, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
    """用 NLU 专用 chat_template 渲染 prompt 字符串(不碰 ASR 的 audio chat_template)。"""
    return processor.apply_chat_template(
        messages,
        chat_template=NLU_CHAT_TEMPLATE,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
    )


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def parse_intent(text: str) -> Optional[Dict]:
    """解析 assistant 输出为意图 dict({"name": ..., "arguments": ...})。失败返回 None。"""
    if not text:
        return None
    cleaned = _FENCE_RE.sub("", text).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or "name" not in obj:
        return None
    return obj


def intent_metrics(preds: List[Optional[Dict]], refs: List[Dict]) -> Dict[str, float]:
    """返回 json_valid_rate / name_acc / args_exact_rate。

    - json_valid_rate: 模型输出能解析为合法意图 JSON 的比例
    - name_acc:        意图 name 命中率
    - args_exact_rate: name + arguments 全等的精确匹配率
    """
    n = len(refs)
    if n == 0:
        return {"json_valid_rate": 0.0, "name_acc": 0.0, "args_exact_rate": 0.0}
    valid = sum(1 for p in preds if p is not None)
    name_ok = sum(1 for p, r in zip(preds, refs) if p and p.get("name") == r.get("name"))
    exact = sum(
        1
        for p, r in zip(preds, refs)
        if p and p.get("name") == r.get("name") and p.get("arguments") == r.get("arguments")
    )
    return {
        "json_valid_rate": valid / n,
        "name_acc": name_ok / n,
        "args_exact_rate": exact / n,
    }


__all__ = [
    "NLU_SYSTEM_PROMPT",
    "NLU_CHAT_TEMPLATE",
    "nlu_messages",
    "build_nlu_prompt",
    "parse_intent",
    "intent_metrics",
]
