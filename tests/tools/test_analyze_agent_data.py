from qwen_asr.tools.analyze_agent_data import (
    extract_candidates,
    extract_question,
    extract_schema,
    parse_agent,
    semantic_key,
)


def test_parse_agent_accepts_bare_action_and_ignores_param_order():
    parsed, reason = parse_agent("phone_reject")
    assert reason is None
    assert parsed == {"action": "phone_reject", "params": {}, "format": "bare-action"}
    assert semantic_key("tool&&a=&1@b=&2") == semantic_key("tool&&b=&2@a=&1")


def test_parse_agent_rejects_nonstandard_parameterized_outputs():
    assert parse_agent("action：auto_q")[1] == "缺少 &&"
    assert parse_agent("tool@param=&value")[1] == "缺少 &&"
    assert parse_agent("tool&&param=value")[1] == "参数片段缺少 =&"


def test_extract_candidates_supports_both_prompt_templates():
    explicit = "Action should be one of [tool_a, tool_b]"
    definitions = "tool_a: Call this tool now\ntool_b: Call this tool now"
    assert extract_candidates(explicit) == ["tool_a", "tool_b"]
    assert extract_candidates(definitions) == ["tool_a", "tool_b"]


def test_extract_schema_supports_both_schema_titles():
    old = 'tool_a: Call this tool. Parameters: [{"name":"a"}]\n\nUse it'
    new = 'tool_a: Call this tool. Parameters of tool_a: [{"name":"b"}]\n\nUse it'
    assert extract_schema(old, "tool_a") == {"a"}
    assert extract_schema(new, "tool_a") == {"b"}


def test_extract_question_supports_markdown_template():
    user = "Example Question: wrong\n\n- 用户问题\n真正的问题"
    assert extract_question(user) == "真正的问题"
