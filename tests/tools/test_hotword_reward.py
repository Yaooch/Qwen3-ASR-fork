from qwen_asr.tools.hotword_reward import (
    normalize,
    parse_text_field,
    parse_hotword_list,
    split_truth,
    hotword_recall,
    false_injection_rate,
    non_hotword_cer,
    compute_reward,
)


def test_normalize_strips_punct_and_lowercases():
    assert normalize("Hello, 世界！ WORLD.") == "hello 世界 world"


def test_parse_text_field_strips_prefix():
    raw = "language Chinese<asr_text>你们看高志森那部电影了吗"
    assert parse_text_field(raw) == "你们看高志森那部电影了吗"


def test_parse_hotword_list():
    prompt = "转写语音，专属名词优先按列表原文输出。\n专属名词：[高志森，小鬼三个爸，洪金宝]"
    assert parse_hotword_list(prompt) == ["高志森", "小鬼三个爸", "洪金宝"]


def test_split_truth_separates_spoken_vs_distractor():
    hw = ["高志森", "小鬼三个爸", "洪金宝", "伊佐美纪"]  # 伊佐美纪 没说到
    gt = "你们看高志森那部小鬼三个爸了吗 洪金宝演的"
    T, D = split_truth(hw, gt)
    assert T == {"高志森", "小鬼三个爸", "洪金宝"}
    assert D == {"伊佐美纪"}


def test_recall_and_fp():
    out = "你们看高志森那部小鬼三个爸了吗"
    T = {"高志森", "小鬼三个爸", "洪金宝"}
    D = {"伊佐美纪"}
    assert hotword_recall(out, T) == 2 / 3
    assert false_injection_rate(out, D) == 0.0
    out2 = "你们看高志森那部伊佐美纪了吗"
    assert false_injection_rate(out2, D) == 1.0


def test_non_hotword_cer_isolated():
    out = "你们看高志森那部电影了吗"
    gt = "你们看高志森那部电影了吗"
    hw = ["高志森", "伊佐美纪"]
    assert non_hotword_cer(out, gt, hw) == 0.0
    out2 = "你们看高志森那部电形了吗"
    assert non_hotword_cer(out2, gt, hw) > 0.0


def test_compute_reward_shape():
    out = "你们看高志森那部小鬼三个爸了吗 洪金宝演的"
    gt = "你们看高志森那部小鬼三个爸了吗 洪金宝演的"
    hw = ["高志森", "小鬼三个爸", "洪金宝", "伊佐美纪"]
    r = compute_reward(out, gt, hw)
    assert r > 0.9  # 召回满、无误注入、CER 0


def test_compute_reward_empty_truth_no_injection():
    out = "今天天气不错"
    gt = "今天天气不错"
    hw = ["伊佐美纪"]  # 无真热词，且未注入
    r = compute_reward(out, gt, hw)
    assert r >= 0.0  # recall 项置 0，CER 0
