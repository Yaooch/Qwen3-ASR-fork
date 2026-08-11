from qwen_asr_ext.hotword import HotwordRetriever


def test_retrieve_fuzzy_chinese_and_english():
    retriever = HotwordRetriever(["撒贝宁", "康辉", "周涛", "Golden Valley"])

    assert retriever.retrieve("撒贝你主持的节目", topk=3) == ["撒贝宁"]
    assert retriever.retrieve("golden vally pharmacy", topk=3) == ["Golden Valley"]


def test_retrieve_empty_input():
    retriever = HotwordRetriever(["撒贝宁"])

    assert retriever.retrieve("") == []


def test_from_file(tmp_path):
    path = tmp_path / "hotword.txt"
    path.write_text("撒贝宁\n\nGolden Valley\n", encoding="utf-8")

    assert HotwordRetriever.from_file(path).hotwords == ["撒贝宁", "Golden Valley"]


def test_retrieve_english_by_pronunciation():
    retriever = HotwordRetriever(["ED SHEERAN", "GLAY", "CHER"])

    assert "ED SHEERAN" in retriever.retrieve("play edd sherron s last album", topk=3)


def test_english_phone_candidate_requires_confidence():
    retriever = HotwordRetriever(["ED SHEERAN"])

    assert retriever.retrieve("play some music", topk=3) == []
