import json
import os
import tempfile

from finetuning.grpo_data import load_samples, split_eval


def _write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_load_samples_parses_fields():
    with tempfile.NamedTemporaryFile(
        "w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        _write(
            f.name,
            [
                {
                    "audio": "/a.wav",
                    "text": "language Chinese<asr_text>洪金宝演的",
                    "prompt": "转写语音。\n专属名词：[洪金宝，伊佐美纪]",
                }
            ],
        )
        path = f.name
    samples = load_samples(path)
    os.unlink(path)
    assert len(samples) == 1
    s = samples[0]
    assert s.audio == "/a.wav"
    assert s.gt_text == "洪金宝演的"  # normalize 后
    assert s.hotwords == ["洪金宝", "伊佐美纪"]
    assert "专属名词" in s.prompt


def test_split_eval_disjoint_and_ratio():
    rows = [
        {
            "audio": f"/{i}.wav",
            "text": f"language Chinese<asr_text>x{i}",
            "prompt": "专属名词：[a，b]",
        }
        for i in range(200)
    ]
    with tempfile.NamedTemporaryFile(
        "w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        _write(f.name, rows)
        path = f.name
    samples = load_samples(path)
    os.unlink(path)
    tr, ev = split_eval(samples, eval_ratio=0.1, seed=42)
    assert len(ev) == 20
    assert len(tr) == 180
    assert set(id(x) for x in tr).isdisjoint(set(id(x) for x in ev))
