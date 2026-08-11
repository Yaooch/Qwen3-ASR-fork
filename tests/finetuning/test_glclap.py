import json
import random
from collections import Counter

import torch

from finetuning.retrieval_eval import normalize_text, read_candidates, read_key_value
from finetuning.train_glclap import (
    iter_jsonl_shard,
    load_word_df,
    parse_text,
    sample_subtext,
)
from qwen_asr.joint.glclap import feature_mask, glclap_loss


def test_parse_text_keeps_ground_truth():
    language, text = parse_text("language Chinese<asr_text>导航去丹尼斯大卫城")
    assert language == "Chinese"
    assert text == "导航去丹尼斯大卫城"
    assert parse_text("plain text") == ("", "plain text")


def test_sample_subtext_is_contiguous_original_slice():
    text = "请播放Taylor Swift的Cruel Summer"
    subtext = sample_subtext(text, max_units=5, rng=random.Random(3))
    assert subtext in text
    assert subtext != text


def test_sample_subtext_uses_language_specific_lengths():
    rng = random.Random(7)
    english = Counter(
        len(sample_subtext("one two three four five six", "English", rng=rng).split())
        for _ in range(10000)
    )
    chinese = Counter(
        len(sample_subtext("一二三四五六七八九十", "Chinese", rng=rng))
        for _ in range(10000)
    )

    assert set(english) == {1, 2, 3, 4}
    for length, expected in {1: 0.4, 2: 0.3, 3: 0.2, 4: 0.1}.items():
        assert abs(english[length] / sum(english.values()) - expected) < 0.02
    assert set(chinese) == set(range(2, 9))
    assert chinese[2] > chinese[4] and chinese[3] > chinese[8]


def test_han_dialects_use_chinese_lengths():
    for language in ("Cantonese", "Sichuanese", "None"):
        lengths = {
            len(sample_subtext("一二三四五六七八九十", language, rng=random.Random(seed)))
            for seed in range(200)
        }
        assert 1 not in lengths
        assert lengths == set(range(2, 9))


def test_one_word_sampling_prefers_low_frequency_content():
    rng = random.Random(11)
    selected = Counter(
        sample_subtext(
            "the quasar",
            "English",
            max_units=1,
            word_df={"the": 10000, "quasar": 5},
            median_df=5,
            rng=rng,
        )
        for _ in range(2000)
    )
    assert selected["quasar"] / sum(selected.values()) > 0.85


def test_load_word_df_uses_valid_content_median(tmp_path):
    path = tmp_path / "word_df.json"
    path.write_text(json.dumps({
        "document_frequency": {
            "the": 1000,
            "quasar": 5,
            "nebula": 9,
            "typo": 2,
        }
    }), encoding="utf-8")
    frequencies, median = load_word_df(str(path))
    assert frequencies["the"] == 1000
    assert median == 7.0


def test_iter_jsonl_shard_reads_each_row_once(tmp_path):
    path = tmp_path / "data.jsonl"
    rows = [{"id": i} for i in range(20)]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    found = []
    for part in range(3):
        found.extend(row["id"] for row in iter_jsonl_shard(str(path), part, 3))
    assert sorted(found) == list(range(20))


def test_feature_mask_matches_conv_lengths():
    attention = torch.tensor([
        [1] * 20 + [0] * 10,
        [1] * 30,
    ])
    mask = feature_mask(attention, 2, [10, 3], [5, 2])
    assert mask.tolist() == [[True, False], [True, True]]


def test_glclap_loss_prefers_aligned_pairs_and_has_gradients():
    text_global = torch.eye(3, requires_grad=True)
    text_local = torch.eye(3, requires_grad=True)
    audio_global = torch.eye(3, requires_grad=True)
    audio_local = torch.stack([
        torch.stack([torch.eye(3)[i], torch.zeros(3)])
        for i in range(3)
    ]).requires_grad_()
    audio_mask = torch.tensor([[True, False]] * 3)
    scale = torch.tensor(0.0, requires_grad=True)

    aligned = glclap_loss(
        text_global,
        text_local,
        audio_global,
        audio_local,
        audio_mask,
        scale,
    )
    shuffled = glclap_loss(
        text_global,
        text_local,
        audio_global.roll(1, 0),
        audio_local.roll(1, 0),
        audio_mask,
        scale,
    )
    assert aligned["loss"] < shuffled["loss"]
    aligned["loss"].backward()
    assert text_global.grad is not None
    assert audio_local.grad is not None


def test_glclap_local_loss_ignores_padding_frames():
    text = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    audio_global = text.clone()
    audio_local = torch.tensor([
        [[1.0, 0.0], [0.0, 1.0]],
        [[0.0, 1.0], [1.0, 0.0]],
    ])
    mask = torch.tensor([[True, False], [True, False]])
    out = glclap_loss(text, text, audio_global, audio_local, mask, torch.tensor(0.0))
    assert out["local_logits"].argmax(dim=1).tolist() == [0, 1]


def test_stop_files_are_normalized_and_deduplicated(tmp_path):
    mapping = tmp_path / "utt_hotword.txt"
    mapping.write_text("utt1  New   York State\nutt2\tAustin\n", encoding="utf-8")
    candidates = tmp_path / "hotword.txt"
    candidates.write_text(" new york state \nAUSTIN\nAustin\n", encoding="utf-8")

    assert read_key_value(str(mapping)) == {
        "utt1": "New   York State",
        "utt2": "Austin",
    }
    assert read_candidates(str(candidates)) == ["NEW YORK STATE", "AUSTIN"]
    assert normalize_text("  New   York State ") == "NEW YORK STATE"
