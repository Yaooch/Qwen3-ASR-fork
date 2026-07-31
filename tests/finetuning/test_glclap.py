import json
import random

import torch

from finetuning.train_glclap import iter_jsonl_shard, parse_text, sample_subtext
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
