import json
import os
import tempfile

import torch

from qwen_asr_ext.grpo.grpo import (
    group_advantages,
    grpo_loss,
    load_samples,
)

# --------------------------------------------------------------------------
# grpo_data: load_samples
# --------------------------------------------------------------------------

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
    assert s.gt_text == "洪金宝演的"
    assert s.hotwords == ["洪金宝", "伊佐美纪"]
    assert "专属名词" in s.prompt

# --------------------------------------------------------------------------
# grpo_math: group_advantages / grpo_loss
# --------------------------------------------------------------------------

def test_group_advantages_zero_mean_per_group():
    r = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    a = group_advantages(r)
    assert a.shape == (1, 4)
    assert abs(float(a.sum())) < 1e-6

def test_grpo_loss_finite_and_sign():
    logp = torch.zeros(2, 4, requires_grad=True)
    old = torch.zeros(2, 4)
    adv = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    ref = torch.zeros(2, 4)
    loss = grpo_loss(logp, old, adv, ref, beta=0.04)
    assert torch.isfinite(loss)
    # 正优势 + logp==old → ratio=1 → surrogate=adv，loss=-mean(adv) < 0
    assert float(loss.detach()) < 0
    loss.backward()
    assert logp.grad is not None

def test_grpo_kl_penalty_is_zero_at_ref_and_positive_when_changed():
    adv = torch.zeros(2)
    same = torch.zeros(2, requires_grad=True)
    moved = torch.tensor([1.0, -1.0], requires_grad=True)
    ref = torch.zeros(2)
    assert float(grpo_loss(same, same.detach(), adv, ref, beta=1.0).detach()) == 0.0
    assert float(grpo_loss(moved, moved.detach(), adv, ref, beta=1.0).detach()) > 0.0


def test_grpo_clip_stops_positive_advantage_above_upper_bound():
    logp = torch.tensor([0.3], requires_grad=True)
    loss = grpo_loss(logp, torch.zeros(1), torch.ones(1), logp.detach(), beta=0.0)
    loss.backward()
    assert float(logp.grad) == 0.0


def test_grpo_clip_stops_negative_advantage_below_lower_bound():
    logp = torch.tensor([-0.3], requires_grad=True)
    loss = grpo_loss(logp, torch.zeros(1), -torch.ones(1), logp.detach(), beta=0.0)
    loss.backward()
    assert float(logp.grad) == 0.0


def test_grpo_ratio_uses_fixed_old_logp_across_updates():
    old_logp = torch.zeros(1)
    first = torch.zeros(1, requires_grad=True)
    grpo_loss(first, old_logp, torch.ones(1), first.detach(), beta=0.0).backward()
    assert float(first.grad) < 0.0

    moved = torch.tensor([0.3], requires_grad=True)
    grpo_loss(moved, old_logp, torch.ones(1), moved.detach(), beta=0.0).backward()
    assert float(moved.grad) == 0.0
