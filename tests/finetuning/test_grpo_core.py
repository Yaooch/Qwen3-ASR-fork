import json
import os
import tempfile

import pytest
import torch

from finetuning.grpo_core import (
    apply_lora,
    assert_only_text_decoder_trainable,
    group_advantages,
    grpo_loss,
    load_samples,
    split_eval,
)

CKPT = "/cfs/data/private/WangYaoChi/model/qwen3-asr-ctc-joint-14-hotword-1/checkpoint-228"


# --------------------------------------------------------------------------
# grpo_data: load_samples / split_eval
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


# --------------------------------------------------------------------------
# grpo_lora: apply_lora（集成，需真实 ckpt，默认跳过）
# --------------------------------------------------------------------------


@pytest.mark.integration
def test_apply_lora_only_text_decoder_trainable():
    from qwen_asr.joint import Qwen3ASRJointModel
    from qwen_asr.joint.defaults import DEFAULT_ATTN_IMPLEMENTATION

    joint = Qwen3ASRJointModel.from_pretrained(
        CKPT,
        dtype=torch.bfloat16,
        device_map=None,
        load_heads=False,
        attn_implementation=DEFAULT_ATTN_IMPLEMENTATION,
    ).to("cuda")
    peft_model = apply_lora(joint)
    assert_only_text_decoder_trainable(peft_model)
    n = sum(1 for p in peft_model.parameters() if p.requires_grad)
    assert n > 0
