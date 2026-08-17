from types import SimpleNamespace

import torch
import torch.nn as nn

from qwen_asr_ext.glclap.model import GLCLAPHead, _pad_batch
from qwen_asr_ext.joint.train import (
    DataCollatorForJointTraining,
    JointTrainer,
    latest_checkpoint,
    set_trainable,
    weights_from_grad_norms,
)


class DummyBert(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.embedding = nn.Embedding(16, 4)

    def gradient_checkpointing_enable(self, **kwargs):
        return None

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def test_glclap_head_uses_existing_audio_features(monkeypatch):
    import transformers

    monkeypatch.setattr(
        transformers.BertModel,
        "from_pretrained",
        lambda *args, **kwargs: DummyBert(),
    )
    head = GLCLAPHead("dummy", audio_dim=4, embed_dim=3)
    audio = torch.randn(2, 3, 4, requires_grad=True)
    audio_mask = torch.tensor([[True, True, False], [True, True, True]])
    ids = torch.tensor([[1, 2], [3, 4]])
    mask = torch.ones_like(ids)
    out = head(audio, audio_mask, ids, mask, ids, mask, gather=False)

    assert set(("loss", "global_loss", "local_loss")) <= set(out)
    out["loss"].backward()
    assert audio.grad is not None
    assert head.audio_projection.weight.grad is not None


def test_glclap_gather_padding_does_not_repeat_samples():
    value = torch.tensor([[1.0, 2.0]])
    padded = _pad_batch(value, 3)

    assert padded.shape == (3, 2)
    assert torch.equal(padded[0], value[0])
    assert torch.count_nonzero(padded[1:]) == 0


def test_joint_collator_skips_audio_beyond_encoder_capacity(monkeypatch):
    import librosa

    monkeypatch.setattr(
        librosa,
        "load",
        lambda *args, **kwargs: (torch.zeros(32001).numpy(), 16000),
    )
    collator = DataCollatorForJointTraining(
        processor=None,
        vocab={},
        sp_model=None,
        max_audio_seconds=2.0,
    )

    assert collator([{"audio": "too-long.wav", "text": "test"}]) is None


def test_joint_log_rounds_metrics_and_keeps_small_learning_rate(monkeypatch):
    captured = {}
    trainer = object.__new__(JointTrainer)
    trainer._loss_count = 0
    monkeypatch.setattr(
        JointTrainer.__mro__[1],
        "log",
        lambda self, logs, *args, **kwargs: captured.update(logs),
    )

    trainer.log({"loss": 1.23456, "ctc_lr": 1.481481e-6})

    assert captured == {"loss": 1.235, "ctc_lr": 1.481e-6}


def test_latest_checkpoint_ignores_incomplete_directory(tmp_path):
    complete = tmp_path / "checkpoint-1000"
    complete.mkdir()
    (complete / "model.safetensors").write_bytes(b"model")
    (complete / "trainer_state.json").write_text("{}", encoding="utf-8")
    incomplete = tmp_path / "checkpoint-5000"
    incomplete.mkdir()
    (incomplete / "config.json").write_text("{}", encoding="utf-8")

    assert latest_checkpoint(str(tmp_path)) == str(complete)


def test_joint_checkpoint_writes_pytorch_model_directly(tmp_path):
    class DummyQwen:
        generation_config = None

        def save_pretrained(self, output_dir, safe_serialization):
            self.saved_to = output_dir
            assert not safe_serialization
            with open(f"{output_dir}/pytorch_model.bin", "wb") as f:
                f.write(b"model")

    class DummyJoint:
        def __init__(self):
            self.qwen_model = DummyQwen()

        def save_aux(self, output_dir, heads, copy_heads_from):
            with open(f"{output_dir}/joint_config.json", "w", encoding="utf-8") as f:
                f.write("{}")

    target = tmp_path / "cfs-target"
    trainer = object.__new__(JointTrainer)
    trainer.args = SimpleNamespace(process_index=0, output_dir=str(target))
    trainer.model = DummyJoint()
    trainer.head_source = str(tmp_path / "source")
    trainer.save_heads = ()
    trainer.save_model()

    assert trainer.model.qwen_model.saved_to == str(target)
    assert (target / "pytorch_model.bin").read_bytes() == b"model"
    assert (target / "joint_config.json").read_text(encoding="utf-8") == "{}"


class DummyTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(2, 2)
        self.proj1 = nn.Linear(2, 2)
        self.act = nn.ReLU()
        self.proj2 = nn.Linear(2, 2)


class DummyThinker(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_tower = DummyTower()
        self.decoder = nn.Linear(2, 2)


class DummyJoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.qwen_model = nn.Module()
        self.qwen_model.thinker = DummyThinker()
        self.ctc = nn.Linear(2, 2)
        self.rnnt = None
        self.glclap = nn.Linear(2, 2)


def test_stage1_only_trains_ctc_and_glclap():
    model = DummyJoint()
    set_trainable(model, ("ctc", "glclap"))

    assert not any(p.requires_grad for p in model.qwen_model.parameters())
    assert all(p.requires_grad for p in model.ctc.parameters())
    assert all(p.requires_grad for p in model.glclap.parameters())


def test_loss_weights_follow_shared_encoder_gradient_ratio():
    weights = weights_from_grad_norms(
        {"llm": 2.0, "ctc": 8.0, "glclap": 0.5},
        {"llm": 1.0, "ctc": 1.0, "glclap": 1.0},
        {"ctc": 0.25, "glclap": 0.25},
    )

    assert weights["llm"] == 1.0
    assert weights["ctc"] == 0.0625
    assert weights["glclap"] == 1.0
