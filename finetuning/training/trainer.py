# training/trainer.py
import os
from typing import Optional

import editdistance
import torch
from transformers import Trainer

from qwen_joint.tokenize_utils import ids_to_text, text_to_ctc_ids, build_id_to_token


class JointTrainer(Trainer):
    def __init__(self, ctc_lr: float = 1e-3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ctc_lr = ctc_lr

    # ---------- 分层学习率 ----------
    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        decay_names = self.get_decay_parameter_names(self.model)
        groups = []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            is_ctc = "ctc." in n
            lr = self.ctc_lr if is_ctc else self.args.learning_rate
            wd = self.args.weight_decay if n in decay_names else 0.0
            for g in groups:
                if g["lr"] == lr and g["weight_decay"] == wd:
                    g["params"].append(p)
                    break
            else:
                groups.append({"params": [p], "lr": lr, "weight_decay": wd})

        cls, kw = Trainer.get_optimizer_cls_and_kwargs(self.args)
        kw.pop("lr", None)
        self.optimizer = cls(groups, **kw)

        if self.args.process_index == 0:
            print(f"\n[Optimizer] Qwen LR={self.args.learning_rate}  CTC LR={self.ctc_lr}\n")
        return self.optimizer

    # ---------- dtype 对齐 ----------
    def _prepare_inputs(self, inputs):
        if inputs is None:
            return None
        inputs = super()._prepare_inputs(inputs)
        mdtype = getattr(self.model, "dtype", None)
        if mdtype is not None:
            for k, v in list(inputs.items()):
                if torch.is_tensor(v) and v.is_floating_point():
                    inputs[k] = v.to(dtype=mdtype)
        return inputs

    # ---------- 评估时额外计算 CTC CER ----------
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        self.model.eval()
        dataloader = self.get_eval_dataloader(eval_dataset)
        base = self.model.module if hasattr(self.model, "module") else self.model
        id2tok = build_id_to_token(self.data_collator.vocab)

        total_e, total_c, shown = 0, 0, 0
        with torch.no_grad():
            for batch in dataloader:
                if batch is None:
                    continue
                inp = self._prepare_inputs(batch)
                out = self.model(**inp)
                pred_ids = base.ctc.greedy_decode(
                    # 需要原始 hs_pad；这里直接用 log_probs 也行
                    base.ctc.log_softmax.__self__.log_softmax(
                        torch.zeros(1)  # placeholder，不走这条
                    ) if False else None,
                    out["output_lengths"],
                ) if False else None  # 见下面简化版

                # 简化：基于 log_probs 做 argmax + 去重
                lp = out["log_probs"]
                preds = lp.argmax(-1)
                pred_texts = []
                for b in range(preds.size(0)):
                    L = int(out["output_lengths"][b].item())
                    ids = preds[b, :L].cpu().tolist()
                    dedup, prev = [], -1
                    for i in ids:
                        if i != base.ctc.blank_id and i != prev:
                            dedup.append(i)
                        prev = i
                    pred_texts.append(ids_to_text(dedup, id2tok))

                refs = inp.get("texts", [])
                for pt, rt in zip(pred_texts, refs):
                    rt_clean = rt.split("<asr_text>")[-1].strip() if "<asr_text>" in rt else rt.strip()
                    ref_ids = text_to_ctc_ids(rt_clean, self.data_collator.vocab, self.data_collator.sp_model)
                    ref_proc = ids_to_text(ref_ids, id2tok)
                    if self.args.process_index == 0 and shown < 5:
                        print(f"[CTC] pred='{pt}'  ref='{ref_proc}'")
                        shown += 1
                    total_e += editdistance.eval(pt, ref_proc)
                    total_c += len(ref_proc)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            t = torch.tensor([total_e, total_c], dtype=torch.float64, device=self.args.device)
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
            total_e, total_c = t[0].item(), t[1].item()
        cer = total_e / total_c if total_c > 0 else 0.0
        if self.args.process_index == 0:
            print(f"[CTC Eval] global CER = {cer:.4f}")
        metrics[f"{metric_key_prefix}_ctc_cer"] = cer
        self.log({f"{metric_key_prefix}_ctc_cer": cer})
        return metrics

    # ---------- 保存（底座 + CTC 分离） ----------
    def save_model(self, output_dir=None, _internal_call=False):
        if self.args.process_index != 0:
            return
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        base = self.model.module if hasattr(self.model, "module") else self.model
        # 1) 底座 -> 标准 HF safetensors
        base.qwen_model.save_pretrained(output_dir, safe_serialization=True)
        # 2) CTC -> 独立 pt + config
        base.save_ctc(output_dir)
        # 清理可能残留的 joint pytorch_model.bin
        legacy = os.path.join(output_dir, "pytorch_model.bin")
        if os.path.exists(legacy):
            os.remove(legacy)

    # ---------- 加载（干净的重构） ----------
    def _load_from_checkpoint(self, resume_from_checkpoint: str, model=None):
        """重构版：直接把 safetensors 加载到 qwen_model，CTC 加载到 ctc。
        不再依赖 JointModel.load_state_dict 的 prefix hack，因此不再出现 missing keys 告警。
        """
        if model is None:
            model = self.model
        base = model.module if hasattr(model, "module") else model

        # 1) 加载底座权重
        self._load_base_weights(base.qwen_model, resume_from_checkpoint)

        # 2) 加载 CTC 权重
        ctc_path = os.path.join(resume_from_checkpoint, "ctc_head.pt")
        if os.path.exists(ctc_path):
            state = torch.load(ctc_path, map_location="cpu")
            missing, unexpected = base.ctc.load_state_dict(state, strict=False)
            if self.args.process_index == 0:
                print(f"[Resume] Loaded CTC head  (missing={len(missing)}, unexpected={len(unexpected)})")
        else:
            if self.args.process_index == 0:
                print(f"[Resume][Warning] CTC head not found at {ctc_path}")

    @staticmethod
    def _load_base_weights(qwen_model, ckpt_dir: str):
        """支持单文件 / 分片 safetensors / pytorch_model.bin"""
        idx = os.path.join(ckpt_dir, "model.safetensors.index.json")
        single = os.path.join(ckpt_dir, "model.safetensors")
        bin_single = os.path.join(ckpt_dir, "pytorch_model.bin")

        if os.path.exists(idx):
            from transformers.modeling_utils import load_sharded_checkpoint
            load_sharded_checkpoint(qwen_model, ckpt_dir, strict=False)
        elif os.path.exists(single):
            from safetensors.torch import load_model
            load_model(qwen_model, single, strict=False)
        elif os.path.exists(bin_single):
            state = torch.load(bin_single, map_location="cpu")
            qwen_model.load_state_dict(state, strict=False)
        else:
            raise FileNotFoundError(f"No model weights found under {ckpt_dir}")

        # tie weights（lm_head <-> embed_tokens）
        if hasattr(qwen_model, "tie_weights"):
            qwen_model.tie_weights()
