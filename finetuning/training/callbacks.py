# training/callbacks.py
import os
from transformers import TrainerCallback, TrainingArguments

from .utils import copy_required_hf_files


class MakeEveryCheckpointInferableCallback(TrainerCallback):
    """训练时复制 tokenizer/processor 配置文件到每个 checkpoint，
    使 checkpoint 可被 Qwen3ASRModel.from_pretrained 直接加载推理。"""
    def __init__(self, base_model_path: str):
        self.base_model_path = base_model_path

    def on_save(self, args: TrainingArguments, state, control, **kwargs):
        if args.process_index != 0:
            return control
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.isdir(ckpt_dir):
            copy_required_hf_files(self.base_model_path, ckpt_dir)
        return control
