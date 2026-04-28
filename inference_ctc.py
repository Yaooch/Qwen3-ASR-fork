#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTC 模型推理脚本（调试版）
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from transformers import AutoModel, AutoProcessor, AutoConfig
from tqdm import tqdm

from qwen_asr.inference.utils import SAMPLE_RATE
from qwen_asr.core.transformers_backend import (
    Qwen3ASRConfig,
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRProcessor,
)

AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)


class CTC(nn.Module):
    """CTC 模块（与训练时一致）"""
    def __init__(self, odim: int, encoder_output_size: int, blank_id: int = 0):
        super().__init__()
        self.ctc_lo = nn.Linear(encoder_output_size, odim)
        self.blank_id = blank_id

    def log_softmax(self, hs_pad):
        return F.log_softmax(self.ctc_lo(hs_pad), dim=2)

    def argmax(self, hs_pad):
        return torch.argmax(self.ctc_lo(hs_pad), dim=2)


class Qwen3ASRCTCInference:
    def __init__(self, base_model_path: str, ctc_checkpoint: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        self.device = device
        self.dtype = dtype
        
        # 加载词汇表
        checkpoint = torch.load(ctc_checkpoint, map_location="cpu")
        self.vocab = checkpoint["vocab"]
        
        # 安全获取blank_id
        blank_val = self.vocab.get("<blank>", 0)
        self.blank_id = blank_val[0] if isinstance(blank_val, list) else blank_val
        
        # 创建id_to_token映射
        first_val = list(self.vocab.values())[0]
        if isinstance(first_val, (list, int)):
            self.id_to_token = {}
            for k, v in self.vocab.items():
                vid = v[0] if isinstance(v, list) else v
                self.id_to_token[vid] = k
        else:
            self.id_to_token = self.vocab
        
        # 记录特殊token
        self.special_ids = {self.blank_id}
        def get_id(key, default=-1):
            val = self.vocab.get(key, default)
            return val[0] if isinstance(val, list) else val
        
        for token in ["<unk>", "<pad>", "<s>", "</s>", "|"]:
            tid = get_id(token, -1)
            if tid != -1:
                self.special_ids.add(tid)
        
        vocab_size = len(self.vocab)
        
        print(f"加载基础模型: {base_model_path}")
        self.processor = Qwen3ASRProcessor.from_pretrained(base_model_path, fix_mistral_regex=True)
        
        # 加载基础模型
        self.qwen_model = AutoModel.from_pretrained(
            base_model_path, 
            torch_dtype=dtype,
            device_map=device
        )
        self.audio_tower = self.qwen_model.thinker.audio_tower
        self.audio_tower.eval()
        
        encoder_output_size = self.audio_tower.config.output_dim
        
        # 初始化CTC
        print(f"加载 CTC 模型: {ctc_checkpoint}")
        self.ctc = CTC(vocab_size, encoder_output_size, self.blank_id).to(device=device, dtype=dtype)
        
        # 加载CTC权重（关键调试点）
        ctc_state = checkpoint["model_state_dict"]
        print(f"CTC checkpoint keys: {list(ctc_state.keys())[:5]}...")
        
        # 检查权重是否全0或随机
        sample_weight = list(ctc_state.values())[0]
        print(f"Sample weight shape: {sample_weight.shape}, mean: {sample_weight.float().mean():.4f}, std: {sample_weight.float().std():.4f}")
        
        self.ctc.load_state_dict(ctc_state)
        self.ctc.eval()
        
        # 验证加载后的权重
        loaded_weight = self.ctc.ctc_lo.weight.data
        print(f"Loaded weight mean: {loaded_weight.float().mean():.4f}, std: {loaded_weight.float().std():.4f}")
        
        print(f"词汇表大小: {vocab_size}")
        print(f"Blank ID: {self.blank_id}")
        print(f"Vocab sample: { {i: self.id_to_token.get(i, 'N/A') for i in range(min(10, vocab_size))} }")

    def _get_feat_extract_output_lengths(self, input_lengths: torch.Tensor):
        """计算音频编码后的长度（与训练时一致）"""
        input_lengths_leave = input_lengths % 100
        feat_lengths = (input_lengths_leave - 1) // 2 + 1
        output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
        return output_lengths

    def _ctc_decode(self, log_probs, input_lengths, debug=False):
        """CTC解码"""
        predictions = torch.argmax(log_probs, dim=2).squeeze(0).cpu().tolist()
        
        output_length = self._get_feat_extract_output_lengths(input_lengths).item()
        predictions = predictions[:output_length]
        
        if debug:
            print(f"  Raw predictions (first 20): {predictions[:20]}")
            print(f"  Output length: {output_length}")
        
        # CTC后处理
        decoded = []
        prev_id = -1
        for idx in predictions:
            if isinstance(idx, list):
                idx = idx[0] if idx else -1
            
            if idx != self.blank_id and idx != prev_id and idx not in self.special_ids:
                decoded.append(idx)
            prev_id = idx
        
        tokens = [self.id_to_token.get(i, f"<{i}>") for i in decoded]
        text = "".join(tokens)
        
        if debug:
            print(f"  Decoded IDs: {decoded}")
            print(f"  Tokens: {tokens}")
            print(f"  Text: '{text}'")
        
        return text

    def transcribe(self, audio_path: str, debug: bool = False) -> str:
        """单条音频推理"""
        wav, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        if len(wav) == 0:
            return ""
        
        # 提取特征
        inputs = self.processor(
            audio=[wav],
            text=[""],
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        
        input_features = inputs["input_features"].to(device=self.device, dtype=self.dtype)
        feature_lengths = torch.tensor([input_features.shape[2]], device=self.device)
        
        if debug:
            print(f"Input features shape: {input_features.shape}")
            print(f"Feature range: [{input_features.min():.2f}, {input_features.max():.2f}]")
        
        # 推理
        with torch.no_grad():
            # 去掉batch维，传入 (n_mels, T)
            feat = input_features.squeeze(0)
            feat_len = feature_lengths
            
            encoder_output = self.audio_tower(
                input_features=feat,
                feature_lens=feat_len
            )
            
            hidden_states = encoder_output.last_hidden_state.unsqueeze(0)
            
            if debug:
                print(f"Encoder output shape: {hidden_states.shape}")
                print(f"Encoder output range: [{hidden_states.min():.2f}, {hidden_states.max():.2f}]")
            
            # CTC前检查
            ctc_input = hidden_states
            if debug:
                print(f"CTC input shape: {ctc_input.shape}")
            
            log_probs = self.ctc.log_softmax(ctc_input).float()
            
            if debug:
                print(f"Log probs shape: {log_probs.shape}")
                print(f"Log probs per position (first 5):")
                for i in range(min(5, log_probs.shape[1])):
                    probs = log_probs[0, i, :]
                    top5 = torch.topk(probs, 5)
                    print(f"  Pos {i}: {[(self.id_to_token.get(idx.item(), '?'), val.item()) for idx, val in zip(top5.indices, top5.values)]}")
            
            text = self._ctc_decode(log_probs, feat_len, debug=debug)
            
        return text

    def transcribe_batch(self, audio_items: list, batch_size: int = 8) -> list:
        """批量推理"""
        results = []
        
        for i in range(0, len(audio_items), batch_size):
            batch = audio_items[i:i+batch_size]
            batch_results = self._transcribe_batch(batch, debug=(i==0))
            results.extend(batch_results)
        
        return results

    def _transcribe_batch(self, audio_items: list, debug: bool = False) -> list:
        """内部批量处理"""
        if isinstance(audio_items[0], tuple):
            uttids = [item[0] for item in audio_items]
            audio_paths = [item[1] for item in audio_items]
        else:
            uttids = [os.path.basename(p) for p in audio_items]
            audio_paths = audio_items
        
        # 加载音频
        wavs = []
        valid_lengths = []
        valid_indices = []
        failed_items = []
        
        for idx, path in enumerate(audio_paths):
            try:
                wav, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
                if len(wav) == 0:
                    raise ValueError("Empty audio")
                wavs.append(wav)
                valid_lengths.append(len(wav))
                valid_indices.append(idx)
            except Exception as e:
                failed_items.append((uttids[idx], path, str(e)))
        
        if not wavs:
            return [], failed_items
        
        # 提取特征
        inputs = self.processor(
            audio=wavs,
            text=[""] * len(wavs),
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        
        input_features = inputs["input_features"].to(device=self.device, dtype=self.dtype)
        batch_size, n_mels, max_time = input_features.shape
        
        # 计算实际帧数
        feature_lengths_list = []
        for l in valid_lengths:
            n_frames = (l - 1) // 160 + 1 if l > 0 else 0
            feature_lengths_list.append(n_frames)
        
        feature_lengths = torch.tensor(feature_lengths_list, device=self.device)
        
        # 推理
        results = []
        with torch.no_grad():
            for b, original_idx in enumerate(valid_indices):
                try:
                    feat = input_features[b, :, :feature_lengths[b]]
                    feat_len = feature_lengths[b].unsqueeze(0)
                    
                    encoder_output = self.audio_tower(
                        input_features=feat,
                        feature_lens=feat_len
                    )
                    hidden = encoder_output.last_hidden_state.unsqueeze(0)
                    log_probs = self.ctc.log_softmax(hidden).float()
                    text = self._ctc_decode(log_probs, feat_len, debug=debug and b==0)
                    
                    results.append((uttids[original_idx], text))
                except Exception as e:
                    results.append((uttids[original_idx], f"ERROR: {str(e)}"))
        
        return results


def parse_scp(scp_path: str):
    """解析 SCP 文件"""
    items = []
    with open(scp_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                items.append((parts[0], parts[1]))
            elif len(parts) == 1:
                uttid = os.path.basename(parts[0]).split('.')[0]
                items.append((uttid, parts[0]))
    return items


def main():
    parser = argparse.ArgumentParser(description='CTC Model Inference')
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--ctc_checkpoint", type=str, required=True)
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--audio", type=str)
    input_group.add_argument("--dir", type=str)
    input_group.add_argument("--scp", type=str)
    
    parser.add_argument("--output", type=str, default="results.txt")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()
    
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    
    model = Qwen3ASRCTCInference(
        base_model_path=args.base_model,
        ctc_checkpoint=args.ctc_checkpoint,
        device=args.device,
        dtype=dtype
    )
    
    # 收集音频
    if args.audio:
        uttid = os.path.basename(args.audio).split('.')[0]
        audio_items = [(uttid, args.audio)]
    elif args.dir:
        audio_files = [os.path.join(args.dir, f) for f in os.listdir(args.dir) 
                      if f.endswith(('.wav', '.mp3', '.flac', '.m4a', '.ogg'))]
        audio_items = [(os.path.basename(p).split('.')[0], p) for p in audio_files]
    elif args.scp:
        audio_items = parse_scp(args.scp)
        print(f"从 SCP 文件加载 {len(audio_items)} 条音频")
    else:
        raise ValueError("必须指定输入")
    
    print(f"处理 {len(audio_items)} 个音频文件...")
    
    # 推理第一个样本时打印调试信息
    results = []
    for i, (uttid, audio_path) in enumerate(tqdm(audio_items, desc="识别中")):
        try:
            # 前3个样本打印调试信息
            text = model.transcribe(audio_path, debug=(i < 3))
            results.append((uttid, text))
            if i < 3:
                print(f"\n结果 {i}: {uttid} -> '{text}'\n")
        except Exception as e:
            print(f"Error: {uttid}: {e}")
            import traceback
            traceback.print_exc()
            results.append((uttid, ""))
    
    # 保存
    with open(args.output, "w", encoding="utf-8") as f:
        for uttid, text in results:
            f.write(f"{uttid}\t{text}\n")
    
    print(f"\n结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
