#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 Qwen3-ASR Audio Encoder 的 CTC 训练脚本 (优化版)
"""

import os
import sys
import json
import argparse
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm
import numpy as np
import librosa
import editdistance

from transformers import AutoModel, AutoProcessor, AutoConfig

from qwen_asr.inference.utils import SAMPLE_RATE, MIN_ASR_INPUT_SECONDS
from qwen_asr.core.transformers_backend import (
    Qwen3ASRConfig,
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRProcessor,
)

AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)


def setup_distributed():
    """初始化分布式训练环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank, world_size, local_rank = 0, 1, 0
    
    if world_size > 1:
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# =============================================================================
# CTC 模块
# =============================================================================

class CTC(nn.Module):
    """CTC 模块"""
    
    def __init__(self, odim: int, encoder_output_size: int, blank_id: int = 0):
        super().__init__()
        self.ctc_lo = nn.Linear(encoder_output_size, odim)
        self.blank_id = blank_id
        self.ctc_loss = nn.CTCLoss(blank=blank_id, zero_infinity=True)

    def log_softmax(self, hs_pad):
        return F.log_softmax(self.ctc_lo(hs_pad), dim=2)

    def forward(self, hs_pad, hs_lengths, targets, target_lengths):
        log_probs = self.log_softmax(hs_pad).transpose(0, 1)
        log_probs = log_probs.float()
        return self.ctc_loss(log_probs, targets, hs_lengths, target_lengths)

    def argmax(self, hs_pad):
        return torch.argmax(self.ctc_lo(hs_pad), dim=2)


# =============================================================================
# Qwen3-ASR + CTC 模型
# =============================================================================

# class Qwen3ASRCTCModel(nn.Module):
#     """Qwen3-ASR Audio Encoder + CTC"""
    
#     def __init__(self, qwen_model_path: str, vocab_size: int, device: str, dtype: torch.dtype):
#         super().__init__()
        
#         self.device = device
#         self.dtype = dtype
        
#         # 加载模型
#         load_kwargs = {"torch_dtype": dtype}
#         if int(os.environ.get("WORLD_SIZE", 1)) <= 1:
#             load_kwargs["device_map"] = device
        
#         print(f"加载模型: {qwen_model_path}")
#         self.qwen_model = AutoModel.from_pretrained(qwen_model_path, **load_kwargs)
#         if int(os.environ.get("WORLD_SIZE", 1)) > 1:
#             self.qwen_model = self.qwen_model.to(device)
        
#         # 提取 audio_tower
#         self.audio_tower = self.qwen_model.thinker.audio_tower
#         self.encoder_output_size = self.audio_tower.config.output_dim
        
#         # 冻结所有参数
#         for param in self.qwen_model.parameters():
#             param.requires_grad = False
#         self.qwen_model.eval()
        
#         # CTC 模块
#         self.ctc = CTC(vocab_size, self.encoder_output_size).to(device=device, dtype=dtype)
        
#         frozen = sum(p.numel() for p in self.qwen_model.parameters())
#         trainable = sum(p.numel() for p in self.ctc.parameters())
#         print(f"冻结参数: {frozen:,}")
#         print(f"CTC模块: encoder_dim={self.encoder_output_size}, vocab={vocab_size}")
#         print(f"可训练参数: {trainable:,}")
    
#     def forward(self, input_features, feature_lengths, targets=None, target_lengths=None):
#         batch_size = input_features.shape[0]
        
#         if input_features.dtype != self.dtype:
#             input_features = input_features.to(self.dtype)
        
#         # 关键修复: 去掉padding并拼接，适配 audio_tower 的输入格式
#         # input_features: [B, n_mels, T] -> 拼接成 [n_mels, sum(T_i)]
#         valid_feats = [input_features[b, :, :feature_lengths[b]] for b in range(batch_size)]
#         concatenated = torch.cat(valid_feats, dim=1)  # [n_mels, sum_lengths]
        
#         with torch.no_grad():
#             encoder_output = self.audio_tower(
#                 input_features=concatenated,
#                 feature_lens=feature_lengths
#             )
#             encoder_hidden_states = encoder_output.last_hidden_state  # [sum_T', hidden_dim]
        
#         # 计算输出长度
#         output_lengths = self._get_feat_extract_output_lengths(feature_lengths)
#         max_len = output_lengths.max()
        
#         # 组装回 [B, max_len, hidden_dim]
#         hs_pad = torch.zeros(batch_size, max_len, self.encoder_output_size,
#                             dtype=encoder_hidden_states.dtype, device=encoder_hidden_states.device)
        
#         idx = 0
#         for b in range(batch_size):
#             length = output_lengths[b].item()
#             hs_pad[b, :length] = encoder_hidden_states[idx:idx+length]
#             idx += length
        
#         # 修复: eval时也能返回loss（validate需要）
#         if targets is not None:
#             loss = self.ctc(hs_pad, output_lengths, targets, target_lengths)
#             if self.training:
#                 return {"loss": loss}
#             else:
#                 return {
#                     "loss": loss,
#                     "log_probs": self.ctc.log_softmax(hs_pad), 
#                     "output_lengths": output_lengths
#                 }
#         else:
#             return {"log_probs": self.ctc.log_softmax(hs_pad), "output_lengths": output_lengths}


        
#     def _get_feat_extract_output_lengths(self, input_lengths: torch.Tensor):
#         input_lengths_leave = input_lengths % 100
#         feat_lengths = (input_lengths_leave - 1) // 2 + 1
#         output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
#         return output_lengths
    
#     def decode(self, log_probs, output_lengths):
#         predictions = torch.argmax(log_probs, dim=2)
#         results = []
#         for b in range(predictions.shape[0]):
#             length = output_lengths[b].item()
#             pred = predictions[b, :length].cpu().tolist()
#             decoded = []
#             prev_id = -1
#             for idx in pred:
#                 if idx != self.ctc.blank_id and idx != prev_id:
#                     decoded.append(idx)
#                 prev_id = idx
#             results.append(decoded)
#         return results

class Qwen3ASRCTCModel(nn.Module):
    """Qwen3-ASR Audio Encoder + CTC（解冻版）"""
    
    def __init__(self, qwen_model_path: str, vocab_size: int, device: str, dtype: torch.dtype):
        super().__init__()
        
        self.device = device
        self.dtype = dtype
        
        # 加载模型
        load_kwargs = {"torch_dtype": dtype}
        if int(os.environ.get("WORLD_SIZE", 1)) <= 1:
            load_kwargs["device_map"] = device
        
        print(f"加载模型: {qwen_model_path}")
        self.qwen_model = AutoModel.from_pretrained(qwen_model_path, **load_kwargs)
        if int(os.environ.get("WORLD_SIZE", 1)) > 1:
            self.qwen_model = self.qwen_model.to(device)
        
        # 提取 audio_tower
        self.audio_tower = self.qwen_model.thinker.audio_tower
        self.encoder_output_size = self.audio_tower.config.output_dim
        
        # ==================== 关键修改 1：解冻 Audio Encoder ====================
        # print("解冻 Audio Tower 参数...")
        # for param in self.audio_tower.parameters():
        #     param.requires_grad = True
        # # 设置为训练模式（取消 eval()）
        # self.audio_tower.train()
        print("冻结 Audio Tower 参数...")
        for param in self.audio_tower.parameters():
            param.requires_grad = False
        self.audio_tower.eval()
        
        # 冻结 LLM 部分（可选，如果你只想优化 Audio Encoder）
        if hasattr(self.qwen_model.thinker, 'model'):
            for param in self.qwen_model.thinker.model.parameters():
                param.requires_grad = False
            self.qwen_model.thinker.model.eval()
        # ===================================================================
        
        # CTC 模块
        self.ctc = CTC(vocab_size, self.encoder_output_size).to(device=device, dtype=dtype)
        
        # 统计参数
        encoder_params = sum(p.numel() for p in self.audio_tower.parameters())
        ctc_params = sum(p.numel() for p in self.ctc.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"Audio Encoder 参数: {encoder_params:,} (已解冻)")
        print(f"CTC 模块参数: {ctc_params:,}")
        print(f"总可训练参数: {trainable:,}")
    
    def forward(self, input_features, feature_lengths, targets=None, target_lengths=None):
        batch_size = input_features.shape[0]
        
        if input_features.dtype != self.dtype:
            input_features = input_features.to(self.dtype)
        
        # 处理输入（去掉padding并拼接）
        valid_feats = [input_features[b, :, :feature_lengths[b]] for b in range(batch_size)]
        concatenated = torch.cat(valid_feats, dim=1)  # [n_mels, sum_lengths]
        
        # ==================== 关键修改 2：移除 torch.no_grad() ====================
        # 之前：with torch.no_grad():
        # 现在：直接前向传播，允许梯度回传
        with torch.no_grad():
            encoder_output = self.audio_tower(
                input_features=concatenated,
                feature_lens=feature_lengths
            )
        encoder_hidden_states = encoder_output.last_hidden_state  # [sum_T', hidden_dim]
        # =======================================================================
        
        # 计算输出长度
        output_lengths = self._get_feat_extract_output_lengths(feature_lengths)
        max_len = output_lengths.max()
        
        # 组装回 [B, max_len, hidden_dim]
        hs_pad = torch.zeros(batch_size, max_len, self.encoder_output_size,
                            dtype=encoder_hidden_states.dtype, device=encoder_hidden_states.device)
        
        idx = 0
        for b in range(batch_size):
            length = output_lengths[b].item()
            hs_pad[b, :length] = encoder_hidden_states[idx:idx+length]
            idx += length
        
            # 在计算 loss 前打印统计信息（只在训练前几个 batch 打印）
        if self.training and torch.rand(1).item() < 0.01:  # 1% 概率打印
            print(f"\n[Debug] hs_pad shape: {hs_pad.shape}")
            print(f"[Debug] hs_pad stats: mean={hs_pad.mean():.3f}, std={hs_pad.std():.3f}")
            print(f"[Debug] output_lengths: {output_lengths[:5]}")  # 前5个
            print(f"[Debug] target_lengths: {target_lengths[:5]}")
            
            # 检查 CTC 输出
            with torch.no_grad():
                logits = self.ctc.ctc_lo(hs_pad)
                probs = torch.softmax(logits, dim=-1)
                print(f"[Debug] Blank prob mean: {probs[:,:,0].mean():.3f}")  # blank (id=0) 的概率
                print(f"[Debug] Max non-blank prob: {probs[:,:,1:].max():.3f}")  # 非 blank 的最大概率


        # 计算 CTC Loss 或推理
        if targets is not None:
            loss = self.ctc(hs_pad, output_lengths, targets, target_lengths)
            if self.training:
                return {"loss": loss}
            else:
                return {
                    "loss": loss,
                    "log_probs": self.ctc.log_softmax(hs_pad), 
                    "output_lengths": output_lengths
                }
        else:
            return {"log_probs": self.ctc.log_softmax(hs_pad), "output_lengths": output_lengths}
    
    def _get_feat_extract_output_lengths(self, input_lengths: torch.Tensor):
        input_lengths_leave = input_lengths % 100
        feat_lengths = (input_lengths_leave - 1) // 2 + 1
        output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
        return output_lengths
    
    def decode(self, log_probs, output_lengths):
        predictions = torch.argmax(log_probs, dim=2)
        results = []
        for b in range(predictions.shape[0]):
            length = output_lengths[b].item()
            pred = predictions[b, :length].cpu().tolist()
            decoded = []
            prev_id = -1
            for idx in pred:
                if idx != self.ctc.blank_id and idx != prev_id:
                    decoded.append(idx)
                prev_id = idx
            results.append(decoded)
        return results


# =============================================================================
# 数据集 - 参考 finetuning/qwen3_asr_sft.py
# =============================================================================

def load_audio(path: str, sr: int = 16000, max_duration: float = 30.0):
    """加载音频，参考 finetuning/qwen3_asr_sft.py"""
    try:
        wav, _ = librosa.load(path, sr=sr, mono=True)
        duration = len(wav) / sr
        if duration > max_duration:
            print(f"Warning: Audio too long ({duration:.1f}s > {max_duration}s), skipping: {path}")
            return None
        return wav
    except Exception as e:
        print(f"Warning: Failed to load audio {path}: {e}")
        return None


class CTCDataset(Dataset):
    """CTC 数据集"""
    
    def __init__(self, manifest_path: str, vocab: Dict[str, int], max_duration: float = 30.0):
        self.vocab = vocab
        self.blank_id = vocab.get("<blank>", 0)
        self.max_duration = max_duration
        
        self.samples = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))
        
        print(f"加载 {len(self.samples)} 条样本 from {manifest_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio_path = sample["audio"]
        text = sample["text"]

        text = text.lower()
        import re
        text = re.sub(r'[^\u4e00-\u9fa5a-z0-9\s]', '', text)
        
        # 检查文件
        if not os.path.exists(audio_path):
            return self._placeholder()
        
        # 加载音频
        wav = load_audio(audio_path, sr=SAMPLE_RATE, max_duration=self.max_duration)
        if wav is None:
            return self._placeholder()
        
        # 文本转 ID
        target_ids = [self.vocab.get(c, self.vocab.get("<unk>", 1)) for c in text]
        
        return {
            "audio": wav,
            "target_ids": target_ids,
            "target_length": len(target_ids),
            "text": text,
        }
    
    def _placeholder(self):
        return {
            "audio": None,
            "target_ids": [self.blank_id],
            "target_length": 1,
            "text": "",
        }


class CTCCollator:
    """CTC Collator - 参考 finetuning 的 DataCollatorForQwen3ASRFinetuning"""
    
    def __init__(self, processor, max_audio_length: int = 3000):
        self.processor = processor
        self.max_audio_length = max_audio_length
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 过滤无效样本
        valid_features = []
        for f in features:
            if f["audio"] is not None:
                valid_features.append(f)
        
        if len(valid_features) == 0:
            return None
        
        audios = [f["audio"] for f in valid_features]
        texts = [f["text"] for f in valid_features]
        
        # 使用 processor 处理音频和文本（参考 finetuning 代码）
        # 这里只处理音频获取 input_features
        audio_inputs = self.processor(
            audio=audios,
            text=[""] * len(audios),
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        
        input_features = audio_inputs["input_features"]  # (B, n_mels, T)
        feature_attention_mask = audio_inputs.get("feature_attention_mask", None)
        
        # 计算 feature_lengths
        if feature_attention_mask is not None:
            feature_lengths = feature_attention_mask.sum(dim=1).long()
        else:
            feature_lengths = torch.tensor([f.shape[1] for f in input_features], dtype=torch.long)
        
        # 限制最大长度
        if input_features.shape[2] > self.max_audio_length:
            input_features = input_features[:, :, :self.max_audio_length]
            feature_lengths = torch.clamp(feature_lengths, max=self.max_audio_length)
        
        # 处理 target
        max_target_len = max(f["target_length"] for f in valid_features)
        target_ids = torch.zeros(len(valid_features), max_target_len, dtype=torch.long)
        target_lengths = torch.zeros(len(valid_features), dtype=torch.long)
        
        for i, f in enumerate(valid_features):
            t_len = f["target_length"]
            target_ids[i, :t_len] = torch.tensor(f["target_ids"], dtype=torch.long)
            target_lengths[i] = t_len
        
        return {
            "input_features": input_features,
            "feature_lengths": feature_lengths,
            "target_ids": target_ids,
            "target_lengths": target_lengths,
            "texts": texts,
        }


# =============================================================================
# 训练函数
# =============================================================================

def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, grad_clip, 
                world_size, rank, grad_accum):
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}") if rank == 0 else dataloader
    
    for batch_idx, batch in enumerate(pbar):
        if batch is None:
            continue
        
        # 优化: 使用 non_blocking=True 实现异步数据传输
        input_features = batch["input_features"].to(device, non_blocking=True)
        feature_lengths = batch["feature_lengths"].to(device, non_blocking=True)
        target_ids = batch["target_ids"].to(device, non_blocking=True)
        target_lengths = batch["target_lengths"].to(device, non_blocking=True)
        
        outputs = model(input_features, feature_lengths, target_ids, target_lengths)
        loss = outputs["loss"] / grad_accum
        loss.backward()
        
        total_loss += loss.item() * grad_accum
        
        if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == num_batches:
            if grad_clip > 0:
                trainable_params = filter(lambda p: p.requires_grad, model.parameters())
                torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
            optimizer.step()
            # 优化: 使用 set_to_none=True 替代默认的 set_to_zero，节省内存并加速
            optimizer.zero_grad(set_to_none=True)
            if scheduler:
                scheduler.step()
        
        if rank == 0:
            pbar.set_postfix({"loss": f"{total_loss / (batch_idx + 1):.4f}"})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    if world_size > 1:
        loss_tensor = torch.tensor(avg_loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = loss_tensor.item()
    
    return {"loss": avg_loss}


# def validate(model, dataloader, device, epoch, vocab, world_size, rank):
#     model.eval()
#     base_model = model.module if hasattr(model, 'module') else model
    
#     total_loss = 0.0
#     num_batches = len(dataloader)
#     id_to_token = {v: k for k, v in vocab.items()}
    
#     total_edits = 0
#     total_chars = 0
    
#     pbar = tqdm(dataloader, desc=f"Val {epoch}") if rank == 0 else dataloader
    
#     with torch.no_grad():
#         for batch in pbar:
#             if batch is None:
#                 continue
            
#             # 优化: 使用 non_blocking=True
#             input_features = batch["input_features"].to(device, non_blocking=True)
#             feature_lengths = batch["feature_lengths"].to(device, non_blocking=True)
#             target_ids = batch["target_ids"].to(device, non_blocking=True)
#             target_lengths = batch["target_lengths"].to(device, non_blocking=True)
#             texts = batch["texts"]
            
#             outputs = model(input_features, feature_lengths, target_ids, target_lengths)
#             total_loss += outputs["loss"].item()
            
#             infer_outputs = model(input_features, feature_lengths)
#             predictions = base_model.decode(infer_outputs["log_probs"], infer_outputs["output_lengths"])
            
#             for pred_ids, ref_text in zip(predictions, texts):
#                 pred_text = "".join([id_to_token.get(i, "") for i in pred_ids])
#                 edits = editdistance.eval(pred_text, ref_text)
#                 total_edits += edits
#                 total_chars += len(ref_text)
    
#     avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
#     cer = total_edits / total_chars if total_chars > 0 else 0.0
    
#     if world_size > 1:
#         metrics = torch.tensor([avg_loss, float(total_edits), float(total_chars)], device=device)
#         dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
#         avg_loss = metrics[0].item() / world_size
#         cer = metrics[1].item() / metrics[2].item() if metrics[2].item() > 0 else 0.0
    
#     if rank == 0:
#         print(f"Val - Loss: {avg_loss:.4f}, CER: {cer:.4f}")
    
#     return {"loss": avg_loss, "cer": cer}

def validate(model, dataloader, device, epoch, vocab, world_size, rank):
    model.eval()
    base_model = model.module if hasattr(model, 'module') else model
    
    total_loss = 0.0
    num_batches = len(dataloader)
    id_to_token = {v: k for k, v in vocab.items()}
    
    total_edits = 0
    total_chars = 0
    
    # 调试计数器
    debug_samples = 0
    max_debug = 3
    
    pbar = tqdm(dataloader, desc=f"Val {epoch}") if rank == 0 else dataloader
    
    with torch.no_grad():
        for batch in pbar:
            if batch is None:
                continue
            
            # 优化: 使用 non_blocking=True
            input_features = batch["input_features"].to(device, non_blocking=True)
            feature_lengths = batch["feature_lengths"].to(device, non_blocking=True)
            target_ids = batch["target_ids"].to(device, non_blocking=True)
            target_lengths = batch["target_lengths"].to(device, non_blocking=True)
            texts = batch["texts"]
            
            outputs = model(input_features, feature_lengths, target_ids, target_lengths)
            total_loss += outputs["loss"].item()
            
            infer_outputs = model(input_features, feature_lengths)
            predictions = base_model.decode(infer_outputs["log_probs"], infer_outputs["output_lengths"])
            
            for pred_ids, ref_text in zip(predictions, texts):
                pred_text = "".join([id_to_token.get(i, "") for i in pred_ids])
                
                # ========== 添加调试打印（只打印前3个样本，只在rank 0打印） ==========
                if rank == 0 and debug_samples < max_debug:
                    print(f"\n[调试 Epoch {epoch}] 样本 {debug_samples}:")
                    print(f"  预测ID序列(前30): {pred_ids[:30]}...")
                    print(f"  预测文本: '{pred_text}'")
                    print(f"  真实文本: '{ref_text}'")
                    print(f"  序列长度: 预测{len(pred_ids)} vs 真实{len(ref_text)}")
                    if pred_ids:
                        unique_ids = set(pred_ids)
                        print(f"  唯一ID数: {len(unique_ids)}, ID范围: {min(pred_ids)}-{max(pred_ids)}")
                        print(f"  是否全为blank(0): {all(x == 0 for x in pred_ids)}")
                        if len(unique_ids) <= 3:
                            print(f"  实际ID值: {list(unique_ids)}")
                    else:
                        print(f"  警告: 预测ID序列为空!")
                    debug_samples += 1
                # ===================================================================
                
                edits = editdistance.eval(pred_text, ref_text)
                total_edits += edits
                total_chars += len(ref_text)
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    cer = total_edits / total_chars if total_chars > 0 else 0.0
    
    if world_size > 1:
        metrics = torch.tensor([avg_loss, float(total_edits), float(total_chars)], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        avg_loss = metrics[0].item() / world_size
        cer = metrics[1].item() / metrics[2].item() if metrics[2].item() > 0 else 0.0
    
    if rank == 0:
        print(f"Val - Loss: {avg_loss:.4f}, CER: {cer:.4f}")
    
    return {"loss": avg_loss, "cer": cer}


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--train_manifest", type=str, required=True)
    parser.add_argument("--val_manifest", type=str, default=None)
    parser.add_argument("--vocab_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./ctc_output")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_duration", type=float, default=30.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()
    
    # 优化: 启用 cuDNN Benchmark 自动寻找最优算法
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    # 分布式
    rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0
    
    # 设备
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
    if world_size > 1:
        dist.barrier()
    
    # 加载 processor
    if is_main:
        print(f"Loading processor from {args.model_path}")
    processor = AutoProcessor.from_pretrained(args.model_path, fix_mistral_regex=True)
    
    # 加载或创建词汇表
    if args.vocab_path and os.path.exists(args.vocab_path):
        with open(args.vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
    else:
        vocab = {"<blank>": 0, "<unk>": 1}
        with open(args.train_manifest, "r", encoding="utf-8") as f:
            for line in f:
                text = json.loads(line.strip())["text"]
                for c in text:
                    if c not in vocab:
                        vocab[c] = len(vocab)
        vocab_path = args.vocab_path or os.path.join(args.output_dir, "vocab.json")
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    if is_main:
        print(f"Vocab size: {len(vocab)}")
    
    # 创建模型
    model = Qwen3ASRCTCModel(args.model_path, len(vocab), device, dtype)
    
    # DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # 数据
    train_dataset = CTCDataset(args.train_manifest, vocab, args.max_duration)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    train_collator = CTCCollator(processor)
    
    # 优化: 添加 persistent_workers 和 prefetch_factor 加速数据加载
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        sampler=train_sampler,
        shuffle=(train_sampler is None), 
        num_workers=args.num_workers,
        collate_fn=train_collator, 
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    
    val_loader = None
    if args.val_manifest:
        val_dataset = CTCDataset(args.val_manifest, vocab, args.max_duration)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
        val_collator = CTCCollator(processor)
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            sampler=val_sampler,
            shuffle=False, 
            num_workers=args.num_workers, 
            collate_fn=val_collator, 
            pin_memory=True,
            persistent_workers=True if args.num_workers > 0 else False,
            prefetch_factor=2 if args.num_workers > 0 else None,
        )
    
    # 优化器
    # optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
   
    # ==================== 关键修改 3：分层学习率 ====================
    # Audio Encoder 使用较小学习率，CTC 使用正常学习率
    base_lr = args.lr  # 建议设为 1e-4 或 5e-5
    
    # optimizer = AdamW([
    #     {
    #         'params': model.audio_tower.parameters() if not hasattr(model, 'module') else model.module.audio_tower.parameters(), 
    #         'lr': base_lr * 0.01,  # Encoder 学习率小 100 倍
    #         'name': 'audio_encoder'
    #     },
    #     {
    #         'params': model.ctc.parameters() if not hasattr(model, 'module') else model.module.ctc.parameters(), 
    #         'lr': base_lr,  # CTC 正常学习率
    #         'name': 'ctc_head'
    #     }
    # ])
    
    # print(f"优化器设置：")
    # print(f"  Audio Encoder LR: {base_lr * 0.1}")
    # print(f"  CTC Head LR: {base_lr}")
    ctc_params = model.ctc.parameters() if not hasattr(model, 'module') else model.module.ctc.parameters()
    optimizer = AdamW(ctc_params, lr=args.lr)  # 可以用更高的学习率，比如 1e-3
    print(f"优化器设置：Encoder 已冻结，只训练 CTC Head，LR: {args.lr}")
    # ============================================================


    total_steps = len(train_loader) * args.num_epochs // args.grad_accum
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    
    # 训练
    best_loss = float("inf")
    for epoch in range(1, args.num_epochs + 1):
        if is_main:
            print(f"\nEpoch {epoch}/{args.num_epochs}")
        
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, device, epoch,
                                    args.grad_clip, world_size, rank, args.grad_accum)
        
        if is_main:
            print(f"Train Loss: {train_metrics['loss']:.4f}")
        
        if val_loader:
            val_metrics = validate(model, val_loader, device, epoch, vocab, world_size, rank)
            
            if is_main and val_metrics["loss"] < best_loss:
                best_loss = val_metrics["loss"]
                save_path = os.path.join(args.output_dir, "best_model.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.module.ctc.state_dict() if hasattr(model, 'module') else model.ctc.state_dict(),
                    "vocab": vocab,
                }, save_path)
                print(f"Saved best model to {save_path}")
        
        if is_main and epoch % 1 == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.module.ctc.state_dict() if hasattr(model, 'module') else model.ctc.state_dict(),
                "vocab": vocab,
            }, save_path)
    
    cleanup_distributed()
    if is_main:
        print("\nTraining completed!")


if __name__ == "__main__":
    main()
