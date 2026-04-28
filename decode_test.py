#!/usr/bin/env python3
# coding=utf-8
"""
批量测试脚本 - 用于 ASR 模型推理（仅输出识别结果到 txt 文件）

支持输入格式：
  - SCP 文件：每行 "文件id\t文件路径"（推荐）
  - JSONL 文件：每行 JSON 对象，包含 "audio" 或 "wav" 字段

使用方法:
    # SCP 文件格式测试（推荐）
    python decode_test.py --test_file /path/to/test.scp

    # 多卡并行测试（4卡，推荐）
    python decode_test.py --test_file /path/to/test.scp --num_gpus 4

    # 指定单卡测试
    python decode_test.py --test_file /path/to/test.scp --gpu_id 0 --batch_size 256

    # 使用其他模型
    python decode_test.py --test_file /path/to/test.scp --model_path Qwen/Qwen3-ASR-0.6B
"""

import argparse
import json
import os
import time
import multiprocessing as mp
from typing import List, Optional
from tqdm import tqdm

import torch


def load_test_data(test_file: str, start_idx: int = 0, max_samples: int = -1) -> List[dict]:
    """
    加载测试数据，支持 SCP 和 JSONL 两种格式
    
    SCP 格式：每行 "文件id\t文件路径"
    JSONL 格式：每行 JSON 对象，包含 "audio" 或 "wav" 字段
    
    返回统一格式的列表，每项包含：
      - key: 文件id（scp格式）或从路径提取（jsonl格式）
      - audio: 音频文件路径
    """
    data = []
    
    # 判断文件格式：检查第一行
    with open(test_file, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        if not first_line:
            return data
        # 尝试解析为 JSON，失败则为 SCP 格式
        try:
            json.loads(first_line)
            is_jsonl = True
        except json.JSONDecodeError:
            is_jsonl = False
    
    with open(test_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < start_idx:
                continue
            if max_samples > 0 and len(data) >= max_samples:
                break
            
            line = line.strip()
            if not line:
                continue
                
            if is_jsonl:
                # JSONL 格式
                item = json.loads(line)
                audio_path = item.get('audio') or item.get('wav')
                if audio_path is None:
                    continue
                # 从路径提取文件id
                key = os.path.splitext(os.path.basename(audio_path))[0]
                data.append({'key': key, 'audio': audio_path})
            else:
                # SCP 格式：key\tpath
                parts = line.split('\t')
                if len(parts) >= 2:
                    key = parts[0]
                    # 去除 .wav 后缀（如果有）
                    if key.lower().endswith('.wav'):
                        key = key[:-4]
                    audio_path = parts[1]
                    data.append({'key': key, 'audio': audio_path})
                    
    return data


def save_summary(summary: dict, output_file: str):
    """
    保存汇总统计（JSON格式）
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def run_single_gpu(
    gpu_id: int,
    model_path: str,
    test_data: List[dict],
    language: Optional[str],
    batch_size: int,
    max_new_tokens: int,
    gpu_memory_utilization: float,
    output_file: str,
    backend: str = "vllm",
):
    """
    单卡推理函数（用于多卡并行）
    直接输出txt格式：文件id + tab + 识别结果
    """
    # 设置当前 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 加载模型
    if backend == "vllm":
        from qwen_asr import Qwen3ASRModel
        model = Qwen3ASRModel.LLM(
            model=model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            max_inference_batch_size=batch_size,
            max_new_tokens=max_new_tokens,
        )
    else:
        from qwen_asr import Qwen3ASRModel
        model = Qwen3ASRModel.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            max_inference_batch_size=batch_size,
            max_new_tokens=max_new_tokens,
        )

    # 直接写入txt文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i + batch_size]
            audio_paths = [item['audio'] for item in batch]
            keys = [item['key'] for item in batch]

            try:
                predictions = model.transcribe(
                    audio=audio_paths,
                    language=language,
                    # context="你是一个拥有超高精度的语音识别引擎。专属名词列表如下:[岚图, 岚图知音]。请根据音频内容进行识别，当遇到音素类似的词汇时，必须优先匹配列表中的专属名词，而不是通用词汇。",
                )
            except Exception as e:
                print(f"[GPU {gpu_id}] 批次 {i} 推理失败: {e}")
                continue

            for j, pred in enumerate(predictions):
                hyp = pred.text.strip()
                # print(pred)
                key = keys[j]
                lag = pred.language
                # 写入：文件id + tab + 识别结果
                f.write(f"{key}\t{hyp}\t{lag}\n")

    return {
        "gpu_id": gpu_id,
        "num_samples": len(test_data),
    }


def run_test(
    model_path: str,
    test_file: str,
    output_dir: str,
    backend: str = "vllm",
    language: Optional[str] = None,
    batch_size: int = 64,
    max_new_tokens: int = 256,
    gpu_memory_utilization: float = 0.8,
    start_idx: int = 0,
    max_samples: int = -1,
    num_gpus: int = 1,
    gpu_id: Optional[int] = None,
):
    """
    运行批量测试（支持多卡并行）
    输出格式：txt文件，每行 "文件id\t识别结果"
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载测试数据
    print(f"加载测试数据: {test_file}")
    all_test_data = load_test_data(test_file, start_idx, max_samples)
    total_samples = len(all_test_data)
    print(f"测试样本数: {total_samples}")

    start_time = time.time()
    results_file = os.path.join(output_dir, "results.txt")

    # 多卡并行
    if num_gpus > 1:
        print(f"使用 {num_gpus} 卡并行推理...")
        
        # 将数据分成 num_gpus 份
        chunk_size = (total_samples + num_gpus - 1) // num_gpus
        data_chunks = []
        for i in range(num_gpus):
            start = i * chunk_size
            end = min(start + chunk_size, total_samples)
            data_chunks.append(all_test_data[start:end])

        # 创建临时输出文件（txt格式）
        temp_files = [os.path.join(output_dir, f"results_gpu_{i}.txt") for i in range(num_gpus)]

        # 多进程并行
        mp.set_start_method('spawn', force=True)
        processes = []
        for i in range(num_gpus):
            p = mp.Process(
                target=run_single_gpu,
                args=(
                    i,
                    model_path,
                    data_chunks[i],
                    language,
                    batch_size,
                    max_new_tokens,
                    gpu_memory_utilization,
                    temp_files[i],
                    backend,
                )
            )
            processes.append(p)
            p.start()

        # 等待所有进程完成
        for p in processes:
            p.join()

        # 合并所有临时txt文件到最终结果文件
        total_lines = 0
        with open(results_file, 'w', encoding='utf-8') as outfile:
            for i, temp_file in enumerate(temp_files):
                if not os.path.exists(temp_file):
                    print(f"警告: GPU {i} 的结果文件不存在")
                    continue
                with open(temp_file, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                    outfile.write(content)
                    total_lines += content.count('\n')
                # 清理临时文件
                os.remove(temp_file)
        
        print(f"\n详细结果已保存到: {results_file}")
        print(f"总识别样本数: {total_lines}")

    # 单卡推理
    else:
        # 指定 GPU
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        print(f"加载模型: {model_path}")
        print(f"后端: {backend}")

        if backend == "vllm":
            from qwen_asr import Qwen3ASRModel
            model = Qwen3ASRModel.LLM(
                model=model_path,
                gpu_memory_utilization=gpu_memory_utilization,
                max_inference_batch_size=batch_size,
                max_new_tokens=max_new_tokens,
            )
        else:
            from qwen_asr import Qwen3ASRModel
            model = Qwen3ASRModel.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                device_map="cuda:0",
                max_inference_batch_size=batch_size,
                max_new_tokens=max_new_tokens,
            )

        # 批量推理并直接写入txt
        with open(results_file, 'w', encoding='utf-8') as f:
            for i in tqdm(range(0, len(all_test_data), batch_size), desc="推理中"):
                batch = all_test_data[i:i + batch_size]
                audio_paths = [item['audio'] for item in batch]
                keys = [item['key'] for item in batch]

                # 推理
                try:
                    predictions = model.transcribe(
                        audio=audio_paths,
                        language=language,
                        # context="你是一个拥有超高精度的语音识别引擎。专属名词列表如下:[岚图]。请根据音频内容进行识别，当遇到音素类似的词汇时，必须优先匹配列表中的专属名词，而不是通用词汇。",
                    )
                except Exception as e:
                    print(f"批次 {i} 推理失败: {e}")
                    continue

                # 写入结果：文件id + tab + 识别结果
                for j, pred in enumerate(predictions):
                    hyp = pred.text.strip()
                    key = keys[j]
                    f.write(f"{key}\t{hyp}\n")

        print(f"\n详细结果已保存到: {results_file}")

    elapsed_time = time.time() - start_time

    # 打印结果
    print("\n" + "=" * 50)
    print("测试完成")
    print("=" * 50)
    print(f"模型路径: {model_path}")
    print(f"使用 GPU 数: {num_gpus}")
    print(f"测试样本数: {total_samples}")
    print(f"总耗时: {elapsed_time:.2f} 秒")
    if total_samples > 0:
        print(f"平均每条耗时: {elapsed_time/total_samples*1000:.2f} 毫秒")
        print(f"吞吐量: {total_samples/elapsed_time:.1f} 条/秒")
    print(f"输出文件: {results_file}")

    # 保存汇总统计（JSON格式，仅包含统计信息）
    summary = {
        "model_path": model_path,
        "backend": backend,
        "num_gpus": num_gpus,
        "test_file": test_file,
        "num_samples": total_samples,
        "total_time_seconds": elapsed_time,
        "avg_time_ms": elapsed_time / total_samples * 1000 if total_samples > 0 else 0,
        "throughput": total_samples / elapsed_time if elapsed_time > 0 else 0,
        "output_format": "txt (file_id\\ttranscription)",
    }
    summary_file = os.path.join(output_dir, "summary.json")
    save_summary(summary, summary_file)
    print(f"汇总统计已保存到: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="批量测试 ASR 模型（输出txt格式：文件id+\t+识别结果）")

    parser.add_argument(
        "--model_path",
        type=str,
        # default="/cfs/data/private/hubk/Qwen3-ASR/Qwen/Qwen3-ASR-1___7B",
        default="/cfs/data/private/WangYaoChi/model/qwen3-asr-ctc-joint-3/checkpoint-18744",
        help="模型路径 (HuggingFace repo ID 或本地路径"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="测试数据文件路径 (SCP 格式: key\\tpath 或 JSONL 格式)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./test_output",
        help="输出目录 (默认: ./joint_llm_result/1/chuan_2)"
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["vllm", "transformers"],
        default="transformers",
        help="推理后端 (默认: vllm)"
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="指定语言 (如 Chinese, English)，不指定则自动检测"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="批量大小 (默认: 64)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="最大生成 token 数 (默认: 256)"
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.8,
        help="GPU 显存利用率 (默认: 0.8, 仅 vLLM 后端)"
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="从第几条数据开始测试 (默认: 0)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="最大测试样本数，-1 表示全部 (默认: -1)"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=8,
        help="使用 GPU 数量，多卡并行推理 (默认: 4)"
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=None,
        help="指定使用哪张 GPU (仅单卡模式有效，默认: None 即使用全部可见 GPU)"
    )

    args = parser.parse_args()

    run_test(
        model_path=args.model_path,
        test_file=args.test_file,
        output_dir=args.output_dir,
        backend=args.backend,
        language=args.language,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        start_idx=args.start_idx,
        max_samples=args.max_samples,
        num_gpus=args.num_gpus,
        gpu_id=args.gpu_id,
    )


if __name__ == "__main__":
    main()
