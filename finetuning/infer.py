# infer.py
import argparse
import json
import multiprocessing as mp
import os
from typing import Dict, List

import torch
from datetime import datetime

from qwen_asr import Qwen3ASRModel
from qwen_asr.joint import Qwen3ASRJointModel

def log(msg: str):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen3-ASR + CTC/RNNT 推理脚本，支持 CTC / RNNT / LLM / Joint 模式，多卡并行，scp 输入。"
    )

    # 模型与输入输出
    parser.add_argument("--ckpt", required=True, help="Joint checkpoint 目录")
    parser.add_argument("--input_scp", required=True, help="输入 scp 文件路径")
    parser.add_argument("--output_dir", required=True, help="推理输出目录")

    # 推理模式
    parser.add_argument(
        "--mode",
        choices=["ctc", "rnnt", "llm", "joint"],
        default="joint",
        help="推理模式：ctc / rnnt / llm / joint",
    )

    # GPU 与 batch 配置
    parser.add_argument(
        "--gpu_ids",
        default="0",
        help="使用的 GPU id，单卡如 0，多卡如 0,1,2,3",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="每个进程的 batch size")
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="模型加载精度",
    )

    # 语言 / prompt
    parser.add_argument(
        "--language",
        default=None,
        help="默认语种；如果 scp 每行已带语种，这里可不传",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="额外 prompt，仅 llm / joint 模式有效",
    )

    # hotword 仅 joint 模式可用
    parser.add_argument("--hotword_file", default=None, help="热词文件，每行一个热词")
    parser.add_argument("--hotword_topk", type=int, default=10, help="召回热词数量")
    parser.add_argument(
        "--hotword_pinyin_style",
        choices=["normal", "tone3"],
        default="normal",
        help="热词拼音召回风格：normal 不带调，tone3 数字声调",
    )
    parser.add_argument(
        "--no_aux_in_prompt",
        action="store_true",
        help="joint 模式下不把 CTC/RNNT 粗识别结果注入 prompt。当前默认就是不注入，保留该参数兼容旧脚本。",
    )
    parser.add_argument(
        "--aux_in_prompt",
        action="store_true",
        help="joint 模式下把 CTC/RNNT 粗识别结果注入 prompt。",
    )
    parser.add_argument(
        "--rnnt_max_symbols_per_step",
        type=int,
        default=5,
        help="RNNT greedy decode 每个 encoder frame 最多吐出的 token 数，调小可加速但可能影响准确率",
    )
    parser.add_argument(
        "--aux_encoder_batch_size",
        type=int,
        default=1,
        help="CTC/RNNT 辅助头 audio encoder micro-batch；1 最稳，调大可加速但可能影响结果",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="RNNT/joint 模式下使用 chunk-wise RNNT 流式粗识别。",
    )
    parser.add_argument("--stream_chunk_sec", type=float, default=0.64)
    parser.add_argument("--stream_left_context_sec", type=float, default=0.64)
    parser.add_argument("--stream_right_context_sec", type=float, default=0.07)
    parser.add_argument("--stream_first_chunk_left_pad_sec", type=float, default=0.0)
    parser.add_argument("--stream_window_batch_size", type=int, default=4)
    parser.add_argument("--stream_window_encoder_batch_size", type=int, default=1)
    return parser.parse_args()


def get_torch_dtype(dtype_name: str):
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def parse_gpu_ids(gpu_ids: str) -> List[int]:
    ids = []
    for part in gpu_ids.split(","):
        part = part.strip()
        if part:
            ids.append(int(part))
    if not ids:
        raise ValueError("gpu_ids 不能为空")
    return ids


def load_scp(path: str, default_language: str = None) -> List[Dict]:
    """读取 scp 文件。

    支持两种格式：
    1. utt_id audio_path
    2. utt_id audio_path language

    输出统一为：
    [
        {
            "utt_id": "...",
            "audio": "...",
            "language": "zh"
        },
        ...
    ]
    """
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"scp 第 {line_no} 行格式错误：{line}")

            utt_id = parts[0]
            audio_path = parts[1]
            language = parts[2] if len(parts) >= 3 else default_language

            items.append(
                {
                    "utt_id": utt_id,
                    "audio": audio_path,
                    "language": language,
                }
            )
    return items


def shard_items(items: List[Dict], world_size: int) -> List[List[Dict]]:
    """把样本按轮询方式切分到不同进程。"""
    shards = [[] for _ in range(world_size)]
    for idx, item in enumerate(items):
        shards[idx % world_size].append(item)
    return shards


def batchify(items: List[Dict], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i: i + batch_size]


def make_hotword_retriever(args):
    """构造热词检索器。未提供热词文件时返回 None。"""
    if not args.hotword_file:
        return None

    from qwen_asr.joint import HotwordRetriever
    return HotwordRetriever.from_file(
        args.hotword_file,
        pinyin_style=args.hotword_pinyin_style,
    )

def extract_text_and_language(output):
    """从模型输出里尽量提取 text 和 language。

    兼容以下几种返回类型：
    1. str
    2. dict，字段中含 text / language
    3. ASRTranscription 对象
    4. list[ASRTranscription]
    5. list[str]
    """
    if output is None:
        return "", None

    # 1) 纯字符串
    if isinstance(output, str):
        return output, None

    # 2) dict
    if isinstance(output, dict):
        text = (
            output.get("text")
            or output.get("prediction")
            or output.get("transcription")
            or ""
        )
        language = output.get("language")
        return text, language

    # 3) list
    if isinstance(output, list):
        if len(output) == 0:
            return "", None

        # 如果是字符串列表，优先取第一个或拼接
        if isinstance(output[0], str):
            if len(output) == 1:
                return output[0], None
            return " ".join(output), None

        # 如果是 ASRTranscription 之类的对象列表，优先取第一个
        first = output[0]

        # dataclass / 普通对象：有 text / language 属性
        text = getattr(first, "text", None)
        language = getattr(first, "language", None)
        if text is not None:
            return text, language

        # 兜底：如果 list 里还是 dict
        if isinstance(first, dict):
            text = (
                first.get("text")
                or first.get("prediction")
                or first.get("transcription")
                or ""
            )
            language = first.get("language")
            return text, language

        # 最后兜底
        return str(first), None

    # 4) 单个对象，例如 ASRTranscription
    text = getattr(output, "text", None)
    language = getattr(output, "language", None)
    if text is not None:
        return text, language

    # 5) 最后兜底
    return str(output), None


def normalize_text_output(output) -> str:
    text, _ = extract_text_and_language(output)
    return text



def normalize_batch_outputs(outputs, batch_size: int):
    """把模型输出统一成长度为 batch_size 的 list。"""
    if isinstance(outputs, list):
        if len(outputs) != batch_size:
            raise ValueError(
                f"模型输出 batch 大小不匹配：expect {batch_size}, got {len(outputs)}"
            )
        return outputs
    return [outputs]


def is_joint_checkpoint(model_path: str) -> bool:
    """判断是否为带 CTC/RNNT 辅助头的联合 checkpoint。"""
    return os.path.exists(os.path.join(model_path, "ctc_config.json"))


def run_batch_infer(model, batch: List[Dict], args, hotword_retriever=None):
    """对一个 batch 执行推理，并返回结构化结果。"""
    audios = [x["audio"] for x in batch]
    languages = [x.get("language") for x in batch]
    aux_loss_type = getattr(model, "aux_loss_type", None)

    if args.mode == "ctc":
        if args.stream:
            raw_outputs = model.transcribe_ctc_streaming(
                audios,
                chunk_sec=args.stream_chunk_sec,
                left_context_sec=args.stream_left_context_sec,
                right_context_sec=args.stream_right_context_sec,
                first_chunk_left_pad_sec=args.stream_first_chunk_left_pad_sec,
                window_batch_size=args.stream_window_batch_size,
                window_encoder_batch_size=args.stream_window_encoder_batch_size,
            )
        else:
            raw_outputs = model.transcribe_ctc(
                audios,
                aux_encoder_batch_size=args.aux_encoder_batch_size,
            )
    elif args.mode == "rnnt":
        if args.stream:
            raw_outputs = model.transcribe_rnnt_streaming(
                audios,
                max_symbols_per_step=args.rnnt_max_symbols_per_step,
                chunk_sec=args.stream_chunk_sec,
                left_context_sec=args.stream_left_context_sec,
                right_context_sec=args.stream_right_context_sec,
                first_chunk_left_pad_sec=args.stream_first_chunk_left_pad_sec,
                window_batch_size=args.stream_window_batch_size,
                window_encoder_batch_size=args.stream_window_encoder_batch_size,
            )
        else:
            raw_outputs = model.transcribe_rnnt(
                audios,
                max_symbols_per_step=args.rnnt_max_symbols_per_step,
                aux_encoder_batch_size=args.aux_encoder_batch_size,
            )
    elif args.mode == "llm":
        if hasattr(model, "transcribe_llm"):
            if args.stream:
                raw_outputs = model.transcribe_llm_streaming(
                    audios,
                    language=languages,
                    context=args.prompt,
                    chunk_sec=args.stream_chunk_sec,
                    left_context_sec=args.stream_left_context_sec,
                    right_context_sec=args.stream_right_context_sec,
                    first_chunk_left_pad_sec=args.stream_first_chunk_left_pad_sec,
                    window_batch_size=args.stream_window_batch_size,
                    window_encoder_batch_size=args.stream_window_encoder_batch_size,
                )
            else:
                raw_outputs = model.transcribe_llm(
                    audios,
                    language=languages,
                    context=args.prompt,
                )
        else:
            if args.stream:
                raise RuntimeError("原始 Qwen3-ASR 模型暂不支持当前脚本的 --stream llm 推理。")
            raw_outputs = model.transcribe(
                audios,
                language=languages,
                context=args.prompt or "",
            )
    else:
        raw_outputs = model.transcribe_joint(
            audios,
            language=languages,
            prompt=args.prompt,
            hotword_retriever=hotword_retriever,
            hotword_topk=args.hotword_topk,
            inject_aux_into_prompt=bool(args.aux_in_prompt and not args.no_aux_in_prompt),
            aux_max_symbols_per_step=args.rnnt_max_symbols_per_step,
            aux_encoder_batch_size=args.aux_encoder_batch_size,
            stream_aux=args.stream,
            stream_chunk_sec=args.stream_chunk_sec,
            stream_left_context_sec=args.stream_left_context_sec,
            stream_right_context_sec=args.stream_right_context_sec,
            stream_first_chunk_left_pad_sec=args.stream_first_chunk_left_pad_sec,
            stream_window_batch_size=args.stream_window_batch_size,
            stream_window_encoder_batch_size=args.stream_window_encoder_batch_size,
        )

    raw_outputs = normalize_batch_outputs(raw_outputs, len(batch))

    records = []
    for item, raw_out in zip(batch, raw_outputs):
        input_lang = item.get("language")

        if args.mode == "joint":
            if isinstance(raw_out, dict):
                # text/llm_refined_text 已经是纯文本，language 应该直接从 raw_out["language"] 里拿
                llm_text = raw_out.get("llm_refined_text") or raw_out.get("text") or ""
                aux_text = raw_out.get("aux_text") or raw_out.get("aux_stream_text", "")
                final_lang = raw_out.get("language") or input_lang or "unknown"

                record = {
                    "utt_id": item["utt_id"],
                    "audio": item["audio"],
                    "input_language": input_lang,
                    "language": final_lang,
                    "llm_language": final_lang,
                    "mode": "joint",
                    "stream": bool(args.stream),
                    "aux_loss_type": aux_loss_type,
                    "text": llm_text,
                    "aux_text": aux_text,
                    "aux_stream_text": aux_text,
                    "ctc_text": raw_out.get("ctc_text", aux_text if aux_loss_type == "ctc" else ""),
                    "rnnt_text": raw_out.get("rnnt_text", aux_text if aux_loss_type == "rnnt" else ""),
                    "llm_text": llm_text,
                    "llm_refined_text": llm_text,
                    "hotwords": raw_out.get("hotwords", []),
                    "prompt": raw_out.get("prompt"),
                }

            else:
                final_text, final_lang = extract_text_and_language(raw_out)
                record = {
                    "utt_id": item["utt_id"],
                    "audio": item["audio"],
                    "input_language": input_lang,
                    "language": final_lang or input_lang or "unknown",
                    "llm_language": final_lang or input_lang or "unknown",
                    "mode": "joint",
                    "stream": bool(args.stream),
                    "aux_loss_type": aux_loss_type,
                    "text": final_text,
                    "aux_text": "",
                    "aux_stream_text": "",
                    "ctc_text": "",
                    "rnnt_text": "",
                    "llm_text": final_text,
                    "llm_refined_text": final_text,
                    "hotwords": [],
                    "prompt": None,
                }

        elif args.mode == "llm":
            if isinstance(raw_out, dict):
                final_text = raw_out.get("text", "")
                final_lang = raw_out.get("language") or input_lang or "unknown"
            else:
                final_text, final_lang = extract_text_and_language(raw_out)
                final_lang = final_lang or input_lang or "unknown"

            record = {
                "utt_id": item["utt_id"],
                "audio": item["audio"],
                "input_language": input_lang,
                "language": final_lang,
                "llm_language": final_lang,
                "mode": "llm",
                "stream": bool(args.stream),
                "text": final_text,
                "llm_text": final_text,
            }

        else:
            final_text, final_lang = extract_text_and_language(raw_out)
            final_lang = final_lang or input_lang or "unknown"
            record = {
                "utt_id": item["utt_id"],
                "audio": item["audio"],
                "input_language": input_lang,
                "language": final_lang,
                "mode": args.mode,
                "stream": bool(args.stream and args.mode in ("ctc", "rnnt")),
                "aux_loss_type": aux_loss_type if args.mode in ("ctc", "rnnt") else None,
                "text": final_text,
                "aux_text": final_text if args.mode in ("ctc", "rnnt") else "",
                "ctc_text": final_text if args.mode == "ctc" else "",
                "rnnt_text": final_text if args.mode == "rnnt" else "",
            }


        records.append(record)


    return records


def worker_main(rank: int, gpu_id: int, shard: List[Dict], args, tmp_output_path: str):
    import time
    from datetime import datetime

    def log(msg: str):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}] {msg}", flush=True)

    if not torch.cuda.is_available():
        raise RuntimeError("当前环境不可用 CUDA，但你传了多卡推理。")

    # 关键修复：
    # 不再依赖 CUDA_VISIBLE_DEVICES 做重映射，
    # 而是直接绑定到真实物理 GPU。
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"

    dtype = get_torch_dtype(args.dtype)

    start_time = time.time()
    log(f"进程{rank}启动：GPU {gpu_id}，样本 {len(shard)}")

    log(f"进程{rank}加载模型")
    if is_joint_checkpoint(args.ckpt):
        model = Qwen3ASRJointModel.from_pretrained(
            args.ckpt,
            dtype=dtype,
            device_map=None,
        )
        model = model.to(device)
        model.eval()
    else:
        if args.mode != "llm":
            raise RuntimeError(
                "当前模型目录不是联合 checkpoint，缺少 ctc_config.json。"
                "原始 Qwen3-ASR 只支持 --mode llm；ctc/rnnt/joint 需要训练后的联合模型。"
            )
        if args.stream:
            raise RuntimeError("原始 Qwen3-ASR 模型暂不支持当前脚本的 --stream llm 推理。")

        model = Qwen3ASRModel.from_pretrained(
            args.ckpt,
            dtype=dtype,
            device_map=None,
            max_inference_batch_size=args.batch_size,
        )
        model.model = model.model.to(device)
        model.model.eval()
        model.device = torch.device(device)
    log(f"进程{rank}模型加载完成")

    aux_loss_type = getattr(model, "aux_loss_type", None)
    if args.mode in ("ctc", "rnnt") and aux_loss_type != args.mode:
        raise RuntimeError(
            f"当前 checkpoint 的 aux_loss_type={aux_loss_type!r}，"
            f"不能使用 --mode {args.mode}。请改用 --mode {aux_loss_type} 或 --mode joint。"
        )

    hotword_retriever = None
    if args.mode == "joint" and args.hotword_file:
        hotword_retriever = make_hotword_retriever(args)

    all_records = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(batchify(shard, args.batch_size), 1):
            batch_records = run_batch_infer(
                model=model,
                batch=batch,
                args=args,
                hotword_retriever=hotword_retriever,
            )
            all_records.extend(batch_records)

    with open(tmp_output_path, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    elapsed = time.time() - start_time
    log(f"进程{rank}完成，用时 {elapsed:.2f}s")



def result_specs(records: List[Dict]):
    """根据实际记录决定需要写哪些结果文件。"""
    specs = []
    has_llm = any(
        r.get("mode") in ("llm", "joint") or "llm_text" in r or "llm_refined_text" in r
        for r in records
    )
    has_ctc = any(
        r.get("mode") == "ctc" or r.get("aux_loss_type") == "ctc" or bool(r.get("ctc_text"))
        for r in records
    )
    has_rnnt = any(
        r.get("mode") == "rnnt" or r.get("aux_loss_type") == "rnnt" or bool(r.get("rnnt_text"))
        for r in records
    )

    if has_ctc:
        specs.append(("ctc", "results_ctc.txt", "ctc_text"))
    if has_rnnt:
        specs.append(("rnnt", "results_rnnt.txt", "rnnt_text"))
    if has_llm:
        specs.append(("llm", "results_llm.txt", "llm_text"))

    if not specs:
        specs.append(("text", "results_text.txt", "text"))
    return specs


def merge_outputs(tmp_files: List[str], output_dir: str):
    """合并各个 worker 的中间结果，并输出最终文件。"""
    os.makedirs(output_dir, exist_ok=True)
    details_dir = os.path.join(output_dir, "details")
    os.makedirs(details_dir, exist_ok=True)

    detail_jsonl = os.path.join(details_dir, "results_detail.jsonl")

    all_records = []
    for path in tmp_files:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_records.append(json.loads(line))

    # 为了输出稳定，按 utt_id 排序
    all_records.sort(key=lambda x: x["utt_id"])

    result_paths = {}
    result_files = []
    for name, filename, field in result_specs(all_records):
        path = os.path.join(output_dir, filename)
        result_paths[name] = path
        result_files.append((name, field, open(path, "w", encoding="utf-8")))

    try:
        with open(detail_jsonl, "w", encoding="utf-8") as f_detail:
            for record in all_records:
                for _name, field, f_txt in result_files:
                    text = record.get(field)
                    if text is None and field == "llm_text":
                        text = record.get("llm_refined_text")
                    if text is None:
                        text = record.get("text", "")
                    language = record.get("language") or record.get("input_language") or "unknown"
                    f_txt.write(f'{record["utt_id"]}\t{text or ""}\t{language}\n')
                f_detail.write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        for _name, _field, f_txt in result_files:
            f_txt.close()

    # 清理中间文件
    for path in tmp_files:
        if os.path.exists(path):
            os.remove(path)

    return result_paths, detail_jsonl


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    items = load_scp(args.input_scp, default_language=args.language)
    if not items:
        raise ValueError(f"scp 文件中没有可用样本：{args.input_scp}")

    gpu_ids = parse_gpu_ids(args.gpu_ids)
    world_size = len(gpu_ids)
    shards = shard_items(items, world_size)

    tmp_files = [
        os.path.join(args.output_dir, f"tmp_rank{rank}.jsonl")
        for rank in range(world_size)
    ]

    print("=" * 80)
    print("推理配置")
    print("=" * 80)
    print(f"模型：{args.ckpt}")
    print(f"模式：{args.mode}")
    print(f"输入：{args.input_scp}")
    print(f"输出：{args.output_dir}")
    print(f"GPU：{gpu_ids}")
    print(f"批量：{args.batch_size}")
    print(f"精度：{args.dtype}")
    print(f"RNNT每帧上限：{args.rnnt_max_symbols_per_step}")
    print(f"辅助编码批量：{args.aux_encoder_batch_size}")
    print(f"流式：{int(args.stream)}")
    if args.stream:
        print(
            "流式配置："
            f"chunk={args.stream_chunk_sec}, left={args.stream_left_context_sec}, "
            f"right={args.stream_right_context_sec}, first_pad={args.stream_first_chunk_left_pad_sec}, "
            f"win_batch={args.stream_window_batch_size}, enc_batch={args.stream_window_encoder_batch_size}"
        )
    print(f"样本数：{len(items)}")
    print("=" * 80)

    if world_size == 1:
        worker_main(
            rank=0,
            gpu_id=gpu_ids[0],
            shard=shards[0],
            args=args,
            tmp_output_path=tmp_files[0],
        )
    else:
        ctx = mp.get_context("spawn")
        procs = []

        for rank, gpu_id in enumerate(gpu_ids):
            p = ctx.Process(
                target=worker_main,
                args=(rank, gpu_id, shards[rank], args, tmp_files[rank]),
            )
            p.start()
            procs.append(p)

        for p in procs:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"子进程失败，exitcode={p.exitcode}")

    result_paths, detail_jsonl = merge_outputs(tmp_files, args.output_dir)

    print("=" * 80)
    print("推理完成")
    print("=" * 80)
    for name, path in result_paths.items():
        print(f"结果[{name}]：{path}")
    print(f"明细：{detail_jsonl}")
    print("=" * 80)


if __name__ == "__main__":
    main()
