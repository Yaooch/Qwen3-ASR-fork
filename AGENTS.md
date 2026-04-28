# Qwen3-ASR Project Context

## Project Overview

**Qwen3-ASR** is a powerful all-in-one automatic speech recognition (ASR) system developed by the Alibaba Qwen team. It supports language identification and speech recognition for 52 languages and dialects, including 30 languages and 22 Chinese dialects.

### Key Features
- **Multi-language Support**: 30 languages + 22 Chinese dialects
- **Two Model Sizes**: Qwen3-ASR-1.7B and Qwen3-ASR-0.6B
- **Dual Backend**: Transformers backend and vLLM backend for inference
- **Streaming Support**: Real-time streaming ASR inference (vLLM backend only)
- **Forced Alignment**: Qwen3-ForcedAligner-0.6B for word/character-level timestamps
- **Timestamp Output**: Optional timestamp prediction via forced aligner

### Model Architecture
- Based on Qwen3-Omni foundation model
- Encoder-Decoder architecture with audio encoder and text decoder
- Supports both offline and streaming inference with a single model

## Project Structure

```
Qwen3-ASR/
├── qwen_asr/                    # Main Python package
│   ├── __init__.py              # Package entry, exports Qwen3ASRModel, Qwen3ForcedAligner
│   ├── cli/                     # Command-line interface tools
│   │   ├── demo.py              # Gradio web UI demo
│   │   ├── demo_streaming.py    # Streaming web demo
│   │   └── serve.py             # vLLM server wrapper
│   ├── core/                    # Core model implementations
│   │   ├── transformers_backend/  # HuggingFace Transformers implementation
│   │   │   ├── configuration_qwen3_asr.py
│   │   │   ├── modeling_qwen3_asr.py
│   │   │   └── processing_qwen3_asr.py
│   │   └── vllm_backend/        # vLLM implementation
│   │       └── qwen3_asr.py
│   └── inference/               # High-level inference APIs
│       ├── qwen3_asr.py         # Qwen3ASRModel main class
│       ├── qwen3_forced_aligner.py  # Qwen3ForcedAligner class
│       ├── utils.py             # Audio processing utilities
│       └── assets/              # Language dictionaries (Korean)
├── examples/                    # Usage examples
│   ├── example_qwen3_asr_transformers.py
│   ├── example_qwen3_asr_vllm.py
│   ├── example_qwen3_asr_vllm_streaming.py
│   └── example_qwen3_forced_aligner.py
├── finetuning/                  # Fine-tuning scripts
│   ├── qwen3_asr_sft.py
│   └── README.md
├── docker/                      # Docker configuration
└── assets/                      # Documentation assets
```

## Building and Running

### Installation

```bash
# Minimal installation (Transformers backend only)
pip install -U qwen-asr

# Full installation with vLLM backend
pip install -U qwen-asr[vllm]

# Install from source
git clone https://github.com/QwenLM/Qwen3-ASR.git
cd Qwen3-ASR
pip install -e .

# Recommended: FlashAttention 2 for better performance
pip install -U flash-attn --no-build-isolation
```

### Environment Requirements
- Python >= 3.9, recommended 3.12
- CUDA-capable GPU (for inference)
- FlashAttention 2 compatible hardware (optional but recommended)

### Quick Inference Commands

**Transformers Backend:**
```python
from qwen_asr import Qwen3ASRModel
import torch

model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
    max_inference_batch_size=32,
    max_new_tokens=256,
)
results = model.transcribe(audio="path/to/audio.wav")
```

**vLLM Backend (faster):**
```python
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.LLM(
    model="Qwen/Qwen3-ASR-1.7B",
    gpu_memory_utilization=0.7,
    max_inference_batch_size=128,
    max_new_tokens=4096,
)
results = model.transcribe(audio="path/to/audio.wav")
```

### CLI Commands

```bash
# Launch Gradio demo
qwen-asr-demo --asr-checkpoint Qwen/Qwen3-ASR-1.7B --backend transformers

# Launch streaming demo
qwen-asr-demo-streaming --asr-model-path Qwen/Qwen3-ASR-1.7B

# Start vLLM server
qwen-asr-serve Qwen/Qwen3-ASR-1.7B --gpu-memory-utilization 0.8 --host 0.0.0.0 --port 8000
```

### Fine-tuning

```bash
# Single GPU
python finetuning/qwen3_asr_sft.py \
  --model_path Qwen/Qwen3-ASR-1.7B \
  --train_file ./train.jsonl \
  --output_dir ./output \
  --batch_size 32 --grad_acc 4 --lr 2e-5 --epochs 1

# Multi-GPU
torchrun --nproc_per_node=2 finetuning/qwen3_asr_sft.py \
  --model_path Qwen/Qwen3-ASR-1.7B \
  --train_file ./train.jsonl \
  --output_dir ./output
```

## Key APIs

### Qwen3ASRModel

Main inference wrapper supporting both backends:

| Method | Description |
|--------|-------------|
| `from_pretrained()` | Initialize with Transformers backend |
| `LLM()` | Initialize with vLLM backend |
| `transcribe()` | Transcribe audio(s) with optional timestamps |
| `init_streaming_state()` | Initialize streaming state (vLLM only) |
| `streaming_transcribe()` | Streaming ASR decode step |
| `finish_streaming_transcribe()` | Finalize streaming |
| `get_supported_languages()` | List supported languages |

### Qwen3ForcedAligner

Forced alignment for timestamps:

| Method | Description |
|--------|-------------|
| `from_pretrained()` | Load model |
| `align()` | Align text-speech pairs, return timestamps |
| `get_supported_languages()` | List supported languages (11 languages) |

### Audio Input Formats

Supported audio inputs:
- Local file path: `"/path/to/audio.wav"`
- URL: `"https://example.com/audio.wav"`
- Base64: `"data:audio/wav;base64,..."`
- NumPy array: `(np.ndarray, sample_rate)`

## Supported Languages

### ASR Models (52 languages/dialects)
**Languages:** Chinese, English, Cantonese, Arabic, German, French, Spanish, Portuguese, Indonesian, Italian, Korean, Russian, Thai, Vietnamese, Japanese, Turkish, Hindi, Malay, Dutch, Swedish, Danish, Finnish, Polish, Czech, Filipino, Persian, Greek, Romanian, Hungarian, Macedonian

**Chinese Dialects:** Anhui, Dongbei, Fujian, Gansu, Guizhou, Hebei, Henan, Hubei, Hunan, Jiangxi, Ningxia, Shandong, Shaanxi, Shanxi, Sichuan, Tianjin, Yunnan, Zhejiang, Cantonese (HK/GD), Wu, Minnan

### Forced Aligner (11 languages)
Chinese, English, Cantonese, French, German, Italian, Japanese, Korean, Portuguese, Russian, Spanish

## Development Conventions

### Code Style
- Python 3.9+ compatibility
- Type hints with `typing` module
- Dataclasses for structured data
- Docstrings in Google style

### Key Constants (utils.py)
- `SAMPLE_RATE = 16000` - All audio resampled to 16kHz
- `MAX_ASR_INPUT_SECONDS = 1200` - Max audio length for ASR
- `MAX_FORCE_ALIGN_INPUT_SECONDS = 180` - Max audio for forced aligner
- `MIN_ASR_INPUT_SECONDS = 0.5` - Minimum audio length

### Audio Processing Pipeline
1. Load audio (URL/local/base64/numpy)
2. Convert to mono
3. Resample to 16kHz
4. Normalize to float32 in [-1, 1]
5. Chunk long audio at low-energy boundaries

### Output Parsing
- Format: `language {Language}<asr_text>{Transcription}`
- Use `parse_asr_output()` to extract language and text
- Automatic repetition detection and fixing

## Dependencies

Core dependencies (from pyproject.toml):
- `transformers==4.57.6`
- `accelerate==1.12.0`
- `qwen-omni-utils`
- `librosa`, `soundfile`, `sox`
- `gradio`, `flask`
- `nagisa`, `soynlp` (for Japanese/Korean tokenization)

Optional:
- `vllm==0.14.0` (for vLLM backend)
- `flash-attn` (recommended for performance)

## Important Notes

1. **vLLM Backend**: Always wrap code under `if __name__ == '__main__':` to avoid multiprocessing errors
2. **Streaming**: Only available with vLLM backend, no timestamps support
3. **Timestamps**: Requires initializing with `forced_aligner` parameter
4. **Long Audio**: Automatically chunked at low-energy boundaries
5. **Language Detection**: Set `language=None` for automatic detection, or specify to force output
