import librosa
import numpy as np
import torch

from qwen_joint.joint_model import Qwen3ASRJointModel

ckpt = "/cfs/data/private/WangYaoChi/model/qwen3-asr-rnnt-2/checkpoint-13283"
wav_path = "/root/hotword_data/爱奇艺播放淮水竹亭第十一集.wav"

model = Qwen3ASRJointModel.from_pretrained(
    ckpt,
    dtype=torch.bfloat16,
    device_map="cuda:0",
)
model.eval()

wav, sr = librosa.load(wav_path, sr=16000, mono=True)

chunk_size_sec = 0.64
chunk_samples = int(chunk_size_sec * 16000)

audio_accum = np.zeros((0,), dtype=np.float32)

for start in range(0, len(wav), chunk_samples):
    chunk = wav[start:start + chunk_samples]
    audio_accum = np.concatenate([audio_accum, chunk], axis=0)

    text = model.transcribe_rnnt(
        audio_accum,
        max_symbols_per_step=3,
        rnnt_decode_strategy="cached",
        aux_encoder_batch_size=1,
    )

    cur_sec = len(audio_accum) / 16000
    print(f"[{cur_sec:.1f}s] {text}")
