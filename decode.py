import torch
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    "/cfs/data/private/WangYaoChi/model/qwen3-asr-ctc-joint-2/checkpoint-5000/",
    dtype=torch.bfloat16,
    device_map="cuda:1",
    # attn_implementation="flash_attention_2",
    max_inference_batch_size=32, # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
    max_new_tokens=256, # Maximum number of tokens to generate. Set a larger value for long audio input.
)

results = model.transcribe(
    audio=[
        # "/root/hotword_data/给蒋善聪打电话.wav",
        # "/root/hotword_data/明天应臻奕要回北京请大家吃饭.wav",
        "/root/hotword_data/爱奇艺播放淮水竹亭第十一集.wav"
        # "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav",
        ],
    language=None, # set "English" to force the language
)

for r in results:
    print(r.language, r.text)