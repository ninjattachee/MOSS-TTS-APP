import importlib.util
from pathlib import Path
import torch
import torchaudio
from transformers import AutoModel, AutoProcessor, GenerationConfig
# Disable the broken cuDNN SDPA backend
torch.backends.cuda.enable_cudnn_sdp(False)
# Keep these enabled as fallbacks
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

class DelayGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layers = kwargs.get("layers", [{} for _ in range(32)])
        self.do_samples = kwargs.get("do_samples", None)
        self.n_vq_for_inference = 32
        
def initial_config(tokenizer, model_name_or_path):
    generation_config = DelayGenerationConfig.from_pretrained(model_name_or_path)
    generation_config.pad_token_id = tokenizer.pad_token_id
    generation_config.eos_token_id = 151653
    generation_config.max_new_tokens = 1000000
    generation_config.temperature = 1.0
    generation_config.top_p = 0.95
    generation_config.top_k = 100
    generation_config.repetition_penalty = 1.1
    generation_config.use_cache = True
    generation_config.do_sample = False
    return generation_config


pretrained_model_name_or_path = "OpenMOSS-Team/MOSS-TTS-Local-Transformer"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

def resolve_attn_implementation() -> str:
    # Prefer FlashAttention 2 when package + device conditions are met.
    if (
        device == "cuda"
        and importlib.util.find_spec("flash_attn") is not None
        and dtype in {torch.float16, torch.bfloat16}
    ):
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return "flash_attention_2"

    # CUDA fallback: use PyTorch SDPA kernels.
    if device == "cuda":
        return "sdpa"

    # CPU fallback.
    return "eager"


attn_implementation = resolve_attn_implementation()
print(f"[INFO] Using attn_implementation={attn_implementation}")

processor = AutoProcessor.from_pretrained(
    pretrained_model_name_or_path,
    trust_remote_code=True,
)
processor.audio_tokenizer = processor.audio_tokenizer.to(device)

text_1 = """亲爱的你，
你好呀。

今天，我想用最认真、最温柔的声音，对你说一些重要的话。
这些话，像一颗小小的星星，希望能在你的心里慢慢发光。

首先，我想祝你——
每天都能平平安安、快快乐乐。

希望你早上醒来的时候，
窗外有光，屋子里很安静，
你的心是轻轻的，没有着急，也没有害怕。
"""
text_2 = """We stand on the threshold of the AI era.
Artificial intelligence is no longer just a concept in laboratories, but is entering every industry, every creative endeavor, and every decision. It has learned to see, hear, speak, and think, and is beginning to become an extension of human capabilities. AI is not about replacing humans, but about amplifying human creativity, making knowledge more equitable, more efficient, and allowing imagination to reach further. A new era, jointly shaped by humans and intelligent systems, has arrived."""
text_3 = "nin2 hao3，qing3 wen4 nin2 lai2 zi4 na3 zuo4 cheng2 shi4？"
text_4 = "nin2 hao3，qing4 wen3 nin2 lai2 zi4 na4 zuo3 cheng4 shi3？"
text_5 = "您好，请问您来自哪 zuo4 cheng2 shi4？"
text_6 = "/həloʊ, meɪ aɪ æsk wɪtʃ sɪti juː ɑːr frʌm?/"

# Use audio from ./assets/audio to avoid downloading from the cloud.
ref_audio_1 = "https://speech-demo.oss-cn-shanghai.aliyuncs.com/moss_tts_demo/tts_readme_demo/reference_zh.wav"
ref_audio_2 = "https://speech-demo.oss-cn-shanghai.aliyuncs.com/moss_tts_demo/tts_readme_demo/reference_en.m4a"

conversations = [
    # Direct TTS (no reference)
    [
        processor.build_user_message(text=text_1)
    ],
    [
        processor.build_user_message(text=text_2)
    ],
    # Pinyin or IPA input
    [
        processor.build_user_message(text=text_3)
    ],
    [
        processor.build_user_message(text=text_4)
    ],
    [
        processor.build_user_message(text=text_5)
    ],
    [
        processor.build_user_message(text=text_6)
    ],
    # Voice cloning (with reference)
    [
        processor.build_user_message(text=text_1, reference=[ref_audio_1])
    ],
    [
        processor.build_user_message(text=text_2, reference=[ref_audio_2])
    ],
]

model = AutoModel.from_pretrained(
    pretrained_model_name_or_path,
    trust_remote_code=True,
    attn_implementation=attn_implementation,
    torch_dtype=dtype,
).to(device)
model.eval()

generation_config = initial_config(processor.tokenizer, pretrained_model_name_or_path)
generation_config.n_vq_for_inference = model.channels - 1
generation_config.do_samples = [True] * model.channels
generation_config.layers = [
    {
        "repetition_penalty": 1.0, 
        "temperature": 1.5, 
        "top_p": 1.0, 
        "top_k": 50
    }
] + [ 
    {
        "repetition_penalty": 1.1, 
        "temperature": 1.0, 
        "top_p": 0.95,
        "top_k": 50
    }
] * (model.channels - 1) 

batch_size = 1

save_dir = Path(f"inference_root_moss_tts_local_transformer_generation")
save_dir.mkdir(exist_ok=True, parents=True)
sample_idx = 0
with torch.no_grad():
    for start in range(0, len(conversations), batch_size):
        batch_conversations = conversations[start : start + batch_size]
        batch = processor(batch_conversations, mode="generation")
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config
        )

        for message in processor.decode(outputs):
            audio = message.audio_codes_list[0]
            out_path = save_dir / f"sample{sample_idx}.wav"
            sample_idx += 1
            torchaudio.save(out_path, audio.unsqueeze(0), processor.model_config.sampling_rate)
