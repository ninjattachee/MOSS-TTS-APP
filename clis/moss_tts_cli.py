#!/usr/bin/env python3
"""MOSS-TTS CLI: Convert text to speech from the command line."""

import argparse
import importlib.util
import sys
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

DEFAULT_MODEL = "OpenMOSS-Team/MOSS-TTS-Local-Transformer"
DEFAULT_OUTPUT = "output.wav"


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


def resolve_attn_implementation(
    requested: str,
    device: torch.device,
    dtype: torch.dtype,
) -> str | None:
    requested_norm = (requested or "").strip().lower()
    if requested_norm in {"none"}:
        return None
    if requested_norm not in {"", "auto"}:
        return requested

    # Prefer FlashAttention 2 when package + device conditions are met.
    if (
        device.type == "cuda"
        and importlib.util.find_spec("flash_attn") is not None
        and dtype in {torch.float16, torch.bfloat16}
    ):
        major, _ = torch.cuda.get_device_capability(device)
        if major >= 8:
            return "flash_attention_2"

    # CUDA fallback: use PyTorch SDPA kernels.
    if device.type == "cuda":
        return "sdpa"

    # CPU fallback.
    return "eager"


def load_model_and_processor(
    model_path: str,
    device_str: str,
    attn_implementation: str,
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    attn_impl = resolve_attn_implementation(
        requested=attn_implementation,
        device=device,
        dtype=dtype,
    )
    if attn_impl is None:
        attn_impl = "eager"

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    processor.audio_tokenizer = processor.audio_tokenizer.to(device)

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "attn_implementation": attn_impl,
    }

    model = AutoModel.from_pretrained(model_path, **model_kwargs).to(device)
    model.eval()

    generation_config = initial_config(processor.tokenizer, model_path)
    generation_config.n_vq_for_inference = model.channels - 1
    generation_config.do_samples = [True] * model.channels
    generation_config.layers = [
        {
            "repetition_penalty": 1.0,
            "temperature": 1.5,
            "top_p": 1.0,
            "top_k": 50,
        }
    ] + [
        {
            "repetition_penalty": 1.1,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 50,
        }
    ] * (model.channels - 1)

    return model, processor, device, generation_config, attn_impl


def resolve_text(args: argparse.Namespace) -> str:
    """Resolve text from -t, -i, or stdin. Exactly one source must provide text."""
    has_text = bool(args.text and args.text.strip())
    has_input_file = args.input_file is not None

    if has_text and has_input_file:
        print("Error: Cannot specify both --text and --input-file.", file=sys.stderr)
        sys.exit(1)

    if has_text:
        return args.text.strip()

    if has_input_file:
        path = Path(args.input_file)
        if not path.exists():
            print(f"Error: Input file not found: {path}", file=sys.stderr)
            sys.exit(1)
        return path.read_text(encoding="utf-8", errors="replace").strip()

    # Read from stdin
    if sys.stdin.isatty():
        print("Error: No text provided. Use --text, --input-file, or pipe text via stdin.", file=sys.stderr)
        sys.exit(1)
    return sys.stdin.read().strip()


def run_inference(
    text: str,
    output_path: Path,
    reference_paths: list[str] | None,
    model_path: str,
    device_str: str,
    attn_implementation: str,
) -> None:
    if not text:
        print("Error: Text is empty.", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Loading model: {model_path}", flush=True)
    model, processor, device, generation_config, attn_impl = load_model_and_processor(
        model_path=model_path,
        device_str=device_str,
        attn_implementation=attn_implementation,
    )
    print(f"[INFO] Using attn_implementation={attn_impl}", flush=True)

    user_kwargs: dict = {"text": text}
    if reference_paths:
        user_kwargs["reference"] = reference_paths

    conversations = [[processor.build_user_message(**user_kwargs)]]
    batch = processor(conversations, mode="generation")
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    print("[INFO] Generating speech...", flush=True)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
        )

    messages = processor.decode(outputs)
    if not messages or messages[0] is None:
        print("Error: Model did not return decodable audio.", file=sys.stderr)
        sys.exit(1)

    audio = messages[0].audio_codes_list[0]
    sampling_rate = processor.model_config.sampling_rate
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(output_path, audio.unsqueeze(0), sampling_rate)
    print(f"[INFO] Saved: {output_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MOSS-TTS CLI: Convert text to speech.",
        epilog="""
Examples:
  %(prog)s -t "Hello, world!" -o hello.wav
  %(prog)s -i script.txt -o speech.wav
  echo "你好" | %(prog)s -o greeting.wav
  %(prog)s -t "Custom voice" -r reference.wav -o cloned.wav
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-t",
        "--text",
        type=str,
        default=None,
        help="Text to synthesize (mutually exclusive with -i and stdin)",
    )
    parser.add_argument(
        "-i",
        "--input-file",
        type=str,
        default=None,
        metavar="FILE",
        help="Path to text file to synthesize",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        metavar="FILE",
        help=f"Output WAV path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "-r",
        "--reference",
        type=str,
        action="append",
        default=None,
        metavar="AUDIO",
        help="Reference audio path(s) for voice cloning (can be repeated)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model path or Hugging Face ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="auto",
        choices=["auto", "sdpa", "eager", "flash_attention_2", "none"],
        help="Attention implementation (default: auto)",
    )
    args = parser.parse_args()

    text = resolve_text(args)
    output_path = Path(args.output)
    reference_paths = args.reference if args.reference else None

    run_inference(
        text=text,
        output_path=output_path,
        reference_paths=reference_paths,
        model_path=args.model,
        device_str=args.device,
        attn_implementation=args.attn_implementation,
    )


if __name__ == "__main__":
    main()
