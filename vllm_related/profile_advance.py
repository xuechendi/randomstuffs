#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Advanced Torch profiling script for vLLM with configurable options.
Profiles offline inference for any Hugging Face-compatible model with
tensor shape tracking and optional MoE kernel selection.

Usage:
    # Basic usage with shape tracking (default small instruct model)
    python profile_advance.py

    # Specific model and MoE backend (MoE models)
    python profile_advance.py --model deepseek-ai/DeepSeek-V2-Lite --moe-backend triton

    # Non-MoE models: --moe-backend auto is fine (default)
    python profile_advance.py --model meta-llama/Llama-3.2-1B-Instruct

    # With custom number of prompts
    python profile_advance.py --num-prompts 10

    # With custom output directory
    python profile_advance.py --output-dir ./my_profile_traces

    # Enable FLOPS tracking (slower but more detailed)
    python profile_advance.py --enable-flops

    # With scheduled profiling (reduced overhead)
    python profile_advance.py --warmup-iters 2 --active-iters 5

    # Custom prompt length (default 2048 input tokens)
    python profile_advance.py --prompt-tokens 4096

    # Disable CUDA graph / compile paths (eager execution; useful for profiling)
    python profile_advance.py --enforce-eager

    # Force a specific attention backend (same names as vLLM --attention-backend)
    python profile_advance.py --model deepseek-ai/DeepSeek-V2-Lite --attention-backend TRITON_MLA

    # Cap context length (same as vLLM --max-model-len)
    python profile_advance.py --max-model-len 8192
"""

import argparse
import time
from pathlib import Path
from typing import get_args

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config.kernel import MoEBackend
from vllm.v1.attention.backends.registry import AttentionBackendEnum

_MOE_BACKEND_CHOICES = get_args(MoEBackend)


def _normalize_moe_backend(s: str) -> str:
    return s.lower().replace("-", "_")


def _parse_attention_backend(s: str) -> AttentionBackendEnum | None:
    """Match vLLM AttentionConfig validation: 'auto' -> None, else enum by name."""
    raw = s.strip()
    if not raw or raw.lower() == "auto":
        return None
    key = raw.upper().replace("-", "_")
    try:
        return AttentionBackendEnum[key]
    except KeyError:
        valid = ", ".join(sorted(AttentionBackendEnum.__members__))
        raise argparse.ArgumentTypeError(
            f"invalid attention backend {s!r}; use 'auto' or one of: {valid}"
        ) from None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile vLLM offline inference with tensor shape tracking"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Hugging Face model id or local path to profile",
    )
    parser.add_argument(
        "--moe-backend",
        type=_normalize_moe_backend,
        choices=_MOE_BACKEND_CHOICES,
        default="auto",
        help=(
            "MoE expert kernel backend (MoE models only; ignored for dense models). "
            "Use 'auto' to let vLLM choose."
        ),
    )
    parser.add_argument(
        "--attention-backend",
        type=_parse_attention_backend,
        default=None,
        help=(
            "Attention backend (vLLM AttentionBackendEnum name), e.g. FLASH_ATTN, "
            "TRITON_MLA, FLASHINFER. Use 'auto' or omit for automatic selection."
        ),
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum model context length in tokens (vLLM --max-model-len); omit for default.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./vllm_profile_traces",
        help="Directory to save profiling traces",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=4,
        help="Number of prompts to process",
    )
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=2048,
        help="Target number of input tokens per prompt (measured with the model tokenizer)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate per prompt",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graph and torch.compile capture; run the model eagerly",
    )
    parser.add_argument(
        "--enable-flops",
        action="store_true",
        help="Enable FLOPS tracking (adds overhead)",
    )
    parser.add_argument(
        "--disable-memory",
        action="store_true",
        help="Disable memory profiling",
    )
    parser.add_argument(
        "--disable-stack",
        action="store_true",
        help="Disable stack traces",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=0,
        help="Number of warmup iterations (schedule-based profiling)",
    )
    parser.add_argument(
        "--active-iters",
        type=int,
        default=5,
        help="Number of active profiling iterations",
    )
    parser.add_argument(
        "--wait-iters",
        type=int,
        default=0,
        help="Number of wait iterations before warmup",
    )
    parser.add_argument(
        "--enable-expert-parallel",
        action="store_true",
        help="Enable expert parallel",
    )
    return parser.parse_args()


def _prompt_text_of_token_length(
    tokenizer,
    target_tokens: int,
    seed: str,
) -> str:
    """Build text that encodes to exactly `target_tokens` tokens (no special tokens)."""
    if target_tokens <= 0:
        return ""

    text = seed
    while True:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) >= target_tokens:
            trimmed = ids[:target_tokens]
            return tokenizer.decode(trimmed, skip_special_tokens=True)
        text = text + seed


def generate_prompts(
    num_prompts: int,
    tokenizer,
    prompt_tokens: int,
) -> list[str]:
    """Generate prompts for profiling, each sized to `prompt_tokens` input tokens."""
    seeds = [
        "Hello, my name is ",
        "The future of artificial intelligence in natural language processing is ",
        "Write a short story about a robot learning to paint. ",
        "Explain quantum computing in simple terms. ",
        "What are the key differences between machine learning and deep learning? ",
        "Describe the process of photosynthesis. ",
        "Write a poem about the ocean. ",
        "How does blockchain technology work? ",
    ]

    prompts = []
    for i in range(num_prompts):
        seed = seeds[i % len(seeds)]
        prompts.append(_prompt_text_of_token_length(tokenizer, prompt_tokens, seed))

    return prompts


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("vLLM Advanced Torch Profiling with Tensor Shape Tracking")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"MoE backend: {args.moe_backend}")
    print(
        "Attention backend: "
        f"{args.attention_backend.name if args.attention_backend else 'auto'}"
    )
    print(
        "max_model_len: "
        f"{args.max_model_len if args.max_model_len is not None else 'default'}"
    )
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Number of prompts: {args.num_prompts}")
    print(f"Prompt length: {args.prompt_tokens} tokens (tokenizer-measured)")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    print(f"Enforce eager: {'Yes' if args.enforce_eager else 'No'}")
    print("-" * 80)
    print("Profiling options:")
    print(f"  Shape tracking: Enabled")
    print(f"  Memory profiling: {'Disabled' if args.disable_memory else 'Enabled'}")
    print(f"  Stack traces: {'Disabled' if args.disable_stack else 'Enabled'}")
    print(f"  FLOPS tracking: {'Enabled' if args.enable_flops else 'Disabled'}")
    if args.warmup_iters > 0:
        print(f"  Schedule-based profiling:")
        print(f"    - Wait iterations: {args.wait_iters}")
        print(f"    - Warmup iterations: {args.warmup_iters}")
        print(f"    - Active iterations: {args.active_iters}")
    print("=" * 80)

    # Size prompts with the same tokenizer the model uses
    print("\nLoading tokenizer for prompt sizing...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )
    prompts = generate_prompts(args.num_prompts, tokenizer, args.prompt_tokens)

    # Create sampling params
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=args.max_tokens,
    )

    # Build profiler config
    profiler_config = {
        "profiler": "torch",
        "torch_profiler_dir": str(output_dir.absolute()),

        # Core features
        "torch_profiler_record_shapes": True,  # Always enable shape tracking
        "torch_profiler_with_memory": not args.disable_memory,
        "torch_profiler_with_stack": not args.disable_stack,
        "torch_profiler_with_flops": args.enable_flops,
        "torch_profiler_use_gzip": True,
        "torch_profiler_dump_cuda_time_total": True,
    }

    # Add schedule-based profiling if configured
    if args.warmup_iters > 0:
        profiler_config.update({
            "warmup_iterations": args.warmup_iters,
            "active_iterations": args.active_iters,
            "wait_iterations": args.wait_iters,
        })

    # Create LLM
    print("\nInitializing vLLM engine...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        enforce_eager=args.enforce_eager,
        enable_expert_parallel=args.enable_expert_parallel,
        profiler_config=profiler_config,
        moe_backend=args.moe_backend,
        attention_backend=args.attention_backend,
        max_model_len=args.max_model_len,
    )

    print("\nStarting profiler...")
    print("Note: Profiling adds significant overhead. This is expected.\n")

    # Start profiling
    llm.start_profile()

    # Generate texts - this is what gets profiled
    print(f"Processing {len(prompts)} prompts...")
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    generation_time = time.time() - start_time

    # Stop profiling
    print("\nStopping profiler...")
    llm.stop_profile()

    # Print results summary
    print("\n" + "=" * 80)
    print("GENERATION RESULTS SUMMARY")
    print("=" * 80)
    print(f"Total generation time: {generation_time:.2f}s")
    print(f"Number of prompts: {len(outputs)}")
    print(f"Average time per prompt: {generation_time/len(outputs):.2f}s")

    # Calculate token statistics
    total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    print(f"Total tokens generated: {total_tokens}")
    print(f"Average tokens per prompt: {total_tokens/len(outputs):.1f}")
    print(f"Throughput: {total_tokens/generation_time:.2f} tokens/s")

    print("\n" + "=" * 80)
    print("SAMPLE OUTPUTS")
    print("=" * 80)
    for i, output in enumerate(outputs[:3], 1):  # Show first 3
        prompt = output.prompt
        generated_text = output.outputs[0].text
        num_tokens = len(output.outputs[0].token_ids)
        prompt_preview = prompt[:120] + ("..." if len(prompt) > 120 else "")
        print(f"\nPrompt {i} ({num_tokens} generated tokens, input truncated in log): {prompt_preview!r}")
        print(f"Generated: {generated_text!r}")
        if i < min(3, len(outputs)):
            print("-" * 80)

    if len(outputs) > 3:
        print(f"\n... and {len(outputs) - 3} more outputs")

    # Wait for profiler to finish writing files
    print("\n" + "=" * 80)
    print("PROFILER FINALIZATION")
    print("=" * 80)
    print("Waiting for profiler to write trace files...")
    print("This may take several seconds to minutes depending on trace size.")
    time.sleep(15)

    print("\n" + "=" * 80)
    print("PROFILING COMPLETE")
    print("=" * 80)
    print(f"Trace files saved to: {output_dir.absolute()}")
    print("\nTo analyze the traces:")
    print("1. Visit https://ui.perfetto.dev/")
    print("2. Click 'Open trace file'")
    print("3. Select the .json.gz file(s) from the output directory")
    print("\nWith shape tracking enabled, you can:")
    print("- See tensor dimensions for each operation")
    print("- Identify memory-intensive operations")
    print("- Track data flow through the model")
    print("\nUseful Perfetto tips:")
    print("- Use 'W/S' keys to zoom in/out")
    print("- Click on operations to see detailed info including tensor shapes")
    print("- Use the search box to find specific operations")
    print("- Filter by thread to focus on specific workers")
    print("=" * 80)


if __name__ == "__main__":
    main()
