#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Advanced Torch profiling script for vLLM with configurable options
This script profiles Deepseek V2 Lite model with customizable profiling parameters.

Usage:
    # Basic usage with shape tracking
    python profile_deepseek_advanced.py

    # With custom number of prompts
    python profile_deepseek_advanced.py --num-prompts 10

    # With custom output directory
    python profile_deepseek_advanced.py --output-dir ./my_profile_traces

    # Enable FLOPS tracking (slower but more detailed)
    python profile_deepseek_advanced.py --enable-flops

    # With scheduled profiling (reduced overhead)
    python profile_deepseek_advanced.py --warmup-iters 2 --active-iters 5

    # Custom prompt length (default 2048 input tokens)
    python profile_deepseek_advanced.py --prompt-tokens 4096

    # Disable CUDA graph / compile paths (eager execution; useful for profiling)
    python profile_deepseek_advanced.py --enforce-eager
"""

import argparse
import time
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile vLLM offline inference with tensor shape tracking"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-V2-Lite",
        help="Model name or path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./vllm_profile_deepseek",
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
        profiler_config=profiler_config,
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
    print("- Analyze attention patterns and MLA operations")
    print("\nUseful Perfetto tips:")
    print("- Use 'W/S' keys to zoom in/out")
    print("- Click on operations to see detailed info including tensor shapes")
    print("- Use the search box to find specific operations")
    print("- Filter by thread to focus on specific workers")
    print("=" * 80)


if __name__ == "__main__":
    main()
