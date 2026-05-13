"""Run a single prompt through the AirLLM runner.

Usage:
    python -m local_llm.scripts.run_airllm --model Qwen/Qwen2.5-32B-Instruct --prompt "hello"
    # or, after `pip install -e .`:
    run-airllm --model Qwen/Qwen2.5-32B-Instruct --prompt "hello"
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HuggingFace model id")
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--compression",
        default="4bit",
        choices=["4bit", "8bit", "none"],
        help="AirLLM weight compression (4bit recommended for 8GB VRAM)",
    )
    parser.add_argument("--max-seq-len", type=int, default=512)
    args = parser.parse_args(argv)
    args.compression = None if args.compression == "none" else args.compression

    # Lazy import so tooling (ruff/mypy) doesn't need airllm installed.
    from local_llm.bench.io import save_result
    from local_llm.engines.airllm_runner import AirLLMRunner

    runner = AirLLMRunner()
    runner.load(args.model, compression=args.compression, max_seq_len=args.max_seq_len)
    result = runner.generate(args.prompt, max_tokens=args.max_tokens)
    print(result.text)
    path = save_result(result)
    print(f"\n[result saved to {path}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
