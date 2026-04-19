"""Run a single prompt through the llama.cpp runner.

Usage:
    python -m local_llm.scripts.run_llamacpp \
        --model models/gguf/qwen2.5-32b-instruct-q4_k_m.gguf \
        --prompt "hello" --n-gpu-layers 20
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to GGUF file")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--n-gpu-layers", type=int, default=20)
    parser.add_argument("--n-ctx", type=int, default=4096)
    args = parser.parse_args(argv)

    from local_llm.engines.llamacpp_runner import LlamaCppRunner

    runner = LlamaCppRunner()
    runner.load(args.model, n_gpu_layers=args.n_gpu_layers, n_ctx=args.n_ctx)
    result = runner.generate(args.prompt, max_tokens=args.max_tokens)
    print(result.text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
