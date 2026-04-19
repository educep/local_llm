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
    args = parser.parse_args(argv)

    # Lazy import so tooling (ruff/mypy) doesn't need airllm installed.
    from local_llm.engines.airllm_runner import AirLLMRunner

    runner = AirLLMRunner()
    runner.load(args.model)
    result = runner.generate(args.prompt, max_tokens=args.max_tokens)
    print(result.text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
