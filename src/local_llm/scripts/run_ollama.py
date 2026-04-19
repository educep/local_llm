"""Run a single prompt through the Ollama runner.

Requires the Ollama service to be running locally (default http://localhost:11434).

Usage:
    python -m local_llm.scripts.run_ollama --model qwen2.5:32b-instruct-q4_K_M --prompt "hello"
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        help="Ollama model tag, e.g. 'qwen2.5:32b-instruct-q4_K_M'",
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--base-url", default="http://localhost:11434")
    args = parser.parse_args(argv)

    from local_llm.engines.ollama_runner import OllamaRunner

    runner = OllamaRunner(base_url=args.base_url)
    runner.load(args.model)
    result = runner.generate(args.prompt, max_tokens=args.max_tokens)
    print(result.text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
