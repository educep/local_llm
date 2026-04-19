"""Benchmark harness — runs prompts across engines x models and records metrics.

STATUS: Phase 4 stub. Will:
  1. Load each runner x model combination requested on the CLI.
  2. For each prompt in `bench.prompts.PROMPTS`, call `runner.generate(...)`.
  3. Poll VRAM (via pynvml) and RAM (via psutil) during generation to capture peaks.
  4. Write one JSON file per run to bench/results/YYYY-MM-DD-<engine>-<model>.json.
  5. Aggregate into bench/results/summary.md.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="local_llm benchmark harness")
    parser.add_argument("--engine", choices=["airllm", "llamacpp", "ollama"], help="Engine to run")
    parser.add_argument("--model", help="Model id or path")
    parser.add_argument("--all", action="store_true", help="Run every engine x every known model")
    args = parser.parse_args(argv)

    if not args.all and not (args.engine and args.model):
        parser.error("pass --all, or both --engine and --model")

    raise NotImplementedError("harness: implement in Phase 4")


if __name__ == "__main__":
    sys.exit(main())
