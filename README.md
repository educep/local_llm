# local_llm

Benchmark and learning lab for running local LLMs on modest hardware (RTX 2070 8GB, 32GB RAM).
Three engines behind one interface:

- **AirLLM** — layer-wise streaming; runs huge models on tiny VRAM, very slowly.
- **llama.cpp** — the de facto standard; GGUF quantization, CPU/GPU hybrid.
- **Ollama** — llama.cpp with better ergonomics.


## Quickstart

```bash
make install_uv      # once per machine
make venv            # creates .venv pinned to Python 3.11
make install_airllm  # or install_llamacpp, install_all
make pre_commit      # installs git hooks
make check-all       # lint + typecheck + tests
```

> **Windows + NVIDIA GPU:** `torch` is pinned to the PyTorch CUDA 12.4 index so the default CPU-only wheel is not picked up. See [`docs/windows-cuda.md`](./docs/windows-cuda.md) for the full rationale and recovery steps.

Full walkthrough: [`docs/setup.md`](./docs/setup.md).

Download and run:
```bash
make download-qwen32b
run-llamacpp --model models/gguf/qwen2.5-32b-instruct-q4_k_m.gguf --prompt "Write a haiku about VRAM."
make bench-all
```

## Project layout

```
src/local_llm/
├── engines/       Runner ABC + per-engine implementations
├── bench/         Prompt set, harness
└── scripts/       CLI entrypoints (exposed as run-airllm / run-llamacpp / run-ollama)
tests/             Fast smoke tests (no big downloads)
models/            Downloaded weights (gitignored)
plans/             Historical plan files
CLAUDE.md          Decision log and primer
```

## Tooling

| Concern | Tool |
|---|---|
| venv + deps | **uv** |
| lint + format + import sort + pyupgrade | **ruff** (replaces black + isort + flake8 + pyupgrade) |
| type check | **mypy** (strict) |
| security | **bandit** |
| dead code | **vulture** |
| typos | **codespell** |
| tests | **pytest** |
| orchestrator | **pre-commit** (Makefile wraps it) |

## Prerequisites

- Windows 11 / macOS / Linux
- NVIDIA GPU with CUDA 12.x driver (for AirLLM/llama.cpp CUDA paths)
- [uv](https://docs.astral.sh/uv/) — `make install_uv` handles it
- Python 3.11 (via `uv venv`)
- [Ollama](https://ollama.com/download) for the Ollama engine (optional)

## Benchmarks

Raw JSON and aggregated `summary.md` land in [`bench/results/`](./bench/results/).
