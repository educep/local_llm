# local_llm

> A benchmark and learning lab for running large language models on modest consumer hardware. Three inference engines behind one interface, so you can feel the tradeoffs.

[![Python](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/mypy-strict-blue)](https://mypy-lang.org/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)](./docs/setup.md)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

---

## Engines

| Engine | What it does | Why it's here |
|---|---|---|
| **AirLLM** | Streams transformer layers one-at-a-time from disk → VRAM. | Runs huge models on tiny VRAM. Very slow, very educational. |
| **llama.cpp** | C++/CUDA hybrid inference on GGUF-quantized weights. | The de facto standard for local inference. |
| **Ollama** | HTTP server wrapping llama.cpp with a model registry. | Same engine, cleaner ergonomics. Ergonomics baseline. |

## Quickstart

```bash
make install_uv      # once per machine
make venv            # .venv pinned to Python 3.11
make install_airllm  # or install_llamacpp, install_all
make pre_commit      # git hooks
make check-all       # lint + typecheck + tests
```

> **Windows + NVIDIA GPU users** — `torch` is pinned to the PyTorch CUDA 12.4 index so the default CPU-only PyPI wheel is never picked up. See [`docs/windows-cuda.md`](./docs/windows-cuda.md) for the full rationale and recovery steps.

Run one:
```bash
make download-qwen32b
run-llamacpp --model models/gguf/qwen2.5-32b-instruct-q4_k_m.gguf --prompt "Write a haiku about VRAM."
make bench-all
```

Full walkthrough: [`docs/setup.md`](./docs/setup.md).

## Project layout

```
src/local_llm/
├── engines/     Runner ABC + per-engine implementations
├── bench/       Prompt set, harness
└── scripts/     CLI entrypoints (run-airllm, run-llamacpp, run-ollama)
tests/           Fast smoke tests (no model downloads)
docs/            Topical reference documentation
models/          Downloaded weights (gitignored)
```

## Tooling

| Concern | Tool |
|---|---|
| venv + deps | **uv** |
| lint + format + imports + pyupgrade | **ruff** (replaces black + isort + flake8 + pyupgrade) |
| type check | **mypy** (strict) |
| security | **bandit** |
| dead code | **vulture** |
| typos | **codespell** |
| tests | **pytest** |
| orchestrator | **pre-commit** (the Makefile wraps it) |

## Prerequisites

- Windows 10/11, macOS, or Linux
- NVIDIA GPU with a CUDA 12.x driver (for AirLLM / llama.cpp GPU paths) — CPU fallback works
- [`uv`](https://docs.astral.sh/uv/) — installed by `make install_uv`
- Python 3.11 (`uv venv` downloads it automatically)
- [Ollama](https://ollama.com/download) — only if using the Ollama engine

## Benchmarks

Per-run JSON and an aggregated `summary.md` land in [`bench/results/`](./bench/results/).

## Documentation

- [Setup walkthrough](./docs/setup.md)
- [Windows: PyTorch CUDA wheels](./docs/windows-cuda.md)

## License

[MIT](./LICENSE).
