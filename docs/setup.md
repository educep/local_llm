# Setup

This document walks through preparing a machine to run this project from a clean checkout. It covers Linux, macOS, and Windows. Windows has an extra CUDA-related step — see the [Windows: PyTorch CUDA wheels](./windows-cuda.md) document for full detail.

## Prerequisites

- A supported operating system: Linux, macOS, or Windows 10/11.
- For GPU-accelerated backends: an NVIDIA GPU with a recent driver and a matching CUDA Toolkit (12.x recommended; this project is tested with CUDA 12.4).
- Git.
- No Python pre-installed is required — `uv` manages the interpreter.

## 1. Install `uv`

`uv` is the package and environment manager used throughout this project. Install it once per machine.

### Linux / macOS
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows
```bash
make install_uv
```
The Makefile target runs the official PowerShell installer:
```powershell
pwsh -Command "irm https://astral.sh/uv/install.ps1 | iex"
```
If PowerShell's execution policy blocks the command, run the installer manually in an elevated terminal.

Verify:
```bash
uv --version
```

## 2. Create the virtual environment

```bash
make venv
```

This runs `uv venv .venv --python 3.11`. Python 3.11 is downloaded automatically if not already available. The version is pinned because AirLLM lags on Python 3.12+.

## 3. Install dependencies

Dependency groups are defined as optional extras in [`pyproject.toml`](../pyproject.toml). Pick the one that matches what you intend to run.

| Target | Installs | Use when |
|---|---|---|
| `make install` | base + dev tooling | You only need linters, tests, and the benchmark harness. No inference yet. |
| `make install_airllm` | dev + `torch` + `airllm` + `transformers` + `accelerate` | You want to run the AirLLM engine. |
| `make install_llamacpp` | dev + `llama-cpp-python` | You want to run the llama.cpp engine. |
| `make install_all` | dev + every engine extra | You plan to benchmark all three engines. |

All install targets call `uv sync --extra <name>`, which respects the source pins in `pyproject.toml` (see §4) and updates `uv.lock` deterministically.

> **Windows users:** the first install target that pulls `torch` will download a ~2.4 GB CUDA-enabled wheel from PyTorch's own index. This is intentional and essential — see the next section. Default PyPI torch wheels on Windows are CPU-only.

## 4. How the Windows CUDA pin works (brief)

The project pins `torch` to the PyTorch CUDA 12.4 index via `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "pytorch-cu124"
url = "https://download.pytorch.org/whl/cu124"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu124" }
```

This block is respected **only by `uv sync`** (and `uv add`), not by `uv pip install`. That is why every install target in the Makefile uses `uv sync`. See [Windows: PyTorch CUDA wheels](./windows-cuda.md) for the full story, the failure mode, and how to verify you got the correct wheel.

Linux and macOS users are unaffected: default PyPI torch wheels on those platforms ship CUDA support for Linux and MPS support for macOS.

## 5. Install the pre-commit hooks

```bash
make pre_commit
```

This installs the git hook at `.git/hooks/pre-commit`. Every subsequent `git commit` will run ruff, mypy, bandit, vulture, and codespell on the staged files.

## 6. Verify the setup

Run the full quality gate:

```bash
make check-all
```

Expected output: every hook passes (ruff, ruff-format, mypy, bandit, vulture, codespell, pytest). If any hook fails, read its output before proceeding — the stubs are small and failures almost always point at a real issue.

## 7. Verify GPU availability (if you installed a GPU backend)

```bash
.venv/Scripts/python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

On Windows with a working setup you should see:
```
2.6.0+cu124 True
```

If you see `+cpu` or `False`, go to [Windows: PyTorch CUDA wheels](./windows-cuda.md) before doing anything else.

## Resetting from scratch

```bash
make clean          # removes .venv and tool caches
make venv
make install_airllm # or whichever backend you need
make pre_commit
```

`make clean-models` is separate and prompts for confirmation — it can delete hundreds of GB of downloaded weights.
