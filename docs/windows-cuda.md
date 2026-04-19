# Windows: PyTorch CUDA wheels

> **Scope.** This document applies to **Windows only**. Linux and macOS users can skip it.

On Windows, installing PyTorch with the default PyPI resolver produces a **CPU-only** build, even on a machine with an NVIDIA GPU and a fully-installed CUDA Toolkit. This document explains why, how this project avoids the problem, how to verify a correct install, and how to recover if something goes wrong.

## 1. The problem

Default PyPI wheels for `torch` on Windows are compiled without CUDA support. Running:

```bash
pip install torch
```

on Windows yields a package whose version string ends in `+cpu`, and whose `torch.cuda.is_available()` returns `False`. No warning is printed. The install succeeds, imports work, tensors allocate on CPU — everything looks fine until a model refuses to move to a GPU device.

PyTorch distributes CUDA-enabled Windows wheels from their own package index, not PyPI. The correct URL depends on the CUDA Toolkit version installed on the machine:

| CUDA Toolkit | Index URL |
|---|---|
| 11.8 | `https://download.pytorch.org/whl/cu118` |
| 12.1 | `https://download.pytorch.org/whl/cu121` |
| 12.4 | `https://download.pytorch.org/whl/cu124` |

`nvcc --version` reports the toolkit version. `nvidia-smi` reports the **driver** version (not the toolkit) — don't mix them up.

## 2. How this project solves it

`pyproject.toml` declares the CUDA index and pins `torch` to it:

```toml
[[tool.uv.index]]
name = "pytorch-cu124"
url = "https://download.pytorch.org/whl/cu124"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu124" }
```

`explicit = true` means this index is consulted **only** for packages that are explicitly sourced from it (via `[tool.uv.sources]`). Every other dependency resolves from the default PyPI.

This configuration is read by `uv sync` and `uv add`. Every install target in the [Makefile](../Makefile) uses `uv sync --extra <name>`, which is why a clean install pulls the CUDA wheel automatically:

```make
install_airllm:
	uv sync --extra dev --extra airllm
```

## 3. Why the Makefile avoids `uv pip install`

`uv pip install` is a pip-compatibility front-end. It **does not read `[tool.uv.sources]`**. If the Makefile invoked `uv pip install -e ".[airllm]"`, the `torch` pin would be silently ignored and the CPU wheel would be installed instead — no error, no warning.

This was the failure mode that originally prompted this document. The commit that switched `install_airllm` to `uv sync` is the fix.

`uv pip install` is still useful for quick one-off installs outside a project's dependency graph. For anything resolved from `pyproject.toml`, use `uv sync`.

## 4. Verification

After any install that should include `torch`, run:

```bash
.venv/Scripts/python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

A correct install prints:

```
2.6.0+cu124 True
```

The `+cuXXX` suffix is the key indicator. If you see `+cpu`, the wrong index was used.

A fuller diagnostic:

```python
import torch

print(f"torch          : {torch.__version__}")
print(f"CUDA available : {torch.cuda.is_available()}")
print(f"CUDA runtime   : {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU            : {torch.cuda.get_device_name(0)}")
    print(f"Compute cap    : {torch.cuda.get_device_capability(0)}")
    print(f"VRAM total     : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    x = torch.randn(1000, 1000, device="cuda")
    assert (x @ x.T).device.type == "cuda"
    print("GPU matmul     : OK")
```

## 5. Recovering from a bad install

If a previous install put `torch==*+cpu` in the environment:

```bash
uv pip uninstall --python .venv/Scripts/python.exe torch
uv sync --extra dev --extra airllm
```

The second command re-resolves `torch` from the pinned CUDA index and updates `uv.lock` to match.

If you need to install `torch` outside a `uv sync` invocation (e.g. an ad-hoc script), pass the index explicitly:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## 6. When the pin becomes wrong

Two cases:

1. **The machine's CUDA Toolkit version changes.** Update the `url` in `[[tool.uv.index]]` to match (e.g. `cu121` → `cu124`) and re-run `uv sync`.
2. **PyTorch drops a channel.** PyTorch eventually stops building for older CUDA versions. If a `uv sync` starts failing with "no matching distribution," check which channels PyTorch currently publishes at `https://pytorch.org/get-started/locally/` and update the index URL.

Both cases should also trigger an entry in the [learning diary](../learning/) explaining the bump.

## 7. Related references

- `pyproject.toml` — the pin lives at the top of the file, next to `[project.scripts]`.
- `uv.lock` — records the resolved `torch` version and its source URL. Commit it.
- PyTorch install matrix: <https://pytorch.org/get-started/locally/>
- `uv` sources and indexes: <https://docs.astral.sh/uv/concepts/dependencies/#index>
