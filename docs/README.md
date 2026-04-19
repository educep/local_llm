# Documentation

Reference documentation for this project. Each file is a self-contained topic; the root [`README.md`](../README.md) links here for anything longer than a quickstart.

## Contents

| Document | Topic |
|---|---|
| [Setup](./setup.md) | Install `uv`, create the virtual environment, install backends. Platform-aware. |
| [Windows: PyTorch CUDA wheels](./windows-cuda.md) | Why `pip install torch` is wrong on Windows, and how this project pins the correct CUDA-enabled wheel. |

## Conventions

- Every document is Markdown, in plain prose. Short enough to read top-to-bottom.
- Commands are shown in fenced code blocks; prefer Makefile targets where one exists.
- Platform-specific notes are called out explicitly. Nothing assumes Linux.
- When a document describes a workaround for a bug or version incompatibility, the surrounding context must explain *why* the workaround exists so a reader can tell when it's safe to drop it.
