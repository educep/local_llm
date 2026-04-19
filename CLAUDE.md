# CLAUDE.md — Local LLM Lab

Living decision log for this project. Update when choices change; don't delete history — append to the decision log at the bottom.

## 1. Project goal

Run large language models locally on modest consumer hardware (8GB GPU, 32GB RAM, Windows 11), and in the process **learn the tradeoffs** between different inference engines. The repo is both a working benchmark suite and a teaching artifact.

Three deliverables:
1. Working runners for AirLLM, llama.cpp, and Ollama, all behind a single `Runner` interface.
2. A benchmark harness that measures tokens/sec, TTFT, peak VRAM, peak RAM across engines × models on a fixed prompt set.
3. Stretch goal: run a ~400B MoE model on this hardware (even if painfully slow) as a demonstration of layer-streaming.

## 2. Hardware (actual)

| Component | Value | Notes |
|---|---|---|
| GPU | **NVIDIA RTX 2070 (Turing, CC 7.5)** | 8GB VRAM. No FP8, no FlashAttention-3. FA2 works. |
| System RAM | 32GB | This is the binding constraint for big MoE models. |
| OS | Windows 11 Pro | Some Linux-native tooling is fiddly here. |
| Driver | 551.61 | Supports CUDA 12.4 runtime. |
| Storage | NVMe assumed | Critical for AirLLM layer streaming and big-model mmap. |

### Turing-specific caveats (RTX 2070)
- **No bfloat16 native support** → prefer `float16` or quantized formats. AirLLM's 4-bit block quant is fine; mixed-bf16 kernels will fall back or fail.
- **No FlashAttention-3**. FA2 works. For llama.cpp this doesn't matter (it has its own kernels).
- **Compute 7.5** → all modern engines support it; no custom builds required.

## 3. Engine choices & honest rationale

### AirLLM — the memory miracle, not a daily driver
- **What it does:** loads transformer layers one-at-a-time from disk → VRAM → runs → discards. The entire model never resides in VRAM.
- **Why we include it:** it's the *only* way a 400B model touches an 8GB GPU without a cluster. Also: great teaching vehicle — makes the layer-wise structure of transformers tangible.
- **Cost:** I/O-bound. Each token pays the full "stream every layer from SSD" cost. Expect seconds-to-minutes per token on big models.
- **Maturity:** moderate. Last substantive updates Aug 2024. Works, but not widely deployed in production.

### llama.cpp — the realistic daily driver
- **What it does:** C++ inference engine with CUDA/CPU hybrid execution. Uses GGUF quantized weights (mmap'd, so "load" is near-instant). Native MoE expert routing.
- **Why we include it:** this is what actually runs well on 8GB. Quantized 30B models at 5–15 tok/s. The community standard.
- **Cost:** less educational than AirLLM (most internals are in C++). Windows CUDA build from source is painful — prefer prebuilt wheels.
- **Maturity:** very high. Active daily development.

### Ollama — llama.cpp with better ergonomics
- **What it does:** Go-based wrapper around llama.cpp with a model registry (`ollama pull`), HTTP API, and automatic model management.
- **Why we include it:** same engine as llama.cpp, but 10× less code to use. Included for the ergonomics comparison, not as a third algorithm.
- **Cost:** less control over tunables (`n_gpu_layers`, `n_ctx`, sampling) — you configure via a Modelfile, not direct flags.
- **Maturity:** very high.

## 4. Engines considered and rejected

| Engine | Why not |
|---|---|
| **vLLM** | Needs the full model in VRAM (no CPU offload for our scale). Requires way more than 8GB. Designed for server deployment. |
| **TGI (Text Generation Inference)** | HuggingFace's server. Overkill for single-user local use; heavier ops. |
| **MLX** | Apple Silicon only. We're on Windows + NVIDIA. |
| **ExLlamaV2** | Excellent speed on GPTQ/EXL2 weights, but narrower model support than llama.cpp and worse MoE support. Might revisit if we want to squeeze more speed. |
| **TensorRT-LLM** | Fastest possible on NVIDIA, but build complexity is brutal and Turing support is deprecated for newer features. |

## 5. Quantization primer (GGUF formats we'll use)

Quantization = shrinking weights from 16-bit floats to fewer bits. Quality/size tradeoff.

| Format | Bits/weight (avg) | Size vs fp16 | Quality | When to use |
|---|---|---|---|---|
| `Q8_0` | ~8.5 | ~50% | ~lossless | Plenty of VRAM/RAM. Rarely used on 8GB. |
| `Q6_K` | ~6.5 | ~40% | Very close to fp16 | Good quality/size balance. |
| `Q5_K_M` | ~5.7 | ~35% | Slight quality loss | Sweet spot for capable models. |
| **`Q4_K_M`** | ~4.8 | ~30% | Noticeable but small | **Our default.** Best fit for 8GB. |
| `Q3_K_M` | ~3.9 | ~25% | Real quality drop | Last resort for huge models. |
| `Q2_K` | ~3.3 | ~20% | Degraded | Useful only for very big models. |

The `_K_M` variants use "k-quants": block-wise quantization that varies precision per-block. Better quality than legacy `Q4_0`/`Q4_1` at the same size.

## 6. MoE primer (and why it matters for 8GB)

A Mixture-of-Experts model has:
- **N experts** per layer (e.g., 128 in Llama 4 Maverick)
- **A router** that picks **K active experts** per token (e.g., 2 of 128)

So: total parameters huge (400B), but **active** parameters per token small (17B). The router decides per-token which experts to consult.

Implication for our hardware: if an engine is MoE-aware, it only needs the K active experts in fast memory at a given moment. llama.cpp supports this. AirLLM streams layers blindly and doesn't special-case MoE — it'll still work, but with less benefit.

This is why Qwen3-30B-A3B (30B total, 3B active) should **feel like a 3B model's speed** but with 30B's quality on llama.cpp.

## 7. Model shortlist

| Tier | Model | Size (Q4_K_M) | Purpose |
|---|---|---|---|
| Small | Qwen2.5-32B-Instruct (dense) | ~18GB | First AirLLM run; llama.cpp baseline. |
| MoE daily | **Qwen3-30B-A3B** (30B/3B active) | ~18GB | The realistic "fast MoE" target. |
| Mid MoE | Mixtral 8x7B | ~26GB | Well-documented classic MoE. |
| Stretch | Llama 4 Maverick (~400B/17B active) | ~230GB | The "can we even run it" stunt. |

## 8. Running conventions

- **`src/` layout**: importable package is `local_llm`. Code lives under `src/local_llm/{engines,bench,scripts}/`.
- **Single interface:** every engine implements `local_llm.engines.base.Runner`. Swappable.
- **Entrypoints** are exposed as console scripts (`run-airllm`, `run-llamacpp`, `run-ollama`, `bench`) via `[project.scripts]`. Also runnable as `python -m local_llm.scripts.run_<engine>`.
- **Bench results** land in `bench/results/YYYY-MM-DD-<engine>-<model>.json` and are summarized into `bench/results/summary.md`.
- **Models** live under `models/` (gitignored). One subdir per format: `models/gguf/`, `models/airllm/`. Ollama has its own store; we just reference the tag.
- **All tunables** that matter (e.g., `n_gpu_layers`) are function arguments, not hard-coded. Defaults recorded here.
- **pre-commit is the single source of truth** for running linters/formatters/tests. The Makefile wraps it so `make <target>` and the git hook behave identically.

### Tool stack

| Concern | Tool | Replaces |
|---|---|---|
| venv + deps | uv | pip + virtualenv |
| lint + format + imports + pyupgrade | **ruff** | black + isort + flake8 (+plugins) + pyupgrade |
| type check | mypy (strict) | — |
| security | bandit | — |
| dead code | vulture | — |
| typos | codespell | — |
| tests | pytest | — |
| orchestrator | pre-commit | — |

## 9. Known limitations & open risks

1. **AirLLM speed on big models** is SSD-bound. A slow drive makes Phase 6 impractical.
2. **llama-cpp-python Windows CUDA build** is unreliable from source. We use prebuilt wheels matching Python 3.11 + CUDA 12.4.
3. **32GB RAM caps MoE scale.** Llama 4 Maverick Q4 is ~230GB — only AirLLM (streaming from disk) can attempt it; llama.cpp will OOM unless we also aggressively `--no-mmap` and page.
4. **Turing GPU (RTX 2070)** lacks bfloat16; stick with fp16 or quantized.
5. **HuggingFace access** — some gated models (Llama 3/4) require accepting a license and a HF token.

## 10. Learning diary — see [`learning/`](./learning/)

Every lesson worth remembering lives as one markdown file in `learning/`. See [`learning/README.md`](./learning/README.md) for the full logging instruction, the entry format, and when to create a new entry.

Quick rule: if you taught the user something, or we changed our minds about a tool, or a non-obvious tradeoff surfaced — drop a file in `learning/`.

## 11. Decision log

*(append-only; newest at bottom)*

- **2026-04-19** — chose Python 3.11 over 3.12+ because AirLLM and some torch extensions still lag on 3.12. Revisit when AirLLM publishes 3.12 wheels.
- **2026-04-19** — chose `uv` over `pip`/`poetry` for venv + deps. Speed + single-tool simplicity; user already wants a tight setup.
- **2026-04-19** — chose `ruff` (lint + format + imports + pyupgrade) over user's previous black+isort+flake8+pyupgrade stack. Kept bandit, vulture, codespell, mypy alongside (ruff doesn't fully replace them). Rationale: 10–100× faster, single config, unified rule namespace, auto-fix. Discussed and confirmed with user.
- **2026-04-19** — chose `mypy` over `pyright`. Stays inside the pre-commit-driven workflow; strict mode on.
- **2026-04-19** — adopted `src/` layout (`src/local_llm/`) with setuptools, matching user's house style. Console scripts via `[project.scripts]`: `run-airllm`, `run-llamacpp`, `run-ollama`, `bench`.
- **2026-04-19** — pre-commit is the execution entrypoint for all linters + tests; Makefile targets wrap `pre-commit run <hook>`. Keeps local and hook behavior identical.
- **2026-04-19** — rejected vLLM/TGI/MLX/ExLlamaV2/TensorRT-LLM (see §4).
- **2026-04-19** — hardware confirmed as **RTX 2070** (not 40-series). Turing caveats logged in §2.
- **2026-04-19** — chose default quant **Q4_K_M** for GGUF models (see §5 rationale).
- **2026-04-19** — chose **Qwen2.5-32B** as first target (Phase 1), **Qwen3-30B-A3B** as MoE showcase (Phase 5), **Llama 4 Maverick** as stretch goal (Phase 6).
