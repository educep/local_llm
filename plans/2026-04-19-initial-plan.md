# Local LLM Lab — AirLLM vs llama.cpp vs Ollama

## Context

You want to run large LLMs locally on an **8GB RTX 40-series GPU + 32GB RAM, Windows 11**, with a stretch goal of running a ~400B-class MoE model. The directory `C:\Users\dev\dev\local_llm` is empty — this is a greenfield learning project.

You asked for three things:
1. **Teach you** the concepts behind local inference (why VRAM is the bottleneck, what each engine does differently).
2. **Implement all three engines** (AirLLM, llama.cpp, Ollama) so we can compare.
3. **Start small** (Qwen 32B on AirLLM) and scale up to a giant MoE model.

This plan reflects all three.

---

## Conceptual primer (what you'll learn as we build)

| Concept | What it means | Why it matters here |
|---|---|---|
| **VRAM vs system RAM vs disk** | GPU VRAM is ~100× faster than SSD. Weights must sit *somewhere*. | 8GB VRAM ≪ any 30B+ model, so we cheat: quantize, offload, or stream. |
| **Quantization** (Q8/Q4/Q2) | Shrink weights from 16-bit floats → 4-bit ints. ~4× smaller, small quality loss. | Lets 30B models fit in ~18GB instead of ~60GB. |
| **Layer-wise streaming** (AirLLM's trick) | Load transformer layer → run it → discard → load next. Keeps only 1 layer in VRAM. | How "405B on 8GB" works. Cost: SSD→RAM→VRAM every token. Very slow. |
| **CPU offload** (llama.cpp's trick) | Put hot layers on GPU, cold layers in system RAM. | Fast for mid-size models; 32GB RAM caps how big "fits." |
| **MoE (Mixture of Experts)** | Model has N experts, router picks K per token (e.g., 128 experts, 8 active). | Total params huge, *active* params small → we only need the active experts in VRAM. llama.cpp exploits this; AirLLM less so. |
| **GGUF** | llama.cpp's weight format. Quantized, mmap-able, portable. | Single-file model. Ollama uses GGUF under the hood. |

---

## Engine landscape (the honest version)

- **AirLLM** — Python library. Layer-by-layer streaming. Memory miracle, speed disaster. Great for "I want to *touch* a 405B model on my gaming PC" demos. Maintenance is spotty (last major updates Aug 2024). **Not** what serious local users run daily.
- **llama.cpp** — The de facto standard. C++ engine, CUDA/CPU hybrid, GGUF quantization, native MoE-aware expert routing. This is what actually gives usable tokens/sec on your hardware.
- **Ollama** — A wrapper around llama.cpp with a nice CLI/HTTP API and a model registry. Same engine, better UX, less control. Worth including for ergonomics comparison but it's *not* a third algorithm.

**Honest recommendation:** AirLLM is the interesting-but-impractical choice. llama.cpp is the one you'll actually use. The benchmark across all three is the point — you'll *feel* the tradeoffs.

---

## Target models (recommended)

| Tier | Model | Size (Q4 GGUF) | Purpose |
|---|---|---|---|
| Small | **Qwen2.5-32B-Instruct** | ~18GB | First AirLLM run, baseline. Dense model, fits CPU offload on llama.cpp. |
| MoE daily-driver | **Qwen3-30B-A3B** (30B MoE, 3B active) | ~18GB | The realistic "fast MoE on your hardware" star. Should hit 10+ tok/s on llama.cpp. |
| Mid MoE | **Mixtral 8x7B** | ~26GB | Classic MoE, well-documented, good baseline. |
| Stretch goal | **Llama 4 Maverick** (~400B MoE, 17B active) | ~230GB Q4 | The "can we even run it" stunt. Expect <1 tok/s, SSD-bound. |

> **Reality check on Maverick:** With 32GB RAM and Q4 at ~230GB, most weights live on SSD. Every token streams ~17B active params' worth from disk. Budget **minutes per token**. If the SSD isn't NVMe with 3+ GB/s read, consider dropping the stretch goal or grabbing cheap RAM (64GB kit helps enormously).

---

## Language choice: Python (confirmed)

You know Python well, and **every engine we're using has a first-class Python interface**:
- AirLLM is Python-native.
- llama.cpp → `llama-cpp-python` bindings.
- Ollama → HTTP API (any language; `httpx` in Python is cleanest).

Alternative considered: **Go** (Ollama is written in Go) or **Rust** (for raw llama.cpp FFI). Neither buys us anything here — we're orchestrating engines, not writing inference kernels. Python wins on ecosystem + your familiarity. If later we want to write a custom kernel or a high-throughput server, we'd revisit (Rust would be the pick).

---

## Project skeleton (tight, Makefile-driven)

```
local_llm/
├── CLAUDE.md                  # living decision log (see below)
├── README.md                  # user-facing quickstart
├── Makefile                   # the ONE interface: make setup / test / bench / etc.
├── pyproject.toml             # deps, ruff, mypy, pytest config
├── .python-version            # pins 3.11
├── .gitignore                 # ignores .venv, models/, *.gguf, __pycache__
├── .venv/                     # created by `make setup`
├── plans/                     # history of plan files (copied from ~/.claude/plans)
│   └── 2026-04-19-initial-plan.md
├── models/                    # gitignored; all downloaded weights
├── engines/
│   ├── __init__.py
│   ├── base.py                # Runner ABC: generate(prompt, max_tokens) -> GenerateResult
│   ├── airllm_runner.py
│   ├── llamacpp_runner.py
│   └── ollama_runner.py
├── bench/
│   ├── prompts.py             # standard prompt set
│   ├── harness.py             # runs runners, records metrics
│   └── results/               # markdown + json outputs
├── scripts/
│   ├── run_airllm.py
│   ├── run_llamacpp.py
│   └── run_ollama.py
└── tests/
    ├── test_runners_smoke.py  # tiny model smoke tests (no big downloads)
    └── test_harness.py
```

### Tooling choices (pyproject.toml)

| Tool | Purpose | Why this one |
|---|---|---|
| **uv** | venv + dependency management | 10–100× faster than pip; replaces pip/pip-tools/virtualenv. Makefile targets use it. |
| **ruff** | linter + formatter | Replaces black + flake8 + isort. Zero-config sensible defaults. |
| **mypy** | static type check | Strict mode on `engines/` — the abstraction must be type-safe across 3 backends. |
| **pytest** | tests | Standard. Smoke tests use a tiny model (e.g., `Qwen2.5-0.5B`) so CI/dev is fast. |
| **pre-commit** | git hook runner | Runs ruff + mypy before commit. |

Python version: **3.11** (AirLLM has had 3.12+ issues, and llama-cpp-python prebuilt wheels lag on 3.13).

### Makefile targets

```makefile
make setup         # uv venv + uv pip install -e ".[dev]"
make lint          # ruff check + ruff format --check
make format        # ruff format
make typecheck     # mypy engines/ bench/
make test          # pytest -q
make check         # lint + typecheck + test (CI-style)
make download-qwen32b
make download-qwen3-moe
make bench-all     # run harness across all engines × downloaded models
make clean         # remove .venv, __pycache__, .pytest_cache
make clean-models  # remove models/ (big — confirms)
```

### CLAUDE.md (the living decision log)

Root-level `CLAUDE.md` captures everything discussed in planning so future sessions (and future-you) have full context. Sections:
1. **Project goal** — run local LLMs on 8GB GPU, learn the tradeoffs.
2. **Hardware** — RTX 40-series 8GB, 32GB RAM, Windows 11, NVMe SSD.
3. **Engine choices & why** — AirLLM (memory miracle / learning), llama.cpp (daily driver), Ollama (ergonomics). Full honest pros/cons from the planning discussion.
4. **Why not other engines** — vLLM (needs way more VRAM), TGI (server-oriented, overkill), MLX (Mac-only), exllama (narrower model support than llama.cpp). One-liner each.
5. **Quantization primer** — Q4_K_M vs Q5_K_M vs Q8 tradeoffs.
6. **MoE primer** — why it matters for our hardware.
7. **Model shortlist & rationale** — the table from this plan.
8. **Running conventions** — every engine implements `engines/base.py:Runner`. Every script takes `--model` and `--prompt`. Bench results land in `bench/results/YYYY-MM-DD-<engine>-<model>.json`.
9. **Known limitations** — AirLLM slow, Maverick SSD-bound, Windows CUDA build of llama-cpp-python is fiddly.
10. **Decision log** — append-only list of "on $DATE we chose X over Y because Z" entries, so drift is visible.

### plans/ folder

On session start we copy `~/.claude/plans/*.md` → `local_llm/plans/YYYY-MM-DD-<slug>.md` so the plan history lives *with the code*. I'll add a `make save-plan` target that does this.

---

## Phased implementation

### Phase 0 — Environment + scaffolding (Day 1)
- Install **uv** (`winget install astral-sh.uv` or the official installer).
- Install **Python 3.11** via `uv python install 3.11`.
- Verify **CUDA 12.x** driver via `nvidia-smi`.
- Create the full skeleton above (all files, empty `engines/base.py` contract + stubs).
- Write `CLAUDE.md`, `README.md`, `Makefile`, `pyproject.toml`, `.gitignore`, `.python-version`.
- Copy this plan file into `plans/2026-04-19-initial-plan.md`.
- `make setup` succeeds; `make check` passes on empty stubs (ruff/mypy green, one trivial test).
- `git init` + first commit.

### Phase 1 — AirLLM + Qwen 32B (the first milestone)
- Implement `engines/base.py`: `Runner` ABC with `generate(prompt, max_tokens) -> GenerateResult` where `GenerateResult` is a `@dataclass(slots=True)` with `text`, `tokens_in`, `tokens_out`, `ttft_s`, `decode_tok_per_s`, `peak_vram_mb`, `peak_ram_mb`.
- Implement `engines/airllm_runner.py`: wraps `AutoModel.from_pretrained(..., compression='4bit')`, caches under `models/airllm/`.
- `scripts/run_airllm.py` entrypoint: `python -m scripts.run_airllm --model Qwen/Qwen2.5-32B-Instruct --prompt "..."`.
- Download is ~60GB on first run.
- Verify: any coherent output, even slow. Record first data point in bench results.

### Phase 2 — llama.cpp + same Qwen 32B (the fast baseline)
- Install **llama-cpp-python** with CUDA. On Windows the source build is fiddly — **first try prebuilt CUDA wheels** from `https://github.com/abetlen/llama-cpp-python/releases` (match Python 3.11, cu124). Fall back to source build with `CMAKE_ARGS="-DGGML_CUDA=on"` if no wheel matches.
- `make download-qwen32b` → fetch **Qwen2.5-32B-Instruct-Q4_K_M.gguf** (~18GB) via `huggingface-hub` CLI to `models/gguf/`.
- Implement `engines/llamacpp_runner.py` with same `Runner` interface.
- Key tunables to teach and record in CLAUDE.md: `n_gpu_layers`, `n_ctx`, `n_batch`. We'll find max `n_gpu_layers` that fits 8GB empirically and document it.
- Expect 5–15 tok/s. You'll feel the gulf vs AirLLM immediately.

### Phase 3 — Ollama (ergonomics baseline)
- Install Ollama for Windows (native installer).
- `ollama pull qwen2.5:32b-instruct-q4_K_M`
- Files:
  ```
  engines/ollama_runner.py      # HTTP client against http://localhost:11434
  run_ollama_qwen32b.py
  ```
- Should match llama.cpp speed (same engine underneath). The point: see how much less code this is.

### Phase 4 — Benchmark harness
- `bench/harness.py` runs the same prompt set across all three runners, records:
  - time-to-first-token (TTFT)
  - tokens/sec (decode)
  - peak VRAM (via `nvidia-smi` polling)
  - peak system RAM
  - output quality (save all outputs to JSON for side-by-side reading)
- Output a markdown table in `bench/results.md`.
- This is where you'll **see the numbers** and internalize the tradeoffs.

### Phase 5 — MoE showcase: Qwen3-30B-A3B
- Re-run Phases 1–4 with the MoE model.
- On llama.cpp, this should be *dramatically* faster than dense 32B (only 3B params active).
- On AirLLM, expect less benefit — it streams layers blindly, not expert-aware.
- This is the moment the MoE concept "clicks."

### Phase 6 — Stretch: Llama 4 Maverick (400B MoE)
- Download Q4 GGUF (~230GB, verify disk space first).
- Try llama.cpp with aggressive CPU offload. May OOM on 32GB RAM — if so, this is the moment we decide: buy more RAM, or accept AirLLM streaming from SSD.
- Run one prompt, record wall time honestly. This is the "I ran a 400B model on my gaming PC" trophy.

---

## Critical files (all new — greenfield)

See the **Project skeleton** section above for the full tree. Phase 0 creates the scaffold; subsequent phases only add to `engines/`, `scripts/`, and `bench/`.

---

## Verification

**After Phase 0:** `make setup && make check` passes. Repo has CLAUDE.md, plans/, Makefile, pyproject.toml, first git commit.

**After Phase 2:** `python -m scripts.run_llamacpp --prompt "Haiku about VRAM"` produces coherent output in <30s and writes a JSON row to `bench/results/`.

**After Phase 4:** `make bench-all` produces a filled markdown table in `bench/results/` comparing AirLLM / llama.cpp / Ollama on Qwen 32B. This is the **core deliverable**.

**Phase 5–6:** Exploration. Success = "we learned something specific and logged it to CLAUDE.md's decision log" rather than a hard numeric target.

---

## Teaching cadence

At each phase I'll explain: **what we're installing**, **what the code is doing**, **why this engine made this choice**, and **what the numbers mean**. You drive the pace — if something's unclear we stop and dig in before moving on.
