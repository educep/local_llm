"""llama.cpp adapter via `llama-cpp-python`.

STATUS: Phase 2 stub. Key tunables (documented in CLAUDE.md §8):
  - n_gpu_layers: how many transformer layers to offload to GPU
  - n_ctx:        context window
  - n_batch:      prompt batch size
"""

from __future__ import annotations

from typing import Any

from local_llm.engines.base import GenerateResult, Runner


class LlamaCppRunner(Runner):
    engine_name = "llamacpp"

    def __init__(self) -> None:
        self._llm: Any = None
        self._model_path: str = ""

    def load(self, model_id: str, **kwargs: Any) -> None:
        del model_id, kwargs
        raise NotImplementedError("llama.cpp runner: implement in Phase 2")

    def generate(self, prompt: str, max_tokens: int = 256, **kwargs: Any) -> GenerateResult:
        del prompt, max_tokens, kwargs
        raise NotImplementedError("llama.cpp runner: implement in Phase 2")
