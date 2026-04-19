"""Ollama adapter — HTTP client against http://localhost:11434.

STATUS: Phase 3 stub. Uses httpx against the /api/generate endpoint.
Ollama uses llama.cpp underneath; this adapter is for ergonomics comparison.
"""

from __future__ import annotations

from typing import Any

from local_llm.engines.base import GenerateResult, Runner


class OllamaRunner(Runner):
    engine_name = "ollama"

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self._base_url = base_url
        self._model_id: str = ""

    def load(self, model_id: str, **kwargs: Any) -> None:
        del model_id, kwargs
        raise NotImplementedError("Ollama runner: implement in Phase 3")

    def generate(self, prompt: str, max_tokens: int = 256, **kwargs: Any) -> GenerateResult:
        del prompt, max_tokens, kwargs
        raise NotImplementedError("Ollama runner: implement in Phase 3")
