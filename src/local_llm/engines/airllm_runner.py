"""AirLLM adapter — layer-wise streaming inference.

STATUS: Phase 1 stub. Wire up in Phase 1 against `airllm.AutoModel`.
Expect `compression='4bit'` for 8GB VRAM targets.
"""

from __future__ import annotations

from typing import Any

from local_llm.engines.base import GenerateResult, Runner


class AirLLMRunner(Runner):
    engine_name = "airllm"

    def __init__(self) -> None:
        self._model: Any = None
        self._tokenizer: Any = None
        self._model_id: str = ""

    def load(self, model_id: str, **kwargs: Any) -> None:
        del model_id, kwargs  # used by the Phase 1 implementation
        raise NotImplementedError("AirLLM runner: implement in Phase 1")

    def generate(self, prompt: str, max_tokens: int = 256, **kwargs: Any) -> GenerateResult:
        del prompt, max_tokens, kwargs
        raise NotImplementedError("AirLLM runner: implement in Phase 1")
