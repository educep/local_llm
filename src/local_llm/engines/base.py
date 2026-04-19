"""Common interface every engine adapter must implement.

The single abstraction used across AirLLM, llama.cpp, and Ollama so the benchmark
harness and entrypoint scripts can treat them interchangeably.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class GenerateResult:
    """One generation's full observability payload.

    Captures both the output and the performance/resource metrics needed by
    `bench/harness.py`. Every runner returns this; no engine-specific fields.
    """

    text: str
    tokens_in: int
    tokens_out: int
    ttft_s: float             # time to first token
    total_s: float            # total wall time including prefill
    decode_tok_per_s: float   # steady-state decode throughput (tokens_out / (total_s - ttft_s))
    peak_vram_mb: float | None = None
    peak_ram_mb: float | None = None
    engine: str = ""
    model: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class Runner(ABC):
    """Abstract base for every inference engine."""

    engine_name: str = "abstract"

    @abstractmethod
    def load(self, model_id: str, **kwargs: Any) -> None:
        """Load / prepare the model. Idempotent per instance.

        For local-file engines (llama.cpp) `model_id` is a path.
        For registry engines (HuggingFace/Ollama) it's a model ID string.
        """

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 256, **kwargs: Any) -> GenerateResult:
        """Run one generation. Must populate all non-optional GenerateResult fields."""

    def unload(self) -> None:
        """Release model resources. Default no-op; override when needed."""
