"""Smoke tests that verify the Runner contract without downloading any model.

Fast (< 1s). Real engine tests that require models live in future phases
and are marked `@pytest.mark.slow`.
"""

from __future__ import annotations

import pytest

from local_llm.engines.airllm_runner import AirLLMRunner
from local_llm.engines.base import GenerateResult, Runner
from local_llm.engines.llamacpp_runner import LlamaCppRunner
from local_llm.engines.ollama_runner import OllamaRunner


def test_generate_result_fields() -> None:
    r = GenerateResult(
        text="hi",
        tokens_in=1,
        tokens_out=1,
        ttft_s=0.1,
        total_s=0.2,
        decode_tok_per_s=10.0,
    )
    assert r.text == "hi"
    assert r.engine == ""
    assert r.extra == {}


@pytest.mark.parametrize("cls", [AirLLMRunner, LlamaCppRunner, OllamaRunner])
def test_runner_subclass(cls: type[Runner]) -> None:
    assert issubclass(cls, Runner)
    assert cls.engine_name != "abstract"


@pytest.mark.parametrize("cls", [AirLLMRunner, LlamaCppRunner, OllamaRunner])
def test_runner_load_not_implemented(cls: type[Runner]) -> None:
    runner = cls()
    with pytest.raises(NotImplementedError):
        runner.load("dummy")
