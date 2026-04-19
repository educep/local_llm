"""Static checks on the benchmark prompt set."""

from __future__ import annotations

from local_llm.bench.prompts import PROMPTS


def test_prompts_nonempty() -> None:
    assert len(PROMPTS) >= 3


def test_prompt_ids_unique() -> None:
    ids = [p.id for p in PROMPTS]
    assert len(ids) == len(set(ids)), f"duplicate prompt ids: {ids}"


def test_prompt_categories_known() -> None:
    allowed = {"short", "medium", "reasoning", "code", "long_context"}
    for p in PROMPTS:
        assert p.category in allowed, f"prompt {p.id!r} has unknown category {p.category!r}"
