"""Tests for the bench harness CLI parsing (no runs yet)."""

from __future__ import annotations

import pytest

from local_llm.bench import harness


def test_harness_requires_engine_or_all() -> None:
    with pytest.raises(SystemExit):
        harness.main([])


def test_harness_rejects_bad_engine() -> None:
    with pytest.raises(SystemExit):
        harness.main(["--engine", "not-a-real-engine", "--model", "x"])
