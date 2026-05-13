"""Persist a GenerateResult to bench/results/ as JSON."""

from __future__ import annotations

import datetime as _dt
import json
import re
from dataclasses import asdict
from pathlib import Path

from local_llm.engines.base import GenerateResult

_RESULTS_DIR = Path("bench/results")


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", s).strip("-")


def save_result(result: GenerateResult, results_dir: Path | str | None = None) -> Path:
    out_dir = Path(results_dir) if results_dir else _RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    date = _dt.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    path = out_dir / f"{date}-{_slug(result.engine)}-{_slug(result.model)}.json"
    path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return path
