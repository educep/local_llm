# Ruff replaces black + isort + flake8 + pyupgrade

- **What I learned:** one Rust-based tool (`ruff`) covers formatting, import sorting, linting (flake8 + many plugins), and syntax upgrades — with a single config block in `pyproject.toml` and auto-fix built in.
- **What I was doing before:** separate `black`, `isort`, `flake8` (+ `flake8-bugbear`, `flake8-comprehensions`, `flake8-pyproject`), and `pyupgrade`, each pinned separately with their own pre-commit entries and config sections.
- **Why the new way is better:** 10–100× faster on large codebases (felt on `pre-commit run --all-files` and in editors); one install, one config, one unified rule namespace (`E501`, `B008`, `UP007` all in one place); `ruff format` is a drop-in for black; `--fix` auto-resolves most issues instead of just flagging them.
- **Tradeoffs / when the old way still wins:** a few niche flake8 plugins don't have ruff equivalents yet; ruff's `S` security rules are shallower than real `bandit`; ruff does NOT do types (keep `mypy`), dead-code detection (keep `vulture`), or prose typos (keep `codespell`). So ruff replaces the *style/lint/format* layer, not the whole linting stack.
