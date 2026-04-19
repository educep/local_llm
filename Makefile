# Makefile for the local_llm project
#
# Philosophy: pre-commit is the single source of truth for running linters/formatters/tests.
# These Makefile targets wrap pre-commit so "run locally" and "run in a hook" are identical.

# --- Variables ------------------------------------------------------------

VENV_NAME = .venv
PACKAGE   = local_llm
SRC_DIR   = src/$(PACKAGE)
TESTS_DIR = tests

ifeq ($(OS),Windows_NT)
    VENV_PYTHON := $(VENV_NAME)/Scripts/python.exe
    OS_NAME     := Windows
else
    VENV_PYTHON := $(VENV_NAME)/bin/python
    OS_NAME     := $(shell uname -s)
endif

# --- uv install -----------------------------------------------------------

.PHONY: install_uv
install_uv:  ## Install uv (cross-platform)
ifeq ($(OS_NAME),Linux)
	@echo "Detected Linux"
	curl -LsSf https://astral.sh/uv/install.sh | sh
else ifeq ($(OS_NAME),Darwin)
	@echo "Detected macOS"
	curl -LsSf https://astral.sh/uv/install.sh | sh
else ifeq ($(OS_NAME),Windows)
	@echo "Detected Windows"
	powershell -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
endif

# --- venv + deps ----------------------------------------------------------

.PHONY: venv
venv:  ## Create the virtual environment (pins Python 3.11)
	uv venv $(VENV_NAME) --python 3.11
	@echo "Virtual environment $(VENV_NAME) created."
	@echo "To activate: source $(VENV_NAME)/Scripts/activate  (Windows bash)"
	@echo "            or  source $(VENV_NAME)/bin/activate   (Linux/macOS)"

.PHONY: install
install:  ## Install project + dev deps in editable mode
ifeq ($(OS_NAME),Windows)
	uv pip install --python $(VENV_PYTHON) pip --upgrade --link-mode=copy
	uv pip install --python $(VENV_PYTHON) -e ".[dev]" --link-mode=copy
else
	@if [ ! -x "$(VENV_PYTHON)" ]; then \
		echo "❌ venv not found at $(VENV_PYTHON). Run 'make venv' first."; exit 1; \
	fi
	uv pip install --python $(VENV_PYTHON) pip --upgrade
	uv pip install --python $(VENV_PYTHON) -e ".[dev]"
endif
	@echo "Dependencies installed (editable)."

.PHONY: install_airllm
install_airllm:  ## Install the AirLLM backend extra
ifeq ($(OS_NAME),Windows)
	uv pip install --python $(VENV_PYTHON) -e ".[airllm]" --link-mode=copy
else
	uv pip install --python $(VENV_PYTHON) -e ".[airllm]"
endif

.PHONY: install_llamacpp
install_llamacpp:  ## Install the llama.cpp backend extra (CUDA wheels recommended on Windows; see README)
ifeq ($(OS_NAME),Windows)
	uv pip install --python $(VENV_PYTHON) -e ".[llamacpp]" --link-mode=copy
else
	uv pip install --python $(VENV_PYTHON) -e ".[llamacpp]"
endif

.PHONY: activate
activate:  ## Print the activate command for your shell
ifeq ($(OS_NAME),Windows)
	@echo "Run: source $(VENV_NAME)/Scripts/activate"
else
	@echo "Run: source $(VENV_NAME)/bin/activate"
endif

# --- pre-commit -----------------------------------------------------------

.PHONY: pre_commit
pre_commit:  ## Install pre-commit git hooks
	pre-commit install
	@echo "Pre-commit hooks installed."

.PHONY: pre-commit-run
pre-commit-run:  ## Run all pre-commit hooks on all files
	pre-commit run --all-files

.PHONY: pre-commit-run-staged
pre-commit-run-staged:  ## Run pre-commit hooks on staged files only
	pre-commit run

# --- individual quality gates (all wrap pre-commit) ----------------------

.PHONY: format
format:  ## Format + auto-fix with ruff (replaces black + isort + pyupgrade)
	pre-commit run ruff --all-files || true
	pre-commit run ruff-format --all-files

.PHONY: ruff
ruff:  ## ruff lint check (no fixes)
	pre-commit run ruff --all-files

.PHONY: mypy
mypy:  ## mypy type check
	pre-commit run mypy --all-files

.PHONY: bandit
bandit:  ## bandit security check
	pre-commit run bandit --all-files

.PHONY: vulture
vulture:  ## vulture dead-code check
	pre-commit run vulture --all-files

.PHONY: codespell
codespell:  ## codespell typo check
	pre-commit run codespell --all-files

.PHONY: lint
lint: ruff mypy bandit vulture codespell  ## Run every linter
	@echo "All linting checks completed."

.PHONY: test
test:  ## Run pytest (via pre-commit so CI and local match)
	pre-commit run pytest --all-files

.PHONY: check-all
check-all: lint test  ## Run every linter + tests
	@echo "All checks and tests completed."

# --- model downloads ------------------------------------------------------

MODELS_DIR := models
GGUF_DIR   := $(MODELS_DIR)/gguf

.PHONY: download-qwen32b
download-qwen32b:  ## Download Qwen2.5-32B-Instruct Q4_K_M GGUF (~18GB)
	@mkdir -p $(GGUF_DIR)
	$(VENV_PYTHON) -m huggingface_hub download Qwen/Qwen2.5-32B-Instruct-GGUF \
		qwen2.5-32b-instruct-q4_k_m.gguf \
		--local-dir $(GGUF_DIR)

.PHONY: download-qwen3-moe
download-qwen3-moe:  ## Download Qwen3-30B-A3B Q4_K_M GGUF (~18GB, MoE showcase)
	@mkdir -p $(GGUF_DIR)
	$(VENV_PYTHON) -m huggingface_hub download Qwen/Qwen3-30B-A3B-GGUF \
		qwen3-30b-a3b-q4_k_m.gguf \
		--local-dir $(GGUF_DIR)

# --- benchmarks -----------------------------------------------------------

.PHONY: bench-all
bench-all:  ## Run benchmark harness across all installed engines/models
	$(VENV_PYTHON) -m local_llm.bench.harness --all

.PHONY: bench-airllm
bench-airllm:
	$(VENV_PYTHON) -m local_llm.bench.harness --engine airllm

.PHONY: bench-llamacpp
bench-llamacpp:
	$(VENV_PYTHON) -m local_llm.bench.harness --engine llamacpp

.PHONY: bench-ollama
bench-ollama:
	$(VENV_PYTHON) -m local_llm.bench.harness --engine ollama

# --- plan management -----------------------------------------------------

.PHONY: save-plan
save-plan:  ## Copy latest plan from ~/.claude/plans/ into ./plans/ with today's date
	@mkdir -p plans
	@latest=$$(ls -t ~/.claude/plans/*.md 2>/dev/null | head -1); \
	if [ -z "$$latest" ]; then echo "no plans found in ~/.claude/plans"; exit 1; fi; \
	dest="plans/$$(date +%Y-%m-%d)-$$(basename $$latest)"; \
	cp "$$latest" "$$dest"; \
	echo "saved $$latest -> $$dest"

# --- cleanup --------------------------------------------------------------

.PHONY: clean
clean:  ## Remove .venv and caches
ifeq ($(OS_NAME),Windows)
	-rmdir /s /q $(VENV_NAME) 2>NUL
	-rmdir /s /q .pytest_cache 2>NUL
	-rmdir /s /q .mypy_cache   2>NUL
	-rmdir /s /q .ruff_cache   2>NUL
else
	rm -rf $(VENV_NAME) .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
endif
	@echo "Cleaned."

.PHONY: clean-models
clean-models:  ## Delete models/ (prompts to confirm — may be hundreds of GB)
	@read -p "delete models/ (may be hundreds of GB)? [y/N] " ans; \
	if [ "$$ans" = "y" ] || [ "$$ans" = "Y" ]; then rm -rf $(MODELS_DIR); echo "removed models/"; else echo "aborted"; fi

# Allow `make <target> <extra args>` without make whining about the extras
%:
	@:

# --- help -----------------------------------------------------------------

.PHONY: help
help:  ## Show this help
	@echo "local_llm — available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
