"""Standard prompt set used by the benchmark harness.

Kept small and diverse so we can eyeball quality side-by-side across engines.
Add more over time; do NOT remove — results reference them by id.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Prompt:
    id: str
    category: str  # one of: short, medium, reasoning, code, long_context
    text: str
    max_tokens: int = 256


PROMPTS: list[Prompt] = [
    Prompt(
        id="haiku",
        category="short",
        text="Write a haiku about GPU VRAM shortages.",
        max_tokens=64,
    ),
    Prompt(
        id="quant_explain",
        category="medium",
        text=(
            "In two paragraphs, explain how 4-bit quantization works for LLM weights "
            "and why it typically loses less quality than naive rounding."
        ),
        max_tokens=256,
    ),
    Prompt(
        id="moe_reasoning",
        category="reasoning",
        text=(
            "A Mixture-of-Experts model has 128 experts per layer and routes each token "
            "to the top-2 experts. If the model has 32 MoE layers and each expert has 1B "
            "parameters, how many parameters are active for a single token pass? Show your work."
        ),
        max_tokens=400,
    ),
    Prompt(
        id="code_fib",
        category="code",
        text=(
            "Write a Python function that returns the nth Fibonacci number using memoization. "
            "Include type hints and a docstring."
        ),
        max_tokens=256,
    ),
    Prompt(
        id="code_review",
        category="code",
        text=(
            "Review this code and list up to three concrete improvements:\n\n"
            "```python\n"
            "def avg(xs):\n"
            "    s = 0\n"
            "    for x in xs: s = s + x\n"
            "    return s / len(xs)\n"
            "```"
        ),
        max_tokens=300,
    ),
]
