"""AirLLM adapter — layer-wise streaming inference.

Wraps `airllm.AutoModel` so the engine plugs into the common Runner interface.
Defaults to 4-bit compression (best fit for 8GB VRAM); pass `compression=None`
to load fp16 (much larger on-disk shard cache, slower per-token).

Layer shards are cached under `models/airllm/`. The first run of a given model
downloads weights and re-splits them; subsequent runs are SSD-bound but skip
the re-shard step.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

from local_llm.engines.base import GenerateResult, Runner
from local_llm.metrics import PeakTracker, now

_DEFAULT_SHARDS_DIR = Path("models/airllm")


class AirLLMRunner(Runner):  # type: ignore[misc]  # Runner resolves to Any under pre-commit's isolated mypy env
    engine_name = "airllm"

    def __init__(self) -> None:
        self._model: Any = None
        self._tokenizer: Any = None
        self._model_id: str = ""
        self._compression: str | None = None

    def load(
        self,
        model_id: str,
        *,
        compression: str | None = "4bit",
        max_seq_len: int = 512,
        shards_dir: Path | str | None = None,
        **kwargs: Any,
    ) -> None:
        from airllm import AutoModel

        shards_path = Path(shards_dir) if shards_dir else _DEFAULT_SHARDS_DIR
        shards_path.mkdir(parents=True, exist_ok=True)

        self._model_id = model_id
        self._compression = compression
        self._model = AutoModel.from_pretrained(
            model_id,
            compression=compression,
            max_seq_len=max_seq_len,
            layer_shards_saving_path=str(shards_path),
            **kwargs,
        )
        self._tokenizer = self._model.tokenizer

    def generate(self, prompt: str, max_tokens: int = 256, **kwargs: Any) -> GenerateResult:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("AirLLMRunner.generate called before load()")

        input_tokens = self._tokenizer(
            prompt,
            return_tensors="pt",
            return_attention_mask=False,
            truncation=True,
            max_length=getattr(self._model, "max_seq_len", 512),
        )
        input_ids = input_tokens["input_ids"]
        with contextlib.suppress(Exception):
            input_ids = input_ids.cuda()

        tokens_in = int(input_ids.shape[-1])

        with PeakTracker() as peaks:
            t_start = now()
            # AirLLM streams every layer per forward pass; first-token cost ~=
            # one full layer-stream pass, so we approximate TTFT by running a
            # 1-token generation, then resuming for the remainder.
            first = self._model.generate(
                input_ids,
                max_new_tokens=1,
                use_cache=True,
                return_dict_in_generate=False,
            )
            ttft_s = now() - t_start

            if max_tokens > 1:
                full = self._model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    use_cache=True,
                    return_dict_in_generate=False,
                )
            else:
                full = first
            total_s = now() - t_start

        out_ids = full[0]
        tokens_out = int(out_ids.shape[-1]) - tokens_in
        text = self._tokenizer.decode(out_ids[tokens_in:], skip_special_tokens=True)

        decode_s = max(total_s - ttft_s, 1e-9)
        decode_tps = max(tokens_out - 1, 0) / decode_s if tokens_out > 1 else 0.0

        return GenerateResult(
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            ttft_s=ttft_s,
            total_s=total_s,
            decode_tok_per_s=decode_tps,
            peak_vram_mb=peaks.peak_vram_mb,
            peak_ram_mb=peaks.peak_ram_mb,
            engine=self.engine_name,
            model=self._model_id,
            extra={"compression": self._compression},
        )

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        with contextlib.suppress(Exception):
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
