"""Resource peak tracker — VRAM via NVML, RAM via psutil.

Used by every Runner. Background thread polls until `stop()`; peak values are
exposed as `peak_vram_mb` / `peak_ram_mb`. Designed to fail soft: if NVML is
unavailable (no GPU, missing driver), VRAM stays `None` and RAM still works.
"""

from __future__ import annotations

import contextlib
import threading
import time
from types import TracebackType
from typing import Any

import psutil

try:  # pynvml ships as `nvidia-ml-py`; absent on hosts without an NVIDIA stack.
    import pynvml
except ImportError:  # pragma: no cover
    pynvml = None  # type: ignore[assignment,unused-ignore]


class PeakTracker:
    """Background sampler for peak VRAM (process+device) and RAM (process RSS)."""

    def __init__(self, interval_s: float = 0.1, gpu_index: int = 0) -> None:
        self._interval_s = interval_s
        self._gpu_index = gpu_index
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_vram_mb: float | None = None
        self.peak_ram_mb: float | None = None
        self._nvml_handle: Any = None
        self._nvml_active: bool = False
        self._proc: psutil.Process | None = None

    def __enter__(self) -> PeakTracker:
        self.start()
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        self.stop()

    def start(self) -> None:
        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self._gpu_index)
                self._nvml_active = True
                self.peak_vram_mb = 0.0
            except Exception:
                self._nvml_handle = None
                self._nvml_active = False
                self.peak_vram_mb = None

        try:
            self._proc = psutil.Process()
            self.peak_ram_mb = 0.0
        except Exception:
            self._proc = None
            self.peak_ram_mb = None

        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2 * self._interval_s + 0.5)
            self._thread = None
        if self._nvml_active and pynvml is not None:
            with contextlib.suppress(Exception):
                pynvml.nvmlShutdown()
        self._nvml_active = False
        self._nvml_handle = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self._interval_s)
        self._sample()

    def _sample(self) -> None:
        if self._nvml_active and pynvml is not None and self._nvml_handle is not None:
            with contextlib.suppress(Exception):
                used_mb = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle).used / (1024 * 1024)
                if self.peak_vram_mb is None or used_mb > self.peak_vram_mb:
                    self.peak_vram_mb = float(used_mb)
        if self._proc is not None:
            with contextlib.suppress(Exception):
                rss_mb = self._proc.memory_info().rss / (1024 * 1024)
                if self.peak_ram_mb is None or rss_mb > self.peak_ram_mb:
                    self.peak_ram_mb = float(rss_mb)


def now() -> float:
    return time.perf_counter()
