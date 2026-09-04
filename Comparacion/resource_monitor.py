from __future__ import annotations

import os
import threading

import psutil


class ResourceMonitor:
    """Sample peak RSS for this process and its recursive children."""

    def __init__(
        self,
        *,
        sample_interval_s: float = 0.05,
        include_children: bool = True,
        use_cuda: bool = False,
    ) -> None:
        if sample_interval_s <= 0:
            raise ValueError("sample_interval_s must be positive")
        self.sample_interval_s = float(sample_interval_s)
        self.include_children = bool(include_children)
        self.use_cuda = bool(use_cuda)
        self._process = psutil.Process(os.getpid())
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_process_bytes = 0
        self._peak_child_bytes = 0
        self._process_status = "measured"
        self._gpu_status = "not_applicable"
        self._peak_gpu_bytes: int | None = None

    def __enter__(self) -> ResourceMonitor:
        if self.use_cuda:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    self._gpu_status = "measured"
                else:
                    self._gpu_status = "cuda_not_available"
            except Exception as error:  # noqa: BLE001 - status is part of the artifact
                self._gpu_status = f"unavailable: {error.__class__.__name__}"
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._sample()
        if self._gpu_status == "measured":
            try:
                import torch

                self._peak_gpu_bytes = int(torch.cuda.max_memory_allocated())
            except Exception as error:  # noqa: BLE001
                self._gpu_status = f"unavailable: {error.__class__.__name__}"

    def _run(self) -> None:
        while not self._stop.wait(self.sample_interval_s):
            self._sample()

    def _sample(self) -> None:
        try:
            own_rss = int(self._process.memory_info().rss)
            child_rss = 0
            if self.include_children:
                for child in self._process.children(recursive=True):
                    try:
                        child_rss += int(child.memory_info().rss)
                    except (psutil.AccessDenied, psutil.NoSuchProcess):
                        continue
            self._peak_child_bytes = max(self._peak_child_bytes, child_rss)
            self._peak_process_bytes = max(self._peak_process_bytes, own_rss + child_rss)
        except (psutil.AccessDenied, psutil.NoSuchProcess) as error:
            self._process_status = f"unavailable: {error.__class__.__name__}"

    def measurement(self) -> dict[str, object]:
        return {
            "peak_process_memory_bytes": self._peak_process_bytes or None,
            "peak_process_memory_status": self._process_status,
            "peak_child_process_memory_bytes": self._peak_child_bytes,
            "peak_gpu_memory_bytes": self._peak_gpu_bytes,
            "peak_gpu_memory_status": self._gpu_status,
            "memory_sample_interval_s": self.sample_interval_s,
            "memory_scope": (
                "process_rss_plus_recursive_children"
                if self.include_children
                else "process_rss"
            ),
        }
