"""Medición explícita de tiempo y memoria para bloques del protocolo.

Las métricas ausentes se conservan como ``None`` con un estado. El monitor no
confunde la RAM instalada con el máximo de memoria residente del proceso.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import threading
import time
from types import TracebackType
from typing import Literal

try:
    import torch
except ImportError:  # Permite verificar los módulos puros antes de instalar ML.
    torch = None  # type: ignore[assignment]

try:
    import psutil
except ImportError:  # pragma: no cover - cubierto por la prueba de degradación
    psutil = None  # type: ignore[assignment]


RESOURCE_FIELD_NAMES = (
    "selection_peak_process_rss_bytes",
    "selection_peak_process_tree_rss_bytes",
    "selection_peak_cuda_allocated_bytes",
    "selection_peak_cuda_reserved_bytes",
    "selection_resource_status",
    "selection_cuda_resource_status",
    "evaluation_peak_process_rss_bytes",
    "evaluation_peak_process_tree_rss_bytes",
    "evaluation_peak_cuda_allocated_bytes",
    "evaluation_peak_cuda_reserved_bytes",
    "evaluation_resource_status",
    "evaluation_cuda_resource_status",
    "selected_fit_peak_process_tree_rss_bytes",
    "selected_fit_peak_cuda_allocated_bytes",
    "selected_fit_resource_status",
    "resource_sample_interval_s",
    "resource_measurement_condition",
    "resource_cost_usable",
)


@dataclass(frozen=True)
class ResourceObservation:
    wall_clock_s: float
    peak_process_rss_bytes: int | None
    peak_process_tree_rss_bytes: int | None
    peak_cuda_allocated_bytes: int | None
    peak_cuda_reserved_bytes: int | None
    process_memory_status: str
    cuda_memory_status: str
    sample_interval_s: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ResourceMonitor:
    """Mide un bloque de forma local y no intrusiva.

    La RAM se muestrea en un hilo auxiliar. CUDA usa los contadores máximos de
    PyTorch y sólo se activa cuando el dispositivo está disponible.
    """

    def __init__(self, *, sample_interval_s: float = 0.05, measure_cuda: bool = False) -> None:
        if sample_interval_s <= 0:
            raise ValueError("sample_interval_s must be positive")
        self.sample_interval_s = float(sample_interval_s)
        self.measure_cuda = bool(measure_cuda)
        self.result: ResourceObservation | None = None
        self._start = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_process_rss = 0
        self._peak_tree_rss = 0
        self._process_status = "psutil_not_installed"
        self._cuda_status = "not_requested"

    def _sample_process_tree(self) -> None:
        if psutil is None:
            return
        try:
            process = psutil.Process(os.getpid())
            own_rss = int(process.memory_info().rss)
            tree_rss = own_rss
            for child in process.children(recursive=True):
                try:
                    tree_rss += int(child.memory_info().rss)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
            self._peak_process_rss = max(self._peak_process_rss, own_rss)
            self._peak_tree_rss = max(self._peak_tree_rss, tree_rss)
            self._process_status = "measured"
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as error:
            self._process_status = f"unavailable:{error.__class__.__name__}"

    def _sampling_loop(self) -> None:
        while not self._stop.wait(self.sample_interval_s):
            self._sample_process_tree()

    def __enter__(self) -> ResourceMonitor:
        self._sample_process_tree()
        if self.measure_cuda:
            if torch is None:
                self._cuda_status = "torch_not_installed"
            elif torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    self._cuda_status = "measured"
                except (RuntimeError, AssertionError) as error:
                    self._cuda_status = f"unavailable:{error.__class__.__name__}"
            else:
                self._cuda_status = "cuda_not_available"
        self._start = time.perf_counter()
        if psutil is not None:
            self._thread = threading.Thread(
                target=self._sampling_loop,
                name="resource-monitor",
                daemon=True,
            )
            self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        elapsed = time.perf_counter() - self._start
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.sample_interval_s * 4))
        self._sample_process_tree()

        peak_allocated: int | None = None
        peak_reserved: int | None = None
        if self._cuda_status == "measured" and torch is not None:
            try:
                torch.cuda.synchronize()
                peak_allocated = int(torch.cuda.max_memory_allocated())
                peak_reserved = int(torch.cuda.max_memory_reserved())
            except (RuntimeError, AssertionError) as error:
                self._cuda_status = f"unavailable:{error.__class__.__name__}"

        self.result = ResourceObservation(
            wall_clock_s=elapsed,
            peak_process_rss_bytes=(
                self._peak_process_rss if self._process_status == "measured" else None
            ),
            peak_process_tree_rss_bytes=(
                self._peak_tree_rss if self._process_status == "measured" else None
            ),
            peak_cuda_allocated_bytes=peak_allocated,
            peak_cuda_reserved_bytes=peak_reserved,
            process_memory_status=self._process_status,
            cuda_memory_status=self._cuda_status,
            sample_interval_s=self.sample_interval_s,
        )
        return False


def protocol_resource_fields(
    selection: ResourceObservation,
    evaluation: ResourceObservation,
    *,
    measurement_condition: str,
) -> dict[str, object]:
    """Aplana dos observaciones sin inventar el máximo del ajuste seleccionado."""

    allowed = {"isolated", "contended", "unknown"}
    if measurement_condition not in allowed:
        raise ValueError(f"measurement_condition must be one of {sorted(allowed)}")
    return {
        "selection_peak_process_rss_bytes": selection.peak_process_rss_bytes,
        "selection_peak_process_tree_rss_bytes": selection.peak_process_tree_rss_bytes,
        "selection_peak_cuda_allocated_bytes": selection.peak_cuda_allocated_bytes,
        "selection_peak_cuda_reserved_bytes": selection.peak_cuda_reserved_bytes,
        "selection_resource_status": selection.process_memory_status,
        "selection_cuda_resource_status": selection.cuda_memory_status,
        "evaluation_peak_process_rss_bytes": evaluation.peak_process_rss_bytes,
        "evaluation_peak_process_tree_rss_bytes": evaluation.peak_process_tree_rss_bytes,
        "evaluation_peak_cuda_allocated_bytes": evaluation.peak_cuda_allocated_bytes,
        "evaluation_peak_cuda_reserved_bytes": evaluation.peak_cuda_reserved_bytes,
        "evaluation_resource_status": evaluation.process_memory_status,
        "evaluation_cuda_resource_status": evaluation.cuda_memory_status,
        "selected_fit_peak_process_tree_rss_bytes": None,
        "selected_fit_peak_cuda_allocated_bytes": None,
        "selected_fit_resource_status": "not_measured_separately",
        "resource_sample_interval_s": max(
            selection.sample_interval_s, evaluation.sample_interval_s
        ),
        "resource_measurement_condition": measurement_condition,
        "resource_cost_usable": measurement_condition == "isolated",
    }
