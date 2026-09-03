"""Comparison models, including a diagnostic PPM-inspired VOMM control.

The public VOMM names are loaded lazily so utilities such as artifact
verification can run before the optional numerical stack is installed.
"""

from __future__ import annotations

from typing import Any

__all__ = ["VariableOrderMarkovModel", "select_vomm_by_validation"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .vomm import VariableOrderMarkovModel, select_vomm_by_validation

        return {
            "VariableOrderMarkovModel": VariableOrderMarkovModel,
            "select_vomm_by_validation": select_vomm_by_validation,
        }[name]
    raise AttributeError(name)
