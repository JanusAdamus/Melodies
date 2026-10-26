"""Rejillas sin techo (C2): cómo crece una rejilla cuya selección toca el máximo."""

from __future__ import annotations

import math


def next_grid_candidates(maximum: int) -> tuple[int, ...]:
    """Siguiente escalón tras ``maximum``: ×1.5 y ×2 (48 -> 72, 96; 96 -> 144, 192).

    Es la escalada preregistrada en ``docs/hmm-grid-sensitivity.md``; para valores
    pequeños garantiza al menos ``maximum + 1``.
    """

    return tuple(sorted({max(maximum + 1, math.ceil(1.5 * maximum)), max(maximum + 1, 2 * maximum)}))


def grid_ceiling_message(*, family: str, parameter: str, limit: int, grid: tuple[int, ...]) -> str:
    return (
        f"{family}: validation keeps selecting the grid maximum {max(grid)} and the next step "
        f"would exceed {parameter}={limit} (grid tried: {list(grid)}). A selection at the "
        f"maximum is a short grid, not a plateau: raise {parameter}, or record a documented "
        "resource limit instead of reporting this run."
    )
