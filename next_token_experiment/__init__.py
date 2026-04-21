"""Base acotada para el experimento de siguiente token en musica simbolica."""

from __future__ import annotations

import warnings

try:
    from requests import RequestsDependencyWarning

    warnings.filterwarnings("ignore", category=RequestsDependencyWarning)
except Exception:  # pragma: no cover - depende del entorno.
    pass

from .config import ExperimentConfig
from .experiment.runner import run_small_transformer_experiment
from .profiles import build_profile_config, list_profiles
from .protocol import build_default_experiment_config, estimate_transformer_parameter_count
from .protocol import validate_experiment_scope

__all__ = [
    "ExperimentConfig",
    "build_profile_config",
    "build_default_experiment_config",
    "estimate_transformer_parameter_count",
    "list_profiles",
    "run_small_transformer_experiment",
    "validate_experiment_scope",
]
