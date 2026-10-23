"""Causal Robust EWMA Warning (CREW)."""

from .features import FEATURE_NAMES, extract_features
from . import pnu_temporal
from .sequential import CrewResult, FrozenRule, apply_crew, causal_ewma, robust_scale

__all__ = [
    "CrewResult",
    "FEATURE_NAMES",
    "FrozenRule",
    "apply_crew",
    "causal_ewma",
    "extract_features",
    "pnu_temporal",
    "robust_scale",
]

__version__ = "1.0.0"
