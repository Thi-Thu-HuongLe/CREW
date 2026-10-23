"""Core causal decision layer for Causal Robust EWMA Warning (CREW)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FrozenRule:
    """A rule selected without using sealed test observations."""

    alpha: float
    threshold: float
    persistence: int

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if self.threshold < 0.0:
            raise ValueError("threshold must be non-negative")
        if self.persistence < 1:
            raise ValueError("persistence must be positive")


@dataclass(frozen=True)
class CrewResult:
    """Exact causal arrays and event indices returned by :func:`apply_crew`."""

    log_score: np.ndarray
    health: np.ndarray
    point_alarm: np.ndarray
    latched_warning: np.ndarray
    center: float
    scale: float
    evidence_start_index: int | None
    confirmation_index: int | None


def robust_scale(values: np.ndarray, minimum: float = 1e-3) -> float:
    """Return the larger of MAD- and IQR-based robust scale estimates."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("values must be a finite non-empty 1-D array")
    median = float(np.median(values))
    mad = 1.4826 * float(np.median(np.abs(values - median)))
    q25, q75 = np.quantile(values, [0.25, 0.75])
    return max(mad, float((q75 - q25) / 1.349), float(minimum))


def causal_ewma(
    log_score: np.ndarray,
    center: float,
    scale: float,
    alpha: float,
) -> np.ndarray:
    """Compute the non-negative causal EWMA health indicator."""

    values = np.asarray(log_score, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("log_score must be a finite 1-D array")
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")

    standardized = np.clip((values - center) / scale, -10.0, 50.0)
    health = np.empty_like(standardized)
    previous = 0.0
    for index, value in enumerate(standardized):
        previous = alpha * float(value) + (1.0 - alpha) * previous
        health[index] = max(0.0, previous)
    return health


def first_persistent(
    above: np.ndarray,
    persistence: int,
    start: int = 0,
    end: int | None = None,
) -> int | None:
    """Return the first sample in the first consecutive persistent exceedance."""

    flags = np.asarray(above, dtype=bool)
    if flags.ndim != 1:
        raise ValueError("above must be one-dimensional")
    if persistence < 1:
        raise ValueError("persistence must be positive")
    stop = len(flags) if end is None else min(int(end), len(flags))
    count = 0
    for index in range(max(0, int(start)), stop):
        count = count + 1 if flags[index] else 0
        if count >= persistence:
            return index - persistence + 1
    return None


def apply_crew(
    novelty_score: np.ndarray,
    commissioning_count: int,
    scale: float,
    rule: FrozenRule,
    *,
    center: float | None = None,
) -> CrewResult:
    """Apply a frozen CREW rule to one chronological non-negative score stream.

    ``scale`` and ``rule`` must be learned from training/calibration assets.
    When ``center`` is omitted, only the unlabeled commissioning prefix of the
    current asset is used to estimate its location. The causal warning is issued
    at the persistence-confirmation sample, not at the retrospective start of
    the exceedance run.
    """

    score = np.asarray(novelty_score, dtype=np.float64)
    if score.ndim != 1 or score.size == 0 or not np.all(np.isfinite(score)):
        raise ValueError("novelty_score must be a finite non-empty 1-D array")
    if np.any(score < 0.0):
        raise ValueError("novelty_score must be non-negative")
    if not 1 <= commissioning_count <= score.size:
        raise ValueError("commissioning_count must be within the score stream")

    log_score = np.log1p(score)
    fitted_center = (
        float(np.median(log_score[:commissioning_count])) if center is None else float(center)
    )
    health = causal_ewma(log_score, fitted_center, float(scale), rule.alpha)
    point_alarm = health > rule.threshold
    evidence_start = first_persistent(point_alarm, rule.persistence, start=commissioning_count)
    confirmation = None if evidence_start is None else evidence_start + rule.persistence - 1
    latched = np.zeros(score.size, dtype=bool)
    if confirmation is not None:
        latched[confirmation:] = True
    return CrewResult(
        log_score=log_score,
        health=health,
        point_alarm=point_alarm,
        latched_warning=latched,
        center=fitted_center,
        scale=float(scale),
        evidence_start_index=evidence_start,
        confirmation_index=confirmation,
    )


def page_cusum(z_score: np.ndarray, allowance: float) -> np.ndarray:
    """One-sided Page CUSUM comparator used in the paper."""

    values = np.asarray(z_score, dtype=np.float64)
    result = np.empty_like(values)
    previous = 0.0
    for index, value in enumerate(values):
        previous = max(0.0, previous + float(value) - allowance)
        result[index] = previous
    return result


def causal_moving_average(z_score: np.ndarray, window: int) -> np.ndarray:
    """Causal trailing mean of positive standardized evidence."""

    if window < 1:
        raise ValueError("window must be positive")
    positive = np.maximum(np.asarray(z_score, dtype=np.float64), 0.0)
    prefix = np.concatenate(([0.0], np.cumsum(positive, dtype=np.float64)))
    result = np.empty_like(positive)
    for index in range(len(positive)):
        start = max(0, index + 1 - window)
        result[index] = (prefix[index + 1] - prefix[start]) / (index + 1 - start)
    return result
