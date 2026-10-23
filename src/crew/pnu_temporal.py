"""Temporal calibration and persistence helpers for PNU-NOMV evaluation."""

from __future__ import annotations

import math

import numpy as np


def quantile_higher(values: np.ndarray, quantile: float) -> float:
    """Return the empirical higher quantile used by the frozen experiments."""

    finite = np.sort(np.asarray(values, dtype=float)[np.isfinite(values)])
    if finite.size == 0:
        raise ValueError("cannot calibrate from empty or non-finite values")
    index = min(len(finite) - 1, max(0, int(math.ceil(quantile * len(finite)) - 1)))
    return float(finite[index])


def robust_log_state(values: np.ndarray, minimum_scale: float) -> tuple[float, float]:
    """Estimate robust log-location and MAD scale."""

    logged = np.log(np.maximum(np.asarray(values, dtype=float), 1e-12))
    location = float(np.median(logged))
    scale = float(1.4826 * np.median(np.abs(logged - location)))
    return location, max(scale, minimum_scale)


def causal_ewma_innovations(
    values: np.ndarray,
    location: float,
    scale: float,
    alpha: float,
    update_clip: float,
    minimum_scale: float,
) -> tuple[np.ndarray, float, float]:
    """Return causal innovations with clipped location/scale state updates."""

    innovations = np.empty(len(values), dtype=float)
    for index, raw_value in enumerate(np.asarray(values, dtype=float)):
        logged = math.log(max(float(raw_value), 1e-12))
        innovation = (logged - location) / scale
        innovations[index] = innovation
        clipped = float(np.clip(innovation, -update_clip, update_clip))
        delta = clipped * scale
        location += alpha * delta
        variance = (1.0 - alpha) * scale * scale + alpha * delta * delta
        scale = max(math.sqrt(max(variance, 0.0)), minimum_scale)
    return innovations, float(location), float(scale)


def lag_one_absolute(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < 3 or np.std(values[:-1]) <= 1e-12 or np.std(values[1:]) <= 1e-12:
        return 0.0
    return float(abs(np.corrcoef(values[:-1], values[1:])[0, 1]))


def stationarity_terms(values: np.ndarray) -> tuple[float, float, float, float]:
    values = np.asarray(values, dtype=float)
    blocks = np.array_split(values, 4)
    iqr = max(float(np.percentile(values, 75) - np.percentile(values, 25)), 0.1)
    medians = [float(np.median(block)) for block in blocks]
    p90s = [float(np.percentile(block, 90)) for block in blocks]
    median_range = (max(medians) - min(medians)) / iqr
    p90_range = (max(p90s) - min(p90s)) / iqr
    lag_one = lag_one_absolute(values)
    loss = median_range + 0.5 * p90_range + 0.25 * lag_one
    return float(loss), float(median_range), float(p90_range), float(lag_one)


def persistence_flags(point_flags: np.ndarray, k: int, n: int) -> np.ndarray:
    """Apply a causal K-of-N persistence rule."""

    point_flags = np.asarray(point_flags, dtype=bool)
    result = np.zeros(len(point_flags), dtype=bool)
    if len(point_flags) < n:
        return result
    cumulative = np.concatenate(([0], np.cumsum(point_flags.astype(int))))
    for index in range(n - 1, len(point_flags)):
        count = cumulative[index + 1] - cumulative[index + 1 - n]
        result[index] = count >= k
    return result


def episode_ranges(flags: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive ranges for contiguous true episodes."""

    flags = np.asarray(flags, dtype=bool)
    starts = np.flatnonzero(flags & ~np.concatenate(([False], flags[:-1])))
    ends = np.flatnonzero(flags & ~np.concatenate((flags[1:], [False])))
    return [(int(start), int(end)) for start, end in zip(starts, ends)]
