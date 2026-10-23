"""Snapshot-local vibration features used by the CREW experiments."""

from __future__ import annotations

import numpy as np


AXES = ("x", "y", "z")
AXIS_FEATURES = (
    "rms_native",
    "peak_to_peak_native",
    "absolute_peak_native",
    "crest_factor",
    "skewness",
    "kurtosis",
    "dominant_frequency_nyquist",
    "spectral_centroid_nyquist",
    "band_0p008_0p080_fraction",
    "band_0p080_0p240_fraction",
    "band_0p240_0p480_fraction",
    "band_0p480_0p800_fraction",
    "band_0p800_1p000_fraction",
)
FEATURE_NAMES = ("vector_ac_rms_native",) + tuple(
    f"{axis}_{name}" for axis in AXES for name in AXIS_FEATURES
)
FREQUENCY_BANDS = (
    (0.008, 0.080),
    (0.080, 0.240),
    (0.240, 0.480),
    (0.480, 0.800),
    (0.800, 1.000),
)


def extract_features(
    samples: np.ndarray,
    sampling_rate_hz: float,
    base_window_seconds: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the 40-feature vector and availability mask for one snapshot.

    The snapshot remains one chronological observation. Features are evaluated
    in complete, non-overlapping base windows and aggregated by the median, so
    this function never bridges gaps between source acquisitions.
    """

    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim == 1:
        samples = samples[:, None]
    if samples.ndim != 2 or not 1 <= samples.shape[1] <= 3:
        raise ValueError(f"expected samples x 1..3 axes, got {samples.shape}")

    window_length = int(round(float(sampling_rate_hz) * base_window_seconds))
    window_count = samples.shape[0] // window_length
    if window_length < 8 or window_count < 1:
        raise ValueError(
            f"snapshot is shorter than one {base_window_seconds:g}-s base window: "
            f"shape={samples.shape}, fs={sampling_rate_hz}"
        )

    blocks = samples[: window_count * window_length].reshape(
        window_count, window_length, samples.shape[1]
    )
    centered = blocks - blocks.mean(axis=1, keepdims=True)
    rms_by_window = np.sqrt(np.mean(centered**2, axis=1))
    feature = np.full(len(FEATURE_NAMES), np.nan, dtype=np.float32)
    mask = np.zeros(len(FEATURE_NAMES), dtype=bool)
    feature[0] = np.median(np.sqrt(np.sum(rms_by_window**2, axis=1)))
    mask[0] = True

    for axis_index in range(samples.shape[1]):
        signal = centered[:, :, axis_index]
        rms = rms_by_window[:, axis_index]
        absolute_peak = np.max(np.abs(signal), axis=1)
        safe_rms = np.where(rms > np.finfo(float).eps, rms, 1.0)
        normalized = signal / safe_rms[:, None]
        normalized[rms <= np.finfo(float).eps] = 0.0

        power = np.abs(
            np.fft.rfft(signal * np.hanning(window_length)[None, :], axis=1)
        ) ** 2
        power[:, 0] = 0.0
        total = power.sum(axis=1)
        safe_total = np.where(total > np.finfo(float).eps, total, 1.0)
        normalized_frequency = np.linspace(0.0, 1.0, power.shape[1])
        dominant = normalized_frequency[np.argmax(power, axis=1)]
        centroid = (power @ normalized_frequency) / safe_total
        spectral_values = [dominant, centroid]
        for band_index, (low, high) in enumerate(FREQUENCY_BANDS):
            upper = normalized_frequency <= high if band_index == 4 else normalized_frequency < high
            selector = (normalized_frequency >= low) & upper
            spectral_values.append(power[:, selector].sum(axis=1) / safe_total)

        values_by_window = np.column_stack(
            (
                rms,
                np.ptp(signal, axis=1),
                absolute_peak,
                absolute_peak / safe_rms,
                np.mean(normalized**3, axis=1),
                np.mean(normalized**4, axis=1),
                *spectral_values,
            )
        )
        start = 1 + axis_index * len(AXIS_FEATURES)
        feature[start : start + len(AXIS_FEATURES)] = np.median(values_by_window, axis=0)
        mask[start : start + len(AXIS_FEATURES)] = True

    return feature, mask


def feature_indices_for_axes(axis_count: int) -> np.ndarray:
    """Return indices available when the source has ``axis_count`` axes."""

    if not 1 <= axis_count <= 3:
        raise ValueError("axis_count must be 1, 2, or 3")
    return np.arange(1 + axis_count * len(AXIS_FEATURES), dtype=int)
