#!/usr/bin/env python3
"""Continuous PNU-NOMV evaluation, ablation, and cross-motor audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from crew import pnu_temporal as d3


SCRIPT_DIR = Path(__file__).resolve().parent


MOTORS_DEFAULT = ("MO01", "MO03", "MO04", "MO05", "MO06", "MO07", "MO08", "MO10")
SPLITS = ("train", "calibration", "test")
RULES = ((1, 1), (2, 3), (3, 5), (4, 7))
METHODS = ("global_static", "motor_static", "motor_causal_ewma", "dual_guardrail")
PRIMARY_FEATURE_SET = "time19"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_shard(root: Path, motor: str, split: str) -> dict[str, np.ndarray]:
    shard = root / motor / split
    return {
        "features": np.load(shard / "features.npy", mmap_mode="r"),
        "starts": np.load(shard / "start_record.npy", mmap_mode="r"),
        "state": np.load(shard / "state_code.npy", mmap_mode="r"),
        "quality_flag": np.load(shard / "quality_flag.npy", mmap_mode="r"),
        "near_full_scale_count": np.load(
            shard / "near_full_scale_count.npy", mmap_mode="r"
        ),
        "exact_full_scale_count": np.load(
            shard / "exact_full_scale_count.npy", mmap_mode="r"
        ),
    }


def active_indices(
    shard: dict[str, np.ndarray],
    active_only: bool,
    exclude_near_full_scale: bool = True,
) -> np.ndarray:
    keep = np.ones(len(shard["features"]), dtype=bool)
    if active_only:
        keep &= np.asarray(shard["state"]) == 2
    if exclude_near_full_scale:
        keep &= np.asarray(shard["quality_flag"]) == 0
    return np.flatnonzero(keep).astype(np.int64)


def deterministic_rows(
    shard: dict[str, np.ndarray],
    count: int,
    active_only: bool,
    exclude_near_full_scale: bool = True,
) -> np.ndarray:
    indices = active_indices(shard, active_only, exclude_near_full_scale)
    if len(indices) > count:
        positions = np.linspace(0, len(indices) - 1, count, dtype=np.int64)
        indices = indices[np.unique(positions)]
    return np.asarray(shard["features"][indices], dtype=float)


def fit_model(values: np.ndarray, feature_indices: np.ndarray, shrinkage: float) -> dict[str, np.ndarray]:
    selected = np.asarray(values[:, feature_indices], dtype=float)
    feature_median = np.median(selected, axis=0)
    q25, q75 = np.percentile(selected, [25, 75], axis=0)
    feature_scale = (q75 - q25) / 1.349
    fallback = np.std(selected, axis=0)
    feature_scale = np.where(
        feature_scale > 1e-9,
        feature_scale,
        np.where(fallback > 1e-9, fallback, 1.0),
    )
    standardized = (selected - feature_median) / feature_scale
    center = np.mean(standardized, axis=0)
    centered = standardized - center
    covariance = np.cov(centered, rowvar=False)
    dimension = covariance.shape[0]
    average_variance = float(np.trace(covariance) / dimension)
    shrunk = (
        (1.0 - shrinkage) * covariance
        + shrinkage * average_variance * np.eye(dimension)
    )
    inverse_covariance = np.linalg.pinv(shrunk, rcond=1e-10)
    return {
        "feature_indices": feature_indices.astype(int),
        "median": feature_median,
        "scale": feature_scale,
        "center": center,
        "inverse_covariance": inverse_covariance,
    }


def score_matrix(values: np.ndarray, model: dict[str, np.ndarray]) -> np.ndarray:
    selected = np.asarray(values[:, model["feature_indices"]], dtype=float)
    standardized = (selected - model["median"]) / model["scale"]
    centered = standardized - model["center"]
    squared = np.sum((centered @ model["inverse_covariance"]) * centered, axis=1)
    return np.sqrt(np.maximum(squared, 0.0) / centered.shape[1])


def score_shard(
    shard: dict[str, np.ndarray],
    model: dict[str, np.ndarray],
    active_only: bool,
    batch_size: int,
    exclude_near_full_scale: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    indices = active_indices(shard, active_only, exclude_near_full_scale)
    scores = np.empty(len(indices), dtype=np.float64)
    for offset in range(0, len(indices), batch_size):
        stop = min(len(indices), offset + batch_size)
        scores[offset:stop] = score_matrix(
            np.asarray(shard["features"][indices[offset:stop]]), model
        )
    starts = np.asarray(shard["starts"][indices], dtype=np.int64)
    return scores, starts


def segment_slices(starts: np.ndarray, stride_seconds: int) -> list[slice]:
    starts = np.asarray(starts, dtype=np.int64)
    if len(starts) == 0:
        return []
    boundaries = np.flatnonzero(np.diff(starts) != stride_seconds) + 1
    edges = np.concatenate(([0], boundaries, [len(starts)]))
    return [slice(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]


def persistent_by_segments(
    point_flags: np.ndarray, starts: np.ndarray, stride_seconds: int, k: int, n: int
) -> np.ndarray:
    result = np.zeros(len(point_flags), dtype=bool)
    for current in segment_slices(starts, stride_seconds):
        result[current] = d3.persistence_flags(point_flags[current], k, n)
    return result


def episode_details(
    warnings: np.ndarray,
    starts: np.ndarray,
    stride_seconds: int,
    window_seconds: int,
) -> list[dict[str, object]]:
    episodes: list[dict[str, object]] = []
    for current in segment_slices(starts, stride_seconds):
        local = warnings[current]
        base = current.start or 0
        for local_start, local_end in d3.episode_ranges(local):
            start = base + local_start
            end = base + local_end
            episodes.append(
                {
                    "start_index": start,
                    "end_index": end,
                    "start_record": int(starts[start]),
                    "end_record": int(starts[end] + window_seconds),
                    "observation_count": end - start + 1,
                    "duration_seconds": (end - start) * stride_seconds + window_seconds,
                }
            )
    return episodes


def monitoring_seconds(starts: np.ndarray, stride_seconds: int, window_seconds: int) -> float:
    total = 0.0
    for current in segment_slices(starts, stride_seconds):
        length = current.stop - current.start
        total += (length - 1) * stride_seconds + window_seconds
    return total


def evaluate_series(
    normalized: np.ndarray,
    starts: np.ndarray,
    stride_seconds: int,
    window_seconds: int,
    k: int,
    n: int,
) -> tuple[dict[str, object], np.ndarray, list[dict[str, object]]]:
    point = np.asarray(normalized > 1.0, dtype=bool)
    warnings = persistent_by_segments(point, starts, stride_seconds, k, n)
    episodes = episode_details(warnings, starts, stride_seconds, window_seconds)
    durations = [float(item["duration_seconds"]) for item in episodes]
    exposure_seconds = monitoring_seconds(starts, stride_seconds, window_seconds)
    statistics: dict[str, object] = {
        "observation_count": len(normalized),
        "point_alarm_count": int(point.sum()),
        "point_alarm_rate": float(np.mean(point)) if len(point) else 0.0,
        "warning_count": int(warnings.sum()),
        "warning_rate": float(np.mean(warnings)) if len(warnings) else 0.0,
        "episode_count": len(episodes),
        "monitoring_hours": exposure_seconds / 3600.0,
        "episodes_per_hour": len(episodes) / max(exposure_seconds / 3600.0, 1e-12),
        "episodes_per_day": len(episodes) / max(exposure_seconds / 86400.0, 1e-12),
        "mean_episode_duration_seconds": float(np.mean(durations)) if durations else 0.0,
        "p95_episode_duration_seconds": float(np.percentile(durations, 95)) if durations else 0.0,
        "max_episode_duration_seconds": max(durations, default=0.0),
    }
    return statistics, warnings, episodes


def merge_statistics(rows: list[dict[str, object]], durations: list[float]) -> dict[str, object]:
    observation_count = sum(int(row["observation_count"]) for row in rows)
    point_count = sum(int(row["point_alarm_count"]) for row in rows)
    warning_count = sum(int(row["warning_count"]) for row in rows)
    episode_count = sum(int(row["episode_count"]) for row in rows)
    hours = sum(float(row["monitoring_hours"]) for row in rows)
    return {
        "observation_count": observation_count,
        "point_alarm_count": point_count,
        "point_alarm_rate": point_count / max(observation_count, 1),
        "warning_count": warning_count,
        "warning_rate": warning_count / max(observation_count, 1),
        "episode_count": episode_count,
        "monitoring_hours": hours,
        "episodes_per_hour": episode_count / max(hours, 1e-12),
        "episodes_per_day": episode_count / max(hours / 24.0, 1e-12),
        "mean_episode_duration_seconds": float(np.mean(durations)) if durations else 0.0,
        "p95_episode_duration_seconds": float(np.percentile(durations, 95)) if durations else 0.0,
        "max_episode_duration_seconds": max(durations, default=0.0),
    }


def append_configuration_results(
    rows: list[dict[str, object]],
    episode_rows: list[dict[str, object]] | None,
    config: dict[str, object],
    normalized_calibration: dict[str, np.ndarray],
    normalized_test: dict[str, np.ndarray],
    starts_calibration: dict[str, np.ndarray],
    starts_test: dict[str, np.ndarray],
    motors: list[str],
    stride_seconds: int,
    window_seconds: int,
    k: int,
    n: int,
) -> dict[str, np.ndarray]:
    motor_rows: list[dict[str, object]] = []
    all_test_durations: list[float] = []
    warning_cache: dict[str, np.ndarray] = {}
    for motor in motors:
        cal_stats, _, _ = evaluate_series(
            normalized_calibration[motor], starts_calibration[motor], stride_seconds, window_seconds, k, n
        )
        test_stats, warnings, episodes = evaluate_series(
            normalized_test[motor], starts_test[motor], stride_seconds, window_seconds, k, n
        )
        warning_cache[motor] = warnings
        all_test_durations.extend(float(item["duration_seconds"]) for item in episodes)
        row = {
            **config,
            "rule": f"{k}of{n}",
            "k": k,
            "n": n,
            "motor": motor,
            "calibration_observation_count": cal_stats["observation_count"],
            "calibration_point_alarm_rate": cal_stats["point_alarm_rate"],
            "calibration_warning_rate": cal_stats["warning_rate"],
            **{f"test_{key}": value for key, value in test_stats.items()},
        }
        rows.append(row)
        motor_rows.append(test_stats)
        if episode_rows is not None:
            for episode_number, item in enumerate(episodes, start=1):
                episode_rows.append(
                    {
                        "config_id": config["config_id"],
                        "method": config["method"],
                        "rule": f"{k}of{n}",
                        "motor": motor,
                        "episode_number": episode_number,
                        **item,
                    }
                )
    aggregate_test = merge_statistics(motor_rows, all_test_durations)
    cal_count = sum(len(normalized_calibration[motor]) for motor in motors)
    cal_point = sum(int(np.sum(normalized_calibration[motor] > 1.0)) for motor in motors)
    cal_warning = 0
    for motor in motors:
        cal_warning += int(
            persistent_by_segments(
                normalized_calibration[motor] > 1.0,
                starts_calibration[motor],
                stride_seconds,
                k,
                n,
            ).sum()
        )
    rows.append(
        {
            **config,
            "rule": f"{k}of{n}",
            "k": k,
            "n": n,
            "motor": "ALL",
            "calibration_observation_count": cal_count,
            "calibration_point_alarm_rate": cal_point / max(cal_count, 1),
            "calibration_warning_rate": cal_warning / max(cal_count, 1),
            **{f"test_{key}": value for key, value in aggregate_test.items()},
        }
    )
    return warning_cache


def stationarity_grid(
    train_scores: dict[str, np.ndarray],
    calibration_scores: dict[str, np.ndarray],
    motors: list[str],
    alpha_grid: list[float],
    update_clip: float,
    minimum_scale: float,
) -> tuple[float, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    losses: dict[float, float] = {}
    for alpha in alpha_grid:
        terms_by_motor = []
        for motor in motors:
            location, scale = d3.robust_log_state(train_scores[motor], minimum_scale)
            innovations, _, _ = d3.causal_ewma_innovations(
                calibration_scores[motor], location, scale, alpha, update_clip, minimum_scale
            )
            terms = d3.stationarity_terms(innovations)
            terms_by_motor.append(terms)
            rows.append(
                {
                    "alpha": alpha,
                    "motor": motor,
                    "stationarity_loss": terms[0],
                    "median_range_scaled": terms[1],
                    "p90_range_scaled": terms[2],
                    "lag1_absolute": terms[3],
                    "selected": False,
                }
            )
        means = np.mean(np.asarray(terms_by_motor), axis=0)
        losses[alpha] = float(means[0])
        rows.append(
            {
                "alpha": alpha,
                "motor": "ALL",
                "stationarity_loss": float(means[0]),
                "median_range_scaled": float(means[1]),
                "p90_range_scaled": float(means[2]),
                "lag1_absolute": float(means[3]),
                "selected": False,
            }
        )
    selected = min(alpha_grid, key=lambda value: (losses[value], value))
    for row in rows:
        row["selected"] = math.isclose(float(row["alpha"]), selected)
    return selected, rows



def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PNU-NOMV Stage D5 continuous evaluation")
    parser.add_argument("--features", type=Path, default=SCRIPT_DIR / "continuous_features")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "outputs")
    parser.add_argument("--fit-windows-per-motor", type=int, default=20000)
    parser.add_argument("--lomo-calibration-per-motor", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--quantiles", default="0.975,0.99,0.995,0.999")
    parser.add_argument(
        "--alpha-grid",
        default="0,0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05",
    )
    parser.add_argument("--covariance-shrinkage", type=float, default=0.10)
    parser.add_argument("--active-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--exclude-near-full-scale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude windows containing abs(raw_count) >= the dataset QC threshold",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()

    feature_root = args.features.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_manifest = json.loads((feature_root / "dataset_manifest.json").read_text(encoding="utf-8"))
    motors = [str(value) for value in dataset_manifest["motors"]]
    feature_names = [str(value) for value in dataset_manifest["feature_names"]]
    window_seconds = int(dataset_manifest["window_seconds"])
    stride_seconds = int(dataset_manifest["stride_seconds"])
    quantiles = sorted({float(value) for value in args.quantiles.split(",")})
    alpha_grid = sorted({float(value) for value in args.alpha_grid.split(",")})
    if any(not 0.5 < value < 1.0 for value in quantiles):
        parser.error("quantiles must be between 0.5 and 1.0")
    if not alpha_grid or any(value < 0.0 or value > 0.25 for value in alpha_grid):
        parser.error("alpha-grid values must be between 0 and 0.25")

    time_names = [
        name
        for name in feature_names
        if name == "vector_ac_rms_counts"
        or not any(
            token in name
            for token in ("dominant_hz", "spectral_centroid_hz", "_band_")
        )
    ]
    feature_sets = {
        "all40": np.arange(len(feature_names), dtype=int),
        PRIMARY_FEATURE_SET: np.asarray(
            [feature_names.index(name) for name in time_names], dtype=int
        ),
    }
    shards = {(motor, split): load_shard(feature_root, motor, split) for motor in motors for split in SPLITS}
    train_samples = {
        motor: deterministic_rows(
            shards[(motor, "train")],
            args.fit_windows_per_motor,
            args.active_only,
            args.exclude_near_full_scale,
        )
        for motor in motors
    }
    calibration_samples = {
        motor: deterministic_rows(
            shards[(motor, "calibration")],
            args.lomo_calibration_per_motor,
            args.active_only,
            args.exclude_near_full_scale,
        )
        for motor in motors
    }
    pooled_train = np.vstack([train_samples[motor] for motor in motors])

    models = {
        name: fit_model(pooled_train, indices, args.covariance_shrinkage)
        for name, indices in feature_sets.items()
    }
    score_data: dict[tuple[str, str, str], np.ndarray] = {}
    start_data: dict[tuple[str, str], np.ndarray] = {}
    for motor in motors:
        for split in SPLITS:
            for feature_set, model in models.items():
                scores, starts = score_shard(
                    shards[(motor, split)],
                    model,
                    args.active_only,
                    args.batch_size,
                    args.exclude_near_full_scale,
                )
                score_data[(feature_set, motor, split)] = scores
                start_data[(motor, split)] = starts
            print(f"scored {motor}/{split}: {len(starts):,} active windows", flush=True)

    static_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    for feature_set in feature_sets:
        calibration_all = np.concatenate(
            [score_data[(feature_set, motor, "calibration")] for motor in motors]
        )
        for quantile in quantiles:
            for scope in ("global_static", "motor_static"):
                thresholds = (
                    {motor: d3.quantile_higher(calibration_all, quantile) for motor in motors}
                    if scope == "global_static"
                    else {
                        motor: d3.quantile_higher(
                            score_data[(feature_set, motor, "calibration")], quantile
                        )
                        for motor in motors
                    }
                )
                for motor in motors:
                    threshold_rows.append(
                        {
                            "analysis": "static_ablation",
                            "feature_set": feature_set,
                            "method": scope,
                            "quantile": quantile,
                            "motor": motor,
                            "threshold": thresholds[motor],
                        }
                    )
                normalized_calibration = {
                    motor: score_data[(feature_set, motor, "calibration")] / thresholds[motor]
                    for motor in motors
                }
                normalized_test = {
                    motor: score_data[(feature_set, motor, "test")] / thresholds[motor]
                    for motor in motors
                }
                for k, n in RULES:
                    config_id = f"{feature_set}_{scope}_q{quantile:g}_{k}of{n}"
                    append_configuration_results(
                        static_rows,
                        None,
                        {
                            "config_id": config_id,
                            "analysis": "static_ablation",
                            "feature_set": feature_set,
                            "method": scope,
                            "quantile": quantile,
                        },
                        normalized_calibration,
                        normalized_test,
                        {motor: start_data[(motor, "calibration")] for motor in motors},
                        {motor: start_data[(motor, "test")] for motor in motors},
                        motors,
                        stride_seconds,
                        window_seconds,
                        k,
                        n,
                    )

    train_scores = {
        motor: score_data[(PRIMARY_FEATURE_SET, motor, "train")] for motor in motors
    }
    calibration_scores = {
        motor: score_data[(PRIMARY_FEATURE_SET, motor, "calibration")]
        for motor in motors
    }
    test_scores = {
        motor: score_data[(PRIMARY_FEATURE_SET, motor, "test")] for motor in motors
    }
    selected_alpha, alpha_rows = stationarity_grid(
        train_scores, calibration_scores, motors, alpha_grid, 3.0, 0.05
    )
    global_threshold = d3.quantile_higher(
        np.concatenate(list(calibration_scores.values())), 0.99
    )
    motor_thresholds = {
        motor: d3.quantile_higher(calibration_scores[motor], 0.99) for motor in motors
    }
    adaptive_calibration: dict[str, np.ndarray] = {}
    adaptive_test: dict[str, np.ndarray] = {}
    adaptive_thresholds: dict[str, float] = {}
    for motor in motors:
        location, scale = d3.robust_log_state(train_scores[motor], 0.05)
        cal_innovations, location, scale = d3.causal_ewma_innovations(
            calibration_scores[motor], location, scale, selected_alpha, 3.0, 0.05
        )
        threshold = d3.quantile_higher(cal_innovations, 0.99)
        test_innovations, _, _ = d3.causal_ewma_innovations(
            test_scores[motor], location, scale, selected_alpha, 3.0, 0.05
        )
        adaptive_calibration[motor] = cal_innovations / threshold
        adaptive_test[motor] = test_innovations / threshold
        adaptive_thresholds[motor] = threshold
        threshold_rows.append(
            {
                "analysis": "reference",
                "feature_set": PRIMARY_FEATURE_SET,
                "method": "motor_causal_ewma",
                "quantile": 0.99,
                "motor": motor,
                "threshold": threshold,
            }
        )

    normalized_reference_cal = {
        "global_static": {motor: calibration_scores[motor] / global_threshold for motor in motors},
        "motor_static": {motor: calibration_scores[motor] / motor_thresholds[motor] for motor in motors},
        "motor_causal_ewma": adaptive_calibration,
    }
    normalized_reference_test = {
        "global_static": {motor: test_scores[motor] / global_threshold for motor in motors},
        "motor_static": {motor: test_scores[motor] / motor_thresholds[motor] for motor in motors},
        "motor_causal_ewma": adaptive_test,
    }
    normalized_reference_cal["dual_guardrail"] = {
        motor: np.maximum(
            normalized_reference_cal["global_static"][motor], adaptive_calibration[motor]
        )
        for motor in motors
    }
    normalized_reference_test["dual_guardrail"] = {
        motor: np.maximum(
            normalized_reference_test["global_static"][motor], adaptive_test[motor]
        )
        for motor in motors
    }
    reference_rows: list[dict[str, object]] = []
    episode_rows: list[dict[str, object]] = []
    reference_warning_cache: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for method in METHODS:
        for k, n in RULES:
            config_id = f"{PRIMARY_FEATURE_SET}_{method}_q0.99_{k}of{n}"
            cache = append_configuration_results(
                reference_rows,
                episode_rows,
                {
                    "config_id": config_id,
                    "analysis": "reference",
                    "feature_set": PRIMARY_FEATURE_SET,
                    "method": method,
                    "quantile": 0.99,
                },
                normalized_reference_cal[method],
                normalized_reference_test[method],
                {motor: start_data[(motor, "calibration")] for motor in motors},
                {motor: start_data[(motor, "test")] for motor in motors},
                motors,
                stride_seconds,
                window_seconds,
                k,
                n,
            )
            reference_warning_cache[(method, f"{k}of{n}")] = cache

    lomo_rows: list[dict[str, object]] = []
    qc_indices = feature_sets[PRIMARY_FEATURE_SET]
    for held_out in motors:
        lomo_train = np.vstack([train_samples[motor] for motor in motors if motor != held_out])
        lomo_model = fit_model(lomo_train, qc_indices, args.covariance_shrinkage)
        held_cal_scores, held_cal_starts = score_shard(
            shards[(held_out, "calibration")],
            lomo_model,
            args.active_only,
            args.batch_size,
            args.exclude_near_full_scale,
        )
        held_test_scores, held_test_starts = score_shard(
            shards[(held_out, "test")],
            lomo_model,
            args.active_only,
            args.batch_size,
            args.exclude_near_full_scale,
        )
        other_calibration_scores = np.concatenate(
            [score_matrix(calibration_samples[motor], lomo_model) for motor in motors if motor != held_out]
        )
        protocol_thresholds = {
            "zero_shot": d3.quantile_higher(other_calibration_scores, 0.99),
            "onboard_calibrated": d3.quantile_higher(held_cal_scores, 0.99),
        }
        for protocol, threshold in protocol_thresholds.items():
            statistics, _, _ = evaluate_series(
                held_test_scores / threshold,
                held_test_starts,
                stride_seconds,
                window_seconds,
                2,
                3,
            )
            lomo_rows.append(
                {
                    "held_out_motor": held_out,
                    "protocol": protocol,
                    "train_motor_count": len(motors) - 1,
                    "train_sample_count": len(lomo_train),
                    "calibration_threshold": threshold,
                    **{f"test_{key}": value for key, value in statistics.items()},
                }
            )
        print(f"LOMO {held_out}: complete", flush=True)

    dual_warnings = reference_warning_cache[("dual_guardrail", "2of3")]
    block_rows: list[tuple[float, float, float, float]] = []
    for motor in motors:
        starts = start_data[(motor, "test")]
        warnings = dual_warnings[motor]
        episode_starts = np.zeros(len(warnings), dtype=bool)
        for current in segment_slices(starts, stride_seconds):
            local = warnings[current]
            episode_starts[current] = local & ~np.concatenate(([False], local[:-1]))
        block_ids = starts // 3600
        for block_id in np.unique(block_ids):
            mask = block_ids == block_id
            block_rows.append(
                (
                    float(mask.sum()),
                    float(warnings[mask].sum()),
                    float(episode_starts[mask].sum()),
                    monitoring_seconds(starts[mask], stride_seconds, window_seconds),
                )
            )
    blocks = np.asarray(block_rows, dtype=float)
    rng = np.random.default_rng(args.seed)
    bootstrap_warning_rates = np.empty(args.bootstrap_replicates)
    bootstrap_episode_rates = np.empty(args.bootstrap_replicates)
    for replicate in range(args.bootstrap_replicates):
        sampled = blocks[rng.integers(0, len(blocks), len(blocks))]
        observations = int(sampled[:, 0].sum())
        bootstrap_warning_rates[replicate] = sampled[:, 1].sum() / max(observations, 1)
        exposure_hours = sampled[:, 3].sum() / 3600.0
        bootstrap_episode_rates[replicate] = sampled[:, 2].sum() / max(exposure_hours / 24.0, 1e-12)
    bootstrap_ci = {
        "warning_rate": [float(value) for value in np.percentile(bootstrap_warning_rates, [2.5, 97.5])],
        "episodes_per_day": [float(value) for value in np.percentile(bootstrap_episode_rates, [2.5, 97.5])],
        "replicates": args.bootstrap_replicates,
        "block_definition": "chronological one-hour start-record bins",
    }

    reference_dual = next(
        row for row in reference_rows if row["method"] == "dual_guardrail" and row["rule"] == "2of3" and row["motor"] == "ALL"
    )
    reference_global = next(
        row for row in reference_rows if row["method"] == "global_static" and row["rule"] == "2of3" and row["motor"] == "ALL"
    )
    reference_adaptive = next(
        row for row in reference_rows if row["method"] == "motor_causal_ewma" and row["rule"] == "2of3" and row["motor"] == "ALL"
    )
    zero_shot_rates = [float(row["test_warning_rate"]) for row in lomo_rows if row["protocol"] == "zero_shot"]
    onboard_rates = [float(row["test_warning_rate"]) for row in lomo_rows if row["protocol"] == "onboard_calibrated"]
    active_count = sum(
        len(score_data[(PRIMARY_FEATURE_SET, motor, split)])
        for motor in motors
        for split in SPLITS
    )
    raw_active_count = sum(
        int(np.count_nonzero(np.asarray(shards[(motor, split)]["state"]) == 2))
        for motor in motors
        for split in SPLITS
    )
    active_quality_excluded = sum(
        int(
            np.count_nonzero(
                (np.asarray(shards[(motor, split)]["state"]) == 2)
                & (np.asarray(shards[(motor, split)]["quality_flag"]) != 0)
            )
        )
        for motor in motors
        for split in SPLITS
    )
    summary = {
        "stage": "D5_PNU_continuous_evaluation",
        "continuous_window_count": int(dataset_manifest["total_window_count"]),
        "active_window_count": active_count,
        "active_window_count_before_quality_filter": raw_active_count,
        "near_full_scale_active_windows_excluded": active_quality_excluded,
        "primary_feature_set": PRIMARY_FEATURE_SET,
        "window_seconds": window_seconds,
        "stride_seconds": stride_seconds,
        "motor_count": len(motors),
        "fit_windows_per_motor": args.fit_windows_per_motor,
        "selected_alpha": selected_alpha,
        "selected_alpha_at_upper_grid_boundary": math.isclose(selected_alpha, max(alpha_grid)),
        "reference_global_static_2of3": {
            "warning_rate": float(reference_global["test_warning_rate"]),
            "episodes_per_day": float(reference_global["test_episodes_per_day"]),
        },
        "reference_causal_ewma_2of3": {
            "warning_rate": float(reference_adaptive["test_warning_rate"]),
            "episodes_per_day": float(reference_adaptive["test_episodes_per_day"]),
        },
        "reference_dual_2of3": {
            "warning_rate": float(reference_dual["test_warning_rate"]),
            "episodes_per_hour": float(reference_dual["test_episodes_per_hour"]),
            "episodes_per_day": float(reference_dual["test_episodes_per_day"]),
            "episode_count": int(reference_dual["test_episode_count"]),
            "monitoring_hours": float(reference_dual["test_monitoring_hours"]),
        },
        "lomo_normal_generalization": {
            "zero_shot_median_warning_rate": float(np.median(zero_shot_rates)),
            "zero_shot_range": [float(min(zero_shot_rates)), float(max(zero_shot_rates))],
            "onboard_calibrated_median_warning_rate": float(np.median(onboard_rates)),
            "onboard_calibrated_range": [float(min(onboard_rates)), float(max(onboard_rates))],
        },
        "block_bootstrap_95_ci": bootstrap_ci,
        "alpha_selection_note": (
            "The calibration-only stationarity objective selected the upper safety cap alpha=0.05. "
            "This value is grid-limited and is not fault-optimal: faster adaptation may mask gradual faults, "
            "so the static guardrail must remain until real-fault validation is available."
        ),
        "claim_boundary": (
            "PNU-NOMV normal-only continuous data establish nuisance-warning stability and cross-motor normal generalization, not fault recall or early-warning lead time."
        ),
    }
    model_arrays: dict[str, np.ndarray] = {}
    for feature_set, model in models.items():
        for key, value in model.items():
            model_arrays[f"{feature_set}__{key}"] = np.asarray(value)
    np.savez_compressed(output_dir / "continuous_models.npz", **model_arrays)
    write_csv(output_dir / "static_ablation_results.csv", static_rows)
    write_csv(output_dir / "reference_results.csv", reference_rows)
    write_csv(output_dir / "reference_episodes.csv", episode_rows)
    write_csv(output_dir / "thresholds.csv", threshold_rows)
    write_csv(output_dir / "alpha_selection.csv", alpha_rows)
    write_csv(output_dir / "lomo_results.csv", lomo_rows)
    evaluation_manifest = {
        "stage": summary["stage"],
        "feature_dataset": str(feature_root),
        "active_only": args.active_only,
        "exclude_near_full_scale": args.exclude_near_full_scale,
        "primary_feature_set": PRIMARY_FEATURE_SET,
        "feature_sets": {name: [feature_names[index] for index in indices] for name, indices in feature_sets.items()},
        "fit_windows_per_motor": args.fit_windows_per_motor,
        "lomo_calibration_per_motor": args.lomo_calibration_per_motor,
        "covariance_shrinkage": args.covariance_shrinkage,
        "quantiles": quantiles,
        "rules": [{"k": k, "n": n} for k, n in RULES],
        "alpha_grid": alpha_grid,
        "selected_alpha": selected_alpha,
        "persistence_resets_at_sequence_gaps": True,
        "bootstrap_seed": args.seed,
        "bootstrap_replicates": args.bootstrap_replicates,
        "leakage_controls": dataset_manifest["leakage_controls"],
    }
    (output_dir / "evaluation_manifest.json").write_text(
        json.dumps(evaluation_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "stage_d5_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"PNU-NOMV Stage D5 outputs written to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
