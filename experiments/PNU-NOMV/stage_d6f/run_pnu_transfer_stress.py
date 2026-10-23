#!/usr/bin/env python3
"""Stress-test locked external temporal policies on PNU-NOMV normal-only data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PNU_DIR = SCRIPT_DIR.parent
PROJECT_CODE = PNU_DIR.parent
D5_DIR = PNU_DIR / "stage_d5"
D5_OUTPUT = D5_DIR / "outputs"
D5_FEATURES = D5_DIR / "continuous_features"
D6C_OUTPUT = PROJECT_CODE / "Real-Fault-Benchmarks" / "stage_d6c" / "outputs"
sys.path.insert(0, str(D5_DIR))
import run_continuous_evaluation as d5  # noqa: E402


MOTORS = ("MO01", "MO03", "MO04", "MO05", "MO06", "MO07", "MO08", "MO10")
PRIMARY_FEATURE_SET = "time19"
STRIDE_SECONDS = 2
WINDOW_SECONDS = 4
TRAIN_ROWS_PER_MOTOR = 20_000
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 20261016


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def robust_scale(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    center = float(np.median(values))
    mad = 1.4826 * float(np.median(np.abs(values - center)))
    q25, q75 = np.quantile(values, [0.25, 0.75])
    return max(mad, float((q75 - q25) / 1.349), 1e-3)


def causal_health(values: np.ndarray, center: float, scale: float, alpha: float) -> np.ndarray:
    logged = np.log1p(np.asarray(values, dtype=np.float64))
    z = np.clip((logged - center) / scale, -10.0, 50.0)
    health = np.empty(len(z), dtype=np.float64)
    previous = 0.0
    for index, value in enumerate(z):
        previous = alpha * float(value) + (1.0 - alpha) * previous
        health[index] = max(0.0, previous)
    return health


def health_by_segments(scores: np.ndarray, starts: np.ndarray, center: float, scale: float, alpha: float) -> np.ndarray:
    result = np.empty(len(scores), dtype=np.float64)
    for current in d5.segment_slices(starts, STRIDE_SECONDS):
        result[current] = causal_health(scores[current], center, scale, alpha)
    return result


def confirmed_warning(point: np.ndarray, starts: np.ndarray, persistence: int) -> np.ndarray:
    result = np.zeros(len(point), dtype=bool)
    for current in d5.segment_slices(starts, STRIDE_SECONDS):
        count = 0
        for index in range(current.start or 0, current.stop):
            count = count + 1 if bool(point[index]) else 0
            if count >= persistence:
                result[index] = True
    return result


def first_warning_exposure_seconds(warnings: np.ndarray, starts: np.ndarray) -> object:
    accumulated = 0.0
    for current in d5.segment_slices(starts, STRIDE_SECONDS):
        local = np.flatnonzero(warnings[current])
        if len(local):
            return accumulated + float(local[0]) * STRIDE_SECONDS + WINDOW_SECONDS
        length = current.stop - (current.start or 0)
        accumulated += (length - 1) * STRIDE_SECONDS + WINDOW_SECONDS
    return ""


def evaluate(point: np.ndarray, warnings: np.ndarray, starts: np.ndarray) -> tuple[dict[str, object], list[dict[str, object]]]:
    episodes = d5.episode_details(warnings, starts, STRIDE_SECONDS, WINDOW_SECONDS)
    durations = [float(item["duration_seconds"]) for item in episodes]
    exposure_seconds = d5.monitoring_seconds(starts, STRIDE_SECONDS, WINDOW_SECONDS)
    first = first_warning_exposure_seconds(warnings, starts)
    return {
        "observation_count": len(point),
        "point_alarm_count": int(point.sum()),
        "point_alarm_rate": float(np.mean(point)) if len(point) else 0.0,
        "warning_count": int(warnings.sum()),
        "warning_rate": float(np.mean(warnings)) if len(warnings) else 0.0,
        "episode_count": len(episodes),
        "monitoring_hours": exposure_seconds / 3600.0,
        "episodes_per_day": len(episodes) / max(exposure_seconds / 86400.0, 1e-12),
        "mean_episode_duration_seconds": float(np.mean(durations)) if durations else 0.0,
        "p95_episode_duration_seconds": float(np.percentile(durations, 95)) if durations else 0.0,
        "max_episode_duration_seconds": max(durations, default=0.0),
        "motor_false_warning": str(bool(len(episodes))).lower(),
        "time_to_first_false_warning_hours": "" if first == "" else float(first) / 3600.0,
    }, episodes


def aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    observations = sum(int(row["test_observation_count"]) for row in rows)
    point = sum(int(row["test_point_alarm_count"]) for row in rows)
    warning = sum(int(row["test_warning_count"]) for row in rows)
    episodes = sum(int(row["test_episode_count"]) for row in rows)
    hours = sum(float(row["test_monitoring_hours"]) for row in rows)
    flagged = sum(row["test_motor_false_warning"] == "true" for row in rows)
    return {
        "test_observation_count": observations,
        "test_point_alarm_count": point,
        "test_point_alarm_rate": point / max(observations, 1),
        "test_warning_count": warning,
        "test_warning_rate": warning / max(observations, 1),
        "test_episode_count": episodes,
        "test_monitoring_hours": hours,
        "test_episodes_per_day": episodes / max(hours / 24.0, 1e-12),
        "test_flagged_motor_count": flagged,
        "test_motor_count": len(rows),
        "test_flagged_motor_rate": flagged / max(len(rows), 1),
    }


def cluster_bootstrap(rows: list[dict[str, object]], rng: np.random.Generator) -> tuple[float, float]:
    values = []
    for _ in range(BOOTSTRAP_REPLICATES):
        selected = rng.integers(0, len(rows), size=len(rows))
        episodes = sum(int(rows[index]["test_episode_count"]) for index in selected)
        hours = sum(float(rows[index]["test_monitoring_hours"]) for index in selected)
        values.append(episodes / max(hours / 24.0, 1e-12))
    return tuple(float(value) for value in np.quantile(values, [0.025, 0.975]))


def safe_id(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=D5_FEATURES)
    parser.add_argument("--d5-output", type=Path, default=D5_OUTPUT)
    parser.add_argument("--d6c-output", type=Path, default=D6C_OUTPUT)
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "outputs")
    args = parser.parse_args()
    features, d5_output, d6c_output, output = (args.features.resolve(), args.d5_output.resolve(), args.d6c_output.resolve(), args.output.resolve())
    output.mkdir(parents=True, exist_ok=True)

    locked = json.loads((d6c_output / "locked_rules.json").read_text(encoding="utf-8"))
    policies = {
        dataset: {
            "alpha": float(rule["alpha"]), "threshold": float(rule["threshold"]),
            "persistence": int(rule["persistence"]), "source": "D6C locked rule",
        }
        for dataset, rule in locked["rules"].items()
    }
    model_archive = np.load(d5_output / "continuous_models.npz")
    model = {
        field: model_archive[f"{PRIMARY_FEATURE_SET}__{field}"]
        for field in ("feature_indices", "median", "scale", "center", "inverse_covariance")
    }

    train_scores: dict[str, np.ndarray] = {}
    split_scores: dict[str, dict[str, np.ndarray]] = {"calibration": {}, "test": {}}
    split_starts: dict[str, dict[str, np.ndarray]] = {"calibration": {}, "test": {}}
    exclusion_rows: list[dict[str, object]] = []
    for motor in MOTORS:
        train_shard = d5.load_shard(features, motor, "train")
        train_values = d5.deterministic_rows(train_shard, TRAIN_ROWS_PER_MOTOR, True)
        train_scores[motor] = d5.score_matrix(train_values, model)
        for split in ("calibration", "test"):
            shard = d5.load_shard(features, motor, split)
            scores, starts = d5.score_shard(shard, model, True, 4096)
            split_scores[split][motor] = scores
            split_starts[split][motor] = starts
            total = len(shard["features"])
            raw_active = int(np.count_nonzero(np.asarray(shard["state"]) == 2))
            quality_excluded = int(
                np.count_nonzero(
                    (np.asarray(shard["state"]) == 2)
                    & (np.asarray(shard["quality_flag"]) != 0)
                )
            )
            active = len(scores)
            exclusion_rows.append({
                "motor": motor, "split": split, "all_valid_windows": total,
                "raw_active_windows": raw_active,
                "active_windows": active,
                "nonactive_excluded": total - raw_active,
                "near_full_scale_excluded": quality_excluded,
                "policy": "primary analysis requires operating_state=active and quality_flag=0",
            })

    motor_centers = {motor: float(np.median(np.log1p(train_scores[motor]))) for motor in MOTORS}
    onboard_residuals = np.concatenate([
        np.log1p(train_scores[motor]) - motor_centers[motor] for motor in MOTORS
    ])
    onboard_scale = robust_scale(onboard_residuals)
    pooled_logs = np.concatenate([np.log1p(train_scores[motor]) for motor in MOTORS])
    global_center = float(np.median(pooled_logs))
    global_scale = robust_scale(pooled_logs - global_center)
    parameter_rows: list[dict[str, object]] = [{
        "normalization": "global_train", "motor": "ALL", "center": global_center,
        "pooled_scale": global_scale, "train_score_count": len(pooled_logs),
        "fit_split": "train_only", "role": "strict_global_PNU-NOMV_score_alignment",
    }]
    for motor in MOTORS:
        parameter_rows.append({
            "normalization": "onboard_motor_train", "motor": motor, "center": motor_centers[motor],
            "pooled_scale": onboard_scale, "train_score_count": len(train_scores[motor]),
            "fit_split": "train_only", "role": "motor_center_only_with_PNU-NOMV_pooled_train_scale",
        })

    motor_rows: list[dict[str, object]] = []
    episode_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    decision_arrays: dict[str, np.ndarray] = {}
    for source_dataset, policy in sorted(policies.items()):
        for normalization in ("global_train", "onboard_motor_train"):
            config_id = f"{safe_id(source_dataset)}__{normalization}"
            for motor in MOTORS:
                center = global_center if normalization == "global_train" else motor_centers[motor]
                scale = global_scale if normalization == "global_train" else onboard_scale
                split_statistics: dict[str, dict[str, object]] = {}
                for split in ("calibration", "test"):
                    scores = split_scores[split][motor]
                    starts = split_starts[split][motor]
                    health = health_by_segments(scores, starts, center, scale, float(policy["alpha"]))
                    point = health > float(policy["threshold"])
                    warning = confirmed_warning(point, starts, int(policy["persistence"]))
                    statistics, episodes = evaluate(point, warning, starts)
                    split_statistics[split] = statistics
                    if split == "test":
                        decision_arrays[f"{config_id}__{motor}__point"] = point
                        decision_arrays[f"{config_id}__{motor}__warning"] = warning
                        for number, episode in enumerate(episodes, start=1):
                            episode_rows.append({
                                "config_id": config_id, "source_dataset": source_dataset,
                                "normalization": normalization, "motor": motor,
                                "episode_number": number, **episode,
                            })
                cal = split_statistics["calibration"]
                test = split_statistics["test"]
                calibration_rows.append({
                    "config_id": config_id, "source_dataset": source_dataset, "normalization": normalization,
                    "motor": motor, **{f"calibration_{key}": value for key, value in cal.items()},
                })
                motor_rows.append({
                    "config_id": config_id, "source_dataset": source_dataset, "normalization": normalization,
                    "alpha": policy["alpha"], "threshold": policy["threshold"], "persistence": policy["persistence"],
                    "motor": motor, **{f"test_{key}": value for key, value in test.items()},
                })

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    aggregate_rows: list[dict[str, object]] = []
    for config_id in sorted({row["config_id"] for row in motor_rows}):
        rows = [row for row in motor_rows if row["config_id"] == config_id]
        combined = aggregate(rows)
        lower, upper = cluster_bootstrap(rows, rng)
        exposure_days = float(combined["test_monitoring_hours"]) / 24.0
        zero_event_upper = -math.log(0.05) / exposure_days if int(combined["test_episode_count"]) == 0 else ""
        aggregate_rows.append({
            "config_id": config_id, "source_dataset": rows[0]["source_dataset"],
            "normalization": rows[0]["normalization"], "alpha": rows[0]["alpha"],
            "threshold": rows[0]["threshold"], "persistence": rows[0]["persistence"],
            **combined, "motor_cluster_bootstrap_episodes_per_day_95_lower": lower,
            "motor_cluster_bootstrap_episodes_per_day_95_upper": upper,
            "zero_episode_one_sided_95_poisson_upper_per_day": zero_event_upper,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        })

    d5_rows = read_csv(d5_output / "reference_results.csv")
    reference_ids = {
        "time19_global_static_q0.99_2of3",
        "time19_motor_causal_ewma_q0.99_2of3",
        "time19_motor_causal_ewma_q0.99_4of7",
        "time19_dual_guardrail_q0.99_2of3",
    }
    d5_reference = [
        row for row in d5_rows if row["motor"] == "ALL" and row["config_id"] in reference_ids
    ]
    if len(d5_reference) != len(reference_ids):
        raise ValueError("D5 reference configurations are incomplete")

    np.savez_compressed(output / "test_decisions.npz", **decision_arrays)
    write_csv(output / "commissioning_parameters.csv", parameter_rows)
    write_csv(output / "operating_state_exclusions.csv", exclusion_rows)
    write_csv(output / "calibration_diagnostics.csv", calibration_rows)
    write_csv(output / "motor_results.csv", motor_rows)
    write_csv(output / "aggregate_results.csv", aggregate_rows)
    if episode_rows:
        write_csv(output / "episode_results.csv", episode_rows)
    else:
        write_csv(output / "episode_results.csv", [{
            "config_id": "", "source_dataset": "", "normalization": "", "motor": "",
            "episode_number": "", "start_index": "", "end_index": "", "start_record": "",
            "end_record": "", "observation_count": "", "duration_seconds": "",
        }])
    write_csv(output / "d5_native_references.csv", d5_reference)

    compatibility = {
        "evaluated_transfer_unit": "locked D6C temporal policy (alpha, health threshold, consecutive persistence) on a PNU-NOMV-native time-only anomaly score after near-full-scale exclusion",
        "normalization_modes": {
            "global_train": "PNU-NOMV train-only global log-score center and scale",
            "onboard_motor_train": "PNU-NOMV train-only motor center plus PNU-NOMV pooled train residual scale",
        },
        "not_evaluated": "strict end-to-end zero-shot transfer of external feature normalizers or anomaly models",
        "reasons": [
            "PNU-NOMV amplitude uses raw IIS3DWB digital output counts while public datasets use other native units.",
            "PNU-NOMV frequency features are excluded from the primary analysis because the 250 Hz register-polling stream has no documented anti-alias filter.",
            "Sampling rates and channel/axis availability differ across datasets.",
        ],
        "interpretation": "This isolates temporal-policy portability after explicit PNU-NOMV commissioning alignment; it is not external-model zero-shot validation.",
    }
    (output / "domain_compatibility.json").write_text(json.dumps(compatibility, indent=2), encoding="utf-8")
    manifest = {
        "schema_version": "D6F-2.0", "status": "ANALYSIS_COMPLETE",
        "pnu_label_scope": "PNU-NOMV is normal-only; all persistent warnings are nuisance/false warnings under the available labels",
        "score_model": "PNU-NOMV D5 train-fitted time19 shrinkage Mahalanobis",
        "amplitude_quality_policy": "exclude windows containing abs(raw_count) >= 32000",
        "external_policy_source": "immutable D6C locked_rules.json",
        "external_policies": policies,
        "train_rows_per_motor_for_alignment": TRAIN_ROWS_PER_MOTOR,
        "primary_operating_state": "active", "window_seconds": WINDOW_SECONDS,
        "stride_seconds": STRIDE_SECONDS, "gap_policy": "health and persistence reset at every sequence gap",
        "continuous_warning_semantics": "warning becomes active at the persistence-confirmation sample and resets when threshold exceedance ends; any episode also counts as an asset-level latched false-warning event",
        "test_rows_used_for_fit_or_selection": 0,
        "bootstrap": {"unit": "motor", "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED},
        "input_sha256": {
            "d6c_locked_rules.json": sha256(d6c_output / "locked_rules.json"),
            "d5_continuous_models.npz": sha256(d5_output / "continuous_models.npz"),
            "d5_reference_results.csv": sha256(d5_output / "reference_results.csv"),
        },
        "claim_boundary": "PNU-NOMV normal-only data estimate nuisance-warning burden and policy portability, not fault sensitivity or warning lead time.",
    }
    (output / "policy_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({
        "configs": len(aggregate_rows), "motor_rows": len(motor_rows),
        "test_episode_rows": len(episode_rows),
        "active_test_windows_per_config": aggregate_rows[0]["test_observation_count"],
        "monitoring_hours_per_config": aggregate_rows[0]["test_monitoring_hours"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
