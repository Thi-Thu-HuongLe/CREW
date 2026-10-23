#!/usr/bin/env python3
"""Stage D8B: stress-test D8A sequential policies on PNU-NOMV normal-only data."""

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
D6F_DIR = PNU_DIR / "stage_d6f"
D6F_OUTPUT = D6F_DIR / "outputs"
D8A_OUTPUT = PROJECT_CODE / "Real-Fault-Benchmarks" / "stage_d8a" / "outputs"
sys.path.insert(0, str(D6F_DIR))
import run_pnu_transfer_stress as d6f  # noqa: E402


MOTORS = d6f.MOTORS
STRIDE_SECONDS = d6f.STRIDE_SECONDS
WINDOW_SECONDS = d6f.WINDOW_SECONDS
TRAIN_ROWS_PER_MOTOR = d6f.TRAIN_ROWS_PER_MOTOR
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 20261019
METHODS = ("crew_ewma", "robust_shewhart", "robust_page_cusum", "robust_causal_ma")
NORMALIZATIONS = ("global_train", "onboard_motor_train")
PRIMARY_FEATURE_SET = "time19"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_id(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_")


def sequential_health(z: np.ndarray, method: str, parameter: float | int | None) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    if method == "robust_shewhart":
        return np.maximum(z, 0.0)
    if method == "crew_ewma":
        alpha = float(parameter)
        result = np.empty(len(z), dtype=np.float64)
        previous = 0.0
        for index, value in enumerate(z):
            previous = alpha * float(value) + (1.0 - alpha) * previous
            result[index] = max(0.0, previous)
        return result
    if method == "robust_page_cusum":
        allowance = float(parameter)
        result = np.empty(len(z), dtype=np.float64)
        previous = 0.0
        for index, value in enumerate(z):
            previous = max(0.0, previous + float(value) - allowance)
            result[index] = previous
        return result
    if method == "robust_causal_ma":
        window = int(parameter)
        positive = np.maximum(z, 0.0)
        prefix = np.concatenate(([0.0], np.cumsum(positive, dtype=np.float64)))
        result = np.empty(len(z), dtype=np.float64)
        for index in range(len(z)):
            start = max(0, index + 1 - window)
            result[index] = (prefix[index + 1] - prefix[start]) / (index + 1 - start)
        return result
    raise ValueError(f"unsupported method: {method}")


def health_by_segments(
    scores: np.ndarray,
    starts: np.ndarray,
    center: float,
    scale: float,
    method: str,
    parameter: float | int | None,
) -> np.ndarray:
    logged = np.log1p(np.asarray(scores, dtype=np.float64))
    z = np.clip((logged - center) / scale, -10.0, 50.0)
    result = np.empty(len(scores), dtype=np.float64)
    for current in d6f.d5.segment_slices(starts, STRIDE_SECONDS):
        result[current] = sequential_health(z[current], method, parameter)
    return result


def cluster_bootstrap(rows: list[dict[str, object]], rng: np.random.Generator) -> tuple[float, float]:
    values = []
    for _ in range(BOOTSTRAP_REPLICATES):
        selected = rng.integers(0, len(rows), size=len(rows))
        episodes = sum(int(rows[index]["test_episode_count"]) for index in selected)
        hours = sum(float(rows[index]["test_monitoring_hours"]) for index in selected)
        values.append(episodes / max(hours / 24.0, 1e-12))
    return tuple(float(value) for value in np.quantile(values, [0.025, 0.975]))


def paired_bootstrap_difference(
    crew_rows: list[dict[str, object]],
    comparator_rows: list[dict[str, object]],
    rng: np.random.Generator,
) -> tuple[float, float]:
    crew_by_motor = {str(row["motor"]): row for row in crew_rows}
    comparator_by_motor = {str(row["motor"]): row for row in comparator_rows}
    motors = sorted(crew_by_motor)
    if motors != sorted(comparator_by_motor):
        raise ValueError("paired motor sets differ")
    differences = []
    for _ in range(BOOTSTRAP_REPLICATES):
        selected = rng.integers(0, len(motors), size=len(motors))
        selected_motors = [motors[index] for index in selected]
        crew_episodes = sum(int(crew_by_motor[m]["test_episode_count"]) for m in selected_motors)
        crew_hours = sum(float(crew_by_motor[m]["test_monitoring_hours"]) for m in selected_motors)
        comparator_episodes = sum(int(comparator_by_motor[m]["test_episode_count"]) for m in selected_motors)
        comparator_hours = sum(float(comparator_by_motor[m]["test_monitoring_hours"]) for m in selected_motors)
        differences.append(
            comparator_episodes / max(comparator_hours / 24.0, 1e-12)
            - crew_episodes / max(crew_hours / 24.0, 1e-12)
        )
    return tuple(float(value) for value in np.quantile(differences, [0.025, 0.975]))


def load_policies(path: Path) -> dict[tuple[str, str], dict[str, object]]:
    rows = read_csv(path / "sequential_method_summary.csv")
    policies: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        method, dataset = row["method"], row["dataset"]
        parameter_name = ""
        parameter: float | int | None = None
        if method == "crew_ewma":
            parameter_name = "alpha"
            parameter = float(row["selected_parameter"])
        elif method == "robust_page_cusum":
            parameter_name = "allowance_k"
            parameter = float(row["selected_parameter"])
        elif method == "robust_causal_ma":
            parameter_name = "window"
            parameter = int(float(row["selected_parameter"]))
        elif method != "robust_shewhart":
            raise ValueError(f"unexpected D8A method: {method}")
        policies[(method, dataset)] = {
            "method": method,
            "source_dataset": dataset,
            "parameter_name": parameter_name,
            "parameter": parameter,
            "threshold": float(row["threshold"]),
            "persistence": int(row["persistence"]),
            "source": "D8A locked train/calibration policy",
        }
    expected = {(method, dataset) for method in METHODS for dataset in ("NASA-IMS", "PRONOSTIA", "XJTU-SY")}
    if set(policies) != expected:
        raise ValueError("D8A policy matrix is incomplete")
    return policies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=D5_FEATURES)
    parser.add_argument("--d5-output", type=Path, default=D5_OUTPUT)
    parser.add_argument("--d8a-output", type=Path, default=D8A_OUTPUT)
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "outputs")
    args = parser.parse_args()
    features = args.features.resolve()
    d5_output = args.d5_output.resolve()
    d8a_output = args.d8a_output.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    policies = load_policies(d8a_output)
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
        train_shard = d6f.d5.load_shard(features, motor, "train")
        train_values = d6f.d5.deterministic_rows(train_shard, TRAIN_ROWS_PER_MOTOR, True)
        train_scores[motor] = d6f.d5.score_matrix(train_values, model)
        for split in ("calibration", "test"):
            shard = d6f.d5.load_shard(features, motor, split)
            scores, starts = d6f.d5.score_shard(shard, model, True, 4096)
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
            exclusion_rows.append({
                "motor": motor, "split": split, "all_valid_windows": total,
                "raw_active_windows": raw_active,
                "active_windows": len(scores),
                "nonactive_excluded": total - raw_active,
                "near_full_scale_excluded": quality_excluded,
                "policy": "primary analysis requires operating_state=active and quality_flag=0",
            })

    motor_centers = {motor: float(np.median(np.log1p(train_scores[motor]))) for motor in MOTORS}
    onboard_residuals = np.concatenate([np.log1p(train_scores[motor]) - motor_centers[motor] for motor in MOTORS])
    onboard_scale = d6f.robust_scale(onboard_residuals)
    pooled_logs = np.concatenate([np.log1p(train_scores[motor]) for motor in MOTORS])
    global_center = float(np.median(pooled_logs))
    global_scale = d6f.robust_scale(pooled_logs - global_center)
    alignment_rows: list[dict[str, object]] = [{
        "normalization": "global_train", "motor": "ALL", "center": global_center,
        "pooled_scale": global_scale, "train_score_count": len(pooled_logs),
        "fit_split": "train_only", "role": "strict_global_PNU-NOMV_score_alignment",
    }]
    for motor in MOTORS:
        alignment_rows.append({
            "normalization": "onboard_motor_train", "motor": motor, "center": motor_centers[motor],
            "pooled_scale": onboard_scale, "train_score_count": len(train_scores[motor]),
            "fit_split": "train_only", "role": "motor_center_only_with_PNU-NOMV_pooled_train_scale",
        })

    motor_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    episode_rows: list[dict[str, object]] = []
    decisions: dict[str, np.ndarray] = {}
    for (method, source_dataset), policy in sorted(policies.items()):
        for normalization in NORMALIZATIONS:
            config_id = f"{method}__{safe_id(source_dataset)}__{normalization}"
            for motor in MOTORS:
                center = global_center if normalization == "global_train" else motor_centers[motor]
                scale = global_scale if normalization == "global_train" else onboard_scale
                split_statistics: dict[str, dict[str, object]] = {}
                for split in ("calibration", "test"):
                    scores = split_scores[split][motor]
                    starts = split_starts[split][motor]
                    health = health_by_segments(scores, starts, center, scale, method, policy["parameter"])
                    point = health > float(policy["threshold"])
                    warning = d6f.confirmed_warning(point, starts, int(policy["persistence"]))
                    statistics, episodes = d6f.evaluate(point, warning, starts)
                    split_statistics[split] = statistics
                    if split == "test":
                        decisions[f"{config_id}__{motor}__point"] = point
                        decisions[f"{config_id}__{motor}__warning"] = warning
                        for number, episode in enumerate(episodes, start=1):
                            episode_rows.append({
                                "config_id": config_id, "method": method, "source_dataset": source_dataset,
                                "normalization": normalization, "motor": motor, "episode_number": number, **episode,
                            })
                common = {
                    "config_id": config_id, "method": method, "source_dataset": source_dataset,
                    "normalization": normalization, "parameter_name": policy["parameter_name"],
                    "parameter_value": "" if policy["parameter"] is None else policy["parameter"],
                    "threshold": policy["threshold"], "persistence": policy["persistence"], "motor": motor,
                }
                calibration_rows.append({**common, **{f"calibration_{key}": value for key, value in split_statistics["calibration"].items()}})
                motor_rows.append({**common, **{f"test_{key}": value for key, value in split_statistics["test"].items()}})

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    aggregate_rows: list[dict[str, object]] = []
    for config_id in sorted({str(row["config_id"]) for row in motor_rows}):
        rows = [row for row in motor_rows if row["config_id"] == config_id]
        combined = d6f.aggregate(rows)
        lower, upper = cluster_bootstrap(rows, rng)
        exposure_days = float(combined["test_monitoring_hours"]) / 24.0
        zero_upper = -math.log(0.05) / exposure_days if int(combined["test_episode_count"]) == 0 else ""
        first = rows[0]
        aggregate_rows.append({
            "config_id": config_id, "method": first["method"], "source_dataset": first["source_dataset"],
            "normalization": first["normalization"], "parameter_name": first["parameter_name"],
            "parameter_value": first["parameter_value"], "threshold": first["threshold"],
            "persistence": first["persistence"], **combined,
            "motor_cluster_bootstrap_episodes_per_day_95_lower": lower,
            "motor_cluster_bootstrap_episodes_per_day_95_upper": upper,
            "zero_episode_one_sided_95_poisson_upper_per_day": zero_upper,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        })

    aggregate_lookup = {
        (str(row["method"]), str(row["source_dataset"]), str(row["normalization"])): row
        for row in aggregate_rows
    }
    paired_rows: list[dict[str, object]] = []
    paired_rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    for source_dataset in ("NASA-IMS", "PRONOSTIA", "XJTU-SY"):
        for normalization in NORMALIZATIONS:
            crew = [row for row in motor_rows if row["method"] == "crew_ewma" and row["source_dataset"] == source_dataset and row["normalization"] == normalization]
            crew_aggregate = aggregate_lookup[("crew_ewma", source_dataset, normalization)]
            for method in METHODS[1:]:
                comparator = [row for row in motor_rows if row["method"] == method and row["source_dataset"] == source_dataset and row["normalization"] == normalization]
                comparator_aggregate = aggregate_lookup[(method, source_dataset, normalization)]
                lower, upper = paired_bootstrap_difference(crew, comparator, paired_rng)
                paired_rows.append({
                    "source_dataset": source_dataset, "normalization": normalization,
                    "primary_method": "crew_ewma", "comparator_method": method,
                    "crew_episodes_per_day": crew_aggregate["test_episodes_per_day"],
                    "comparator_episodes_per_day": comparator_aggregate["test_episodes_per_day"],
                    "comparator_minus_crew_episodes_per_day": float(comparator_aggregate["test_episodes_per_day"]) - float(crew_aggregate["test_episodes_per_day"]),
                    "paired_motor_bootstrap_difference_95_lower": lower,
                    "paired_motor_bootstrap_difference_95_upper": upper,
                    "crew_flagged_motors": crew_aggregate["test_flagged_motor_count"],
                    "comparator_flagged_motors": comparator_aggregate["test_flagged_motor_count"],
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                })

    public_summary = read_csv(d8a_output / "sequential_method_summary.csv")
    public_lookup = {(row["method"], row["dataset"]): row for row in public_summary}
    tradeoff_rows: list[dict[str, object]] = []
    for row in aggregate_rows:
        public = public_lookup[(str(row["method"]), str(row["source_dataset"]))]
        tradeoff_rows.append({
            "method": row["method"], "source_dataset": row["source_dataset"], "normalization": row["normalization"],
            "public_known_failed_detected": public["test_known_failed_detected"],
            "public_known_failed_total": public["test_known_failed_total"],
            "public_median_terminal_lead_seconds": public["test_lead_seconds_median"],
            "pnu_test_monitoring_hours": row["test_monitoring_hours"],
            "pnu_nuisance_episodes_per_day": row["test_episodes_per_day"],
            "pnu_flagged_motors": row["test_flagged_motor_count"], "pnu_motor_count": row["test_motor_count"],
            "interpretation": "fault evidence and normal-only nuisance evidence are complementary; neither supplies physical-onset accuracy",
        })

    np.savez_compressed(output / "test_decisions.npz", **decisions)
    write_csv(output / "commissioning_parameters.csv", alignment_rows)
    write_csv(output / "operating_state_exclusions.csv", exclusion_rows)
    write_csv(output / "calibration_diagnostics.csv", calibration_rows)
    write_csv(output / "motor_results.csv", motor_rows)
    write_csv(output / "aggregate_results.csv", aggregate_rows)
    write_csv(output / "paired_nuisance_comparison.csv", paired_rows)
    write_csv(output / "cross_evidence_tradeoff.csv", tradeoff_rows)
    if episode_rows:
        write_csv(output / "episode_results.csv", episode_rows)
    else:
        write_csv(output / "episode_results.csv", [{
            "config_id": "", "method": "", "source_dataset": "", "normalization": "", "motor": "",
            "episode_number": "", "start_index": "", "end_index": "", "start_record": "",
            "end_record": "", "observation_count": "", "duration_seconds": "",
        }])

    manifest = {
        "schema_version": "D8B-2.0", "status": "ANALYSIS_COMPLETE",
        "pnu_label_scope": "PNU-NOMV is normal-only; every persistent episode is a nuisance warning under available labels",
        "score_model": "PNU-NOMV D5 train-fitted time19 shrinkage Mahalanobis",
        "amplitude_quality_policy": "exclude windows containing abs(raw_count) >= 32000",
        "external_policy_source": "D8A locked train/calibration sequential policies",
        "methods": list(METHODS), "source_datasets": ["NASA-IMS", "PRONOSTIA", "XJTU-SY"],
        "normalizations": list(NORMALIZATIONS), "train_rows_per_motor_for_alignment": TRAIN_ROWS_PER_MOTOR,
        "primary_operating_state": "active", "window_seconds": WINDOW_SECONDS, "stride_seconds": STRIDE_SECONDS,
        "gap_policy": "health state, moving window, CUSUM state, and persistence reset at every sequence gap",
        "test_rows_used_for_fit_or_selection": 0, "pnu_calibration_used_for_retuning": False,
        "bootstrap": {"unit": "motor", "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED},
        "input_sha256": {
            "d8a_sequential_method_summary.csv": sha256(d8a_output / "sequential_method_summary.csv"),
            "d5_continuous_models.npz": sha256(d5_output / "continuous_models.npz"),
            "d6f_aggregate_results.csv": sha256(D6F_OUTPUT / "aggregate_results.csv"),
        },
        "claim_boundary": "D8B compares nuisance-warning burden after PNU-NOMV train-only score alignment; it does not measure PNU-NOMV fault sensitivity or physical fault onset.",
    }
    (output / "policy_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "BUILT", "configs": len(aggregate_rows), "motor_rows": len(motor_rows),
        "test_decision_arrays": len(decisions), "active_test_windows_per_config": aggregate_rows[0]["test_observation_count"],
        "monitoring_hours_per_config": aggregate_rows[0]["test_monitoring_hours"], "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
