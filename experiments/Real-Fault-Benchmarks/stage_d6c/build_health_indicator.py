#!/usr/bin/env python3
"""Build a train-fitted causal health indicator and lock calibration warning rules."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
D6B_OUTPUT = SCRIPT_DIR.parent / "stage_d6b" / "outputs"
ALPHA_CANDIDATES = (0.05, 0.10, 0.20, 0.30, 0.50)
QUANTILE_CANDIDATES = (0.990, 0.995, 0.999, 1.000)
PERSISTENCE_CANDIDATES = (3, 5, 10)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def group_key(row: dict[str, str]) -> tuple[str, str]:
    return row["dataset"], row["condition_id"]


def outcome(row: dict[str, str]) -> tuple[str, str]:
    dataset, asset = row["dataset"], row["asset_id"]
    if dataset in {"XJTU-SY", "PRONOSTIA"}:
        return "known_failed", row["terminal_fault_label"]
    if dataset == "NASA-IMS":
        bearing = asset.rsplit(":", 1)[-1]
        if row["run_id"] == "NASA-IMS:set_1":
            labels = {
                "bearing1": ("known_control", "no_documented_terminal_fault"),
                "bearing2": ("known_control", "no_documented_terminal_fault"),
                "bearing3": ("known_failed", "inner_race"),
                "bearing4": ("known_failed", "roller_element"),
            }
            return labels[bearing]
        if row["run_id"] == "NASA-IMS:set_2":
            return ("known_failed", "outer_race") if bearing == "bearing1" else ("known_control", "no_documented_terminal_fault")
        return "unknown", "unknown_due_to_archive_provenance_conflict"
    return "excluded", "stationary_auxiliary"


def robust_scale(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    mad = 1.4826 * float(np.median(np.abs(values - median)))
    q25, q75 = np.quantile(values, [0.25, 0.75])
    return max(mad, float((q75 - q25) / 1.349), 1e-3)


def causal_health(log_score: np.ndarray, center: float, scale: float, alpha: float) -> np.ndarray:
    z_score = np.clip((log_score - center) / scale, -10.0, 50.0)
    smoothed = np.empty_like(z_score, dtype=np.float64)
    previous = 0.0
    for index, value in enumerate(z_score):
        previous = alpha * float(value) + (1.0 - alpha) * previous
        smoothed[index] = max(0.0, previous)
    return smoothed


def correlation_with_time(values: np.ndarray) -> float:
    if values.size < 3 or float(np.std(values)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(np.arange(values.size, dtype=float), values)[0, 1])


def first_persistent(above: np.ndarray, persistence: int, start: int = 0, end: int | None = None) -> int | None:
    end = len(above) if end is None else min(end, len(above))
    count = 0
    for index in range(max(0, start), end):
        count = count + 1 if bool(above[index]) else 0
        if count >= persistence:
            return index - persistence + 1
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d6b-output", type=Path, default=D6B_OUTPUT)
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "outputs")
    args = parser.parse_args()
    d6b, output_dir = args.d6b_output.resolve(), args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(d6b / "feature_metadata.csv")
    robust_distance = np.load(d6b / "robust_distance.npy", mmap_mode="r")
    if len(rows) != robust_distance.shape[0]:
        raise ValueError("D6B metadata/score mismatch")

    asset_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row["early_warning_eligible"] == "true":
            asset_indices[row["asset_id"]].append(index)
    for asset, indices in asset_indices.items():
        indices.sort(key=lambda value: int(rows[value]["sequence_index"]))
        if [int(rows[value]["sequence_index"]) for value in indices] != list(range(1, len(indices) + 1)):
            raise ValueError(f"non-contiguous sequence for {asset}")

    train_assets_by_group: dict[tuple[str, str], list[str]] = defaultdict(list)
    for asset, indices in asset_indices.items():
        row = rows[indices[0]]
        if row["split"] == "train":
            train_assets_by_group[group_key(row)].append(asset)

    group_parameters: dict[tuple[str, str], dict[str, float | int]] = {}
    group_parameter_rows: list[dict[str, object]] = []
    for key, assets in sorted(train_assets_by_group.items()):
        ten_percent_counts = [max(5, int(math.ceil(0.10 * len(asset_indices[asset])))) for asset in assets]
        commissioning_count = max(5, int(round(float(np.median(ten_percent_counts)))))
        residuals: list[float] = []
        for asset in assets:
            indices = asset_indices[asset]
            count = min(commissioning_count, len(indices))
            values = np.log1p(np.asarray(robust_distance[indices[:count]], dtype=np.float64))
            center = float(np.median(values))
            residuals.extend((values - center).tolist())
        scale = robust_scale(np.asarray(residuals))
        group_parameters[key] = {"commissioning_count": commissioning_count, "pooled_log_scale": scale}
        group_parameter_rows.append({
            "dataset": key[0], "condition_id": key[1], "train_asset_count": len(assets),
            "commissioning_count": commissioning_count, "pooled_log_scale": scale,
            "fit_split": "train_only",
        })

    def asset_log_and_center(asset: str) -> tuple[np.ndarray, float, int, float]:
        indices = asset_indices[asset]
        key = group_key(rows[indices[0]])
        parameters = group_parameters[key]
        count = min(int(parameters["commissioning_count"]), len(indices))
        values = np.log1p(np.asarray(robust_distance[indices], dtype=np.float64))
        center = float(np.median(values[:count]))
        return values, center, count, float(parameters["pooled_log_scale"])

    alpha_rows: list[dict[str, object]] = []
    selected_alpha: dict[str, float] = {}
    datasets = sorted({rows[indices[0]]["dataset"] for indices in asset_indices.values()})
    for dataset in datasets:
        failed_train_assets = [
            asset for asset, indices in asset_indices.items()
            if rows[indices[0]]["dataset"] == dataset
            and rows[indices[0]]["split"] == "train"
            and outcome(rows[indices[0]])[0] == "known_failed"
        ]
        if not failed_train_assets:
            raise ValueError(f"no known-failed training trajectory for {dataset}")
        candidates: list[dict[str, object]] = []
        for alpha in ALPHA_CANDIDATES:
            correlations, gains = [], []
            for asset in failed_train_assets:
                values, center, count, scale = asset_log_and_center(asset)
                health = causal_health(values, center, scale, alpha)
                tail_count = max(count, int(math.ceil(0.10 * len(health))))
                correlations.append(correlation_with_time(health[count:]))
                gains.append(float(np.median(health[-tail_count:]) - np.median(health[:count])))
            median_correlation = float(np.median(correlations))
            median_terminal_gain = float(np.median(gains))
            effective_lag = (1.0 - alpha) / alpha
            selection_score = median_correlation + 0.10 * math.tanh(median_terminal_gain / 5.0) - 0.002 * effective_lag
            candidates.append({
                "dataset": dataset, "alpha": alpha, "known_failed_train_assets": len(failed_train_assets),
                "median_time_correlation": median_correlation, "median_terminal_gain": median_terminal_gain,
                "effective_lag_observations": effective_lag, "selection_score": selection_score,
            })
        best = max(candidates, key=lambda row: (float(row["selection_score"]), float(row["alpha"])))
        selected_alpha[dataset] = float(best["alpha"])
        for row in candidates:
            row["selected"] = str(row is best).lower()
            alpha_rows.append(row)

    health = np.full(len(rows), np.nan, dtype=np.float32)
    asset_runtime: dict[str, dict[str, object]] = {}
    for asset, indices in asset_indices.items():
        first = rows[indices[0]]
        values, center, count, scale = asset_log_and_center(asset)
        alpha = selected_alpha[first["dataset"]]
        trajectory = causal_health(values, center, scale, alpha)
        health[indices] = trajectory.astype(np.float32)
        asset_runtime[asset] = {
            "indices": indices, "health": trajectory, "center": center,
            "commissioning_count": count, "scale": scale, "alpha": alpha,
            "outcome": outcome(first),
        }

    rule_search_rows: list[dict[str, object]] = []
    locked_rules: dict[str, dict[str, object]] = {}
    for dataset in datasets:
        calibration_assets = [
            asset for asset, indices in asset_indices.items()
            if rows[indices[0]]["dataset"] == dataset and rows[indices[0]]["split"] == "calibration"
        ]
        failed_assets = [asset for asset in calibration_assets if asset_runtime[asset]["outcome"][0] == "known_failed"]
        reference_end: dict[str, int] = {}
        reference_values: list[float] = []
        for asset in calibration_assets:
            trajectory = np.asarray(asset_runtime[asset]["health"])
            commissioning = int(asset_runtime[asset]["commissioning_count"])
            status = str(asset_runtime[asset]["outcome"][0])
            end = len(trajectory) if status == "known_control" else max(commissioning, int(math.ceil(0.20 * len(trajectory))))
            reference_end[asset] = end
            reference_values.extend(trajectory[:end].tolist())
        reference_array = np.asarray(reference_values, dtype=np.float64)
        candidates: list[dict[str, object]] = []
        for quantile in QUANTILE_CANDIDATES:
            threshold = float(np.quantile(reference_array, quantile))
            if quantile == 1.0:
                threshold = float(np.nextafter(threshold, np.inf))
            for persistence in PERSISTENCE_CANDIDATES:
                false_assets = 0
                for asset in calibration_assets:
                    trajectory = np.asarray(asset_runtime[asset]["health"])
                    commissioning = int(asset_runtime[asset]["commissioning_count"])
                    boundary_onset = first_persistent(
                        trajectory > threshold, persistence, commissioning,
                        min(len(trajectory), reference_end[asset] + persistence - 1),
                    )
                    if boundary_onset is not None and boundary_onset < reference_end[asset]:
                        false_assets += 1
                detected, lead_fractions = 0, []
                for asset in failed_assets:
                    trajectory = np.asarray(asset_runtime[asset]["health"])
                    onset = first_persistent(
                        trajectory > threshold, persistence,
                        max(int(asset_runtime[asset]["commissioning_count"]), reference_end[asset]),
                    )
                    if onset is not None:
                        detected += 1
                        confirmation = onset + persistence - 1
                        lead_fractions.append((len(trajectory) - 1 - confirmation) / max(len(trajectory) - 1, 1))
                candidates.append({
                    "dataset": dataset, "quantile": quantile, "threshold": threshold,
                    "persistence": persistence, "calibration_asset_count": len(calibration_assets),
                    "calibration_failed_asset_count": len(failed_assets),
                    "reference_false_asset_count": false_assets,
                    "failed_detected_count": detected,
                    "failed_detection_rate": detected / len(failed_assets) if failed_assets else "",
                    "median_lead_fraction": float(np.median(lead_fractions)) if lead_fractions else -1.0,
                })
        best = min(
            candidates,
            key=lambda row: (
                int(row["reference_false_asset_count"]),
                -int(row["failed_detected_count"]),
                -float(row["median_lead_fraction"]),
                float(row["quantile"]),
                int(row["persistence"]),
            ),
        )
        locked_rules[dataset] = {
            "alpha": selected_alpha[dataset], "threshold_quantile": best["quantile"],
            "threshold": best["threshold"], "persistence": best["persistence"],
            "threshold_semantics": "empirical_max_plus_epsilon" if float(best["quantile"]) == 1.0 else "empirical_quantile",
            "fit_splits": {"alpha": "train", "threshold_and_persistence": "calibration"},
            "reference_policy": "known controls use the full trajectory; known failures use the first 20%; test is unused",
        }
        for row in candidates:
            row["selected"] = str(row is best).lower()
            rule_search_rows.append(row)

    point_alarm = np.zeros(len(rows), dtype=bool)
    warning = np.zeros(len(rows), dtype=bool)
    trajectory_rows: list[dict[str, object]] = []
    for asset, runtime in sorted(asset_runtime.items()):
        indices = list(runtime["indices"])
        first = rows[indices[0]]
        trajectory = np.asarray(runtime["health"])
        rule = locked_rules[first["dataset"]]
        threshold, persistence = float(rule["threshold"]), int(rule["persistence"])
        above = trajectory > threshold
        point_alarm[indices] = above
        commissioning = int(runtime["commissioning_count"])
        onset = first_persistent(above, persistence, commissioning)
        confirmation = None if onset is None else onset + persistence - 1
        if confirmation is not None:
            warning[np.asarray(indices[confirmation:], dtype=int)] = True
        status, asset_label = runtime["outcome"]
        terminal_elapsed = float(rows[indices[-1]]["elapsed_seconds"])
        onset_elapsed = float(rows[indices[onset]]["elapsed_seconds"]) if onset is not None else None
        confirmation_elapsed = float(rows[indices[confirmation]]["elapsed_seconds"]) if confirmation is not None else None
        failed = status == "known_failed"
        trajectory_rows.append({
            "dataset": first["dataset"], "condition_id": first["condition_id"], "split": first["split"],
            "run_id": first["run_id"], "asset_id": asset, "outcome_status": status,
            "asset_terminal_label": asset_label, "observation_count": len(indices),
            "commissioning_count": commissioning, "commissioning_log_center": runtime["center"],
            "pooled_train_log_scale": runtime["scale"], "ewma_alpha": runtime["alpha"],
            "threshold_quantile": rule["threshold_quantile"], "threshold": threshold,
            "persistence": persistence, "detected": str(onset is not None).lower(),
            "evidence_start_zero_based": "" if onset is None else onset,
            "evidence_start_sequence_index": "" if onset is None else rows[indices[onset]]["sequence_index"],
            "evidence_start_feature_row_id": "" if onset is None else indices[onset],
            "evidence_start_elapsed_seconds": "" if onset_elapsed is None else onset_elapsed,
            "warning_confirmation_zero_based": "" if confirmation is None else confirmation,
            "warning_confirmation_sequence_index": "" if confirmation is None else rows[indices[confirmation]]["sequence_index"],
            "warning_confirmation_feature_row_id": "" if confirmation is None else indices[confirmation],
            "warning_confirmation_elapsed_seconds": "" if confirmation_elapsed is None else confirmation_elapsed,
            "terminal_elapsed_seconds": terminal_elapsed,
            "lead_observations_to_terminal": "" if confirmation is None or not failed else len(indices) - 1 - confirmation,
            "lead_seconds_to_terminal": "" if confirmation_elapsed is None or not failed else terminal_elapsed - confirmation_elapsed,
            "evidence_start_life_fraction": "" if onset is None else onset / max(len(indices) - 1, 1),
            "warning_confirmation_life_fraction": "" if confirmation is None else confirmation / max(len(indices) - 1, 1),
            "early_warning_before_20pct": str(confirmation is not None and confirmation / max(len(indices) - 1, 1) < 0.20).lower(),
            "interpretation": "algorithmic_evidence_start_and_causal_warning_confirmation_not_physical_ground_truth" if onset is not None else "no_locked_rule_detection",
        })

    np.save(output_dir / "health_indicator.npy", health)
    np.save(output_dir / "point_alarm.npy", point_alarm)
    np.save(output_dir / "latched_warning.npy", warning)

    def write_rows(name: str, values: list[dict[str, object]]) -> None:
        with (output_dir / name).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader(); writer.writerows(values)

    write_rows("group_parameters.csv", group_parameter_rows)
    write_rows("alpha_selection.csv", alpha_rows)
    write_rows("rule_search.csv", rule_search_rows)
    write_rows("trajectory_results.csv", trajectory_rows)
    lock_manifest = {
        "status": "LOCKED", "schema_version": "D6C-1.0", "indicator": "causal EWMA of positive train-scaled log1p robust distance",
        "commissioning_policy": "count is median train-run 10% count per dataset-condition; asset center uses only that initial commissioning prefix",
        "scale_policy": "pooled robust log-score residual scale fitted on train commissioning prefixes only",
        "winsorization": [-10.0, 50.0], "alpha_candidates": list(ALPHA_CANDIDATES),
        "threshold_quantile_candidates": list(QUANTILE_CANDIDATES), "persistence_candidates": list(PERSISTENCE_CANDIDATES),
        "rules": locked_rules,
        "test_rows_used_for_global_fit_or_selection": 0,
        "test_commissioning_note": "Each deployed/test asset uses its locked-length unlabeled initial prefix only for center adaptation; it does not alter scale, alpha, threshold, or persistence.",
        "onset_semantics": "Evidence start is the first sample in a locked persistent exceedance; causal warning time is the final sample that confirms persistence. Neither is annotated physical fault onset.",
    }
    (output_dir / "locked_rules.json").write_text(json.dumps(lock_manifest, indent=2), encoding="utf-8")
    fit_audit = {
        "group_parameter_assets": {f"{key[0]}::{key[1]}": sorted(value) for key, value in train_assets_by_group.items()},
        "alpha_selection_assets": {
            dataset: sorted(
                asset for asset, indices in asset_indices.items()
                if rows[indices[0]]["dataset"] == dataset and rows[indices[0]]["split"] == "train"
                and outcome(rows[indices[0]])[0] == "known_failed"
            ) for dataset in datasets
        },
        "calibration_rule_assets": {
            dataset: sorted(
                asset for asset, indices in asset_indices.items()
                if rows[indices[0]]["dataset"] == dataset and rows[indices[0]]["split"] == "calibration"
            ) for dataset in datasets
        },
        "test_assets_used_for_fit": [], "test_rows_used_for_fit": 0,
    }
    (output_dir / "fit_audit.json").write_text(json.dumps(fit_audit, indent=2), encoding="utf-8")

    summary_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in trajectory_rows:
        summary_groups[f"{row['dataset']}::{row['split']}::{row['outcome_status']}"] .append(row)
    summaries = {}
    for key, values in sorted(summary_groups.items()):
        detected = [row for row in values if row["detected"] == "true"]
        lead_seconds = [float(row["lead_seconds_to_terminal"]) for row in detected if row["lead_seconds_to_terminal"] != ""]
        summaries[key] = {
            "trajectory_count": len(values), "detected_count": len(detected),
            "detection_rate": len(detected) / len(values),
            "median_lead_seconds_to_terminal_known_failed": float(np.median(lead_seconds)) if lead_seconds else None,
            "early_warning_count_before_20pct": sum(row["early_warning_before_20pct"] == "true" for row in values),
        }
    summary = {
        "status": "PASS", "trajectory_count": len(trajectory_rows),
        "fit_partition_counts": dict(Counter(row["split"] for row in trajectory_rows)),
        "outcome_counts": dict(Counter(str(row["outcome_status"]) for row in trajectory_rows)),
        "results": summaries,
        "claim_boundary": "Detection means persistent health-index evidence before a known terminal failure. It is not accuracy against annotated physical onset, which is unavailable.",
    }
    (output_dir / "d6c_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"locked_rules": locked_rules, **summary}, indent=2))


if __name__ == "__main__":
    main()
