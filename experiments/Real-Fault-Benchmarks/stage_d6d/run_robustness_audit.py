#!/usr/bin/env python3
"""Run the locked-rule robustness and baseline audit for Stage D6D."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
D6B_OUTPUT = SCRIPT_DIR.parent / "stage_d6b" / "outputs"
D6C_OUTPUT = SCRIPT_DIR.parent / "stage_d6c" / "outputs"
THRESHOLD_MULTIPLIERS = (0.9, 1.0, 1.1)
PERSISTENCE_VALUES = (1, 3, 5)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def first_persistent(values: np.ndarray, persistence: int, start: int = 0, stop: int | None = None) -> int | None:
    stop = len(values) if stop is None else min(stop, len(values))
    run = 0
    for index in range(max(0, start), stop):
        run = run + 1 if bool(values[index]) else 0
        if run >= persistence:
            return index - persistence + 1
    return None


def reference_end(result: dict[str, str]) -> int:
    count = int(result["observation_count"])
    commissioning = int(result["commissioning_count"])
    if result["outcome_status"] == "known_control":
        return count
    return min(count, max(commissioning, int(math.ceil(0.20 * count))))


def evaluate(values: np.ndarray, threshold: float, persistence: int, commissioning: int) -> tuple[int | None, int | None]:
    evidence = first_persistent(values > threshold, persistence, commissioning)
    confirmation = None if evidence is None else evidence + persistence - 1
    return evidence, confirmation


def reference_false(values: np.ndarray, threshold: float, persistence: int, commissioning: int, end: int) -> bool:
    # Extend by p-1 samples so a run that begins inside the reference region is
    # still counted when causal confirmation falls just beyond its boundary.
    evidence = first_persistent(values > threshold, persistence, commissioning, min(len(values), end + persistence - 1))
    return evidence is not None and evidence < end


def elapsed_at(metadata: list[dict[str, str]], indices: list[int], position: int) -> float:
    return float(metadata[indices[position]]["elapsed_seconds"])


def safe_stats(values: list[float]) -> tuple[object, object, object]:
    if not values:
        return "", "", ""
    array = np.asarray(values, dtype=float)
    return float(np.min(array)), float(np.median(array)), float(np.max(array))


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[object, object, object]:
    if total == 0:
        return "", "", ""
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return p, max(0.0, center - half), min(1.0, center + half)


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d6b-output", type=Path, default=D6B_OUTPUT)
    parser.add_argument("--d6c-output", type=Path, default=D6C_OUTPUT)
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "outputs")
    args = parser.parse_args()
    d6b, d6c, output = args.d6b_output.resolve(), args.d6c_output.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    metadata = read_csv(d6b / "feature_metadata.csv")
    trajectory_rows = read_csv(d6c / "trajectory_results.csv")
    trajectory_by_asset = {row["asset_id"]: row for row in trajectory_rows}
    locked = json.loads((d6c / "locked_rules.json").read_text(encoding="utf-8"))
    health = np.load(d6c / "health_indicator.npy", mmap_mode="r")
    locked_point_alarm = np.load(d6c / "point_alarm.npy", mmap_mode="r")
    raw_distance = np.load(d6b / "robust_distance.npy", mmap_mode="r")
    threshold_rows = read_csv(d6b / "thresholds.csv")
    raw_thresholds = {(row["dataset"], row["condition_id"]): float(row["q0.990"]) for row in threshold_rows}
    raw_support = {(row["dataset"], row["condition_id"]): row["q0.990_supported_n_ge_99"] for row in threshold_rows}

    by_asset: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        if row["early_warning_eligible"] == "true":
            by_asset[row["asset_id"]].append(index)
    for asset, indices in by_asset.items():
        indices.sort(key=lambda value: int(metadata[value]["sequence_index"]))
        if asset not in trajectory_by_asset:
            raise ValueError(f"missing D6C trajectory result for {asset}")

    sensitivity_rows: list[dict[str, object]] = []
    for dataset in sorted(locked["rules"]):
        rule = locked["rules"][dataset]
        dataset_assets = [row for row in trajectory_rows if row["dataset"] == dataset]
        for multiplier in THRESHOLD_MULTIPLIERS:
            for persistence in PERSISTENCE_VALUES:
                threshold = float(rule["threshold"]) * multiplier
                calibration_reference_assets = 0
                calibration_false_assets = 0
                calibration_failed_total = 0
                calibration_failed_detected = 0
                test_failed_total = 0
                test_failed_detected = 0
                test_unknown_total = 0
                test_unknown_flagged = 0
                early_test_count = 0
                leads: list[float] = []
                for result in dataset_assets:
                    indices = by_asset[result["asset_id"]]
                    values = np.asarray(health[indices], dtype=float)
                    commissioning = int(result["commissioning_count"])
                    # At multiplier 1.0, preserve the immutable D6C comparison
                    # bits. D6C formed them before the health array was cast to
                    # float32, so re-thresholding the stored array can move an
                    # exact-boundary event by one sample.
                    above = np.asarray(locked_point_alarm[indices], dtype=bool) if multiplier == 1.0 else values > threshold
                    evidence = first_persistent(above, persistence, commissioning)
                    confirmation = None if evidence is None else evidence + persistence - 1
                    detected = confirmation is not None
                    if result["split"] == "calibration":
                        calibration_reference_assets += 1
                        ref_end = reference_end(result)
                        ref_evidence = first_persistent(above, persistence, commissioning, min(len(above), ref_end + persistence - 1))
                        if ref_evidence is not None and ref_evidence < ref_end:
                            calibration_false_assets += 1
                        if result["outcome_status"] == "known_failed":
                            calibration_failed_total += 1
                            calibration_failed_detected += int(detected)
                    if result["split"] == "test" and result["outcome_status"] == "known_failed":
                        test_failed_total += 1
                        test_failed_detected += int(detected)
                        if detected:
                            life_fraction = confirmation / max(len(values) - 1, 1)
                            early_test_count += int(life_fraction < 0.20)
                            leads.append(elapsed_at(metadata, indices, len(indices) - 1) - elapsed_at(metadata, indices, confirmation))
                    if result["split"] == "test" and result["outcome_status"] == "unknown":
                        test_unknown_total += 1
                        test_unknown_flagged += int(detected)
                lead_min, lead_median, lead_max = safe_stats(leads)
                sensitivity_rows.append({
                    "dataset": dataset,
                    "threshold_multiplier": multiplier,
                    "threshold": threshold,
                    "persistence": persistence,
                    "is_locked_primary": bool_text(multiplier == 1.0 and persistence == int(rule["persistence"])),
                    "calibration_reference_asset_count": calibration_reference_assets,
                    "calibration_reference_false_asset_count": calibration_false_assets,
                    "calibration_known_failed_detected": calibration_failed_detected,
                    "calibration_known_failed_total": calibration_failed_total,
                    "test_known_failed_detected": test_failed_detected,
                    "test_known_failed_total": test_failed_total,
                    "test_detection_rate": "" if test_failed_total == 0 else test_failed_detected / test_failed_total,
                    "test_detected_before_20pct": early_test_count,
                    "test_lead_seconds_min": lead_min,
                    "test_lead_seconds_median": lead_median,
                    "test_lead_seconds_max": lead_max,
                    "test_unknown_flagged": test_unknown_flagged,
                    "test_unknown_total": test_unknown_total,
                })
    sensitivity_fields = list(sensitivity_rows[0])
    write_csv(output / "rule_sensitivity.csv", sensitivity_rows, sensitivity_fields)

    baseline_trajectories: list[dict[str, object]] = []
    for result in trajectory_rows:
        asset = result["asset_id"]
        indices = by_asset[asset]
        commissioning = int(result["commissioning_count"])
        condition_key = (result["dataset"], result["condition_id"])
        methods = (
            ("locked_d6c_ewma", np.asarray(health[indices], dtype=float), np.asarray(locked_point_alarm[indices], dtype=bool), float(locked["rules"][result["dataset"]]["threshold"]), int(locked["rules"][result["dataset"]]["persistence"]), "true"),
            ("raw_d6b_q99_p3", np.asarray(raw_distance[indices], dtype=float), None, raw_thresholds[condition_key], 3, raw_support[condition_key]),
        )
        for method, values, fixed_above, threshold, persistence, support in methods:
            above = fixed_above if fixed_above is not None else values > threshold
            evidence = first_persistent(above, persistence, commissioning)
            confirmation = None if evidence is None else evidence + persistence - 1
            detected = confirmation is not None
            lead_seconds: object = ""
            life_fraction: object = ""
            if detected:
                life_fraction = confirmation / max(len(values) - 1, 1)
                if result["outcome_status"] == "known_failed":
                    lead_seconds = elapsed_at(metadata, indices, len(indices) - 1) - elapsed_at(metadata, indices, confirmation)
            false_ref = ""
            if result["split"] == "calibration":
                ref_end = reference_end(result)
                ref_evidence = first_persistent(above, persistence, commissioning, min(len(above), ref_end + persistence - 1))
                false_ref = bool_text(ref_evidence is not None and ref_evidence < ref_end)
            baseline_trajectories.append({
                "method": method,
                "dataset": result["dataset"],
                "condition_id": result["condition_id"],
                "split": result["split"],
                "run_id": result["run_id"],
                "asset_id": asset,
                "outcome_status": result["outcome_status"],
                "threshold": threshold,
                "threshold_support_n_ge_99": support,
                "persistence": persistence,
                "commissioning_count": commissioning,
                "detected": bool_text(detected),
                "evidence_start_zero_based": "" if evidence is None else evidence,
                "warning_confirmation_zero_based": "" if confirmation is None else confirmation,
                "warning_confirmation_life_fraction": life_fraction,
                "lead_seconds_to_terminal": lead_seconds,
                "calibration_reference_false_event": false_ref,
            })
    baseline_fields = list(baseline_trajectories[0])
    write_csv(output / "baseline_trajectory_results.csv", baseline_trajectories, baseline_fields)

    comparison_rows: list[dict[str, object]] = []
    for dataset in sorted(locked["rules"]):
        for method in ("locked_d6c_ewma", "raw_d6b_q99_p3"):
            rows = [row for row in baseline_trajectories if row["dataset"] == dataset and row["method"] == method]
            calibration = [row for row in rows if row["split"] == "calibration"]
            test_failed = [row for row in rows if row["split"] == "test" and row["outcome_status"] == "known_failed"]
            test_unknown = [row for row in rows if row["split"] == "test" and row["outcome_status"] == "unknown"]
            detected_failed = [row for row in test_failed if row["detected"] == "true"]
            leads = [float(row["lead_seconds_to_terminal"]) for row in detected_failed]
            lead_min, lead_median, lead_max = safe_stats(leads)
            unsupported_conditions = sorted({row["condition_id"] for row in rows if row["threshold_support_n_ge_99"] != "true"})
            comparison_rows.append({
                "dataset": dataset,
                "method": method,
                "calibration_reference_false_assets": sum(row["calibration_reference_false_event"] == "true" for row in calibration),
                "calibration_reference_assets": len(calibration),
                "test_known_failed_detected": len(detected_failed),
                "test_known_failed_total": len(test_failed),
                "test_detection_rate": "" if not test_failed else len(detected_failed) / len(test_failed),
                "test_detected_before_20pct": sum(float(row["warning_confirmation_life_fraction"]) < 0.20 for row in detected_failed),
                "test_lead_seconds_min": lead_min,
                "test_lead_seconds_median": lead_median,
                "test_lead_seconds_max": lead_max,
                "test_unknown_flagged": sum(row["detected"] == "true" for row in test_unknown),
                "test_unknown_total": len(test_unknown),
                "unsupported_q99_conditions": ";".join(unsupported_conditions) if method == "raw_d6b_q99_p3" else "",
            })
    comparison_fields = list(comparison_rows[0])
    write_csv(output / "baseline_comparison.csv", comparison_rows, comparison_fields)

    confidence_rows: list[dict[str, object]] = []
    primary_rows = [row for row in comparison_rows if row["method"] == "locked_d6c_ewma"]
    for row in primary_rows:
        successes = int(row["test_known_failed_detected"])
        total = int(row["test_known_failed_total"])
        estimate, lower, upper = wilson_interval(successes, total)
        confidence_rows.append({
            "dataset": row["dataset"],
            "estimand": "trajectory-level known-failed test detection probability",
            "successes": successes,
            "total": total,
            "estimate": estimate,
            "wilson_95_lower": lower,
            "wilson_95_upper": upper,
            "lead_seconds_min_detected_only": row["test_lead_seconds_min"],
            "lead_seconds_median_detected_only": row["test_lead_seconds_median"],
            "lead_seconds_max_detected_only": row["test_lead_seconds_max"],
            "annotated_physical_onset_available": "false",
        })
    confidence_fields = list(confidence_rows[0])
    write_csv(output / "locked_test_confidence.csv", confidence_rows, confidence_fields)

    summary = {
        "status": "D6D_ANALYSIS_COMPLETE",
        "schema_version": "D6D-1.0",
        "primary_method": "locked_d6c_ewma",
        "primary_rule_source": "immutable Stage D6C locked_rules.json",
        "sensitivity_grid": {"threshold_multipliers": list(THRESHOLD_MULTIPLIERS), "persistence_values": list(PERSISTENCE_VALUES)},
        "sensitivity_role": "pre-specified robustness reporting only; no test-based selection",
        "baseline": "raw D6B robust distance with the existing condition-specific q0.990 threshold and persistence=3",
        "confidence_interval": "two-sided Wilson score interval at 95%, trajectory as the independent unit",
        "claim_boundary": "Event detection is against documented terminal outcome, not accuracy against annotated physical fault onset.",
        "test_rows_used_for_fit_or_selection": 0,
        "input_sha256": {
            "d6c_locked_rules.json": sha256(d6c / "locked_rules.json"),
            "d6c_health_indicator.npy": sha256(d6c / "health_indicator.npy"),
            "d6c_point_alarm.npy": sha256(d6c / "point_alarm.npy"),
            "d6c_trajectory_results.csv": sha256(d6c / "trajectory_results.csv"),
            "d6b_robust_distance.npy": sha256(d6b / "robust_distance.npy"),
            "d6b_thresholds.csv": sha256(d6b / "thresholds.csv"),
        },
    }
    (output / "d6d_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "sensitivity_rows": len(sensitivity_rows),
        "baseline_trajectory_rows": len(baseline_trajectories),
        "comparison_rows": len(comparison_rows),
        "confidence_rows": len(confidence_rows),
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
