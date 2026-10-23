#!/usr/bin/env python3
"""Validate D6C fit isolation, causal arrays, locked rules, and result semantics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from build_health_indicator import causal_health, first_persistent


SCRIPT_DIR = Path(__file__).resolve().parent
D6B_OUTPUT = SCRIPT_DIR.parent / "stage_d6b" / "outputs"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d6b-output", type=Path, default=D6B_OUTPUT)
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "outputs")
    args = parser.parse_args()
    d6b, output = args.d6b_output.resolve(), args.output.resolve()
    metadata = read_csv(d6b / "feature_metadata.csv")
    source_score = np.load(d6b / "robust_distance.npy", mmap_mode="r")
    health = np.load(output / "health_indicator.npy", mmap_mode="r")
    point_alarm = np.load(output / "point_alarm.npy", mmap_mode="r")
    warning = np.load(output / "latched_warning.npy", mmap_mode="r")
    trajectories = read_csv(output / "trajectory_results.csv")
    group_parameters = read_csv(output / "group_parameters.csv")
    alpha_selection = read_csv(output / "alpha_selection.csv")
    rule_search = read_csv(output / "rule_search.csv")
    locked = json.loads((output / "locked_rules.json").read_text(encoding="utf-8"))
    audit = json.loads((output / "fit_audit.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "d6c_summary.json").read_text(encoding="utf-8"))

    expected_shape = (len(metadata),)
    if health.shape != expected_shape or point_alarm.shape != expected_shape or warning.shape != expected_shape:
        raise ValueError("D6C arrays are not aligned to D6B feature rows")
    eligible = np.asarray([row["early_warning_eligible"] == "true" for row in metadata])
    if not np.isfinite(health[eligible]).all() or not (health[eligible] >= 0).all():
        raise ValueError("eligible health values must be finite and nonnegative")
    if not np.isnan(health[~eligible]).all() or point_alarm[~eligible].any() or warning[~eligible].any():
        raise ValueError("non-eligible rows entered the early-warning indicator")

    selected_alpha = [row for row in alpha_selection if row["selected"] == "true"]
    selected_rules = [row for row in rule_search if row["selected"] == "true"]
    if len(selected_alpha) != 3 or len(selected_rules) != 3:
        raise ValueError("expected one selected alpha and rule per early-warning dataset")
    for row in selected_alpha:
        rule = locked["rules"][row["dataset"]]
        if float(row["alpha"]) != float(rule["alpha"]):
            raise ValueError("selected alpha/locked rule mismatch")
    for row in selected_rules:
        rule = locked["rules"][row["dataset"]]
        if not np.isclose(float(row["threshold"]), float(rule["threshold"])) or int(row["persistence"]) != int(rule["persistence"]):
            raise ValueError("selected calibration rule/lock mismatch")
        if int(row["reference_false_asset_count"]) != 0:
            raise ValueError("selected rule has a calibration-reference false event")

    if locked["test_rows_used_for_global_fit_or_selection"] != 0 or audit["test_rows_used_for_fit"] != 0 or audit["test_assets_used_for_fit"]:
        raise ValueError("test entered global fitting or rule selection")
    metadata_asset_split: dict[str, str] = {}
    for row in metadata:
        if row["early_warning_eligible"] == "true":
            metadata_asset_split[row["asset_id"]] = row["split"]
    for assets in audit["group_parameter_assets"].values():
        if any(metadata_asset_split[asset] != "train" for asset in assets):
            raise ValueError("group parameters used a non-train asset")
    for assets in audit["alpha_selection_assets"].values():
        if any(metadata_asset_split[asset] != "train" for asset in assets):
            raise ValueError("alpha selection used a non-train asset")
    for assets in audit["calibration_rule_assets"].values():
        if any(metadata_asset_split[asset] != "calibration" for asset in assets):
            raise ValueError("rule selection used a non-calibration asset")

    by_asset_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        if row["early_warning_eligible"] == "true":
            by_asset_indices[row["asset_id"]].append(index)
    trajectory_lookup = {row["asset_id"]: row for row in trajectories}
    if set(trajectory_lookup) != set(by_asset_indices) or len(trajectories) != 44:
        raise ValueError("trajectory manifest coverage mismatch")
    for asset, indices in by_asset_indices.items():
        indices.sort(key=lambda value: int(metadata[value]["sequence_index"]))
        result = trajectory_lookup[asset]
        rule = locked["rules"][result["dataset"]]
        expected_alarm = health[indices] > float(rule["threshold"])
        if not np.array_equal(point_alarm[indices], expected_alarm):
            raise ValueError(f"point alarm mismatch for {asset}")
        onset = first_persistent(expected_alarm, int(rule["persistence"]), int(result["commissioning_count"]))
        expected_warning = np.zeros(len(indices), dtype=bool)
        confirmation = None if onset is None else onset + int(rule["persistence"]) - 1
        if onset is not None:
            expected_warning[confirmation:] = True
        if not np.array_equal(warning[indices], expected_warning):
            raise ValueError(f"latched warning mismatch for {asset}")
        recorded_onset = None if result["evidence_start_zero_based"] == "" else int(result["evidence_start_zero_based"])
        recorded_confirmation = None if result["warning_confirmation_zero_based"] == "" else int(result["warning_confirmation_zero_based"])
        if recorded_onset != onset or recorded_confirmation != confirmation:
            raise ValueError(f"evidence/warning timing mismatch for {asset}")

    # Direct prefix-invariance check on one trajectory per dataset.
    prefix_checks = []
    for dataset in sorted(locked["rules"]):
        result = next(row for row in trajectories if row["dataset"] == dataset and row["split"] == "test")
        indices = by_asset_indices[result["asset_id"]]
        prefix_length = max(int(result["commissioning_count"]) + 3, len(indices) // 2)
        log_score = np.log1p(np.asarray(source_score[indices], dtype=np.float64))
        full = causal_health(log_score, float(result["commissioning_log_center"]), float(result["pooled_train_log_scale"]), float(result["ewma_alpha"]))
        prefix = causal_health(log_score[:prefix_length], float(result["commissioning_log_center"]), float(result["pooled_train_log_scale"]), float(result["ewma_alpha"]))
        if not np.allclose(full[:prefix_length], prefix, rtol=0, atol=0):
            raise ValueError(f"prefix causality failed for {dataset}")
        prefix_checks.append(result["asset_id"])

    nasa_counts = Counter(row["outcome_status"] for row in trajectories if row["dataset"] == "NASA-IMS")
    if nasa_counts != Counter({"known_control": 5, "known_failed": 3, "unknown": 4}):
        raise ValueError(f"NASA asset-specific outcome mapping failed: {nasa_counts}")
    if summary["claim_boundary"].find("not accuracy against annotated physical onset") < 0:
        raise ValueError("claim boundary is missing")

    report = {
        "status": "PASS", "schema_version": locked["schema_version"],
        "trajectory_count": len(trajectories), "point_count": int(eligible.sum()),
        "group_parameter_count": len(group_parameters), "locked_rule_count": len(locked["rules"]),
        "prefix_causality_assets": prefix_checks,
        "outcome_counts": dict(Counter(row["outcome_status"] for row in trajectories)),
        "partition_counts": dict(Counter(row["split"] for row in trajectories)),
        "guards": {
            "group_scale_train_only": True, "alpha_train_only": True,
            "threshold_and_persistence_calibration_only": True,
            "test_global_fit_rows": 0, "causal_prefix_invariance": True,
            "noneligible_rows_excluded": True, "nasa_asset_outcomes_not_campaign_copied": True,
            "selected_calibration_reference_false_assets": 0,
            "onset_explicitly_algorithmic_not_ground_truth": True,
            "warning_latched_at_causal_persistence_confirmation": True,
        },
        "input_sha256": {
            "d6b_feature_metadata.csv": sha256(d6b / "feature_metadata.csv"),
            "d6b_robust_distance.npy": sha256(d6b / "robust_distance.npy"),
        },
        "artifact_sha256": {
            name: sha256(output / name) for name in (
                "health_indicator.npy", "point_alarm.npy", "latched_warning.npy",
                "group_parameters.csv", "alpha_selection.csv", "rule_search.csv",
                "trajectory_results.csv", "locked_rules.json", "fit_audit.json", "d6c_summary.json",
            )
        },
    }
    (output / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
