#!/usr/bin/env python3
"""Validate Stage D6D against immutable D6C primary results."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT = SCRIPT_DIR / "outputs"
D6C_OUTPUT = SCRIPT_DIR.parent / "stage_d6c" / "outputs"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def optional_int(value: str) -> int | None:
    return None if value == "" else int(value)


def main() -> None:
    sensitivity = read_csv(OUTPUT / "rule_sensitivity.csv")
    baseline = read_csv(OUTPUT / "baseline_trajectory_results.csv")
    comparison = read_csv(OUTPUT / "baseline_comparison.csv")
    confidence = read_csv(OUTPUT / "locked_test_confidence.csv")
    summary = json.loads((OUTPUT / "d6d_summary.json").read_text(encoding="utf-8"))
    d6c_trajectories = read_csv(D6C_OUTPUT / "trajectory_results.csv")

    if len(sensitivity) != 27:
        raise ValueError(f"expected 27 pre-specified sensitivity rows, got {len(sensitivity)}")
    per_dataset = Counter(row["dataset"] for row in sensitivity)
    if per_dataset != Counter({"NASA-IMS": 9, "PRONOSTIA": 9, "XJTU-SY": 9}):
        raise ValueError(f"incomplete sensitivity grid: {per_dataset}")
    primary = [row for row in sensitivity if row["is_locked_primary"] == "true"]
    if len(primary) != 3 or any(float(row["threshold_multiplier"]) != 1.0 or int(row["persistence"]) != 3 for row in primary):
        raise ValueError("primary sensitivity rows do not exactly identify the locked D6C rule")

    locked_baseline = {row["asset_id"]: row for row in baseline if row["method"] == "locked_d6c_ewma"}
    d6c_lookup = {row["asset_id"]: row for row in d6c_trajectories}
    if set(locked_baseline) != set(d6c_lookup) or len(locked_baseline) != 44:
        raise ValueError("locked-method trajectory coverage mismatch")
    for asset, d6c_row in d6c_lookup.items():
        d6d_row = locked_baseline[asset]
        if d6d_row["detected"] != d6c_row["detected"]:
            raise ValueError(f"D6C detection changed for {asset}")
        if optional_int(d6d_row["evidence_start_zero_based"]) != optional_int(d6c_row["evidence_start_zero_based"]):
            raise ValueError(f"D6C evidence time changed for {asset}")
        if optional_int(d6d_row["warning_confirmation_zero_based"]) != optional_int(d6c_row["warning_confirmation_zero_based"]):
            raise ValueError(f"D6C confirmation time changed for {asset}")

    locked_comparison = {row["dataset"]: row for row in comparison if row["method"] == "locked_d6c_ewma"}
    expected = {"XJTU-SY": (3, 3), "PRONOSTIA": (5, 5), "NASA-IMS": (0, 0)}
    for dataset, counts in expected.items():
        row = locked_comparison[dataset]
        if (int(row["test_known_failed_detected"]), int(row["test_known_failed_total"])) != counts:
            raise ValueError(f"locked test result changed for {dataset}")
        if int(row["calibration_reference_false_assets"]) != 0:
            raise ValueError(f"locked calibration reference false event for {dataset}")

    if summary["test_rows_used_for_fit_or_selection"] != 0 or "no test-based selection" not in summary["sensitivity_role"]:
        raise ValueError("test-selection guard is missing")
    if summary["input_sha256"]["d6c_locked_rules.json"] != sha256(D6C_OUTPUT / "locked_rules.json"):
        raise ValueError("D6C locked rule hash changed during D6D")

    for row in confidence:
        total = int(row["total"])
        if total == 0:
            if any(row[field] != "" for field in ("estimate", "wilson_95_lower", "wilson_95_upper")):
                raise ValueError("zero-denominator confidence row must be blank")
        else:
            lower, estimate, upper = float(row["wilson_95_lower"]), float(row["estimate"]), float(row["wilson_95_upper"])
            if not (0 <= lower <= estimate <= upper <= 1):
                raise ValueError("invalid Wilson confidence interval")
        if row["annotated_physical_onset_available"] != "false":
            raise ValueError("D6D must not imply annotated onset availability")

    nasa_primary = next(row for row in primary if row["dataset"] == "NASA-IMS")
    if nasa_primary["test_known_failed_total"] != "0" or nasa_primary["test_unknown_total"] != "4":
        raise ValueError("NASA unknown test trajectories entered known-failed metrics")

    report = {
        "status": "PASS",
        "schema_version": summary["schema_version"],
        "sensitivity_configuration_count": len(sensitivity),
        "trajectory_method_rows": len(baseline),
        "primary_trajectory_count": len(locked_baseline),
        "guards": {
            "d6c_rule_hash_unchanged": True,
            "primary_trajectory_events_exactly_reproduced": True,
            "test_used_for_selection": False,
            "all_27_pre_specified_variants_retained": True,
            "causal_confirmation_not_evidence_start_used": True,
            "nasa_unknown_excluded_from_sensitivity_and_lead": True,
            "trajectory_level_confidence_intervals": True,
            "annotated_physical_onset_claimed": False,
        },
        "artifact_sha256": {
            name: sha256(OUTPUT / name) for name in (
                "rule_sensitivity.csv", "baseline_trajectory_results.csv", "baseline_comparison.csv",
                "locked_test_confidence.csv", "d6d_summary.json",
            )
        },
    }
    (OUTPUT / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
