#!/usr/bin/env python3
"""Validate D8B policy transfer, D6F reproduction, and leakage boundaries."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
D6F = HERE.parent / "stage_d6f" / "outputs"
D8A = HERE.parent.parent / "Real-Fault-Benchmarks" / "stage_d8a" / "outputs"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def close(a: str, b: str, tolerance: float = 1e-12) -> bool:
    return abs(float(a) - float(b)) <= tolerance * max(1.0, abs(float(a)), abs(float(b)))


def main() -> None:
    required = (
        "commissioning_parameters.csv", "operating_state_exclusions.csv", "calibration_diagnostics.csv",
        "motor_results.csv", "aggregate_results.csv", "episode_results.csv", "paired_nuisance_comparison.csv",
        "cross_evidence_tradeoff.csv", "test_decisions.npz", "policy_manifest.json",
    )
    checks: list[dict[str, object]] = []
    for name in required:
        checks.append({"check": f"exists::{name}", "pass": (OUT / name).is_file()})

    manifest = json.loads((OUT / "policy_manifest.json").read_text(encoding="utf-8"))
    checks.append({"check": "no_PNU_NOMV_test_fit_or_selection", "pass": manifest["test_rows_used_for_fit_or_selection"] == 0 and manifest["pnu_calibration_used_for_retuning"] is False})

    aggregate = read_csv(OUT / "aggregate_results.csv")
    methods = {row["method"] for row in aggregate}
    sources = {row["source_dataset"] for row in aggregate}
    normalizations = {row["normalization"] for row in aggregate}
    checks.append({"check": "complete_4x3x2_policy_matrix", "pass": len(aggregate) == 24 and methods == {"crew_ewma", "robust_shewhart", "robust_page_cusum", "robust_causal_ma"} and sources == {"NASA-IMS", "PRONOSTIA", "XJTU-SY"} and normalizations == {"global_train", "onboard_motor_train"}})
    observation_counts = {int(row["test_observation_count"]) for row in aggregate}
    monitoring_hours = {float(row["test_monitoring_hours"]) for row in aggregate}
    checks.append({
        "check": "constant_quality_eligible_test_exposure",
        "pass": len(observation_counts) == 1 and len(monitoring_hours) == 1,
    })

    d6f = read_csv(D6F / "aggregate_results.csv")
    d6f_lookup = {(row["source_dataset"], row["normalization"]): row for row in d6f}
    crew = [row for row in aggregate if row["method"] == "crew_ewma"]
    exact_fields = ("test_observation_count", "test_point_alarm_count", "test_warning_count", "test_episode_count", "test_flagged_motor_count")
    reproduced = True
    for row in crew:
        reference = d6f_lookup[(row["source_dataset"], row["normalization"])]
        reproduced &= all(row[field] == reference[field] for field in exact_fields)
        reproduced &= close(row["test_episodes_per_day"], reference["test_episodes_per_day"])
    checks.append({"check": "CREW_exactly_reproduces_D6F_point_metrics", "pass": reproduced})

    align = read_csv(OUT / "commissioning_parameters.csv")
    d6f_align = read_csv(D6F / "commissioning_parameters.csv")
    align_lookup = {(row["normalization"], row["motor"]): row for row in align}
    align_reference = {(row["normalization"], row["motor"]): row for row in d6f_align}
    alignment_match = set(align_lookup) == set(align_reference)
    for key in align_lookup:
        alignment_match &= close(align_lookup[key]["center"], align_reference[key]["center"])
        alignment_match &= close(align_lookup[key]["pooled_scale"], align_reference[key]["pooled_scale"])
    checks.append({"check": "train_only_alignment_reproduces_D6F", "pass": alignment_match})

    policy_rows = read_csv(D8A / "sequential_method_summary.csv")
    policy_lookup = {(row["method"], row["dataset"]): row for row in policy_rows}
    policy_match = True
    for row in aggregate:
        reference = policy_lookup[(row["method"], row["source_dataset"])]
        policy_match &= close(row["threshold"], reference["threshold"])
        policy_match &= int(row["persistence"]) == int(reference["persistence"])
        if row["method"] != "robust_shewhart":
            policy_match &= close(row["parameter_value"], reference["selected_parameter"])
    checks.append({"check": "all_temporal_policies_match_locked_D8A", "pass": policy_match})

    tradeoff = read_csv(OUT / "cross_evidence_tradeoff.csv")
    checks.append({"check": "cross_evidence_rows_cover_every_config", "pass": len(tradeoff) == 24})
    public_match = all(
        row["public_known_failed_detected"] == policy_lookup[(row["method"], row["source_dataset"])]["test_known_failed_detected"]
        and row["public_known_failed_total"] == policy_lookup[(row["method"], row["source_dataset"])]["test_known_failed_total"]
        for row in tradeoff
    )
    checks.append({"check": "public_fault_evidence_copied_without_recomputation", "pass": public_match})

    paired = read_csv(OUT / "paired_nuisance_comparison.csv")
    checks.append({"check": "paired_comparison_matrix_3x2x3", "pass": len(paired) == 18})
    decisions = np.load(OUT / "test_decisions.npz")
    checks.append({"check": "exact_decision_array_count", "pass": len(decisions.files) == 24 * 8 * 2})

    zero_bound_ok = all(bool(row["zero_episode_one_sided_95_poisson_upper_per_day"]) == (int(row["test_episode_count"]) == 0) for row in aggregate)
    checks.append({"check": "zero_episode_Poisson_bounds_present_iff_needed", "pass": zero_bound_ok})

    passed = all(bool(item["pass"]) for item in checks)
    report = {"stage": "D8B", "status": "PASS" if passed else "FAIL", "checks": checks}
    (OUT / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
