#!/usr/bin/env python3
"""Validate Stage D8A outputs and leakage guards."""

from __future__ import annotations

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"


def read_csv(name: str) -> list[dict[str, str]]:
    with (OUT / name).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    required = (
        "parameter_selection.csv", "rule_search.csv", "trajectory_results.csv",
        "sequential_method_summary.csv", "paired_comparison.csv", "integrated_method_summary.csv",
        "sequential_health_indicators.npz", "fit_audit.json",
    )
    checks: list[dict[str, object]] = []
    for name in required:
        checks.append({"check": f"exists::{name}", "pass": (OUT / name).is_file()})

    audit = json.loads((OUT / "fit_audit.json").read_text(encoding="utf-8"))
    checks.append({"check": "no_test_fit_or_selection", "pass": audit["test_rows_used_for_fit_or_selection"] == 0 and not audit["test_assets_used_for_fit_or_selection"]})

    trajectories = read_csv("trajectory_results.csv")
    expected = {"crew_ewma", "robust_shewhart", "robust_page_cusum", "robust_causal_ma"}
    checks.append({"check": "expected_sequential_methods", "pass": {row["method"] for row in trajectories} == expected})
    counts = {method: sum(row["method"] == method for row in trajectories) for method in expected}
    checks.append({"check": "trajectory_alignment", "pass": len(set(counts.values())) == 1})

    primary_test = [row for row in trajectories if row["method"] == "crew_ewma" and row["split"] == "test" and row["outcome_status"] == "known_failed"]
    checks.append({"check": "crew_primary_preserved_8_of_8", "pass": len(primary_test) == 8 and all(row["detected"] == "true" for row in primary_test)})

    summaries = read_csv("sequential_method_summary.csv")
    experimental_calibration = [row for row in summaries if row["method"] != "crew_ewma"]
    checks.append({"check": "selected_rules_zero_calibration_reference_false_assets", "pass": all(int(row["calibration_reference_false_assets"]) == 0 for row in experimental_calibration)})

    parameter_rows = read_csv("parameter_selection.csv")
    groups: dict[tuple[str, str], int] = {}
    for row in parameter_rows:
        key = (row["method"], row["dataset"])
        groups[key] = groups.get(key, 0) + int(row["selected"] == "true")
    checks.append({"check": "one_train_selected_parameter_per_method_dataset", "pass": groups and all(value == 1 for value in groups.values())})

    integrated = read_csv("integrated_method_summary.csv")
    integrated_methods = {row["method"] for row in integrated}
    expected_integrated = {
        "crew_ewma", "raw_distance_q99_p3", "pca_spe", "shrinkage_mahalanobis", "knn5_novelty",
        "time_feature_ablation", "frequency_feature_ablation", "robust_shewhart", "robust_page_cusum", "robust_causal_ma",
    }
    checks.append({"check": "integrated_comparator_coverage", "pass": integrated_methods == expected_integrated})

    passed = all(bool(item["pass"]) for item in checks)
    report = {"stage": "D8A", "status": "PASS" if passed else "FAIL", "checks": checks}
    (OUT / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
