#!/usr/bin/env python3
"""Validate D6E fit isolation, causal decisions, and D6C preservation."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT = SCRIPT_DIR / "outputs"
D6B_OUTPUT = SCRIPT_DIR.parent / "stage_d6b" / "outputs"
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


def first_persistent(values: np.ndarray, persistence: int, start: int = 0) -> int | None:
    count = 0
    for index in range(max(0, start), len(values)):
        count = count + 1 if bool(values[index]) else 0
        if count >= persistence:
            return index - persistence + 1
    return None


def optional_int(value: str) -> int | None:
    return None if value == "" else int(value)


def main() -> None:
    metadata = read_csv(D6B_OUTPUT / "feature_metadata.csv")
    method_names = json.loads((OUTPUT / "method_names.json").read_text(encoding="utf-8"))["axis_0_order"]
    registry = json.loads((OUTPUT / "model_registry.json").read_text(encoding="utf-8"))
    audit = json.loads((OUTPUT / "fit_audit.json").read_text(encoding="utf-8"))
    alpha_rows = read_csv(OUTPUT / "alpha_selection.csv")
    rule_rows = read_csv(OUTPUT / "rule_search.csv")
    selected_rules = read_csv(OUTPUT / "selected_rules.csv")
    trajectories = read_csv(OUTPUT / "trajectory_results.csv")
    summaries = read_csv(OUTPUT / "method_summary.csv")
    paired = read_csv(OUTPUT / "paired_comparison.csv")
    source = np.load(OUTPUT / "source_scores.npy", mmap_mode="r")
    health = np.load(OUTPUT / "health_indicators.npy", mmap_mode="r")
    point = np.load(OUTPUT / "point_alarms.npy", mmap_mode="r")
    warning = np.load(OUTPUT / "latched_warnings.npy", mmap_mode="r")
    d6b_score = np.load(D6B_OUTPUT / "robust_distance.npy", mmap_mode="r")
    d6c_health = np.load(D6C_OUTPUT / "health_indicator.npy", mmap_mode="r")
    d6c_point = np.load(D6C_OUTPUT / "point_alarm.npy", mmap_mode="r")
    d6c_warning = np.load(D6C_OUTPUT / "latched_warning.npy", mmap_mode="r")
    d6c_trajectories = read_csv(D6C_OUTPUT / "trajectory_results.csv")

    expected_methods = [
        "d6c_robust_ewma", "pca_spe", "shrinkage_mahalanobis", "knn5_novelty",
        "time_feature_ablation", "frequency_feature_ablation",
    ]
    if method_names != expected_methods:
        raise ValueError("method order changed")
    expected_shape = (len(method_names), len(metadata))
    if any(array.shape != expected_shape for array in (source, health, point, warning)):
        raise ValueError("D6E arrays are not aligned to method x D6B feature rows")
    eligible = np.asarray([row["early_warning_eligible"] == "true" for row in metadata])
    if not np.isfinite(source[:, eligible]).all() or not np.isfinite(health[:, eligible]).all():
        raise ValueError("eligible scores/health must be finite")
    if not np.isnan(source[:, ~eligible]).all() or not np.isnan(health[:, ~eligible]).all():
        raise ValueError("non-eligible rows entered D6E scores")
    if point[:, ~eligible].any() or warning[:, ~eligible].any():
        raise ValueError("non-eligible rows entered D6E alarms")

    if not np.array_equal(source[0], d6b_score, equal_nan=True):
        raise ValueError("D6B primary source score changed")
    if not np.array_equal(health[0], d6c_health, equal_nan=True):
        raise ValueError("D6C health indicator changed")
    if not np.array_equal(point[0], d6c_point) or not np.array_equal(warning[0], d6c_warning):
        raise ValueError("D6C locked decisions changed")

    if len(registry) != 35 or Counter(item["method"] for item in registry) != Counter({method: 7 for method in expected_methods[1:]}):
        raise ValueError("expected five experimental methods across seven dataset-condition groups")
    if any(item["fit_split"] != "train_only" for item in registry):
        raise ValueError("non-train data entered model fit")
    if audit["test_rows_used_for_fit_or_selection"] != 0 or audit["test_assets_used_for_fit_or_selection"]:
        raise ValueError("test entered fitting or selection")
    if audit["input_sha256"]["d6c_locked_rules.json"] != sha256(D6C_OUTPUT / "locked_rules.json"):
        raise ValueError("D6C locked rules changed")

    selected_alphas = [row for row in alpha_rows if row["selected"] == "true"]
    selected_experimental_rules = [row for row in rule_rows if row["selected"] == "true"]
    if len(alpha_rows) != 75 or len(selected_alphas) != 15:
        raise ValueError("incomplete train-only alpha search")
    if len(rule_rows) != 180 or len(selected_experimental_rules) != 15:
        raise ValueError("incomplete calibration-only rule search")
    if len(selected_rules) != 18 or any(row["test_rows_used"] != "0" for row in selected_rules):
        raise ValueError("selected rule manifest mismatch")
    if any(int(row["reference_false_asset_count"]) != 0 for row in selected_experimental_rules):
        raise ValueError("a selected experimental rule has a calibration-reference false event")

    by_asset: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        if row["early_warning_eligible"] == "true":
            by_asset[row["asset_id"]].append(index)
    for asset, indices in by_asset.items():
        indices.sort(key=lambda index: int(metadata[index]["sequence_index"]))
    if len(trajectories) != 264 or Counter(row["method"] for row in trajectories) != Counter({method: 44 for method in method_names}):
        raise ValueError("method-trajectory coverage mismatch")
    for row in trajectories:
        mi = method_names.index(row["method"])
        indices = by_asset[row["asset_id"]]
        above = np.asarray(point[mi, indices], dtype=bool)
        evidence = first_persistent(above, int(row["persistence"]), int(row["commissioning_count"]))
        confirmation = None if evidence is None else evidence + int(row["persistence"]) - 1
        if evidence != optional_int(row["evidence_start_zero_based"]) or confirmation != optional_int(row["warning_confirmation_zero_based"]):
            raise ValueError(f"causal event mismatch: {row['method']} {row['asset_id']}")
        expected_warning = np.zeros(len(indices), dtype=bool)
        if confirmation is not None:
            expected_warning[confirmation:] = True
        if not np.array_equal(expected_warning, warning[mi, indices]):
            raise ValueError(f"latched warning mismatch: {row['method']} {row['asset_id']}")

    d6c_lookup = {row["asset_id"]: row for row in d6c_trajectories}
    primary_lookup = {row["asset_id"]: row for row in trajectories if row["method"] == "d6c_robust_ewma"}
    for asset, original in d6c_lookup.items():
        reproduced = primary_lookup[asset]
        for field in ("detected", "evidence_start_zero_based", "warning_confirmation_zero_based", "lead_seconds_to_terminal"):
            if reproduced[field] != original[field]:
                raise ValueError(f"D6C primary trajectory field changed: {asset} {field}")

    if len(summaries) != 18:
        raise ValueError("method summary coverage mismatch")
    for row in summaries:
        if row["dataset"] == "NASA-IMS" and (row["test_known_failed_total"] != "0" or row["test_unknown_total"] != "4"):
            raise ValueError("NASA unknown outcomes entered the labeled denominator")
        if row["calibration_reference_false_assets"] != "0":
            raise ValueError("selected method summary contains a calibration-reference false asset")
    if len(paired) != 5 or any(row["known_failed_test_assets"] != "8" for row in paired):
        raise ValueError("paired comparison must use the same eight known-failed test assets")
    if any(row["inferential_role"] != "descriptive_small_n_unadjusted_for_multiple_comparisons" for row in paired):
        raise ValueError("small-n statistical qualification is missing")

    # Feature ablations must be disjoint and reconstruct the compatible D6B set.
    registry_lookup = {(item["method"], item["dataset"], item["condition_id"]): item for item in registry}
    for dataset, condition in sorted({(item["dataset"], item["condition_id"]) for item in registry}):
        full = set(registry_lookup[("pca_spe", dataset, condition)]["feature_indices"])
        time = set(registry_lookup[("time_feature_ablation", dataset, condition)]["feature_indices"])
        frequency = set(registry_lookup[("frequency_feature_ablation", dataset, condition)]["feature_indices"])
        if time & frequency or time | frequency != full:
            raise ValueError(f"feature ablation partition mismatch: {dataset} {condition}")

    report = {
        "status": "PASS", "schema_version": audit["schema_version"],
        "method_count": len(method_names), "trajectory_method_rows": len(trajectories),
        "model_registry_entries": len(registry),
        "guards": {
            "model_fit_train_early_life_only": True,
            "alpha_selected_train_only": True,
            "threshold_and_persistence_selected_calibration_only": True,
            "test_rows_used_for_fit_or_selection": 0,
            "d6c_primary_arrays_unchanged": True,
            "d6c_primary_trajectory_events_unchanged": True,
            "causal_persistence_confirmation_verified": True,
            "noneligible_rows_excluded": True,
            "nasa_unknown_excluded_from_labeled_denominator": True,
            "feature_ablation_partition_verified": True,
            "paired_test_is_descriptive_small_n": True,
        },
        "artifact_sha256": {
            name: sha256(OUTPUT / name) for name in (
                "source_scores.npy", "health_indicators.npy", "point_alarms.npy", "latched_warnings.npy",
                "model_registry.json", "selected_rules.csv", "trajectory_results.csv", "method_summary.csv",
                "feature_ablation.csv", "paired_comparison.csv", "fit_audit.json",
            )
        },
    }
    (OUTPUT / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
