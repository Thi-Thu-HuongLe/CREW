#!/usr/bin/env python3
"""Validate D6B split, feature, missing-axis, and fit-isolation guarantees."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from feature_schema import FEATURE_NAMES


SCRIPT_DIR = Path(__file__).resolve().parent


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
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "outputs")
    args = parser.parse_args()
    output = args.output.resolve()
    split_rows = read_csv(output / "split_manifest.csv")
    metadata = read_csv(output / "feature_metadata.csv")
    baselines = read_csv(output / "baseline_rows.csv")
    thresholds = read_csv(output / "thresholds.csv")
    features = np.load(output / "features.npy", mmap_mode="r")
    masks = np.load(output / "feature_mask.npy", mmap_mode="r")
    standardized = np.load(output / "standardized_features.npy", mmap_mode="r")
    scores = np.load(output / "robust_distance.npy", mmap_mode="r")
    schema = json.loads((output / "feature_schema.json").read_text(encoding="utf-8"))
    preprocessing = json.loads((output / "preprocessing_manifest.json").read_text(encoding="utf-8"))

    expected_rows = sum(
        int(row["record_count"]) * (4 if row["dataset"] == "NASA-IMS" else 1)
        for row in split_rows if row["primary"] == "true"
    )
    expected_shape = (expected_rows, len(FEATURE_NAMES))
    if features.shape != expected_shape or masks.shape != expected_shape or standardized.shape != expected_shape:
        raise ValueError(f"array shape mismatch; expected {expected_shape}")
    if scores.shape != (expected_rows,) or len(metadata) != expected_rows:
        raise ValueError("score or metadata row mismatch")
    if schema["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("feature schema order mismatch")
    if [int(row["feature_row_id"]) for row in metadata] != list(range(expected_rows)):
        raise ValueError("feature_row_id is not contiguous")
    if not np.array_equal(np.isfinite(features), masks):
        raise ValueError("feature mask does not exactly match finite raw features")
    expected_mask_counts = np.asarray([1 + 13 * int(row["axis_count"]) for row in metadata])
    if not np.array_equal(masks.sum(axis=1), expected_mask_counts):
        raise ValueError("axis_count does not match feature mask")

    run_split: dict[tuple[str, str], set[str]] = defaultdict(set)
    group_split: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in metadata:
        run_split[(row["dataset"], row["run_id"])].add(row["split"])
        group_split[(row["dataset"], row["group_id"])].add(row["split"])
    if any(len(value) != 1 for value in run_split.values()) or any(len(value) != 1 for value in group_split.values()):
        raise ValueError("run/group split leakage")

    by_asset: dict[str, list[int]] = defaultdict(list)
    for row in metadata:
        by_asset[row["asset_id"]].append(int(row["sequence_index"]))
    for asset, sequence in by_asset.items():
        ordered = sorted(sequence)
        if ordered != list(range(1, len(ordered) + 1)):
            raise ValueError(f"non-contiguous sequence for {asset}")

    metadata_by_id = {row["feature_row_id"]: row for row in metadata}
    baseline_ids: set[int] = set()
    for row in baselines:
        index = int(row["feature_row_id"])
        if index in baseline_ids:
            raise ValueError("baseline row appears more than once")
        baseline_ids.add(index)
        source = metadata_by_id[str(index)]
        expected_split = "train" if row["role"] == "fit_normalization" else "calibration"
        if row["split"] != expected_split or source["split"] != expected_split:
            raise ValueError("fit role uses the wrong split")
        if source["dataset"] not in {"XJTU-SY", "PRONOSTIA", "NASA-IMS"}:
            raise ValueError("a dataset outside the selected scope entered fitting")
    if preprocessing["test_rows_used_for_fitting"] != 0:
        raise ValueError("test split was used for fitting")

    early_warning = np.asarray([row["early_warning_eligible"] == "true" for row in metadata])
    if not np.isfinite(scores[early_warning]).all() or not np.isnan(scores[~early_warning]).all():
        raise ValueError("robust scores are invalid or an ineligible row was not isolated")
    if not np.isnan(standardized[~early_warning]).all():
        raise ValueError("ineligible rows should remain unstandardized in D6B")
    if len(thresholds) != 7:
        raise ValueError(f"expected 7 dataset-condition thresholds, found {len(thresholds)}")
    for row in thresholds:
        values = [float(row[key]) for key in ("q0.990", "q0.995", "q0.999")]
        if not all(np.isfinite(values)) or values != sorted(values):
            raise ValueError(f"invalid thresholds for {row['dataset']} {row['condition_id']}")

    feature_counts = Counter(f"{row['dataset']}::{row['split']}" for row in metadata)
    axis_counts = Counter(f"{row['dataset']}::{row['axis_count']}_axis" for row in metadata)
    report = {
        "status": "PASS", "schema_version": schema["schema_version"],
        "source_record_count": schema["source_record_count"], "feature_row_count": expected_rows,
        "feature_count": len(FEATURE_NAMES), "feature_row_counts": dict(sorted(feature_counts.items())),
        "axis_availability_counts": dict(sorted(axis_counts.items())),
        "trajectory_count": len(by_asset), "threshold_group_count": len(thresholds),
        "baseline_fit_rows": len(baselines),
        "artifact_sha256": {
            name: sha256(output / name) for name in (
                "split_manifest.csv", "features.npy", "feature_mask.npy", "feature_metadata.csv",
                "standardized_features.npy", "robust_distance.npy", "normalization.npz", "thresholds.csv",
            )
        },
        "leakage_guards": {
            "run_level_split_exclusive": True, "group_level_split_exclusive": True,
            "chronological_sequences_contiguous": True, "normalization_train_only": True,
            "threshold_calibration_only": True, "test_fit_rows": 0,
            "missing_axes_not_imputed": True, "dataset_scope_limited_to_selected_sources": True,
        },
        "scope_guard": "No fault-onset estimation and no final early-warning metric are claimed in D6B.",
    }
    (output / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
