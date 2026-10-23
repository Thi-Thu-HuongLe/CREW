#!/usr/bin/env python3
"""Fit training-only robust scaling and calibration-only provisional thresholds."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from feature_schema import FEATURE_NAMES, feature_indices_for_axes


SCRIPT_DIR = Path(__file__).resolve().parent


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def group_key(row: dict[str, str]) -> tuple[str, str]:
    return row["dataset"], row["condition_id"]


def slug(key: tuple[str, str]) -> str:
    return re.sub(r"[^a-z0-9]+", "_", "_".join(key).lower()).strip("_")


def baseline_indices(rows: list[dict[str, str]], split: str, fraction: float) -> list[int]:
    trajectories: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row["split"] == split and row["early_warning_eligible"] == "true":
            trajectories[row["asset_id"]].append((int(row["sequence_index"]), index))
    selected: list[int] = []
    for values in trajectories.values():
        ordered = [index for _, index in sorted(values)]
        count = max(5, int(math.ceil(len(ordered) * fraction)))
        selected.extend(ordered[: min(count, len(ordered))])
    return sorted(selected)


def robust_parameters(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.median(values, axis=0)
    q25, q75 = np.quantile(values, [0.25, 0.75], axis=0)
    iqr_scale = (q75 - q25) / 1.349
    mad_scale = 1.4826 * np.median(np.abs(values - center), axis=0)
    floor = np.maximum(np.abs(center) * 1e-9, 1e-12)
    scale = np.where(iqr_scale > floor, iqr_scale, np.where(mad_scale > floor, mad_scale, floor))
    near_constant = (iqr_scale <= floor) & (mad_scale <= floor)
    return center, scale, near_constant


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "outputs")
    parser.add_argument("--baseline-fraction", type=float, default=0.10)
    args = parser.parse_args()
    output = args.output.resolve()
    if not 0 < args.baseline_fraction <= 0.5:
        raise ValueError("baseline fraction must be in (0, 0.5]")
    rows = read_csv(output / "feature_metadata.csv")
    features = np.load(output / "features.npy", mmap_mode="r")
    masks = np.load(output / "feature_mask.npy", mmap_mode="r")
    if len(rows) != features.shape[0] or features.shape != masks.shape:
        raise ValueError("feature/metadata/mask shape mismatch")

    train_baseline = baseline_indices(rows, "train", args.baseline_fraction)
    calibration_baseline = baseline_indices(rows, "calibration", args.baseline_fraction)
    train_set, calibration_set = set(train_baseline), set(calibration_baseline)
    if train_set & calibration_set:
        raise ValueError("train/calibration baseline overlap")

    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row["early_warning_eligible"] == "true":
            groups[group_key(row)].append(index)

    standardized = np.lib.format.open_memmap(
        output / "standardized_features.partial.npy", mode="w+", dtype=np.float32, shape=features.shape
    )
    standardized[:] = np.nan
    scores = np.lib.format.open_memmap(
        output / "robust_distance.partial.npy", mode="w+", dtype=np.float32, shape=(features.shape[0],)
    )
    scores[:] = np.nan
    normalization_arrays: dict[str, np.ndarray] = {}
    registry: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []

    for key, group_rows in sorted(groups.items()):
        dataset, condition = key
        # NASA set 1 has x/y while sets 2 and 4 have x only. Index 0 (vector RMS)
        # therefore is not cross-campaign compatible and must also be excluded.
        candidate_indices = np.arange(1, 14, dtype=int) if dataset == "NASA-IMS" else feature_indices_for_axes(2)
        train = [index for index in group_rows if index in train_set]
        calibration = [index for index in group_rows if index in calibration_set]
        if not train or not calibration:
            raise ValueError(f"missing train/calibration baseline for {key}: {len(train)}, {len(calibration)}")
        train_values = np.asarray(features[np.ix_(train, candidate_indices)], dtype=np.float64)
        if not np.isfinite(train_values).all():
            raise ValueError(f"non-finite training values for {key}")
        center, scale, near_constant = robust_parameters(train_values)
        excluded_indices = candidate_indices[near_constant]
        indices = candidate_indices[~near_constant]
        center = center[~near_constant]
        scale = scale[~near_constant]
        if not indices.size:
            raise ValueError(f"all candidate features are near-constant for {key}")
        all_values = np.asarray(features[np.ix_(group_rows, indices)], dtype=np.float64)
        transformed = (all_values - center) / scale
        standardized[np.ix_(group_rows, indices)] = transformed.astype(np.float32)
        scores[group_rows] = np.sqrt(np.mean(transformed ** 2, axis=1)).astype(np.float32)
        calibration_scores = np.asarray(scores[calibration], dtype=np.float64)
        quantiles = np.quantile(calibration_scores, [0.99, 0.995, 0.999])
        identifier = slug(key)
        normalization_arrays[f"{identifier}__feature_indices"] = indices
        normalization_arrays[f"{identifier}__center"] = center
        normalization_arrays[f"{identifier}__scale"] = scale
        registry.append({
            "id": identifier, "dataset": dataset, "condition_id": condition,
            "candidate_feature_count": int(candidate_indices.size),
            "feature_count": int(indices.size), "feature_indices": indices.tolist(),
            "feature_names": [FEATURE_NAMES[index] for index in indices],
            "train_baseline_rows": len(train), "calibration_baseline_rows": len(calibration),
            "near_constant_feature_count": int(near_constant.sum()),
            "excluded_near_constant_feature_names": [FEATURE_NAMES[index] for index in excluded_indices],
        })
        threshold_rows.append({
            "dataset": dataset, "condition_id": condition, "score": "robust_rms_z_distance",
            "calibration_baseline_rows": len(calibration), "q0.990": float(quantiles[0]),
            "q0.995": float(quantiles[1]), "q0.999": float(quantiles[2]),
            "empirical_tail_resolution": float(1.0 / (len(calibration_scores) + 1)),
            "q0.990_supported_n_ge_99": str(len(calibration_scores) >= 99).lower(),
            "q0.995_supported_n_ge_199": str(len(calibration_scores) >= 199).lower(),
            "q0.999_supported_n_ge_999": str(len(calibration_scores) >= 999).lower(),
            "status": "provisional_no_onset_labels",
        })

    standardized.flush(); scores.flush()
    del standardized, scores
    (output / "standardized_features.partial.npy").replace(output / "standardized_features.npy")
    (output / "robust_distance.partial.npy").replace(output / "robust_distance.npy")
    np.savez_compressed(output / "normalization.npz", **normalization_arrays)
    with (output / "thresholds.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(threshold_rows[0]))
        writer.writeheader(); writer.writerows(threshold_rows)
    baseline_rows = []
    for index in train_baseline:
        baseline_rows.append({"feature_row_id": index, "role": "fit_normalization", **{key: rows[index][key] for key in ("dataset", "condition_id", "run_id", "asset_id", "sequence_index", "split")}})
    for index in calibration_baseline:
        baseline_rows.append({"feature_row_id": index, "role": "fit_threshold", **{key: rows[index][key] for key in ("dataset", "condition_id", "run_id", "asset_id", "sequence_index", "split")}})
    with (output / "baseline_rows.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(baseline_rows[0]))
        writer.writeheader(); writer.writerows(baseline_rows)
    manifest = {
        "status": "PASS", "schema_version": "D6B-1.0", "baseline_fraction": args.baseline_fraction,
        "baseline_definition": "first ceil(fraction*N), minimum five observations, independently per asset trajectory",
        "normalization_fit_split": "train only", "threshold_fit_split": "calibration only",
        "test_rows_used_for_fitting": 0, "normalization": "median and max(IQR/1.349, MAD*1.4826, numerical floor); features constant by both IQR and MAD on train are excluded from scoring",
        "score": "root mean square of robust z-scores over the dataset-compatible feature subset",
        "thresholds": [0.99, 0.995, 0.999],
        "threshold_status": "provisional; D6B does not estimate fault onset or final warning performance",
        "missing_axis_policy": "NASA uses the common x-only 13-feature subset across its 8- and 4-channel campaigns; vector RMS is excluded because it contains y in campaign 1; missing axes are not imputed",
        "group_registry": registry,
        "baseline_role_counts": dict(Counter(row["role"] for row in baseline_rows)),
    }
    (output / "preprocessing_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
