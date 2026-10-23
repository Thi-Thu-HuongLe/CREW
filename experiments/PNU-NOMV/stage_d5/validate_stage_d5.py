#!/usr/bin/env python3
"""Validate Stage D5 feature shards and generated machine-readable outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
SPLITS = ("train", "calibration", "test")
ARRAYS = (
    "features",
    "start_record",
    "sequence_start",
    "environment",
    "vector_rms",
    "state_code",
    "near_full_scale_count",
    "exact_full_scale_count",
    "quality_flag",
)
REQUIRED_CSV_FILES = (
    "static_ablation_results.csv",
    "reference_results.csv",
    "lomo_results.csv",
    "alpha_selection.csv",
    "thresholds.csv",
)


def csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=SCRIPT_DIR / "continuous_features")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "outputs")
    parser.add_argument("--batch-size", type=int, default=32768)
    args = parser.parse_args()

    feature_root = args.features.resolve()
    output_root = args.output.resolve()
    manifest = json.loads((feature_root / "dataset_manifest.json").read_text(encoding="utf-8"))
    feature_count = int(manifest["feature_count"])
    total_windows = 0
    active_windows = 0
    quality_excluded_windows = 0
    shard_count = 0

    for motor in manifest["motors"]:
        for split in SPLITS:
            shard = feature_root / str(motor) / split
            shard_manifest = json.loads((shard / "shard_manifest.json").read_text(encoding="utf-8"))
            count = int(shard_manifest["window_count"])
            arrays = {name: np.load(shard / f"{name}.npy", mmap_mode="r") for name in ARRAYS}
            expected_shapes = {
                "features": (count, feature_count),
                "start_record": (count,),
                "sequence_start": (count,),
                "environment": (count, 2),
                "vector_rms": (count,),
                "state_code": (count,),
                "near_full_scale_count": (count,),
                "exact_full_scale_count": (count,),
                "quality_flag": (count,),
            }
            for name, expected in expected_shapes.items():
                if arrays[name].shape != expected:
                    raise ValueError(f"{motor}/{split}/{name}: {arrays[name].shape} != {expected}")
            starts = arrays["start_record"]
            if count > 1 and np.any(np.diff(starts) <= 0):
                raise ValueError(f"{motor}/{split}: start_record is not strictly increasing")
            if not np.isin(arrays["state_code"], (0, 1, 2)).all():
                raise ValueError(f"{motor}/{split}: invalid state code")
            if not np.isin(arrays["quality_flag"], (0, 1)).all():
                raise ValueError(f"{motor}/{split}: invalid quality flag")
            expected_quality = np.asarray(arrays["near_full_scale_count"]) > 0
            if not np.array_equal(
                np.asarray(arrays["quality_flag"], dtype=bool), expected_quality
            ):
                raise ValueError(f"{motor}/{split}: quality flag/count mismatch")
            if np.any(
                np.asarray(arrays["exact_full_scale_count"])
                > np.asarray(arrays["near_full_scale_count"])
            ):
                raise ValueError(f"{motor}/{split}: exact count exceeds near count")
            for begin in range(0, count, args.batch_size):
                end = min(begin + args.batch_size, count)
                for name in ("features", "environment", "vector_rms"):
                    if not np.isfinite(arrays[name][begin:end]).all():
                        raise ValueError(f"{motor}/{split}/{name}: non-finite value in [{begin}:{end}]")
            state_counts = shard_manifest["state_counts"]
            observed_active = int(np.count_nonzero(arrays["state_code"] == 2))
            if observed_active != int(state_counts["active"]):
                raise ValueError(f"{motor}/{split}: active-state count mismatch")
            validation_error = float(shard_manifest["vectorized_scalar_validation_max_abs_error"])
            if not math.isfinite(validation_error) or validation_error > 1e-3:
                raise ValueError(f"{motor}/{split}: scalar/vector feature mismatch {validation_error}")
            total_windows += count
            active_windows += observed_active
            quality_excluded_windows += int(
                np.count_nonzero(
                    (np.asarray(arrays["state_code"]) == 2)
                    & (np.asarray(arrays["quality_flag"]) != 0)
                )
            )
            shard_count += 1

    if total_windows != int(manifest["total_window_count"]):
        raise ValueError("dataset total does not equal shard total")

    row_counts = {}
    for name in REQUIRED_CSV_FILES:
        observed = csv_rows(output_root / name)
        if observed <= 0:
            raise ValueError(f"{name}: expected a non-empty table")
        row_counts[name] = observed

    for path in output_root.glob("*.json"):
        json.loads(path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "status": "PASS",
                "shard_count": shard_count,
                "total_windows": total_windows,
                "active_windows": active_windows,
                "near_full_scale_active_windows": quality_excluded_windows,
                "csv_rows": row_counts,
                "json_files": len(list(output_root.glob("*.json"))),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
