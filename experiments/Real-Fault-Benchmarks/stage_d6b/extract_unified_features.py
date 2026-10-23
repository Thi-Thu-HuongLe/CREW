#!/usr/bin/env python3
"""Extract the causal 40-feature representation for all primary D6A records."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Mapping
from zipfile import ZipFile

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[2]
D6A_DIR = SCRIPT_DIR.parent / "stage_d6a"
sys.path.insert(0, str(D6A_DIR))

from unified_loader import _read_delimited  # noqa: E402
from feature_schema import FEATURE_NAMES, extract_features  # noqa: E402


METADATA_FIELDS = (
    "feature_row_id", "dataset", "run_id", "group_id", "asset_id", "record_id",
    "sequence_index", "capture_time_iso", "elapsed_seconds", "split", "fold_id",
    "condition_id", "sampling_rate_hz", "terminal_fault_label", "provenance_status",
    "axis_count", "channel_mapping_json", "early_warning_eligible", "source_path", "member_path",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_signal(row: Mapping[str, str], dataset_root: Path, archives: Mapping[str, ZipFile]) -> np.ndarray:
    source_path = dataset_root / row["source_path"]
    channels = json.loads(row["channels_json"])
    columns = [int(value) for value in json.loads(row["signal_columns_json"])]
    delimiter_value = row.get("delimiter", "")
    delimiter = None if delimiter_value in ("", "whitespace") else delimiter_value
    skip_rows = int(row.get("header_rows", "0") or 0)
    source_kind = row["source_kind"]
    if source_kind == "filesystem_csv":
        return _read_delimited(source_path, delimiter, skip_rows, columns)
    if source_kind == "zip_csv":
        return _read_delimited(archives[row["source_path"]].read(row["member_path"]), delimiter, skip_rows, columns)
    raise ValueError(f"read_signal cannot directly read {source_kind}")


def axis_views(row: Mapping[str, str], samples: np.ndarray) -> list[tuple[str, np.ndarray, list[str]]]:
    channels = list(json.loads(row["channels_json"]))
    if row["dataset"] != "NASA-IMS":
        mapped = ["x", "y"][: samples.shape[1]]
        return [(row["run_id"], samples[:, : len(mapped)], mapped)]
    if samples.shape[1] == 8:
        return [
            (f"{row['run_id']}:bearing{index + 1}", samples[:, index * 2 : index * 2 + 2], ["x", "y"])
            for index in range(4)
        ]
    if samples.shape[1] == 4:
        return [
            (f"{row['run_id']}:bearing{index + 1}", samples[:, index : index + 1], ["x"])
            for index in range(4)
        ]
    raise ValueError(f"unexpected NASA shape {samples.shape} for {row['record_id']}")


def metadata_row(row: Mapping[str, str], split: Mapping[str, str], asset_id: str, axes: list[str], row_index: int) -> dict[str, object]:
    return {
        "feature_row_id": row_index,
        "dataset": row["dataset"], "run_id": row["run_id"], "group_id": row["group_id"],
        "asset_id": asset_id, "record_id": row["record_id"], "sequence_index": row["sequence_index"],
        "capture_time_iso": row["capture_time_iso"], "elapsed_seconds": row["elapsed_seconds"],
        "split": split["split"], "fold_id": split["fold_id"], "condition_id": row["condition_id"],
        "sampling_rate_hz": row["sampling_rate_hz"], "terminal_fault_label": row["terminal_fault_label"],
        "provenance_status": row["provenance_status"], "axis_count": len(axes),
        "channel_mapping_json": json.dumps(axes, separators=(",", ":")),
        "early_warning_eligible": row["early_warning_eligible"],
        "source_path": row["source_path"], "member_path": row["member_path"],
    }


def process_record(row: dict[str, str], samples: np.ndarray, split: dict[str, str], features: np.ndarray, masks: np.ndarray, metadata: list[dict[str, object]], offset: int) -> int:
    expected = int(row["expected_sample_count"])
    if samples.shape[0] != expected or not np.isfinite(samples).all():
        raise ValueError(f"invalid samples for {row['record_id']}: {samples.shape}, expected {expected}")
    for asset_id, view, axes in axis_views(row, samples):
        vector, mask = extract_features(view, float(row["sampling_rate_hz"]))
        features[offset] = vector
        masks[offset] = mask
        metadata.append(metadata_row(row, split, asset_id, axes, offset))
        offset += 1
    return offset


def extract_rar_once(archive: Path, temporary_parent: Path) -> Path:
    executable = shutil.which("tar") or shutil.which("tar.exe")
    if executable is None:
        raise RuntimeError("tar/bsdtar is required for NASA-IMS RAR archives")
    temporary = Path(tempfile.mkdtemp(prefix="nasa_", dir=temporary_parent))
    subprocess.run([executable, "-xf", str(archive), "-C", str(temporary)], check=True)
    return temporary


def print_progress(done: int, total: int, started: float, label: str) -> None:
    elapsed = max(time.perf_counter() - started, 1e-9)
    print(f"[{done:>5}/{total}] {label}; {done / elapsed:.1f} source records/s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=WORKSPACE_ROOT / "Dataset")
    parser.add_argument("--d6a-output", type=Path, default=D6A_DIR / "outputs")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "outputs")
    parser.add_argument("--dataset", action="append", help="Development-only dataset filter; repeat as needed")
    parser.add_argument("--limit", type=int, default=0, help="Development-only source-record limit; zero means all")
    args = parser.parse_args()
    dataset_root, output = args.dataset_root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    temporary_parent = SCRIPT_DIR / "tmp"
    temporary_parent.mkdir(parents=True, exist_ok=True)

    records = [row for row in read_csv(args.d6a_output.resolve() / "record_manifest.csv") if row["primary"] == "true"]
    if args.dataset:
        records = [row for row in records if row["dataset"] in set(args.dataset)]
    if args.limit:
        records = records[: args.limit]
    splits = {row["run_id"]: row for row in read_csv(output / "split_manifest.csv")}
    if any(row["run_id"] not in splits for row in records):
        raise ValueError("split_manifest.csv is incomplete")
    feature_count = sum(4 if row["dataset"] == "NASA-IMS" else 1 for row in records)
    feature_temp = output / "features.partial.npy"
    mask_temp = output / "feature_mask.partial.npy"
    features = np.lib.format.open_memmap(feature_temp, mode="w+", dtype=np.float32, shape=(feature_count, len(FEATURE_NAMES)))
    masks = np.lib.format.open_memmap(mask_temp, mode="w+", dtype=bool, shape=(feature_count, len(FEATURE_NAMES)))
    metadata: list[dict[str, object]] = []
    started, offset, done = time.perf_counter(), 0, 0

    non_nasa = [row for row in records if row["dataset"] != "NASA-IMS"]
    nasa_by_source: dict[str, list[dict[str, str]]] = {}
    for row in records:
        if row["dataset"] == "NASA-IMS":
            nasa_by_source.setdefault(row["source_path"], []).append(row)

    with ExitStack() as stack:
        archive_paths = sorted({row["source_path"] for row in non_nasa if row["source_kind"].startswith("zip_")})
        archives = {path: stack.enter_context(ZipFile(dataset_root / path)) for path in archive_paths}
        for row in non_nasa:
            try:
                samples = read_signal(row, dataset_root, archives)
                offset = process_record(row, samples, splits[row["run_id"]], features, masks, metadata, offset)
            except Exception as error:
                raise RuntimeError(f"failed source record {row['record_id']}") from error
            done += 1
            if done % 500 == 0:
                print_progress(done, len(records), started, row["dataset"])

    for source_path, source_rows in nasa_by_source.items():
        extracted = extract_rar_once(dataset_root / source_path, temporary_parent)
        try:
            for row in source_rows:
                try:
                    member_path = extracted / Path(row["member_path"])
                    columns = [int(value) for value in json.loads(row["signal_columns_json"])]
                    samples = _read_delimited(member_path, None, int(row["header_rows"] or 0), columns)
                    offset = process_record(row, samples, splits[row["run_id"]], features, masks, metadata, offset)
                except Exception as error:
                    raise RuntimeError(f"failed source record {row['record_id']}") from error
                done += 1
                if done % 500 == 0:
                    print_progress(done, len(records), started, row["dataset"])
        finally:
            shutil.rmtree(extracted)

    if offset != feature_count or len(metadata) != feature_count:
        raise ValueError(f"feature row mismatch: {offset}, {len(metadata)}, expected {feature_count}")
    features.flush()
    masks.flush()
    del features, masks
    feature_temp.replace(output / "features.npy")
    mask_temp.replace(output / "feature_mask.npy")
    with (output / "feature_metadata.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(metadata)
    schema = {
        "schema_version": "D6B-1.0",
        "feature_count": len(FEATURE_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "observation_unit": "one source acquisition snapshot; NASA expands to one row per bearing",
        "base_window_seconds": 0.1,
        "within_snapshot_aggregation": "median over complete non-overlapping base windows; incomplete tail discarded",
        "causal": True,
        "cross_snapshot_concatenation": False,
        "amplitude_units": "native source units; robust scaling is fitted later by dataset",
        "frequency_units": "fraction of Nyquist frequency",
        "missing_axis_policy": "NaN plus boolean feature mask; axes are never copied or imputed",
        "source_record_count": len(records),
        "feature_row_count": feature_count,
    }
    (output / "feature_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    print_progress(done, len(records), started, "complete")
    print(json.dumps({"status": "PASS", **schema}, indent=2))


if __name__ == "__main__":
    main()
