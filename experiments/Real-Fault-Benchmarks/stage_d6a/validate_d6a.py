#!/usr/bin/env python3
"""Validate D6A manifests, provenance exclusions, and representative loader paths."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from unified_loader import load_record


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[2]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def choose_samples(records: list[dict[str, str]]) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for condition in ("condition_1", "condition_2", "condition_3"):
        selected.append(next(row for row in records if row["dataset"] == "XJTU-SY" and row["condition_id"] == condition))
        selected.append(
            next(
                row
                for row in records
                if row["dataset"] == "PRONOSTIA"
                and row["primary"] == "true"
                and row["condition_id"] == condition
            )
        )
    for run_id in ("NASA-IMS:set_1", "NASA-IMS:set_2", "NASA-IMS:supplied_4th_test"):
        selected.append(next(row for row in records if row["run_id"] == run_id))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=WORKSPACE_ROOT / "Dataset")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "outputs")
    args = parser.parse_args()
    dataset_root = args.dataset_root.resolve()
    output = args.output.resolve()

    inventory = json.loads((output / "dataset_inventory.json").read_text(encoding="utf-8"))
    records = read_csv(output / "record_manifest.csv")
    runs = read_csv(output / "run_manifest.csv")
    if len(records) != int(inventory["all_record_manifest_rows"]):
        raise ValueError("record count does not match dataset_inventory.json")
    if len(runs) != int(inventory["all_run_count"]):
        raise ValueError("run count does not match dataset_inventory.json")
    if len({row["record_id"] for row in records}) != len(records):
        raise ValueError("record_id is not unique")
    if len({row["run_id"] for row in runs}) != len(runs):
        raise ValueError("run_id is not unique")

    run_lookup = {row["run_id"]: row for row in runs}
    by_run: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in records:
        if row["run_id"] not in run_lookup:
            raise ValueError(f"record references unknown run {row['run_id']}")
        if row["primary"] != run_lookup[row["run_id"]]["primary"]:
            raise ValueError(f"primary flag mismatch for {row['run_id']}")
        if not (dataset_root / row["source_path"]).is_file():
            raise FileNotFoundError(row["source_path"])
        by_run[row["run_id"]].append(row)

    for run_id, rows in by_run.items():
        sequences = sorted(int(row["sequence_index"]) for row in rows)
        if sequences != list(range(1, len(rows) + 1)):
            raise ValueError(f"non-contiguous sequence_index for {run_id}")
        if len(rows) != int(run_lookup[run_id]["record_count"]):
            raise ValueError(f"record count mismatch for {run_id}")

    if any(row["primary"] != "false" for row in runs if row["role"] == "challenge_truncated_duplicate"):
        raise ValueError("a PRONOSTIA truncated duplicate is marked primary")
    if set(row["dataset"] for row in runs) != {"XJTU-SY", "PRONOSTIA", "NASA-IMS"}:
        raise ValueError("D6A contains a dataset outside the selected experimental scope")
    fourth = run_lookup["NASA-IMS:supplied_4th_test"]
    if fourth["provenance_status"] != "provenance_conflict" or fourth["terminal_fault_label"] != "unknown":
        raise ValueError("NASA supplied fourth-test provenance guard failed")

    loaded = []
    for row in choose_samples(records):
        record = load_record(row, dataset_root)
        loaded.append(
            {
                "record_id": row["record_id"],
                "source_kind": row["source_kind"],
                "shape": list(record.samples.shape),
                "sampling_rate_hz": record.sampling_rate_hz,
                "channels": list(record.channel_names),
                "rms": [float(value) for value in np.sqrt(np.mean(np.square(record.samples), axis=0))],
            }
        )

    dataset_run_counts = Counter(row["dataset"] for row in runs)
    dataset_record_counts = Counter(row["dataset"] for row in records)
    report = {
        "status": "PASS",
        "run_count": len(runs),
        "primary_run_count": sum(row["primary"] == "true" for row in runs),
        "primary_early_warning_run_count": sum(
            row["primary"] == "true" and row["early_warning_eligible"] == "true" for row in runs
        ),
        "record_manifest_rows": len(records),
        "primary_record_manifest_rows": sum(row["primary"] == "true" for row in records),
        "primary_early_warning_record_rows": sum(
            row["primary"] == "true" and row["early_warning_eligible"] == "true" for row in records
        ),
        "dataset_run_counts": dict(sorted(dataset_run_counts.items())),
        "dataset_record_counts": dict(sorted(dataset_record_counts.items())),
        "loader_smoke_test_count": len(loaded),
        "loader_smoke_tests": loaded,
        "leakage_guards": {
            "unique_record_ids": True,
            "unique_run_ids": True,
            "contiguous_sequence_indices": True,
            "pronostia_truncated_duplicates_excluded": True,
            "dataset_scope_limited_to_selected_sources": True,
            "nasa_fourth_test_fault_label_not_inferred": True,
        },
    }
    (output / "validation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
