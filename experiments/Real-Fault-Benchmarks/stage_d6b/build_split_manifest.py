#!/usr/bin/env python3
"""Build leakage-safe, run-level Stage D6B split assignments."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
D6A_OUTPUT = SCRIPT_DIR.parent / "stage_d6a" / "outputs"


XJTU_SPLITS = {
    "Bearing1_1": "train", "Bearing1_4": "train", "Bearing1_5": "train",
    "Bearing1_2": "calibration", "Bearing1_3": "test",
    "Bearing2_1": "train", "Bearing2_2": "train", "Bearing2_4": "train",
    "Bearing2_3": "calibration", "Bearing2_5": "test",
    "Bearing3_1": "train", "Bearing3_3": "train", "Bearing3_5": "train",
    "Bearing3_2": "calibration", "Bearing3_4": "test",
}

PRONOSTIA_SPLITS = {
    "Learning_set:Bearing1_1": "train", "Learning_set:Bearing1_2": "train",
    "Full_Test_Set:Bearing1_3": "train", "Full_Test_Set:Bearing1_4": "train",
    "Full_Test_Set:Bearing1_5": "calibration",
    "Full_Test_Set:Bearing1_6": "test", "Full_Test_Set:Bearing1_7": "test",
    "Learning_set:Bearing2_1": "train", "Learning_set:Bearing2_2": "train",
    "Full_Test_Set:Bearing2_3": "train", "Full_Test_Set:Bearing2_4": "train",
    "Full_Test_Set:Bearing2_5": "calibration",
    "Full_Test_Set:Bearing2_6": "test", "Full_Test_Set:Bearing2_7": "test",
    "Learning_set:Bearing3_1": "train",
    "Learning_set:Bearing3_2": "calibration",
    "Full_Test_Set:Bearing3_3": "test",
}

NASA_SPLITS = {
    "NASA-IMS:set_1": "train",
    "NASA-IMS:set_2": "calibration",
    "NASA-IMS:supplied_4th_test": "test",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def assign_primary(row: dict[str, str]) -> tuple[str, str, str]:
    dataset, run_id = row["dataset"], row["run_id"]
    if dataset == "XJTU-SY":
        key = run_id.rsplit(":", 1)[-1]
        return XJTU_SPLITS[key], "run_holdout", "fixed bearing/run split within operating condition"
    if dataset == "PRONOSTIA":
        key = run_id.split(":", 1)[1]
        return PRONOSTIA_SPLITS[key], "run_holdout", "fixed bearing/run split within operating condition"
    if dataset == "NASA-IMS":
        return NASA_SPLITS[run_id], "campaign_holdout", "complete acquisition campaign held out"
    raise ValueError(dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d6a-output", type=Path, default=D6A_OUTPUT)
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "outputs")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runs = read_csv(args.d6a_output.resolve() / "run_manifest.csv")

    fields = [
        "dataset", "run_id", "group_id", "split", "fold_id", "protocol", "reason",
        "primary", "early_warning_eligible", "condition_id", "terminal_fault_label",
        "record_count", "provenance_status",
    ]
    assignments: list[dict[str, object]] = []
    for row in runs:
        if row["primary"] != "true":
            split, protocol, reason = "excluded_duplicate", "provenance_exclusion", "truncated duplicate of complete PRONOSTIA trajectory"
            fold_id: int | str = ""
        else:
            split, protocol, reason = assign_primary(row)
            fold_id = ""
        assignments.append({field: row.get(field, "") for field in fields} | {
            "split": split, "fold_id": fold_id, "protocol": protocol, "reason": reason,
        })

    with (output / "split_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(assignments)

    group_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in assignments:
        if row["primary"] == "true":
            group_splits[(str(row["dataset"]), str(row["group_id"]))].add(str(row["split"]))
    leaked = {f"{key[0]}::{key[1]}": sorted(value) for key, value in group_splits.items() if len(value) != 1}
    if leaked:
        raise ValueError(f"group leakage detected: {leaked}")

    summary = {
        "status": "PASS",
        "policy": "deterministic run/group-level assignment; no random snapshot/window split",
        "dataset_policies": {
            "XJTU-SY": "Within each condition: three train bearings, one calibration bearing, one test bearing; calibration trajectories were selected a priori for adequate tail-quantile support and tests remain independent.",
            "PRONOSTIA": "Conditions 1/2: four train, one calibration, two test bearings; condition 3: one train, one calibration, one test bearing.",
            "NASA-IMS": "Whole-campaign holdout: set 1 train, set 2 calibration, supplied fourth-test archive test.",
        },
        "run_counts": dict(sorted(Counter(f"{row['dataset']}::{row['split']}" for row in assignments).items())),
        "record_counts": {},
        "group_leakage_count": len(leaked),
        "nasa_test_caveat": "The supplied fourth-test archive has a provenance conflict and unknown terminal fault label.",
    }
    record_counts: Counter[str] = Counter()
    for row in assignments:
        record_counts[f"{row['dataset']}::{row['split']}"] += int(str(row["record_count"]))
    summary["record_counts"] = dict(sorted(record_counts.items()))
    (output / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
