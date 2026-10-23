#!/usr/bin/env python3
"""Inventory local real-fault datasets and build leakage-aware D6A manifests."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from zipfile import ZipFile

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[2]

RECORD_FIELDS = [
    "dataset",
    "run_id",
    "group_id",
    "record_id",
    "record_unit",
    "sequence_index",
    "capture_time_iso",
    "elapsed_seconds",
    "primary",
    "role",
    "source_kind",
    "source_path",
    "member_path",
    "sampling_rate_hz",
    "expected_sample_count",
    "snapshot_duration_seconds",
    "snapshot_interval_seconds",
    "channels_json",
    "signal_columns_json",
    "delimiter",
    "header_rows",
    "condition_id",
    "speed_rpm",
    "load_value",
    "load_unit",
    "terminal_fault_label",
    "fault_size_inch",
    "fault_position",
    "fault_onset_available",
    "early_warning_eligible",
    "duplicate_group_id",
    "provenance_status",
    "notes",
]

RUN_FIELDS = [
    "dataset",
    "run_id",
    "group_id",
    "role",
    "primary",
    "early_warning_eligible",
    "record_count",
    "auxiliary_record_count",
    "sampling_rates_hz_json",
    "channels_json",
    "condition_id",
    "speed_rpm",
    "load_value",
    "load_unit",
    "terminal_fault_label",
    "fault_onset_available",
    "first_capture_time",
    "last_capture_time",
    "source_path",
    "provenance_status",
    "notes",
]

XJTU_FAULTS = {
    "Bearing1_1": "outer_race",
    "Bearing1_2": "outer_race",
    "Bearing1_3": "outer_race",
    "Bearing1_4": "cage",
    "Bearing1_5": "inner_and_outer_race",
    "Bearing2_1": "inner_race",
    "Bearing2_2": "outer_race",
    "Bearing2_3": "cage",
    "Bearing2_4": "outer_race",
    "Bearing2_5": "outer_race",
    "Bearing3_1": "outer_race",
    "Bearing3_2": "inner_race_ball_cage_outer_race",
    "Bearing3_3": "inner_race",
    "Bearing3_4": "inner_race",
    "Bearing3_5": "outer_race",
}

XJTU_CONDITIONS = {
    "35Hz12kN": ("condition_1", 2100, 12.0),
    "37.5Hz11kN": ("condition_2", 2250, 11.0),
    "40Hz10kN": ("condition_3", 2400, 10.0),
}

PRONOSTIA_CONDITIONS = {
    "1": ("condition_1", 1800, 4000.0),
    "2": ("condition_2", 1650, 4200.0),
    "3": ("condition_3", 1500, 5000.0),
}


def relative(path: Path, dataset_root: Path) -> str:
    return path.resolve().relative_to(dataset_root.resolve()).as_posix()


def text_bool(value: bool) -> str:
    return "true" if value else "false"


def json_compact(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def csv_write(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def xjtu_inventory(dataset_root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    base = dataset_root / "XJTU-SY" / "XJTU-SY_Bearing_Datasets"
    records: list[dict[str, object]] = []
    runs: list[dict[str, object]] = []
    for condition_dir in sorted(path for path in base.iterdir() if path.is_dir()):
        condition_id, speed_rpm, load_kn = XJTU_CONDITIONS[condition_dir.name]
        for bearing_dir in sorted(condition_dir.iterdir(), key=lambda path: path.name):
            if not bearing_dir.is_dir():
                continue
            run_id = f"XJTU-SY:{bearing_dir.name}"
            files = sorted(bearing_dir.glob("*.csv"), key=lambda path: int(path.stem))
            for index, path in enumerate(files, start=1):
                records.append(
                    {
                        "dataset": "XJTU-SY",
                        "run_id": run_id,
                        "group_id": run_id,
                        "record_id": f"{run_id}:{index:05d}",
                        "record_unit": "two_channel_snapshot",
                        "sequence_index": index,
                        "capture_time_iso": "",
                        "elapsed_seconds": (index - 1) * 60,
                        "primary": "true",
                        "role": "run_to_failure_complete",
                        "source_kind": "filesystem_csv",
                        "source_path": relative(path, dataset_root),
                        "member_path": "",
                        "sampling_rate_hz": 25600,
                        "expected_sample_count": 32768,
                        "snapshot_duration_seconds": 1.28,
                        "snapshot_interval_seconds": 60,
                        "channels_json": json_compact(["horizontal", "vertical"]),
                        "signal_columns_json": json_compact([0, 1]),
                        "delimiter": ",",
                        "header_rows": 1,
                        "condition_id": condition_id,
                        "speed_rpm": speed_rpm,
                        "load_value": load_kn,
                        "load_unit": "kN",
                        "terminal_fault_label": XJTU_FAULTS[bearing_dir.name],
                        "fault_size_inch": "",
                        "fault_position": "",
                        "fault_onset_available": "false",
                        "early_warning_eligible": "true",
                        "duplicate_group_id": "",
                        "provenance_status": "documented",
                        "notes": "Terminal fault applies to the run; individual snapshots are not fault-labelled.",
                    }
                )
            runs.append(
                {
                    "dataset": "XJTU-SY",
                    "run_id": run_id,
                    "group_id": run_id,
                    "role": "run_to_failure_complete",
                    "primary": "true",
                    "early_warning_eligible": "true",
                    "record_count": len(files),
                    "auxiliary_record_count": 0,
                    "sampling_rates_hz_json": json_compact([25600]),
                    "channels_json": json_compact(["horizontal", "vertical"]),
                    "condition_id": condition_id,
                    "speed_rpm": speed_rpm,
                    "load_value": load_kn,
                    "load_unit": "kN",
                    "terminal_fault_label": XJTU_FAULTS[bearing_dir.name],
                    "fault_onset_available": "false",
                    "first_capture_time": "elapsed_0_s",
                    "last_capture_time": f"elapsed_{max(0, len(files)-1)*60}_s",
                    "source_path": relative(bearing_dir, dataset_root),
                    "provenance_status": "documented",
                    "notes": "Complete accelerated run-to-failure; failure onset is not annotated.",
                }
            )
    summary = {
        "status": "ready",
        "primary_run_count": len(runs),
        "primary_record_count": len(records),
        "sampling_rate_hz": 25600,
        "snapshot_samples": 32768,
        "snapshot_duration_seconds": 1.28,
        "snapshot_interval_seconds": 60,
        "channels": ["horizontal", "vertical"],
        "split_unit": "bearing run",
    }
    return records, runs, summary


def pronostia_inventory(dataset_root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    archive_path = dataset_root / "PRONOSTIA" / "phm-ieee-2012-data-challenge-dataset-master.zip"
    records: list[dict[str, object]] = []
    runs: list[dict[str, object]] = []
    with ZipFile(archive_path) as archive:
        members = archive.namelist()
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    temperatures: dict[tuple[str, str], int] = defaultdict(int)
    for member in members:
        parts = member.split("/")
        if len(parts) < 4 or not re.fullmatch(r"Bearing\d+_\d+", parts[2]):
            continue
        key = (parts[1], parts[2])
        if re.fullmatch(r"acc_\d+\.csv", parts[-1]):
            grouped[key].append(member)
        elif re.fullmatch(r"temp_\d+\.csv", parts[-1]):
            temperatures[key] += 1

    run_delimiters: dict[tuple[str, str], str] = {}
    with ZipFile(archive_path) as archive:
        for key, acc_members in grouped.items():
            first_member = min(acc_members, key=lambda value: int(Path(value).stem.split("_")[-1]))
            first_line = archive.read(first_member).splitlines()[0]
            counts = {candidate: first_line.count(candidate.encode("ascii")) for candidate in (",", ";")}
            delimiter = max(counts, key=counts.get)
            if counts[delimiter] < 5:
                raise ValueError(f"cannot identify PRONOSTIA delimiter for {first_member}")
            run_delimiters[key] = delimiter

    for (set_name, bearing), acc_members in sorted(grouped.items()):
        condition_id, speed_rpm, load_n = PRONOSTIA_CONDITIONS[bearing[len("Bearing")]]
        primary = set_name in {"Learning_set", "Full_Test_Set"}
        role = "run_to_failure_complete" if primary else "challenge_truncated_duplicate"
        run_id = f"PRONOSTIA:{set_name}:{bearing}"
        duplicate_group = f"PRONOSTIA:{bearing}" if set_name in {"Full_Test_Set", "Test_set"} else ""
        ordered = sorted(acc_members, key=lambda value: int(Path(value).stem.split("_")[-1]))
        for index, member in enumerate(ordered, start=1):
            records.append(
                {
                    "dataset": "PRONOSTIA",
                    "run_id": run_id,
                    "group_id": run_id,
                    "record_id": f"{run_id}:{index:05d}",
                    "record_unit": "two_channel_snapshot",
                    "sequence_index": index,
                    "capture_time_iso": "",
                    "elapsed_seconds": (index - 1) * 10,
                    "primary": text_bool(primary),
                    "role": role,
                    "source_kind": "zip_csv",
                    "source_path": relative(archive_path, dataset_root),
                    "member_path": member,
                    "sampling_rate_hz": 25600,
                    "expected_sample_count": 2560,
                    "snapshot_duration_seconds": 0.1,
                    "snapshot_interval_seconds": 10,
                    "channels_json": json_compact(["horizontal", "vertical"]),
                    "signal_columns_json": json_compact([4, 5]),
                    "delimiter": run_delimiters[(set_name, bearing)],
                    "header_rows": 0,
                    "condition_id": condition_id,
                    "speed_rpm": speed_rpm,
                    "load_value": load_n,
                    "load_unit": "N",
                    "terminal_fault_label": "mixed_bearing_degradation_unspecified",
                    "fault_size_inch": "",
                    "fault_position": "",
                    "fault_onset_available": "false",
                    "early_warning_eligible": text_bool(primary),
                    "duplicate_group_id": duplicate_group,
                    "provenance_status": "documented",
                    "notes": (
                        "Full trajectory; stop criterion is vibration amplitude above 20 g."
                        if primary
                        else "Truncated challenge prefix duplicated by Full_Test_Set; excluded from primary use."
                    ),
                }
            )
        runs.append(
            {
                "dataset": "PRONOSTIA",
                "run_id": run_id,
                "group_id": run_id,
                "role": role,
                "primary": text_bool(primary),
                "early_warning_eligible": text_bool(primary),
                "record_count": len(ordered),
                "auxiliary_record_count": temperatures[(set_name, bearing)],
                "sampling_rates_hz_json": json_compact([25600]),
                "channels_json": json_compact(["horizontal", "vertical"]),
                "condition_id": condition_id,
                "speed_rpm": speed_rpm,
                "load_value": load_n,
                "load_unit": "N",
                "terminal_fault_label": "mixed_bearing_degradation_unspecified",
                "fault_onset_available": "false",
                "first_capture_time": "elapsed_0_s",
                "last_capture_time": f"elapsed_{max(0, len(ordered)-1)*10}_s",
                "source_path": relative(archive_path, dataset_root),
                "provenance_status": "documented",
                "notes": (
                    "Primary complete run."
                    if primary
                    else f"Excluded duplicate of Full_Test_Set; duplicate_group_id={duplicate_group}."
                ),
            }
        )
    primary_runs = [row for row in runs if row["primary"] == "true"]
    primary_records = [row for row in records if row["primary"] == "true"]
    summary = {
        "status": "ready_archive_stream",
        "all_run_count": len(runs),
        "primary_run_count": len(primary_runs),
        "excluded_truncated_duplicate_run_count": len(runs) - len(primary_runs),
        "all_vibration_record_count": len(records),
        "primary_vibration_record_count": len(primary_records),
        "sampling_rate_hz": 25600,
        "snapshot_samples": 2560,
        "snapshot_duration_seconds": 0.1,
        "snapshot_interval_seconds": 10,
        "channels": ["horizontal", "vertical"],
        "split_unit": "bearing run",
        "primary_sets": ["Learning_set", "Full_Test_Set"],
        "excluded_set": "Test_set",
    }
    return records, runs, summary


def tar_members(path: Path) -> list[str]:
    executable = shutil.which("tar") or shutil.which("tar.exe")
    if executable is None:
        raise RuntimeError("tar/bsdtar is required to inventory NASA-IMS RAR archives")
    completed = subprocess.run(
        [executable, "-tf", str(path)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    return completed.stdout.splitlines()


def nasa_inventory(dataset_root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    base = dataset_root / "NASA-IMS" / "raw" / "source_bundle"
    configs = [
        {
            "archive": "1st_test.rar",
            "run_id": "NASA-IMS:set_1",
            "channels": ["bearing1_x", "bearing1_y", "bearing2_x", "bearing2_y", "bearing3_x", "bearing3_y", "bearing4_x", "bearing4_y"],
            "fault": "bearing3_inner_race_and_bearing4_roller_element",
            "status": "documented",
            "notes": "First 43 snapshots nominally use 5-minute intervals; later snapshots use 10-minute intervals.",
        },
        {
            "archive": "2nd_test.rar",
            "run_id": "NASA-IMS:set_2",
            "channels": ["bearing1", "bearing2", "bearing3", "bearing4"],
            "fault": "bearing1_outer_race",
            "status": "documented",
            "notes": "Documented second run-to-failure experiment.",
        },
        {
            "archive": "3rd_test.rar",
            "run_id": "NASA-IMS:supplied_4th_test",
            "channels": ["bearing1", "bearing2", "bearing3", "bearing4"],
            "fault": "unknown",
            "status": "provenance_conflict",
            "notes": "Archive filename says third test, but internal root is 4th_test and timestamps extend to 2004-04-18; no terminal-fault label is assigned.",
        },
    ]
    timestamp_pattern = re.compile(r"(\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2})$")
    records: list[dict[str, object]] = []
    runs: list[dict[str, object]] = []
    for config in configs:
        archive_path = base / str(config["archive"])
        timestamped: list[tuple[datetime, str]] = []
        for member in tar_members(archive_path):
            match = timestamp_pattern.search(member)
            if match:
                timestamped.append((datetime.strptime(match.group(1), "%Y.%m.%d.%H.%M.%S"), member))
        timestamped.sort()
        first_time = timestamped[0][0]
        channels = list(config["channels"])
        for index, (capture_time, member) in enumerate(timestamped, start=1):
            records.append(
                {
                    "dataset": "NASA-IMS",
                    "run_id": config["run_id"],
                    "group_id": config["run_id"],
                    "record_id": f"{config['run_id']}:{index:05d}",
                    "record_unit": "multi_channel_snapshot",
                    "sequence_index": index,
                    "capture_time_iso": capture_time.isoformat(),
                    "elapsed_seconds": int((capture_time - first_time).total_seconds()),
                    "primary": "true",
                    "role": "run_to_failure_complete",
                    "source_kind": "rar_ascii",
                    "source_path": relative(archive_path, dataset_root),
                    "member_path": member,
                    "sampling_rate_hz": 20000,
                    "expected_sample_count": 20480,
                    "snapshot_duration_seconds": 1.024,
                    "snapshot_interval_seconds": 600,
                    "channels_json": json_compact(channels),
                    "signal_columns_json": json_compact(list(range(len(channels)))),
                    "delimiter": "whitespace",
                    "header_rows": 0,
                    "condition_id": "2000rpm_6000lb",
                    "speed_rpm": 2000,
                    "load_value": 6000,
                    "load_unit": "lb",
                    "terminal_fault_label": config["fault"],
                    "fault_size_inch": "",
                    "fault_position": "",
                    "fault_onset_available": "false",
                    "early_warning_eligible": "true",
                    "duplicate_group_id": "",
                    "provenance_status": config["status"],
                    "notes": config["notes"],
                }
            )
        runs.append(
            {
                "dataset": "NASA-IMS",
                "run_id": config["run_id"],
                "group_id": config["run_id"],
                "role": "run_to_failure_complete",
                "primary": "true",
                "early_warning_eligible": "true",
                "record_count": len(timestamped),
                "auxiliary_record_count": 0,
                "sampling_rates_hz_json": json_compact([20000]),
                "channels_json": json_compact(channels),
                "condition_id": "2000rpm_6000lb",
                "speed_rpm": 2000,
                "load_value": 6000,
                "load_unit": "lb",
                "terminal_fault_label": config["fault"],
                "fault_onset_available": "false",
                "first_capture_time": timestamped[0][0].isoformat(),
                "last_capture_time": timestamped[-1][0].isoformat(),
                "source_path": relative(archive_path, dataset_root),
                "provenance_status": config["status"],
                "notes": config["notes"],
            }
        )
    summary = {
        "status": "ready_via_bsdtar",
        "primary_run_count": len(runs),
        "primary_record_count": len(records),
        "sampling_rate_hz": 20000,
        "snapshot_samples": 20480,
        "snapshot_duration_seconds": 1.024,
        "nominal_snapshot_interval_seconds": 600,
        "split_unit": "test-to-failure experiment",
        "known_issue": "The supplied 3rd_test.rar is internally labelled 4th_test and does not match documented Set 3; its terminal fault remains unknown.",
    }
    return records, runs, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=WORKSPACE_ROOT / "Dataset")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "outputs")
    args = parser.parse_args()
    dataset_root = args.dataset_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    all_records: list[dict[str, object]] = []
    all_runs: list[dict[str, object]] = []
    summaries: dict[str, object] = {}
    for name, builder in (
        ("XJTU-SY", xjtu_inventory),
        ("PRONOSTIA", pronostia_inventory),
        ("NASA-IMS", nasa_inventory),
    ):
        records, runs, summary = builder(dataset_root)
        all_records.extend(records)
        all_runs.extend(runs)
        summaries[name] = summary
        print(f"{name}: {len(runs):,} runs, {len(records):,} record rows", flush=True)

    csv_write(output / "record_manifest.csv", RECORD_FIELDS, all_records)
    csv_write(output / "run_manifest.csv", RUN_FIELDS, all_runs)
    primary_records = [row for row in all_records if row["primary"] == "true"]
    primary_runs = [row for row in all_runs if row["primary"] == "true"]
    early_warning_records = [row for row in primary_records if row["early_warning_eligible"] == "true"]
    early_warning_runs = [row for row in primary_runs if row["early_warning_eligible"] == "true"]
    manifest = {
        "stage": "D6A_real_fault_inventory_and_unified_loaders",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "dataset_count": 3,
        "all_run_count": len(all_runs),
        "primary_run_count": len(primary_runs),
        "primary_early_warning_run_count": len(early_warning_runs),
        "all_record_manifest_rows": len(all_records),
        "primary_record_manifest_rows": len(primary_records),
        "primary_early_warning_record_rows": len(early_warning_records),
        "datasets": summaries,
        "primary_policy": {
            "XJTU-SY": "all 15 complete run-to-failure bearing runs",
            "PRONOSTIA": "Learning_set plus Full_Test_Set; Test_set truncated duplicates excluded",
            "NASA-IMS": "three supplied RAR experiments; supplied fourth-test label remains unknown",
        },
        "leakage_controls": [
            "Split by group_id/run_id before window generation; never randomly split windows from the same run.",
            "PRONOSTIA Test_set is excluded because it duplicates prefixes present in Full_Test_Set.",
            "Terminal fault labels are run-level metadata and must not be copied to every pre-fault snapshot as a binary fault label.",
            "Fault-onset labels are unavailable and must be estimated only inside training data, then locked for evaluation.",
        ],
        "claim_boundaries": {
            "run_to_failure": ["XJTU-SY", "PRONOSTIA", "NASA-IMS"],
            "no_direct_early_warning_label": True,
        },
        "known_issues": [
            "NASA-IMS 3rd_test.rar internally contains 4th_test and conflicts with the supplied PDF's Set 3 description.",
            "NASA-IMS supplied fourth test has no verified terminal fault label in the local documentation.",
            "PRONOSTIA terminal damage component is unspecified; the documented stop criterion is vibration above 20 g.",
        ],
        "artifacts": {
            "record_manifest": "record_manifest.csv",
            "run_manifest": "run_manifest.csv",
            "loader": "../unified_loader.py",
        },
    }
    (output / "dataset_inventory.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "all_run_count",
                    "primary_run_count",
                    "primary_early_warning_run_count",
                    "all_record_manifest_rows",
                    "primary_record_manifest_rows",
                    "primary_early_warning_record_rows",
                )
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
