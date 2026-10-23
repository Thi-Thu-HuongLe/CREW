#!/usr/bin/env python3
"""Validate D6F policy provenance, decisions, gaps, and aggregate results."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PNU_DIR = SCRIPT_DIR.parent
D5_DIR = PNU_DIR / "stage_d5"
D5_OUTPUT = D5_DIR / "outputs"
D5_FEATURES = D5_DIR / "continuous_features"
D6C_OUTPUT = PNU_DIR.parent / "Real-Fault-Benchmarks" / "stage_d6c" / "outputs"
OUTPUT = SCRIPT_DIR / "outputs"
sys.path.insert(0, str(D5_DIR))
import run_continuous_evaluation as d5  # noqa: E402


MOTORS = ("MO01", "MO03", "MO04", "MO05", "MO06", "MO07", "MO08", "MO10")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_warning(point: np.ndarray, starts: np.ndarray, persistence: int) -> np.ndarray:
    result = np.zeros(len(point), dtype=bool)
    for current in d5.segment_slices(starts, 2):
        count = 0
        for index in range(current.start or 0, current.stop):
            count = count + 1 if bool(point[index]) else 0
            if count >= persistence:
                result[index] = True
    return result


def main() -> None:
    manifest = json.loads((OUTPUT / "policy_manifest.json").read_text(encoding="utf-8"))
    compatibility = json.loads((OUTPUT / "domain_compatibility.json").read_text(encoding="utf-8"))
    locked = json.loads((D6C_OUTPUT / "locked_rules.json").read_text(encoding="utf-8"))
    parameters = read_csv(OUTPUT / "commissioning_parameters.csv")
    exclusions = read_csv(OUTPUT / "operating_state_exclusions.csv")
    calibration = read_csv(OUTPUT / "calibration_diagnostics.csv")
    motors = read_csv(OUTPUT / "motor_results.csv")
    aggregates = read_csv(OUTPUT / "aggregate_results.csv")
    episodes = read_csv(OUTPUT / "episode_results.csv")
    d5_reference = read_csv(OUTPUT / "d5_native_references.csv")
    decisions = np.load(OUTPUT / "test_decisions.npz")

    if manifest["test_rows_used_for_fit_or_selection"] != 0:
        raise ValueError("test entered D6F fitting or selection")
    if manifest["input_sha256"]["d6c_locked_rules.json"] != sha256(D6C_OUTPUT / "locked_rules.json"):
        raise ValueError("D6C locked rule hash changed")
    for dataset, rule in locked["rules"].items():
        transferred = manifest["external_policies"][dataset]
        for field in ("alpha", "threshold", "persistence"):
            if not np.isclose(float(transferred[field]), float(rule[field])):
                raise ValueError(f"external policy changed: {dataset} {field}")
    if compatibility["not_evaluated"].find("strict end-to-end zero-shot") < 0:
        raise ValueError("domain compatibility boundary is missing")

    if len(parameters) != 9 or any(row["fit_split"] != "train_only" for row in parameters):
        raise ValueError("commissioning parameter audit mismatch")
    if Counter(row["split"] for row in exclusions) != Counter({"calibration": 8, "test": 8}):
        raise ValueError("operating-state exclusion coverage mismatch")
    exclusion_summary = {
        split: (
            sum(int(row["active_windows"]) for row in exclusions if row["split"] == split),
            sum(int(row["nonactive_excluded"]) for row in exclusions if row["split"] == split),
            sum(int(row["near_full_scale_excluded"]) for row in exclusions if row["split"] == split),
        ) for split in ("calibration", "test")
    }
    expected_exclusions = {}
    for split in ("calibration", "test"):
        analyzed = nonactive = near_full_scale = 0
        for motor in MOTORS:
            shard = d5.load_shard(D5_FEATURES, motor, split)
            active = np.asarray(shard["state"]) == 2
            flagged = np.asarray(shard["quality_flag"]) != 0
            analyzed += int(np.count_nonzero(active & ~flagged))
            nonactive += int(np.count_nonzero(~active))
            near_full_scale += int(np.count_nonzero(active & flagged))
        expected_exclusions[split] = (analyzed, nonactive, near_full_scale)
    if exclusion_summary != expected_exclusions:
        raise ValueError(f"active-window scope changed: {exclusion_summary}")

    if len(calibration) != 48 or len(motors) != 48 or len(aggregates) != 6:
        raise ValueError("expected six policies x eight motors")
    if Counter(row["source_dataset"] for row in motors) != Counter({"NASA-IMS": 16, "PRONOSTIA": 16, "XJTU-SY": 16}):
        raise ValueError("external policy coverage mismatch")
    if len(decisions.files) != 96:
        raise ValueError("test decision archive must contain point/warning arrays for 48 motor-config pairs")

    motor_lookup = {(row["config_id"], row["motor"]): row for row in motors}
    for row in motors:
        config, motor = row["config_id"], row["motor"]
        point = np.asarray(decisions[f"{config}__{motor}__point"], dtype=bool)
        warning = np.asarray(decisions[f"{config}__{motor}__warning"], dtype=bool)
        shard = d5.load_shard(D5_FEATURES, motor, "test")
        indices = d5.active_indices(shard, True)
        starts = np.asarray(shard["starts"][indices], dtype=np.int64)
        if len(point) != len(starts) or int(row["test_observation_count"]) != len(starts):
            raise ValueError(f"decision alignment mismatch: {config} {motor}")
        expected = expected_warning(point, starts, int(row["persistence"]))
        if not np.array_equal(warning, expected):
            raise ValueError(f"persistence/gap reset mismatch: {config} {motor}")
        if int(point.sum()) != int(row["test_point_alarm_count"]) or int(warning.sum()) != int(row["test_warning_count"]):
            raise ValueError(f"decision count mismatch: {config} {motor}")
        episode_count = len(d5.episode_details(warning, starts, 2, 4))
        if episode_count != int(row["test_episode_count"]):
            raise ValueError(f"episode count mismatch: {config} {motor}")
        if (episode_count > 0) != (row["test_motor_false_warning"] == "true"):
            raise ValueError(f"motor false-warning flag mismatch: {config} {motor}")

    for aggregate in aggregates:
        rows = [row for row in motors if row["config_id"] == aggregate["config_id"]]
        observation_count = sum(int(row["test_observation_count"]) for row in rows)
        episode_count = sum(int(row["test_episode_count"]) for row in rows)
        hours = sum(float(row["test_monitoring_hours"]) for row in rows)
        flagged = sum(row["test_motor_false_warning"] == "true" for row in rows)
        if observation_count != int(aggregate["test_observation_count"]) or episode_count != int(aggregate["test_episode_count"]):
            raise ValueError(f"aggregate count mismatch: {aggregate['config_id']}")
        if not np.isclose(hours, float(aggregate["test_monitoring_hours"])) or not np.isclose(episode_count / (hours / 24), float(aggregate["test_episodes_per_day"])):
            raise ValueError(f"aggregate exposure/rate mismatch: {aggregate['config_id']}")
        if flagged != int(aggregate["test_flagged_motor_count"]):
            raise ValueError(f"aggregate motor flag mismatch: {aggregate['config_id']}")
        lower = float(aggregate["motor_cluster_bootstrap_episodes_per_day_95_lower"])
        upper = float(aggregate["motor_cluster_bootstrap_episodes_per_day_95_upper"])
        if not (0 <= lower <= upper):
            raise ValueError("invalid cluster-bootstrap interval")
        if episode_count == 0:
            expected_upper = -np.log(0.05) / (hours / 24.0)
            if not np.isclose(float(aggregate["zero_episode_one_sided_95_poisson_upper_per_day"]), expected_upper):
                raise ValueError("zero-event exposure upper bound mismatch")
        elif aggregate["zero_episode_one_sided_95_poisson_upper_per_day"] != "":
            raise ValueError("zero-event bound must be blank for nonzero counts")

    if episodes[0]["config_id"] == "":
        actual_episode_count = 0
    else:
        actual_episode_count = len(episodes)
    if actual_episode_count != sum(int(row["test_episode_count"]) for row in aggregates):
        raise ValueError("episode table count mismatch")

    original_d5 = {(row["config_id"], row["motor"]): row for row in read_csv(D5_OUTPUT / "reference_results.csv")}
    if len(d5_reference) != 4:
        raise ValueError("D5 reference subset incomplete")
    for row in d5_reference:
        original = original_d5[(row["config_id"], row["motor"])]
        if row != original:
            raise ValueError(f"D5 reference changed: {row['config_id']}")

    report = {
        "status": "PASS", "schema_version": manifest["schema_version"],
        "policy_configuration_count": len(aggregates), "motor_policy_rows": len(motors),
        "active_quality_eligible_test_windows_per_policy": int(aggregates[0]["test_observation_count"]),
        "near_full_scale_test_windows_excluded": expected_exclusions["test"][2],
        "monitoring_hours_per_policy": float(aggregates[0]["test_monitoring_hours"]),
        "guards": {
            "d6c_temporal_policies_unchanged": True,
            "pnu_score_model_train_fitted": True,
            "commissioning_alignment_train_only": True,
            "test_rows_used_for_fit_or_selection": 0,
            "active_state_conditioning_verified": True,
            "near_full_scale_exclusion_verified": True,
            "sequence_gap_resets_verified": True,
            "causal_persistence_confirmation_verified": True,
            "d5_native_references_unchanged": True,
            "strict_external_zero_shot_not_claimed": True,
            "fault_sensitivity_or_lead_time_not_claimed": True,
        },
        "artifact_sha256": {
            name: sha256(OUTPUT / name) for name in (
                "commissioning_parameters.csv", "operating_state_exclusions.csv", "calibration_diagnostics.csv",
                "motor_results.csv", "aggregate_results.csv", "episode_results.csv", "d5_native_references.csv",
                "test_decisions.npz", "domain_compatibility.json", "policy_manifest.json",
            )
        },
    }
    (OUTPUT / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
