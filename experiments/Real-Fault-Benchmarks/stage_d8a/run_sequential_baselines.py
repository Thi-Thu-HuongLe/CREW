#!/usr/bin/env python3
"""Stage D8A: compare CREW with classical causal sequential baselines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
D6B = ROOT / "stage_d6b" / "outputs"
D6C = ROOT / "stage_d6c" / "outputs"
D6D = ROOT / "stage_d6d" / "outputs"
D6E = ROOT / "stage_d6e" / "outputs"
METHODS = ("crew_ewma", "robust_shewhart", "robust_page_cusum", "robust_causal_ma")
EXPERIMENTAL = METHODS[1:]
CUSUM_ALLOWANCES = (0.25, 0.50, 0.75, 1.00)
MA_WINDOWS = (3, 5, 10, 20)
QUANTILES = (0.990, 0.995, 0.999, 1.000)
PERSISTENCE = (3, 5, 10)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def correlation_with_time(values: np.ndarray) -> float:
    if len(values) < 3 or float(np.std(values)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(np.arange(len(values), dtype=float), values)[0, 1])


def first_persistent(values: np.ndarray, persistence: int, start: int = 0, end: int | None = None) -> int | None:
    end = len(values) if end is None else min(end, len(values))
    count = 0
    for index in range(max(0, start), end):
        count = count + 1 if bool(values[index]) else 0
        if count >= persistence:
            return index - persistence + 1
    return None


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[object, object, object]:
    if total == 0:
        return "", "", ""
    estimate = successes / total
    denominator = 1 + z * z / total
    center = (estimate + z * z / (2 * total)) / denominator
    half = z * math.sqrt(estimate * (1 - estimate) / total + z * z / (4 * total * total)) / denominator
    return estimate, max(0.0, center - half), min(1.0, center + half)


def exact_mcnemar_two_sided(primary_only: int, comparator_only: int) -> object:
    discordant = primary_only + comparator_only
    if discordant == 0:
        return ""
    tail = sum(math.comb(discordant, value) for value in range(min(primary_only, comparator_only) + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def page_cusum(z: np.ndarray, allowance: float) -> np.ndarray:
    result = np.empty_like(z, dtype=np.float64)
    previous = 0.0
    for index, value in enumerate(z):
        previous = max(0.0, previous + float(value) - allowance)
        result[index] = previous
    return result


def causal_moving_average(z: np.ndarray, window: int) -> np.ndarray:
    positive = np.maximum(z, 0.0)
    prefix = np.concatenate(([0.0], np.cumsum(positive, dtype=np.float64)))
    result = np.empty_like(positive, dtype=np.float64)
    for index in range(len(positive)):
        start = max(0, index + 1 - window)
        result[index] = (prefix[index + 1] - prefix[start]) / (index + 1 - start)
    return result


def candidate_score(health_by_asset: list[tuple[np.ndarray, int]]) -> dict[str, float]:
    correlations: list[float] = []
    gains: list[float] = []
    reference_levels: list[float] = []
    for health, commissioning in health_by_asset:
        tail = max(commissioning, int(math.ceil(0.10 * len(health))))
        correlations.append(correlation_with_time(health[commissioning:]))
        gains.append(float(np.median(health[-tail:]) - np.median(health[:commissioning])))
        reference_levels.append(float(np.quantile(health[: max(commissioning, int(math.ceil(0.20 * len(health))))], 0.99)))
    median_corr = float(np.median(correlations))
    median_gain = float(np.median(gains))
    median_reference = float(np.median(reference_levels))
    score = median_corr + 0.10 * math.tanh(median_gain / 10.0) - 0.02 * math.tanh(median_reference / 10.0)
    return {
        "median_time_correlation": median_corr,
        "median_terminal_gain": median_gain,
        "median_reference_q99": median_reference,
        "selection_score": score,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "outputs")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    metadata = read_csv(D6B / "feature_metadata.csv")
    robust_distance = np.load(D6B / "robust_distance.npy", mmap_mode="r")
    group_rows = read_csv(D6C / "group_parameters.csv")
    crew_rows = read_csv(D6C / "trajectory_results.csv")
    crew_health = np.load(D6C / "health_indicator.npy", mmap_mode="r")
    d6d_rows = read_csv(D6D / "baseline_trajectory_results.csv")
    d6e_summary = read_csv(D6E / "method_summary.csv")

    trajectory_lookup = {row["asset_id"]: row for row in crew_rows}
    asset_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        if row["early_warning_eligible"] == "true":
            asset_indices[row["asset_id"]].append(index)
    for indices in asset_indices.values():
        indices.sort(key=lambda index: int(metadata[index]["sequence_index"]))

    group_parameters = {
        (row["dataset"], row["condition_id"]): {
            "commissioning": int(row["commissioning_count"]),
            "scale": float(row["pooled_log_scale"]),
        }
        for row in group_rows
    }
    runtime: dict[str, dict[str, object]] = {}
    for asset, indices in asset_indices.items():
        first = metadata[indices[0]]
        params = group_parameters[(first["dataset"], first["condition_id"])]
        commissioning = min(int(params["commissioning"]), len(indices))
        log_score = np.log1p(np.asarray(robust_distance[indices], dtype=np.float64))
        center = float(np.median(log_score[:commissioning]))
        z = np.clip((log_score - center) / float(params["scale"]), -10.0, 50.0)
        runtime[asset] = {
            "indices": indices,
            "dataset": first["dataset"],
            "condition_id": first["condition_id"],
            "split": first["split"],
            "run_id": first["run_id"],
            "commissioning": commissioning,
            "z": z,
        }

    datasets = sorted({row["dataset"] for row in crew_rows})
    parameter_rows: list[dict[str, object]] = []
    selected_parameter: dict[tuple[str, str], float] = {}
    for dataset in datasets:
        train_failed = [
            asset for asset, data in runtime.items()
            if data["dataset"] == dataset
            and data["split"] == "train"
            and trajectory_lookup[asset]["outcome_status"] == "known_failed"
        ]
        for allowance in CUSUM_ALLOWANCES:
            items = [(page_cusum(np.asarray(runtime[a]["z"]), allowance), int(runtime[a]["commissioning"])) for a in train_failed]
            metrics = candidate_score(items)
            parameter_rows.append({
                "method": "robust_page_cusum", "dataset": dataset, "parameter_name": "allowance_k",
                "parameter_value": allowance, "known_failed_train_assets": len(train_failed), **metrics,
            })
        candidates = [row for row in parameter_rows if row["method"] == "robust_page_cusum" and row["dataset"] == dataset]
        best = max(candidates, key=lambda row: (float(row["selection_score"]), float(row["parameter_value"])))
        selected_parameter[("robust_page_cusum", dataset)] = float(best["parameter_value"])
        for row in candidates:
            row["selected"] = str(row is best).lower()

        for window in MA_WINDOWS:
            items = [(causal_moving_average(np.asarray(runtime[a]["z"]), window), int(runtime[a]["commissioning"])) for a in train_failed]
            metrics = candidate_score(items)
            parameter_rows.append({
                "method": "robust_causal_ma", "dataset": dataset, "parameter_name": "window",
                "parameter_value": window, "known_failed_train_assets": len(train_failed), **metrics,
            })
        candidates = [row for row in parameter_rows if row["method"] == "robust_causal_ma" and row["dataset"] == dataset]
        best = max(candidates, key=lambda row: (float(row["selection_score"]), -float(row["parameter_value"])))
        selected_parameter[("robust_causal_ma", dataset)] = float(best["parameter_value"])
        for row in candidates:
            row["selected"] = str(row is best).lower()

    health: dict[tuple[str, str], np.ndarray] = {}
    for asset, data in runtime.items():
        dataset = str(data["dataset"])
        z = np.asarray(data["z"], dtype=np.float64)
        health[("robust_shewhart", asset)] = np.maximum(z, 0.0)
        health[("robust_page_cusum", asset)] = page_cusum(z, selected_parameter[("robust_page_cusum", dataset)])
        health[("robust_causal_ma", asset)] = causal_moving_average(z, int(selected_parameter[("robust_causal_ma", dataset)]))

    rule_search_rows: list[dict[str, object]] = []
    selected_rules: dict[tuple[str, str], dict[str, object]] = {}
    for method in EXPERIMENTAL:
        for dataset in datasets:
            calibration_assets = [a for a, data in runtime.items() if data["dataset"] == dataset and data["split"] == "calibration"]
            failed_assets = [a for a in calibration_assets if trajectory_lookup[a]["outcome_status"] == "known_failed"]
            reference_ends: dict[str, int] = {}
            reference_values: list[float] = []
            for asset in calibration_assets:
                values = health[(method, asset)]
                commissioning = int(runtime[asset]["commissioning"])
                status = trajectory_lookup[asset]["outcome_status"]
                end = len(values) if status == "known_control" else max(commissioning, int(math.ceil(0.20 * len(values))))
                reference_ends[asset] = end
                reference_values.extend(values[:end].tolist())
            reference = np.asarray(reference_values, dtype=np.float64)
            candidates: list[dict[str, object]] = []
            for quantile in QUANTILES:
                threshold = float(np.quantile(reference, quantile))
                if quantile == 1.0:
                    threshold = float(np.nextafter(threshold, np.inf))
                for persistence in PERSISTENCE:
                    false_assets = 0
                    for asset in calibration_assets:
                        values = health[(method, asset)]
                        commissioning = int(runtime[asset]["commissioning"])
                        evidence = first_persistent(values > threshold, persistence, commissioning, min(len(values), reference_ends[asset] + persistence - 1))
                        false_assets += int(evidence is not None and evidence < reference_ends[asset])
                    detected, leads = 0, []
                    for asset in failed_assets:
                        values = health[(method, asset)]
                        start = max(int(runtime[asset]["commissioning"]), reference_ends[asset])
                        evidence = first_persistent(values > threshold, persistence, start)
                        if evidence is not None:
                            detected += 1
                            confirmation = evidence + persistence - 1
                            leads.append((len(values) - 1 - confirmation) / max(len(values) - 1, 1))
                    candidates.append({
                        "method": method, "dataset": dataset, "quantile": quantile, "threshold": threshold,
                        "persistence": persistence, "calibration_asset_count": len(calibration_assets),
                        "calibration_failed_asset_count": len(failed_assets), "reference_false_asset_count": false_assets,
                        "failed_detected_count": detected, "failed_detection_rate": "" if not failed_assets else detected / len(failed_assets),
                        "median_lead_fraction": float(np.median(leads)) if leads else -1.0,
                    })
            best = min(candidates, key=lambda row: (
                int(row["reference_false_asset_count"]), -int(row["failed_detected_count"]),
                -float(row["median_lead_fraction"]), float(row["quantile"]), int(row["persistence"]),
            ))
            selected_rules[(method, dataset)] = dict(best)
            for row in candidates:
                row["selected"] = str(row is best).lower()
                rule_search_rows.append(row)

    trajectory_rows: list[dict[str, object]] = []
    for original in crew_rows:
        trajectory_rows.append({
            "method": "crew_ewma", "dataset": original["dataset"], "condition_id": original["condition_id"],
            "split": original["split"], "run_id": original["run_id"], "asset_id": original["asset_id"],
            "outcome_status": original["outcome_status"], "observation_count": original["observation_count"],
            "commissioning_count": original["commissioning_count"], "selected_parameter": original["ewma_alpha"],
            "threshold": original["threshold"], "persistence": original["persistence"],
            "detected": original["detected"], "evidence_start_zero_based": original["evidence_start_zero_based"],
            "warning_confirmation_zero_based": original["warning_confirmation_zero_based"],
            "warning_confirmation_life_fraction": original["warning_confirmation_life_fraction"],
            "lead_seconds_to_terminal": original["lead_seconds_to_terminal"],
            "calibration_reference_false_event": "false" if original["split"] == "calibration" else "",
        })

    for method in EXPERIMENTAL:
        for asset in sorted(asset_indices):
            data = runtime[asset]
            original = trajectory_lookup[asset]
            values = health[(method, asset)]
            rule = selected_rules[(method, str(data["dataset"]))]
            above = values > float(rule["threshold"])
            evidence = first_persistent(above, int(rule["persistence"]), int(data["commissioning"]))
            confirmation = None if evidence is None else evidence + int(rule["persistence"]) - 1
            lead_seconds: object = ""
            life_fraction: object = ""
            indices = data["indices"]
            if confirmation is not None:
                life_fraction = confirmation / max(len(indices) - 1, 1)
                if original["outcome_status"] == "known_failed":
                    lead_seconds = float(metadata[indices[-1]]["elapsed_seconds"]) - float(metadata[indices[confirmation]]["elapsed_seconds"])
            ref_false: object = ""
            if data["split"] == "calibration":
                ref_end = len(values) if original["outcome_status"] == "known_control" else max(int(data["commissioning"]), int(math.ceil(0.20 * len(values))))
                ref_evidence = first_persistent(above, int(rule["persistence"]), int(data["commissioning"]), min(len(values), ref_end + int(rule["persistence"]) - 1))
                ref_false = str(ref_evidence is not None and ref_evidence < ref_end).lower()
            parameter = ""
            if method == "robust_page_cusum":
                parameter = selected_parameter[(method, str(data["dataset"]))]
            elif method == "robust_causal_ma":
                parameter = int(selected_parameter[(method, str(data["dataset"]))])
            trajectory_rows.append({
                "method": method, "dataset": data["dataset"], "condition_id": data["condition_id"],
                "split": data["split"], "run_id": data["run_id"], "asset_id": asset,
                "outcome_status": original["outcome_status"], "observation_count": len(indices),
                "commissioning_count": data["commissioning"], "selected_parameter": parameter,
                "threshold": rule["threshold"], "persistence": rule["persistence"],
                "detected": str(evidence is not None).lower(),
                "evidence_start_zero_based": "" if evidence is None else evidence,
                "warning_confirmation_zero_based": "" if confirmation is None else confirmation,
                "warning_confirmation_life_fraction": life_fraction, "lead_seconds_to_terminal": lead_seconds,
                "calibration_reference_false_event": ref_false,
            })

    summary_rows: list[dict[str, object]] = []
    for method in METHODS:
        for dataset in datasets:
            rows = [row for row in trajectory_rows if row["method"] == method and row["dataset"] == dataset]
            calibration = [row for row in rows if row["split"] == "calibration"]
            failed = [row for row in rows if row["split"] == "test" and row["outcome_status"] == "known_failed"]
            detected = [row for row in failed if row["detected"] == "true"]
            unknown = [row for row in rows if row["split"] == "test" and row["outcome_status"] == "unknown"]
            estimate, lower, upper = wilson(len(detected), len(failed))
            leads = [float(row["lead_seconds_to_terminal"]) for row in detected]
            summary_rows.append({
                "method": method, "category": "proposed" if method == "crew_ewma" else "classical_sequential",
                "dataset": dataset, "selected_parameter": rows[0]["selected_parameter"] if rows else "",
                "threshold": rows[0]["threshold"] if rows else "", "persistence": rows[0]["persistence"] if rows else "",
                "calibration_reference_false_assets": sum(row["calibration_reference_false_event"] == "true" for row in calibration),
                "calibration_reference_assets": len(calibration), "test_known_failed_detected": len(detected),
                "test_known_failed_total": len(failed), "test_detection_rate": estimate,
                "wilson_95_lower": lower, "wilson_95_upper": upper,
                "test_lead_seconds_min": "" if not leads else float(np.min(leads)),
                "test_lead_seconds_median": "" if not leads else float(np.median(leads)),
                "test_lead_seconds_max": "" if not leads else float(np.max(leads)),
                "test_unknown_flagged": sum(row["detected"] == "true" for row in unknown), "test_unknown_total": len(unknown),
            })

    primary = {row["asset_id"]: row for row in trajectory_rows if row["method"] == "crew_ewma" and row["split"] == "test" and row["outcome_status"] == "known_failed"}
    paired_rows: list[dict[str, object]] = []
    for method in EXPERIMENTAL:
        comparator = {row["asset_id"]: row for row in trajectory_rows if row["method"] == method and row["split"] == "test" and row["outcome_status"] == "known_failed"}
        both = primary_only = comparator_only = neither = crew_earlier = comparator_earlier = equal = 0
        for asset, p_row in primary.items():
            c_row = comparator[asset]
            p_det, c_det = p_row["detected"] == "true", c_row["detected"] == "true"
            if p_det and c_det:
                both += 1
                p_lead, c_lead = float(p_row["lead_seconds_to_terminal"]), float(c_row["lead_seconds_to_terminal"])
                crew_earlier += int(p_lead > c_lead)
                comparator_earlier += int(c_lead > p_lead)
                equal += int(c_lead == p_lead)
            elif p_det:
                primary_only += 1
            elif c_det:
                comparator_only += 1
            else:
                neither += 1
        paired_rows.append({
            "primary_method": "crew_ewma", "comparator_method": method, "known_failed_test_assets": len(primary),
            "both_detected": both, "primary_only_detected": primary_only, "comparator_only_detected": comparator_only,
            "neither_detected": neither, "exact_mcnemar_two_sided_p": exact_mcnemar_two_sided(primary_only, comparator_only),
            "crew_earlier_among_both": crew_earlier, "comparator_earlier_among_both": comparator_earlier,
            "equal_warning_time_among_both": equal, "inferential_role": "descriptive_small_n_unadjusted",
        })

    integrated: list[dict[str, object]] = []
    for row in d6e_summary:
        method = "crew_ewma" if row["method"] == "d6c_robust_ewma" else row["method"]
        integrated.append({
            "method": method, "category": "proposed" if method == "crew_ewma" else row["category"], "dataset": row["dataset"],
            "calibration_reference_false_assets": row["calibration_reference_false_assets"],
            "calibration_reference_assets": row["calibration_reference_assets"],
            "test_known_failed_detected": row["test_known_failed_detected"], "test_known_failed_total": row["test_known_failed_total"],
            "test_detection_rate": row["test_detection_rate"], "test_lead_seconds_median": row["test_lead_seconds_median"],
            "test_unknown_flagged": row["test_unknown_flagged"], "test_unknown_total": row["test_unknown_total"],
            "source_stage": "D6E",
        })
    for row in read_csv(D6D / "baseline_comparison.csv"):
        if row["method"] != "raw_d6b_q99_p3":
            continue
        integrated.append({
            "method": "raw_distance_q99_p3", "category": "classical_unsmoothed", "dataset": row["dataset"],
            "calibration_reference_false_assets": row["calibration_reference_false_assets"],
            "calibration_reference_assets": row["calibration_reference_assets"],
            "test_known_failed_detected": row["test_known_failed_detected"], "test_known_failed_total": row["test_known_failed_total"],
            "test_detection_rate": row["test_detection_rate"], "test_lead_seconds_median": row["test_lead_seconds_median"],
            "test_unknown_flagged": row["test_unknown_flagged"], "test_unknown_total": row["test_unknown_total"],
            "source_stage": "D6D",
        })
    for row in summary_rows:
        if row["method"] == "crew_ewma":
            continue
        integrated.append({
            "method": row["method"], "category": row["category"], "dataset": row["dataset"],
            "calibration_reference_false_assets": row["calibration_reference_false_assets"],
            "calibration_reference_assets": row["calibration_reference_assets"],
            "test_known_failed_detected": row["test_known_failed_detected"], "test_known_failed_total": row["test_known_failed_total"],
            "test_detection_rate": row["test_detection_rate"], "test_lead_seconds_median": row["test_lead_seconds_median"],
            "test_unknown_flagged": row["test_unknown_flagged"], "test_unknown_total": row["test_unknown_total"],
            "source_stage": "D8A",
        })

    write_csv(output / "parameter_selection.csv", parameter_rows)
    write_csv(output / "rule_search.csv", rule_search_rows)
    write_csv(output / "trajectory_results.csv", trajectory_rows)
    write_csv(output / "sequential_method_summary.csv", summary_rows)
    write_csv(output / "paired_comparison.csv", paired_rows)
    write_csv(output / "integrated_method_summary.csv", integrated)
    np.savez_compressed(
        output / "sequential_health_indicators.npz",
        **{f"{method}__{asset}".replace(":", "_"): values.astype(np.float32) for (method, asset), values in health.items()},
    )
    audit = {
        "schema_version": "D8A-1.0", "primary_method": "CREW (Causal Robust EWMA Warning)",
        "candidate_parameter_fit_split": "train known-failed trajectories only",
        "threshold_and_persistence_fit_split": "calibration reference portions only",
        "test_rows_used_for_fit_or_selection": 0, "test_assets_used_for_fit_or_selection": [],
        "comparison_role": "classical sequential decision-layer baselines on the same robust source score",
        "input_sha256": {
            "d6b_robust_distance.npy": sha256(D6B / "robust_distance.npy"),
            "d6c_trajectory_results.csv": sha256(D6C / "trajectory_results.csv"),
            "d6e_method_summary.csv": sha256(D6E / "method_summary.csv"),
        },
        "claim_boundary": "Warning is measured before documented terminal recording, not annotated physical fault onset.",
    }
    (output / "fit_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "BUILT", "methods": list(METHODS), "trajectory_rows": len(trajectory_rows),
        "integrated_methods": sorted({row["method"] for row in integrated}), "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
