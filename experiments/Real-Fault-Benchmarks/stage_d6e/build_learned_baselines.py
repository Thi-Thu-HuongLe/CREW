#!/usr/bin/env python3
"""Fit leakage-safe learned baselines and feature ablations for Stage D6E."""

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
D6B_OUTPUT = SCRIPT_DIR.parent / "stage_d6b" / "outputs"
D6C_OUTPUT = SCRIPT_DIR.parent / "stage_d6c" / "outputs"
METHODS = (
    "d6c_robust_ewma",
    "pca_spe",
    "shrinkage_mahalanobis",
    "knn5_novelty",
    "time_feature_ablation",
    "frequency_feature_ablation",
)
EXPERIMENTAL_METHODS = METHODS[1:]
ALPHAS = (0.05, 0.10, 0.20, 0.30, 0.50)
QUANTILES = (0.990, 0.995, 0.999, 1.000)
PERSISTENCE = (3, 5, 10)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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


def robust_scale(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    center = float(np.median(values))
    mad = 1.4826 * float(np.median(np.abs(values - center)))
    q25, q75 = np.quantile(values, [0.25, 0.75])
    return max(mad, float((q75 - q25) / 1.349), 1e-3)


def causal_health(log_score: np.ndarray, center: float, scale: float, alpha: float) -> np.ndarray:
    z = np.clip((log_score - center) / scale, -10.0, 50.0)
    result = np.empty_like(z, dtype=np.float64)
    previous = 0.0
    for index, value in enumerate(z):
        previous = alpha * float(value) + (1.0 - alpha) * previous
        result[index] = max(0.0, previous)
    return result


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
    tail = sum(math.comb(discordant, value) for value in range(0, min(primary_only, comparator_only) + 1)) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def score_pca(train: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    mean = np.mean(train, axis=0)
    centered = train - mean
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    variance = singular * singular
    cumulative = np.cumsum(variance) / max(float(np.sum(variance)), 1e-12)
    components = int(np.searchsorted(cumulative, 0.90) + 1)
    components = max(1, min(components, vt.shape[0], query.shape[1] - 1))
    basis = vt[:components]
    q_centered = query - mean
    residual = q_centered - (q_centered @ basis.T) @ basis
    score = np.sqrt(np.mean(residual * residual, axis=1))
    return score, {
        "variance_target": 0.90,
        "component_count": components,
        "retained_train_variance": float(cumulative[components - 1]),
    }


def score_mahalanobis(train: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    mean = np.mean(train, axis=0)
    centered = train - mean
    covariance = centered.T @ centered / max(len(train) - 1, 1)
    shrinkage = 0.10
    diagonal = np.diag(np.diag(covariance))
    regularized = (1.0 - shrinkage) * covariance + shrinkage * diagonal
    floor = max(1e-6, 1e-6 * float(np.median(np.diag(regularized))))
    regularized += np.eye(regularized.shape[0]) * floor
    inverse = np.linalg.pinv(regularized, rcond=1e-10)
    q_centered = query - mean
    squared = np.einsum("ij,jk,ik->i", q_centered, inverse, q_centered, optimize=True)
    score = np.sqrt(np.maximum(squared, 0.0) / query.shape[1])
    condition = float(np.linalg.cond(regularized))
    return score, {"shrinkage": shrinkage, "ridge_floor": floor, "regularized_condition_number": condition}


def score_knn(train: np.ndarray, query: np.ndarray, train_ids: np.ndarray, query_ids: np.ndarray, k: int = 5) -> tuple[np.ndarray, dict[str, object]]:
    effective_k = min(k, max(1, len(train) - 1))
    train_norm = np.sum(train * train, axis=1)
    train_position = {int(row_id): position for position, row_id in enumerate(train_ids)}
    result = np.empty(len(query), dtype=np.float64)
    chunk_size = 512
    for start in range(0, len(query), chunk_size):
        stop = min(start + chunk_size, len(query))
        chunk = query[start:stop]
        squared = np.maximum(np.sum(chunk * chunk, axis=1)[:, None] + train_norm[None, :] - 2.0 * chunk @ train.T, 0.0)
        for local, row_id in enumerate(query_ids[start:stop]):
            position = train_position.get(int(row_id))
            if position is not None:
                squared[local, position] = np.inf
        nearest = np.partition(squared, effective_k - 1, axis=1)[:, :effective_k]
        result[start:stop] = np.mean(np.sqrt(nearest / query.shape[1]), axis=1)
    return result, {"k": effective_k, "distance": "mean Euclidean RMS to k neighbors", "train_self_policy": "leave_one_out"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d6b-output", type=Path, default=D6B_OUTPUT)
    parser.add_argument("--d6c-output", type=Path, default=D6C_OUTPUT)
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "outputs")
    args = parser.parse_args()
    d6b, d6c, output = args.d6b_output.resolve(), args.d6c_output.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    metadata = read_csv(d6b / "feature_metadata.csv")
    baseline_rows = read_csv(d6b / "baseline_rows.csv")
    standardized = np.load(d6b / "standardized_features.npy", mmap_mode="r")
    robust_distance = np.load(d6b / "robust_distance.npy", mmap_mode="r")
    schema = json.loads((d6b / "feature_schema.json").read_text(encoding="utf-8"))
    preprocessing = json.loads((d6b / "preprocessing_manifest.json").read_text(encoding="utf-8"))
    d6c_locked = json.loads((d6c / "locked_rules.json").read_text(encoding="utf-8"))
    d6c_group_rows = read_csv(d6c / "group_parameters.csv")
    d6c_trajectories = read_csv(d6c / "trajectory_results.csv")
    d6c_health = np.load(d6c / "health_indicator.npy", mmap_mode="r")
    d6c_point = np.load(d6c / "point_alarm.npy", mmap_mode="r")
    d6c_warning = np.load(d6c / "latched_warning.npy", mmap_mode="r")

    if standardized.shape != (len(metadata), len(schema["feature_names"])):
        raise ValueError("D6B feature/metadata alignment failed")
    trajectory_lookup = {row["asset_id"]: row for row in d6c_trajectories}
    asset_indices: dict[str, list[int]] = defaultdict(list)
    group_indices: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        if row["early_warning_eligible"] == "true":
            asset_indices[row["asset_id"]].append(index)
            group_indices[(row["dataset"], row["condition_id"])].append(index)
    for asset, indices in asset_indices.items():
        indices.sort(key=lambda item: int(metadata[item]["sequence_index"]))
    train_baseline_ids: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in baseline_rows:
        if row["role"] == "fit_normalization":
            if row["split"] != "train":
                raise ValueError("non-train row entered model fitting baseline")
            train_baseline_ids[(row["dataset"], row["condition_id"])].append(int(row["feature_row_id"]))

    registry_by_group = {(item["dataset"], item["condition_id"]): item for item in preprocessing["group_registry"]}
    feature_names = list(schema["feature_names"])
    method_index = {method: index for index, method in enumerate(METHODS)}
    source_scores = np.full((len(METHODS), len(metadata)), np.nan, dtype=np.float32)
    source_scores[0] = np.asarray(robust_distance, dtype=np.float32)
    model_registry: list[dict[str, object]] = []

    for key in sorted(group_indices):
        dataset, condition = key
        query_ids = np.asarray(group_indices[key], dtype=int)
        train_ids = np.asarray(train_baseline_ids[key], dtype=int)
        base_features = [int(value) for value in registry_by_group[key]["feature_indices"]]
        time_features = [index for index in base_features if "frequency" not in feature_names[index] and "band_" not in feature_names[index]]
        frequency_features = [index for index in base_features if "frequency" in feature_names[index] or "band_" in feature_names[index]]
        all_train = np.asarray(standardized[train_ids][:, base_features], dtype=np.float64)
        all_query = np.asarray(standardized[query_ids][:, base_features], dtype=np.float64)
        if not np.isfinite(all_train).all() or not np.isfinite(all_query).all():
            raise ValueError(f"non-finite compatible features in {key}")

        pca_score, pca_parameters = score_pca(all_train, all_query)
        mahal_score, mahal_parameters = score_mahalanobis(all_train, all_query)
        knn_score, knn_parameters = score_knn(all_train, all_query, train_ids, query_ids)
        time_query = np.asarray(standardized[query_ids][:, time_features], dtype=np.float64)
        frequency_query = np.asarray(standardized[query_ids][:, frequency_features], dtype=np.float64)
        time_score = np.sqrt(np.mean(time_query * time_query, axis=1))
        frequency_score = np.sqrt(np.mean(frequency_query * frequency_query, axis=1))
        definitions = (
            ("pca_spe", pca_score, base_features, pca_parameters, "learned_one_class"),
            ("shrinkage_mahalanobis", mahal_score, base_features, mahal_parameters, "learned_one_class"),
            ("knn5_novelty", knn_score, base_features, knn_parameters, "learned_one_class"),
            ("time_feature_ablation", time_score, time_features, {"aggregation": "RMS robust z-score"}, "feature_ablation"),
            ("frequency_feature_ablation", frequency_score, frequency_features, {"aggregation": "RMS robust z-score"}, "feature_ablation"),
        )
        for method, values, selected_features, parameters, category in definitions:
            source_scores[method_index[method], query_ids] = values.astype(np.float32)
            model_registry.append({
                "method": method,
                "category": category,
                "dataset": dataset,
                "condition_id": condition,
                "fit_split": "train_only",
                "train_early_life_row_count": len(train_ids),
                "query_row_count": len(query_ids),
                "feature_count": len(selected_features),
                "feature_indices": selected_features,
                "feature_names": [feature_names[index] for index in selected_features],
                "parameters": parameters,
            })

    group_parameters = {
        (row["dataset"], row["condition_id"]): int(row["commissioning_count"])
        for row in d6c_group_rows
    }
    experimental_health = np.full_like(source_scores, np.nan, dtype=np.float32)
    experimental_health[0] = np.asarray(d6c_health, dtype=np.float32)
    point_alarm = np.zeros_like(source_scores, dtype=bool)
    latched_warning = np.zeros_like(source_scores, dtype=bool)
    point_alarm[0] = np.asarray(d6c_point, dtype=bool)
    latched_warning[0] = np.asarray(d6c_warning, dtype=bool)

    method_group_scale: dict[tuple[str, str, str], float] = {}
    asset_runtime: dict[tuple[str, str], dict[str, object]] = {}
    scale_rows: list[dict[str, object]] = []
    for method in EXPERIMENTAL_METHODS:
        mi = method_index[method]
        for key in sorted(group_indices):
            dataset, condition = key
            count = group_parameters[key]
            train_assets = [
                asset for asset, indices in asset_indices.items()
                if metadata[indices[0]]["dataset"] == dataset
                and metadata[indices[0]]["condition_id"] == condition
                and metadata[indices[0]]["split"] == "train"
            ]
            residuals: list[float] = []
            for asset in train_assets:
                indices = asset_indices[asset]
                used = min(count, len(indices))
                values = np.log1p(np.asarray(source_scores[mi, indices], dtype=np.float64))
                center = float(np.median(values[:used]))
                residuals.extend((values[:used] - center).tolist())
            scale = robust_scale(np.asarray(residuals))
            method_group_scale[(method, dataset, condition)] = scale
            scale_rows.append({
                "method": method, "dataset": dataset, "condition_id": condition,
                "train_asset_count": len(train_assets), "commissioning_count": count,
                "pooled_log_scale": scale, "fit_split": "train_only",
            })
        for asset, indices in asset_indices.items():
            first = metadata[indices[0]]
            key = (first["dataset"], first["condition_id"])
            count = min(group_parameters[key], len(indices))
            values = np.log1p(np.asarray(source_scores[mi, indices], dtype=np.float64))
            center = float(np.median(values[:count]))
            asset_runtime[(method, asset)] = {
                "values": values, "center": center, "count": count,
                "scale": method_group_scale[(method, *key)],
            }

    alpha_rows: list[dict[str, object]] = []
    selected_alpha: dict[tuple[str, str], float] = {}
    datasets = sorted(d6c_locked["rules"])
    for method in EXPERIMENTAL_METHODS:
        for dataset in datasets:
            train_failed = [
                asset for asset, indices in asset_indices.items()
                if metadata[indices[0]]["dataset"] == dataset
                and metadata[indices[0]]["split"] == "train"
                and trajectory_lookup[asset]["outcome_status"] == "known_failed"
            ]
            candidates: list[dict[str, object]] = []
            for alpha in ALPHAS:
                correlations, gains = [], []
                for asset in train_failed:
                    runtime = asset_runtime[(method, asset)]
                    health = causal_health(runtime["values"], float(runtime["center"]), float(runtime["scale"]), alpha)
                    count = int(runtime["count"])
                    tail = max(count, int(math.ceil(0.10 * len(health))))
                    correlations.append(correlation_with_time(health[count:]))
                    gains.append(float(np.median(health[-tail:]) - np.median(health[:count])))
                median_corr = float(np.median(correlations))
                median_gain = float(np.median(gains))
                lag = (1.0 - alpha) / alpha
                selection_score = median_corr + 0.10 * math.tanh(median_gain / 5.0) - 0.002 * lag
                candidates.append({
                    "method": method, "dataset": dataset, "alpha": alpha,
                    "known_failed_train_assets": len(train_failed),
                    "median_time_correlation": median_corr, "median_terminal_gain": median_gain,
                    "effective_lag_observations": lag, "selection_score": selection_score,
                })
            best = max(candidates, key=lambda row: (float(row["selection_score"]), float(row["alpha"])))
            selected_alpha[(method, dataset)] = float(best["alpha"])
            for row in candidates:
                row["selected"] = str(row is best).lower()
                alpha_rows.append(row)

    for method in EXPERIMENTAL_METHODS:
        mi = method_index[method]
        for asset, indices in asset_indices.items():
            runtime = asset_runtime[(method, asset)]
            dataset = metadata[indices[0]]["dataset"]
            alpha = selected_alpha[(method, dataset)]
            health = causal_health(runtime["values"], float(runtime["center"]), float(runtime["scale"]), alpha)
            runtime["health"] = health
            runtime["alpha"] = alpha
            experimental_health[mi, indices] = health.astype(np.float32)

    rule_search_rows: list[dict[str, object]] = []
    selected_rules: dict[tuple[str, str], dict[str, object]] = {}
    for method in EXPERIMENTAL_METHODS:
        for dataset in datasets:
            calibration_assets = [
                asset for asset, indices in asset_indices.items()
                if metadata[indices[0]]["dataset"] == dataset and metadata[indices[0]]["split"] == "calibration"
            ]
            failed_assets = [asset for asset in calibration_assets if trajectory_lookup[asset]["outcome_status"] == "known_failed"]
            reference_ends: dict[str, int] = {}
            reference_values: list[float] = []
            for asset in calibration_assets:
                runtime = asset_runtime[(method, asset)]
                health = np.asarray(runtime["health"], dtype=np.float64)
                count = int(runtime["count"])
                status = trajectory_lookup[asset]["outcome_status"]
                end = len(health) if status == "known_control" else max(count, int(math.ceil(0.20 * len(health))))
                reference_ends[asset] = end
                reference_values.extend(health[:end].tolist())
            reference = np.asarray(reference_values, dtype=np.float64)
            candidates: list[dict[str, object]] = []
            for quantile in QUANTILES:
                threshold = float(np.quantile(reference, quantile))
                if quantile == 1.0:
                    threshold = float(np.nextafter(threshold, np.inf))
                for persistence in PERSISTENCE:
                    false_assets = 0
                    for asset in calibration_assets:
                        runtime = asset_runtime[(method, asset)]
                        health = np.asarray(runtime["health"])
                        count = int(runtime["count"])
                        evidence = first_persistent(health > threshold, persistence, count, min(len(health), reference_ends[asset] + persistence - 1))
                        false_assets += int(evidence is not None and evidence < reference_ends[asset])
                    detected, leads = 0, []
                    for asset in failed_assets:
                        runtime = asset_runtime[(method, asset)]
                        health = np.asarray(runtime["health"])
                        evidence = first_persistent(health > threshold, persistence, max(int(runtime["count"]), reference_ends[asset]))
                        if evidence is not None:
                            detected += 1
                            confirmation = evidence + persistence - 1
                            leads.append((len(health) - 1 - confirmation) / max(len(health) - 1, 1))
                    candidates.append({
                        "method": method, "dataset": dataset, "quantile": quantile, "threshold": threshold,
                        "persistence": persistence, "calibration_asset_count": len(calibration_assets),
                        "calibration_failed_asset_count": len(failed_assets),
                        "reference_false_asset_count": false_assets, "failed_detected_count": detected,
                        "failed_detection_rate": "" if not failed_assets else detected / len(failed_assets),
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
    # Preserve the exact serialized D6C decisions as the immutable reference.
    for original in d6c_trajectories:
        asset = original["asset_id"]
        indices = asset_indices[asset]
        ref_end = len(indices) if original["outcome_status"] == "known_control" else max(int(original["commissioning_count"]), int(math.ceil(0.20 * len(indices))))
        false_reference: object = ""
        if original["split"] == "calibration":
            evidence = first_persistent(np.asarray(d6c_point[indices], dtype=bool), int(original["persistence"]), int(original["commissioning_count"]), min(len(indices), ref_end + int(original["persistence"]) - 1))
            false_reference = str(evidence is not None and evidence < ref_end).lower()
        trajectory_rows.append({
            "method": "d6c_robust_ewma", "dataset": original["dataset"], "condition_id": original["condition_id"],
            "split": original["split"], "run_id": original["run_id"], "asset_id": asset,
            "outcome_status": original["outcome_status"], "observation_count": original["observation_count"],
            "commissioning_count": original["commissioning_count"], "alpha": original["ewma_alpha"],
            "threshold": original["threshold"], "persistence": original["persistence"],
            "detected": original["detected"], "evidence_start_zero_based": original["evidence_start_zero_based"],
            "warning_confirmation_zero_based": original["warning_confirmation_zero_based"],
            "warning_confirmation_life_fraction": original["warning_confirmation_life_fraction"],
            "lead_seconds_to_terminal": original["lead_seconds_to_terminal"],
            "calibration_reference_false_event": false_reference,
        })

    for method in EXPERIMENTAL_METHODS:
        mi = method_index[method]
        for asset in sorted(asset_indices):
            indices = asset_indices[asset]
            first = metadata[indices[0]]
            original = trajectory_lookup[asset]
            runtime = asset_runtime[(method, asset)]
            rule = selected_rules[(method, first["dataset"])]
            health = np.asarray(runtime["health"], dtype=np.float64)
            above = health > float(rule["threshold"])
            point_alarm[mi, indices] = above
            evidence = first_persistent(above, int(rule["persistence"]), int(runtime["count"]))
            confirmation = None if evidence is None else evidence + int(rule["persistence"]) - 1
            if confirmation is not None:
                latched_warning[mi, np.asarray(indices[confirmation:], dtype=int)] = True
            terminal_elapsed = float(metadata[indices[-1]]["elapsed_seconds"])
            lead_seconds: object = ""
            life_fraction: object = ""
            if confirmation is not None:
                life_fraction = confirmation / max(len(indices) - 1, 1)
                if original["outcome_status"] == "known_failed":
                    lead_seconds = terminal_elapsed - float(metadata[indices[confirmation]]["elapsed_seconds"])
            ref_false: object = ""
            if first["split"] == "calibration":
                ref_end = len(indices) if original["outcome_status"] == "known_control" else max(int(runtime["count"]), int(math.ceil(0.20 * len(indices))))
                ref_evidence = first_persistent(above, int(rule["persistence"]), int(runtime["count"]), min(len(indices), ref_end + int(rule["persistence"]) - 1))
                ref_false = str(ref_evidence is not None and ref_evidence < ref_end).lower()
            trajectory_rows.append({
                "method": method, "dataset": first["dataset"], "condition_id": first["condition_id"],
                "split": first["split"], "run_id": first["run_id"], "asset_id": asset,
                "outcome_status": original["outcome_status"], "observation_count": len(indices),
                "commissioning_count": runtime["count"], "alpha": runtime["alpha"],
                "threshold": rule["threshold"], "persistence": rule["persistence"],
                "detected": str(evidence is not None).lower(),
                "evidence_start_zero_based": "" if evidence is None else evidence,
                "warning_confirmation_zero_based": "" if confirmation is None else confirmation,
                "warning_confirmation_life_fraction": life_fraction,
                "lead_seconds_to_terminal": lead_seconds,
                "calibration_reference_false_event": ref_false,
            })

    method_summary: list[dict[str, object]] = []
    for method in METHODS:
        category = "primary_reference" if method == "d6c_robust_ewma" else ("learned_one_class" if method in METHODS[1:4] else "feature_ablation")
        for dataset in datasets:
            rows = [row for row in trajectory_rows if row["method"] == method and row["dataset"] == dataset]
            calibration = [row for row in rows if row["split"] == "calibration"]
            test_failed = [row for row in rows if row["split"] == "test" and row["outcome_status"] == "known_failed"]
            detected = [row for row in test_failed if row["detected"] == "true"]
            test_unknown = [row for row in rows if row["split"] == "test" and row["outcome_status"] == "unknown"]
            estimate, lower, upper = wilson(len(detected), len(test_failed))
            leads = [float(row["lead_seconds_to_terminal"]) for row in detected]
            primary_rule = d6c_locked["rules"][dataset] if method == "d6c_robust_ewma" else selected_rules[(method, dataset)]
            alpha = float(primary_rule["alpha"]) if method == "d6c_robust_ewma" else selected_alpha[(method, dataset)]
            method_summary.append({
                "method": method, "category": category, "dataset": dataset,
                "alpha": alpha, "threshold": primary_rule["threshold"], "persistence": primary_rule["persistence"],
                "calibration_reference_false_assets": sum(row["calibration_reference_false_event"] == "true" for row in calibration),
                "calibration_reference_assets": len(calibration),
                "test_known_failed_detected": len(detected), "test_known_failed_total": len(test_failed),
                "test_detection_rate": estimate, "wilson_95_lower": lower, "wilson_95_upper": upper,
                "test_detected_before_20pct": sum(float(row["warning_confirmation_life_fraction"]) < 0.20 for row in detected),
                "test_lead_seconds_min": "" if not leads else float(np.min(leads)),
                "test_lead_seconds_median": "" if not leads else float(np.median(leads)),
                "test_lead_seconds_max": "" if not leads else float(np.max(leads)),
                "test_unknown_flagged": sum(row["detected"] == "true" for row in test_unknown),
                "test_unknown_total": len(test_unknown),
            })

    paired_rows: list[dict[str, object]] = []
    primary_by_asset = {
        row["asset_id"]: row for row in trajectory_rows
        if row["method"] == "d6c_robust_ewma" and row["split"] == "test" and row["outcome_status"] == "known_failed"
    }
    for method in EXPERIMENTAL_METHODS:
        comparator_by_asset = {
            row["asset_id"]: row for row in trajectory_rows
            if row["method"] == method and row["split"] == "test" and row["outcome_status"] == "known_failed"
        }
        primary_only = comparator_only = both = neither = 0
        primary_earlier = comparator_earlier = equal_time = 0
        for asset, primary in primary_by_asset.items():
            comparator = comparator_by_asset[asset]
            p_detected, c_detected = primary["detected"] == "true", comparator["detected"] == "true"
            if p_detected and c_detected:
                both += 1
                p_lead, c_lead = float(primary["lead_seconds_to_terminal"]), float(comparator["lead_seconds_to_terminal"])
                if p_lead > c_lead:
                    primary_earlier += 1
                elif c_lead > p_lead:
                    comparator_earlier += 1
                else:
                    equal_time += 1
            elif p_detected:
                primary_only += 1
            elif c_detected:
                comparator_only += 1
            else:
                neither += 1
        paired_rows.append({
            "primary_method": "d6c_robust_ewma", "comparator_method": method,
            "known_failed_test_assets": len(primary_by_asset), "both_detected": both,
            "primary_only_detected": primary_only, "comparator_only_detected": comparator_only,
            "neither_detected": neither,
            "exact_mcnemar_two_sided_p": exact_mcnemar_two_sided(primary_only, comparator_only),
            "primary_earlier_among_both": primary_earlier,
            "comparator_earlier_among_both": comparator_earlier,
            "equal_warning_time_among_both": equal_time,
            "inferential_role": "descriptive_small_n_unadjusted_for_multiple_comparisons",
        })

    selected_rule_rows: list[dict[str, object]] = []
    for dataset in datasets:
        rule = d6c_locked["rules"][dataset]
        selected_rule_rows.append({
            "method": "d6c_robust_ewma", "dataset": dataset, "alpha": rule["alpha"],
            "quantile": rule["threshold_quantile"], "threshold": rule["threshold"],
            "persistence": rule["persistence"], "alpha_fit_split": "train",
            "threshold_persistence_fit_split": "calibration", "test_rows_used": 0,
        })
    for method in EXPERIMENTAL_METHODS:
        for dataset in datasets:
            rule = selected_rules[(method, dataset)]
            selected_rule_rows.append({
                "method": method, "dataset": dataset, "alpha": selected_alpha[(method, dataset)],
                "quantile": rule["quantile"], "threshold": rule["threshold"],
                "persistence": rule["persistence"], "alpha_fit_split": "train",
                "threshold_persistence_fit_split": "calibration", "test_rows_used": 0,
            })

    np.save(output / "source_scores.npy", source_scores)
    np.save(output / "health_indicators.npy", experimental_health)
    np.save(output / "point_alarms.npy", point_alarm)
    np.save(output / "latched_warnings.npy", latched_warning)
    write_csv(output / "group_scales.csv", scale_rows)
    write_csv(output / "alpha_selection.csv", alpha_rows)
    write_csv(output / "rule_search.csv", rule_search_rows)
    write_csv(output / "selected_rules.csv", selected_rule_rows)
    write_csv(output / "trajectory_results.csv", trajectory_rows)
    write_csv(output / "method_summary.csv", method_summary)
    write_csv(output / "paired_comparison.csv", paired_rows)
    write_csv(output / "feature_ablation.csv", [row for row in method_summary if row["method"] in {"d6c_robust_ewma", "time_feature_ablation", "frequency_feature_ablation"}])
    (output / "method_names.json").write_text(json.dumps({"axis_0_order": list(METHODS)}, indent=2), encoding="utf-8")
    (output / "model_registry.json").write_text(json.dumps(model_registry, indent=2), encoding="utf-8")
    audit = {
        "schema_version": "D6E-1.0",
        "model_fit_split": "train early-life rows only",
        "alpha_fit_split": "train known-failed trajectories only",
        "threshold_and_persistence_fit_split": "calibration only",
        "test_rows_used_for_fit_or_selection": 0,
        "test_assets_used_for_fit_or_selection": [],
        "train_early_life_rows_by_group": {f"{key[0]}::{key[1]}": len(value) for key, value in sorted(train_baseline_ids.items())},
        "input_sha256": {
            "d6b_standardized_features.npy": sha256(d6b / "standardized_features.npy"),
            "d6b_baseline_rows.csv": sha256(d6b / "baseline_rows.csv"),
            "d6c_locked_rules.json": sha256(d6c / "locked_rules.json"),
            "d6c_point_alarm.npy": sha256(d6c / "point_alarm.npy"),
        },
        "claim_boundary": "Known terminal-event detection is reported; annotated physical fault onset is unavailable.",
    }
    (output / "fit_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({
        "methods": list(METHODS), "model_registry_entries": len(model_registry),
        "trajectory_rows": len(trajectory_rows), "summary_rows": len(method_summary),
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
