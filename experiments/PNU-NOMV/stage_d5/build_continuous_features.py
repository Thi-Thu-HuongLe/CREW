#!/usr/bin/env python3
"""Build sharded continuous-window PNU-NOMV feature data without loading recordings.

The extractor reuses the Stage ABC binary parser and feature schema, but writes
only compact 40-dimensional features and chronology metadata. Each motor/split
shard is independently restartable and stored as memory-mappable NumPy arrays.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PNU_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PNU_DIR / "stage_abc"))

import prepare_pnu_inputs as abc  # noqa: E402


STATE_LABELS = {0: "quiet", 1: "transition", 2: "active"}
QUALITY_SCHEMA_VERSION = "pnu-near-full-scale-v1"
DEFAULT_NEAR_FULL_SCALE_ABS_COUNTS = 32000
EXACT_FULL_SCALE_ABS_COUNTS = 32764


def feature_schema() -> list[str]:
    names = ["vector_ac_rms_counts"]
    base_names = (
        "ac_rms_counts",
        "peak_to_peak_counts",
        "abs_peak_counts",
        "crest_factor",
        "skewness",
        "kurtosis",
        "dominant_hz",
        "spectral_centroid_hz",
    )
    for axis in abc.AXES:
        names.extend(f"{axis}_{name}" for name in base_names)
        for low, high in abc.BANDS_HZ:
            label = f"{axis}_band_{low:g}_{high if high <= 125 else 125:g}_fraction"
            names.append(label)
    return names


def vectorized_features(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return Stage ABC-compatible features and vector RMS for BxTx3 raw data."""
    raw32 = np.asarray(raw, dtype=np.float32)
    centered = (
        raw32 - np.mean(raw32, axis=1, keepdims=True, dtype=np.float32)
    ).astype(np.float64)
    variance = np.mean(centered * centered, axis=1)
    std = np.sqrt(variance)
    peak = np.max(np.abs(centered), axis=1)
    peak_to_peak = np.ptp(centered, axis=1)
    crest = np.divide(peak, std, out=np.zeros_like(std), where=std > 0)
    third = np.mean(centered**3, axis=1)
    fourth = np.mean(centered**4, axis=1)
    skew = np.divide(third, std**3, out=np.zeros_like(std), where=std > 0)
    kurtosis = np.divide(
        fourth, variance**2, out=np.zeros_like(std), where=variance > 0
    )

    taper = np.hanning(centered.shape[1]).astype(np.float64)
    transformed = np.fft.rfft(centered * taper[None, :, None], axis=1)
    power = np.abs(transformed) ** 2
    power[:, 0, :] = 0.0
    frequencies = np.fft.rfftfreq(centered.shape[1], d=1.0 / abc.FS_HZ)
    total = power.sum(axis=1)
    dominant = frequencies[np.argmax(power, axis=1)]
    centroid = np.divide(
        (power * frequencies[None, :, None]).sum(axis=1),
        total,
        out=np.zeros_like(total),
        where=total > 0,
    )

    output = np.empty((len(raw32), 40), dtype=np.float32)
    vector_rms = np.sqrt(np.sum(variance, axis=1))
    output[:, 0] = vector_rms.astype(np.float32)
    column = 1
    for axis_index in range(3):
        base_values = (
            std[:, axis_index],
            peak_to_peak[:, axis_index],
            peak[:, axis_index],
            crest[:, axis_index],
            skew[:, axis_index],
            kurtosis[:, axis_index],
            dominant[:, axis_index],
            centroid[:, axis_index],
        )
        for values in base_values:
            output[:, column] = values.astype(np.float32)
            column += 1
        for low, high in abc.BANDS_HZ:
            mask = (frequencies >= low) & (frequencies < high)
            fractions = np.divide(
                power[:, mask, axis_index].sum(axis=1),
                total[:, axis_index],
                out=np.zeros(len(raw32), dtype=float),
                where=total[:, axis_index] > 0,
            )
            output[:, column] = fractions.astype(np.float32)
            column += 1
    return output, vector_rms.astype(np.float32)


def load_batch(
    mapped: np.memmap,
    descending: bool,
    starts: np.ndarray,
    window_records: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    chronological = starts[:, None] + np.arange(window_records, dtype=np.int64)[None, :]
    physical = len(mapped) - 1 - chronological if descending else chronological
    records = mapped[physical]
    sequences = records["seq"].astype(np.int64)
    if np.any(np.diff(sequences, axis=1) != 1):
        raise ValueError("Batch contains a window crossing a sequence gap")
    raw = records["data"].reshape(
        len(starts), window_records * abc.SAMPLES_PER_RECORD, 3
    )
    temperature = np.mean(abc.temperature_c(records["temp_raw"]), axis=1)
    humidity = np.mean(abc.humidity_pct(records["humi_raw"]), axis=1)
    return raw, sequences[:, 0], temperature, humidity


def amplitude_quality_counts(
    raw: np.ndarray, near_full_scale_abs_counts: int
) -> tuple[np.ndarray, np.ndarray]:
    """Count near/full-scale samples in each BxTx3 raw window.

    IIS3DWB data are stored as signed 16-bit digital output counts.  The
    released PNU-NOMV files repeatedly reach +/-32764, so an exact hit is retained
    separately from the conservative near-full-scale screening threshold.
    """
    absolute = np.abs(np.asarray(raw, dtype=np.int32))
    near = np.count_nonzero(absolute >= near_full_scale_abs_counts, axis=(1, 2))
    exact = np.count_nonzero(absolute == EXACT_FULL_SCALE_ABS_COUNTS, axis=(1, 2))
    return near.astype(np.int32), exact.astype(np.int32)


def shard_is_complete(
    shard_dir: Path,
    expected_count: int,
    window_seconds: int,
    stride_seconds: int,
    near_full_scale_abs_counts: int,
) -> bool:
    manifest_path = shard_dir / "shard_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    required = (
        "features.npy",
        "start_record.npy",
        "sequence_start.npy",
        "environment.npy",
        "vector_rms.npy",
        "state_code.npy",
        "near_full_scale_count.npy",
        "exact_full_scale_count.npy",
        "quality_flag.npy",
    )
    return (
        int(manifest.get("window_count", -1)) == expected_count
        and int(manifest.get("window_seconds", -1)) == window_seconds
        and int(manifest.get("stride_seconds", -1)) == stride_seconds
        and manifest.get("quality_schema_version") == QUALITY_SCHEMA_VERSION
        and int(manifest.get("near_full_scale_abs_counts", -1))
        == near_full_scale_abs_counts
        and all((shard_dir / name).is_file() for name in required)
    )


def process_shard(
    mapped: np.memmap,
    descending: bool,
    starts: np.ndarray,
    shard_dir: Path,
    motor: str,
    split: str,
    source_file: str,
    window_records: int,
    stride_records: int,
    batch_size: int,
    quiet_limit: float,
    active_limit: float,
    near_full_scale_abs_counts: int,
    overwrite: bool,
) -> dict[str, object]:
    shard_dir.mkdir(parents=True, exist_ok=True)
    if not overwrite and shard_is_complete(
        shard_dir,
        len(starts),
        window_records,
        stride_records,
        near_full_scale_abs_counts,
    ):
        manifest = json.loads(
            (shard_dir / "shard_manifest.json").read_text(encoding="utf-8")
        )
        print(f"{motor}/{split}: reusing {len(starts):,} completed windows", flush=True)
        return manifest

    count = len(starts)
    features_out = np.lib.format.open_memmap(
        shard_dir / "features.npy", mode="w+", dtype=np.float32, shape=(count, 40)
    )
    starts_out = np.lib.format.open_memmap(
        shard_dir / "start_record.npy", mode="w+", dtype=np.int64, shape=(count,)
    )
    sequence_out = np.lib.format.open_memmap(
        shard_dir / "sequence_start.npy", mode="w+", dtype=np.uint32, shape=(count,)
    )
    environment_out = np.lib.format.open_memmap(
        shard_dir / "environment.npy", mode="w+", dtype=np.float32, shape=(count, 2)
    )
    rms_out = np.lib.format.open_memmap(
        shard_dir / "vector_rms.npy", mode="w+", dtype=np.float32, shape=(count,)
    )
    state_out = np.lib.format.open_memmap(
        shard_dir / "state_code.npy", mode="w+", dtype=np.uint8, shape=(count,)
    )
    near_full_scale_out = np.lib.format.open_memmap(
        shard_dir / "near_full_scale_count.npy",
        mode="w+",
        dtype=np.int32,
        shape=(count,),
    )
    exact_full_scale_out = np.lib.format.open_memmap(
        shard_dir / "exact_full_scale_count.npy",
        mode="w+",
        dtype=np.int32,
        shape=(count,),
    )
    quality_flag_out = np.lib.format.open_memmap(
        shard_dir / "quality_flag.npy", mode="w+", dtype=np.uint8, shape=(count,)
    )

    started = time.perf_counter()
    validation_error = 0.0
    for offset in range(0, count, batch_size):
        stop = min(count, offset + batch_size)
        batch_starts = starts[offset:stop]
        raw, sequence_start, temperature, humidity = load_batch(
            mapped, descending, batch_starts, window_records
        )
        features, vector_rms = vectorized_features(raw)
        near_full_scale_count, exact_full_scale_count = amplitude_quality_counts(
            raw, near_full_scale_abs_counts
        )
        if offset == 0 and len(raw):
            scalar_centered = (
                raw[0].astype(np.float32)
                - raw[0].astype(np.float32).mean(axis=0, keepdims=True)
            )
            scalar_features, scalar_names = abc.feature_vector(scalar_centered)
            if scalar_names != feature_schema():
                raise RuntimeError("Continuous feature schema differs from Stage ABC")
            validation_error = float(
                np.max(np.abs(features[0].astype(float) - scalar_features.astype(float)))
            )
            if validation_error > 1e-3:
                raise RuntimeError(
                    f"Vectorized feature validation failed: {validation_error}"
                )
        state_code = np.where(
            vector_rms < quiet_limit,
            0,
            np.where(vector_rms >= active_limit, 2, 1),
        ).astype(np.uint8)
        features_out[offset:stop] = features
        starts_out[offset:stop] = batch_starts
        sequence_out[offset:stop] = sequence_start.astype(np.uint32)
        environment_out[offset:stop, 0] = temperature.astype(np.float32)
        environment_out[offset:stop, 1] = humidity.astype(np.float32)
        rms_out[offset:stop] = vector_rms
        state_out[offset:stop] = state_code
        near_full_scale_out[offset:stop] = near_full_scale_count
        exact_full_scale_out[offset:stop] = exact_full_scale_count
        quality_flag_out[offset:stop] = (near_full_scale_count > 0).astype(np.uint8)
        if stop == count or stop % max(batch_size * 50, 1) == 0:
            elapsed = time.perf_counter() - started
            rate = stop / max(elapsed, 1e-9)
            print(
                f"{motor}/{split}: {stop:,}/{count:,} windows ({rate:,.0f}/s)",
                flush=True,
            )

    for array in (
        features_out,
        starts_out,
        sequence_out,
        environment_out,
        rms_out,
        state_out,
        near_full_scale_out,
        exact_full_scale_out,
        quality_flag_out,
    ):
        array.flush()
    state_counts = {
        STATE_LABELS[code]: int(np.sum(np.asarray(state_out) == code))
        for code in STATE_LABELS
    }
    start_differences = np.diff(starts)
    segment_count = int(1 + np.sum(start_differences != stride_records)) if count else 0
    manifest = {
        "motor": motor,
        "split": split,
        "source_file": source_file,
        "window_count": count,
        "window_seconds": window_records,
        "stride_seconds": stride_records,
        "window_samples_per_axis": window_records * abc.SAMPLES_PER_RECORD,
        "feature_count": 40,
        "state_counts": state_counts,
        "quality_schema_version": QUALITY_SCHEMA_VERSION,
        "near_full_scale_abs_counts": near_full_scale_abs_counts,
        "exact_full_scale_abs_counts": EXACT_FULL_SCALE_ABS_COUNTS,
        "near_full_scale_window_count": int(np.count_nonzero(quality_flag_out)),
        "exact_full_scale_window_count": int(
            np.count_nonzero(np.asarray(exact_full_scale_out) > 0)
        ),
        "chronological_segment_count": segment_count,
        "first_start_record": int(starts[0]) if count else None,
        "last_start_record": int(starts[-1]) if count else None,
        "vectorized_scalar_validation_max_abs_error": validation_error,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (shard_dir / "shard_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    project_root = SCRIPT_DIR.parents[2]
    parser = argparse.ArgumentParser(
        description="Build continuous sharded PNU-NOMV tabular feature data"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "Dataset" / "PNU-NOMV" / "data" / "records",
    )
    parser.add_argument(
        "--output", type=Path, default=SCRIPT_DIR / "continuous_features"
    )
    parser.add_argument("--window-seconds", type=int, default=4)
    parser.add_argument("--stride-seconds", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--quiet-rms-limit", type=float, default=200.0)
    parser.add_argument("--active-rms-limit", type=float, default=1000.0)
    parser.add_argument(
        "--near-full-scale-abs-counts",
        type=int,
        default=DEFAULT_NEAR_FULL_SCALE_ABS_COUNTS,
        help="Conservative raw-count threshold used to flag a window (default: 32000)",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.window_seconds < 1 or args.stride_seconds < 1 or args.batch_size < 1:
        parser.error("window, stride, and batch size must be positive integers")
    if not 1 <= args.near_full_scale_abs_counts <= EXACT_FULL_SCALE_ABS_COUNTS:
        parser.error(
            f"near-full-scale threshold must be in [1, {EXACT_FULL_SCALE_ABS_COUNTS}]"
        )
    input_dir = args.input.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(input_dir.glob("MO*_vibration_*.dat"))
    if not paths:
        raise SystemExit(f"No PNU-NOMV DAT files found in {input_dir}")

    schema = feature_schema()
    shard_manifests: list[dict[str, object]] = []
    started = time.perf_counter()
    for path in paths:
        motor = abc.motor_from_path(path)
        mapped, descending = abc.file_map(path)
        sequences = abc.chronological_sequences(mapped, descending)
        segments = abc.contiguous_segments(sequences)
        ranges = abc.split_ranges(len(mapped))
        print(
            f"{motor}: {len(mapped):,} records, {len(segments)} contiguous segments",
            flush=True,
        )
        for split, (split_start, split_stop) in ranges.items():
            starts = abc.valid_window_starts(
                segments,
                split_start,
                split_stop,
                args.window_seconds,
                args.stride_seconds,
            )
            manifest = process_shard(
                mapped,
                descending,
                starts,
                output_dir / motor / split,
                motor,
                split,
                path.name,
                args.window_seconds,
                args.stride_seconds,
                args.batch_size,
                args.quiet_rms_limit,
                args.active_rms_limit,
                args.near_full_scale_abs_counts,
                args.overwrite,
            )
            manifest["candidate_split_start_record"] = split_start
            manifest["candidate_split_stop_record_exclusive"] = split_stop
            shard_manifests.append(manifest)
        del mapped

    total_windows = int(sum(int(row["window_count"]) for row in shard_manifests))
    dataset_manifest = {
        "stage": "D5_continuous_feature_dataset",
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "file_count": len(paths),
        "motors": [abc.motor_from_path(path) for path in paths],
        "sample_rate_hz": abc.FS_HZ,
        "effective_register_polling_rate_hz": abc.FS_HZ,
        "sensor_model": abc.SENSOR_MODEL,
        "sensor_full_scale_g": abc.SENSOR_FULL_SCALE_G,
        "nominal_sensitivity_mg_per_lsb": abc.NOMINAL_SENSITIVITY_MG_PER_LSB,
        "anti_alias_filter_documented": False,
        "window_seconds": args.window_seconds,
        "stride_seconds": args.stride_seconds,
        "overlap_fraction": 1.0 - args.stride_seconds / args.window_seconds,
        "feature_names": schema,
        "feature_count": len(schema),
        "state_code_labels": {str(code): label for code, label in STATE_LABELS.items()},
        "quality_control": {
            "schema_version": QUALITY_SCHEMA_VERSION,
            "near_full_scale_abs_counts": args.near_full_scale_abs_counts,
            "exact_full_scale_abs_counts": EXACT_FULL_SCALE_ABS_COUNTS,
            "primary_policy": "exclude windows containing any sample with abs(raw_count) >= threshold",
            "raw_data_mutated": False,
        },
        "split_fractions": {name: [low, high] for name, low, high in abc.SPLITS},
        "total_window_count": total_windows,
        "shards": shard_manifests,
        "elapsed_seconds": time.perf_counter() - started,
        "storage_note": (
            "Only tabular features and compact metadata are materialized. Raw/FFT/STFT tensors are read lazily from source DAT files for future deep models."
        ),
        "leakage_controls": [
            "Chronological 60/20/20 boundaries are computed independently per motor.",
            "No window crosses a split boundary or sequence discontinuity.",
            "Chronological start records are retained for event reconstruction.",
        ],
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(dataset_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"D5 continuous dataset: {total_windows:,} windows in {time.perf_counter()-started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
