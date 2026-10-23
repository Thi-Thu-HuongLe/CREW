#!/usr/bin/env python3
"""Build deterministic model-ready samples from PNU-NOMV DAT files.

The script preserves chronological order, prevents windows from crossing
sequence gaps or split boundaries, and fits robust scaling on training data
only. It writes a compact representative sample for pipeline verification;
training code can reuse the same functions for lazy full-dataset loading.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


FS_HZ = 250
SAMPLES_PER_RECORD = 250
RECORD_BYTES = 1512
SENSOR_MODEL = "IIS3DWB"
SENSOR_FULL_SCALE_G = 8.0
NOMINAL_SENSITIVITY_MG_PER_LSB = 0.244
AXES = ("x", "y", "z")
SPLITS = (("train", 0.0, 0.6), ("calibration", 0.6, 0.8), ("test", 0.8, 1.0))
BANDS_HZ = ((1.0, 10.0), (10.0, 30.0), (30.0, 60.0), (60.0, 100.0), (100.0, 125.01))

RECORD_DTYPE = np.dtype(
    {
        "names": ("header", "seq", "data", "temp_raw", "humi_raw"),
        "formats": ("S4", "<u4", ("<i2", 750), "<u2", "<u2"),
        "offsets": (0, 4, 8, 1508, 1510),
        "itemsize": RECORD_BYTES,
    }
)


def temperature_c(raw: np.ndarray) -> np.ndarray:
    return -45.0 + 175.0 * raw.astype(np.float64) / 65535.0


def humidity_pct(raw: np.ndarray) -> np.ndarray:
    return -6.0 + 125.0 * raw.astype(np.float64) / 65535.0


def motor_from_path(path: Path) -> str:
    return path.name.split("_", 1)[0].upper()


def file_map(path: Path) -> tuple[np.memmap, bool]:
    size = path.stat().st_size
    if size % RECORD_BYTES:
        raise ValueError(f"{path.name}: file size is not divisible by {RECORD_BYTES}")
    count = size // RECORD_BYTES
    mapped = np.memmap(path, dtype=RECORD_DTYPE, mode="r", shape=(count,))
    return mapped, int(mapped[0]["seq"]) > int(mapped[-1]["seq"])


def chronological_sequences(mapped: np.memmap, descending: bool) -> np.ndarray:
    seq = np.asarray(mapped["seq"], dtype=np.uint32)
    return seq[::-1] if descending else seq


def contiguous_segments(seq: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open chronological ranges whose sequence increments by one."""
    boundaries = np.flatnonzero(np.diff(seq.astype(np.int64)) != 1) + 1
    starts = np.r_[0, boundaries]
    stops = np.r_[boundaries, len(seq)]
    return [(int(start), int(stop)) for start, stop in zip(starts, stops)]


def split_ranges(record_count: int) -> dict[str, tuple[int, int]]:
    return {
        name: (int(math.floor(lo * record_count)), int(math.floor(hi * record_count)))
        for name, lo, hi in SPLITS
    }


def valid_window_starts(
    segments: list[tuple[int, int]],
    split_start: int,
    split_stop: int,
    window_records: int,
    stride_records: int,
) -> np.ndarray:
    arrays: list[np.ndarray] = []
    for seg_start, seg_stop in segments:
        start = max(seg_start, split_start)
        stop = min(seg_stop, split_stop)
        last_start = stop - window_records
        if last_start >= start:
            arrays.append(np.arange(start, last_start + 1, stride_records, dtype=np.int64))
    return np.concatenate(arrays) if arrays else np.empty(0, dtype=np.int64)


def evenly_spaced(values: np.ndarray, count: int) -> np.ndarray:
    if len(values) <= count:
        return values
    indices = np.linspace(0, len(values) - 1, count, dtype=np.int64)
    return values[np.unique(indices)]


def proportional_sample_counts(total_per_motor: int) -> dict[str, int]:
    """Allocate a per-motor sample budget using the chronological split ratios."""
    if total_per_motor < 1:
        raise ValueError("total_per_motor must be positive")
    names = [name for name, _, _ in SPLITS]
    weights = np.asarray([high - low for _, low, high in SPLITS], dtype=float)
    raw = weights / weights.sum() * total_per_motor
    counts = np.floor(raw).astype(int)
    remainder = total_per_motor - int(counts.sum())
    order = np.argsort(-(raw - counts), kind="stable")
    for index in order[:remainder]:
        counts[index] += 1
    return {name: int(count) for name, count in zip(names, counts)}


def load_window(
    mapped: np.memmap,
    descending: bool,
    chronological_start: int,
    window_records: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    chronological = np.arange(
        chronological_start, chronological_start + window_records, dtype=np.int64
    )
    physical = len(mapped) - 1 - chronological if descending else chronological
    records = mapped[physical]
    sequences = records["seq"].astype(np.int64)
    if np.any(np.diff(sequences) != 1):
        raise ValueError("Attempted to create a window across a sequence discontinuity")
    raw = records["data"].reshape(window_records, SAMPLES_PER_RECORD, 3)
    raw = raw.reshape(window_records * SAMPLES_PER_RECORD, 3).astype(np.float32)
    temp = float(np.mean(temperature_c(records["temp_raw"])))
    humi = float(np.mean(humidity_pct(records["humi_raw"])))
    return raw, sequences, temp, humi


def state_from_rms(value: float, quiet_limit: float, active_limit: float) -> str:
    if value < quiet_limit:
        return "quiet"
    if value >= active_limit:
        return "active"
    return "transition"


def spectrum_log_power(signal: np.ndarray) -> np.ndarray:
    window = np.hanning(len(signal)).astype(np.float32)[:, None]
    transformed = np.fft.rfft(signal * window, axis=0)
    power = np.abs(transformed) ** 2
    return np.log1p(power).T.astype(np.float32)


def stft_log_power(signal: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    starts = range(0, len(signal) - frame_length + 1, hop_length)
    taper = np.hanning(frame_length).astype(np.float32)[:, None]
    frames = []
    for start in starts:
        transformed = np.fft.rfft(signal[start : start + frame_length] * taper, axis=0)
        frames.append(np.log1p(np.abs(transformed) ** 2).T.astype(np.float32))
    return np.stack(frames, axis=-1)


def feature_vector(centered: np.ndarray) -> tuple[np.ndarray, list[str]]:
    values = centered.astype(np.float64)
    variance = np.mean(values * values, axis=0)
    std = np.sqrt(variance)
    peak = np.max(np.abs(values), axis=0)
    peak_to_peak = np.ptp(values, axis=0)
    crest = np.divide(peak, std, out=np.zeros_like(std), where=std > 0)
    third = np.mean(values**3, axis=0)
    fourth = np.mean(values**4, axis=0)
    skew = np.divide(third, std**3, out=np.zeros_like(std), where=std > 0)
    kurtosis = np.divide(fourth, variance**2, out=np.zeros_like(std), where=variance > 0)

    transformed = np.fft.rfft(values * np.hanning(len(values))[:, None], axis=0)
    power = np.abs(transformed) ** 2
    power[0, :] = 0.0
    frequencies = np.fft.rfftfreq(len(values), d=1.0 / FS_HZ)
    total = power.sum(axis=0)
    dominant = frequencies[np.argmax(power, axis=0)]
    centroid = np.divide(
        (power * frequencies[:, None]).sum(axis=0),
        total,
        out=np.zeros_like(total),
        where=total > 0,
    )

    feature_names = ["vector_ac_rms_counts"]
    feature_values = [float(np.sqrt(np.sum(variance)))]
    for axis_idx, axis in enumerate(AXES):
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
        base_values = (
            std[axis_idx],
            peak_to_peak[axis_idx],
            peak[axis_idx],
            crest[axis_idx],
            skew[axis_idx],
            kurtosis[axis_idx],
            dominant[axis_idx],
            centroid[axis_idx],
        )
        feature_names.extend(f"{axis}_{name}" for name in base_names)
        feature_values.extend(float(value) for value in base_values)
        for low, high in BANDS_HZ:
            mask = (frequencies >= low) & (frequencies < high)
            fraction = float(power[mask, axis_idx].sum() / total[axis_idx]) if total[axis_idx] > 0 else 0.0
            label = f"{axis}_band_{low:g}_{high if high <= 125 else 125:g}_fraction"
            feature_names.append(label)
            feature_values.append(fraction)
    return np.asarray(feature_values, dtype=np.float32), feature_names


def robust_scales(
    centered_windows: np.ndarray, metadata: list[dict[str, object]]
) -> tuple[np.ndarray, dict[str, list[float]]]:
    scales = np.empty((len(centered_windows), 3), dtype=np.float32)
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(metadata):
        if row["split"] == "train":
            key = f"{row['motor']}|{row['operating_state']}"
            groups.setdefault(key, []).append(index)

    global_indices = [index for index, row in enumerate(metadata) if row["split"] == "train"]
    global_scale = np.median(
        np.abs(centered_windows[global_indices]), axis=(0, 2)
    ) / 0.67448975
    global_scale = np.where(global_scale > 1e-6, global_scale, 1.0).astype(np.float32)

    learned: dict[str, list[float]] = {"global": global_scale.tolist()}
    for key, indices in groups.items():
        scale = np.median(np.abs(centered_windows[indices]), axis=(0, 2)) / 0.67448975
        scale = np.where(scale > 1e-6, scale, global_scale).astype(np.float32)
        learned[key] = scale.tolist()

    for index, row in enumerate(metadata):
        key = f"{row['motor']}|{row['operating_state']}"
        scales[index] = np.asarray(learned.get(key, learned["global"]), dtype=np.float32)
    return scales, learned


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "Dataset" / "PNU-NOMV" / "data" / "records",
    )
    parser.add_argument("--output", type=Path, default=script_dir / "outputs")
    parser.add_argument("--window-seconds", type=int, default=4)
    parser.add_argument("--stride-seconds", type=int, default=2)
    parser.add_argument(
        "--windows-per-motor",
        type=int,
        default=1000,
        help="Representative-window budget per motor, allocated by split ratio (default: 1000 = 600/200/200)",
    )
    parser.add_argument(
        "--windows-per-motor-split",
        type=int,
        default=None,
        help="Legacy mode: use the same count in every split instead of proportional allocation",
    )
    parser.add_argument("--quiet-rms-limit", type=float, default=200.0)
    parser.add_argument("--active-rms-limit", type=float, default=1000.0)
    parser.add_argument("--stft-frame", type=int, default=256)
    parser.add_argument("--stft-hop", type=int, default=128)
    args = parser.parse_args()

    input_dir = args.input.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(input_dir.glob("MO*_vibration_*.dat"))
    if not paths:
        raise SystemExit(f"No PNU-NOMV DAT files found in {input_dir}")

    window_records = int(args.window_seconds)
    stride_records = int(args.stride_seconds)
    if window_records < 1 or stride_records < 1:
        raise SystemExit("Window and stride must be positive whole seconds")
    if args.windows_per_motor < 1:
        raise SystemExit("--windows-per-motor must be positive")
    if args.windows_per_motor_split is not None and args.windows_per_motor_split < 1:
        raise SystemExit("--windows-per-motor-split must be positive when supplied")
    if args.stft_frame > window_records * SAMPLES_PER_RECORD:
        raise SystemExit("STFT frame is longer than the signal window")

    if args.windows_per_motor_split is not None:
        target_counts = {
            name: args.windows_per_motor_split for name, _, _ in SPLITS
        }
        sampling_strategy = "legacy_equal_count_per_motor_split"
    else:
        target_counts = proportional_sample_counts(args.windows_per_motor)
        sampling_strategy = "proportional_to_chronological_split"

    centered_windows: list[np.ndarray] = []
    metadata: list[dict[str, object]] = []
    feature_rows: list[np.ndarray] = []
    inventory_rows: list[dict[str, object]] = []
    feature_names: list[str] | None = None

    for path in paths:
        motor = motor_from_path(path)
        mapped, descending = file_map(path)
        seq = chronological_sequences(mapped, descending)
        segments = contiguous_segments(seq)
        ranges = split_ranges(len(mapped))
        print(f"{motor}: {len(mapped):,} records, {len(segments)} contiguous segments", flush=True)
        for split, (split_start, split_stop) in ranges.items():
            candidates = valid_window_starts(
                segments,
                split_start,
                split_stop,
                window_records,
                stride_records,
            )
            selected = evenly_spaced(candidates, target_counts[split])
            inventory_rows.append(
                {
                    "motor": motor,
                    "split": split,
                    "chronological_start_record": split_start,
                    "chronological_stop_record_exclusive": split_stop,
                    "valid_candidate_windows": len(candidates),
                    "sampled_windows": len(selected),
                    "window_seconds": window_records,
                    "stride_seconds": stride_records,
                    "contiguous_segments_in_file": len(segments),
                }
            )
            for start in selected:
                raw, sequences, temp, humi = load_window(
                    mapped, descending, int(start), window_records
                )
                centered = (raw - raw.mean(axis=0, keepdims=True)).T.astype(np.float32)
                vector_rms = float(np.sqrt(np.sum(np.mean(centered.astype(np.float64) ** 2, axis=1))))
                state = state_from_rms(
                    vector_rms, args.quiet_rms_limit, args.active_rms_limit
                )
                features, names = feature_vector(centered.T)
                if feature_names is None:
                    feature_names = names
                elif names != feature_names:
                    raise RuntimeError("Feature schema changed while processing")
                centered_windows.append(centered)
                feature_rows.append(features)
                metadata.append(
                    {
                        "sample_id": len(metadata),
                        "motor": motor,
                        "split": split,
                        "source_file": path.name,
                        "chronological_start_second": int(start),
                        "chronological_stop_second_exclusive": int(start + window_records),
                        "sequence_start": int(sequences[0]),
                        "sequence_stop": int(sequences[-1]),
                        "temperature_c": temp,
                        "humidity_pct": humi,
                        "vector_ac_rms_counts": vector_rms,
                        "operating_state": state,
                    }
                )
        del mapped

    centered_array = np.stack(centered_windows).astype(np.float32)
    scales, learned_scales = robust_scales(centered_array, metadata)
    normalized = centered_array / scales[:, :, None]
    spectra = np.stack([spectrum_log_power(window.T) for window in normalized])
    stft = np.stack(
        [
            stft_log_power(window.T, args.stft_frame, args.stft_hop)
            for window in normalized
        ]
    )
    features_array = np.stack(feature_rows).astype(np.float32)

    np.savez_compressed(
        output_dir / "pnu_model_inputs_sample.npz",
        time_normalized=normalized.astype(np.float32),
        fft_log_power=spectra.astype(np.float32),
        stft_log_power=stft.astype(np.float32),
        tabular_features=features_array,
        feature_names=np.asarray(feature_names, dtype="U64"),
        motor=np.asarray([row["motor"] for row in metadata], dtype="U4"),
        split=np.asarray([row["split"] for row in metadata], dtype="U11"),
        operating_state=np.asarray(
            [row["operating_state"] for row in metadata], dtype="U10"
        ),
    )
    write_csv(output_dir / "pnu_model_inputs_metadata.csv", metadata)
    write_csv(output_dir / "pnu_window_inventory.csv", inventory_rows)
    config = {
        "input_directory": str(input_dir),
        "file_count": len(paths),
        "motors": [motor_from_path(path) for path in paths],
        "sample_rate_hz": FS_HZ,
        "effective_register_polling_rate_hz": FS_HZ,
        "sensor_model": SENSOR_MODEL,
        "sensor_full_scale_g": SENSOR_FULL_SCALE_G,
        "nominal_sensitivity_mg_per_lsb": NOMINAL_SENSITIVITY_MG_PER_LSB,
        "anti_alias_filter_documented": False,
        "window_seconds": window_records,
        "window_samples_per_axis": window_records * SAMPLES_PER_RECORD,
        "stride_seconds": stride_records,
        "split_fractions": {name: [lo, hi] for name, lo, hi in SPLITS},
        "quiet_rms_limit_counts": args.quiet_rms_limit,
        "active_rms_limit_counts": args.active_rms_limit,
        "stft_frame_samples": args.stft_frame,
        "stft_hop_samples": args.stft_hop,
        "frequency_resolution_hz": FS_HZ / (window_records * SAMPLES_PER_RECORD),
        "sampling_strategy": sampling_strategy,
        "windows_per_motor": (
            None if args.windows_per_motor_split is not None else args.windows_per_motor
        ),
        "sampled_windows_per_motor_split": target_counts,
        "legacy_windows_per_motor_split": args.windows_per_motor_split,
        "sample_count": len(metadata),
        "time_shape": list(normalized.shape),
        "fft_shape": list(spectra.shape),
        "stft_shape": list(stft.shape),
        "feature_shape": list(features_array.shape),
        "robust_scales_by_motor_state": learned_scales,
        "notes": [
            "All scaling is fitted on training samples only.",
            "Representative-window counts preserve the chronological 60/20/20 split ratio unless legacy equal-count mode is explicitly requested.",
            "Windows never cross split boundaries or sequence gaps.",
            "Vibration is retained as raw IIS3DWB digital output counts; +/-8 g permits a nominal 0.244 mg/LSB conversion, but no traceable calibration is claimed.",
            "Frequency-domain PNU-NOMV features are exploratory because no anti-alias filter is documented before 250 Hz register polling.",
            "Order spectra require RPM or shaft-angle metadata and are not generated.",
        ],
    }
    (output_dir / "pnu_preprocessing_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(metadata):,} samples to {output_dir}", flush=True)


if __name__ == "__main__":
    main()

