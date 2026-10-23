#!/usr/bin/env python3
"""Unified, archive-aware signal loaders for Stage D6A benchmark records."""

from __future__ import annotations

import csv
import io
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping
from zipfile import ZipFile

import numpy as np


@dataclass(frozen=True)
class SignalRecord:
    samples: np.ndarray
    channel_names: tuple[str, ...]
    sampling_rate_hz: float
    metadata: dict[str, object]


def read_record_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def iter_run_records(rows: Iterable[Mapping[str, str]], dataset: str, run_id: str) -> Iterable[Mapping[str, str]]:
    for row in rows:
        if row["dataset"] == dataset and row["run_id"] == run_id:
            yield row


def _read_delimited(
    payload: bytes | Path,
    delimiter: str | None,
    skip_rows: int,
    columns: list[int],
) -> np.ndarray:
    """Parse the benchmark ASCII formats without ``loadtxt``'s per-token overhead."""
    raw = payload.read_bytes() if isinstance(payload, Path) else payload
    lines = [line.strip() for line in raw.splitlines()[skip_rows:] if line.strip()]
    if not lines:
        return np.empty((0, len(columns)), dtype=np.float64)

    try:
        if delimiter is None:
            width = len(lines[0].split())
            flattened = b" ".join(lines)
            values = np.fromstring(flattened.decode("ascii"), sep=" ", dtype=np.float64)
        else:
            delimiter_bytes = delimiter.encode("ascii")
            if lines[0].count(delimiter_bytes) == 0:
                candidates = (b",", b";", b"\t")
                delimiter_bytes = max(candidates, key=lines[0].count)
                if lines[0].count(delimiter_bytes) == 0:
                    raise ValueError("declared and fallback delimiters are absent")
                delimiter = delimiter_bytes.decode("ascii")
            width = lines[0].count(delimiter_bytes) + 1
            flattened = delimiter_bytes.join(lines)
            values = np.fromstring(flattened.decode("ascii"), sep=delimiter, dtype=np.float64)
        if width <= 0 or values.size % width:
            raise ValueError(
                f"parsed {values.size} values with inferred width {width}"
            )
        return values.reshape(-1, width)[:, columns]
    except (UnicodeDecodeError, ValueError):
        # A small number of source CSVs contain formatting accepted by loadtxt
        # but rejected by fromstring. Preserve correctness with a slow fallback.
        values = np.loadtxt(
            io.BytesIO(raw), delimiter=delimiter, skiprows=skip_rows,
            usecols=columns, dtype=np.float64,
        )
        return values[:, None] if values.ndim == 1 else values


def _tar_member(archive: Path, member: str) -> bytes:
    executable = shutil.which("tar") or shutil.which("tar.exe")
    if executable is None:
        raise RuntimeError("tar/bsdtar is required to stream NASA-IMS RAR members")
    completed = subprocess.run(
        [executable, "-xOf", str(archive), member],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def load_record(row: Mapping[str, str], dataset_root: Path) -> SignalRecord:
    """Load one row from ``record_manifest.csv`` into samples x channels."""
    source_path = dataset_root / row["source_path"]
    member = row.get("member_path", "")
    source_kind = row["source_kind"]
    channels = tuple(json.loads(row["channels_json"]))
    columns = [int(value) for value in json.loads(row["signal_columns_json"])]
    delimiter_value = row.get("delimiter", "")
    delimiter = None if delimiter_value in ("", "whitespace") else delimiter_value
    skip_rows = int(row.get("header_rows", "0") or 0)
    extra: dict[str, object] = {}

    if source_kind == "filesystem_csv":
        samples = _read_delimited(source_path, delimiter, skip_rows, columns)
    elif source_kind == "zip_csv":
        with ZipFile(source_path) as archive:
            samples = _read_delimited(archive.read(member), delimiter, skip_rows, columns)
    elif source_kind == "rar_ascii":
        samples = _read_delimited(_tar_member(source_path, member), delimiter, skip_rows, columns)
    else:
        raise ValueError(f"unsupported source_kind={source_kind!r}")

    expected = int(row["expected_sample_count"])
    if expected > 0 and samples.shape[0] != expected:
        raise ValueError(f"{row['record_id']}: expected {expected} samples, found {samples.shape[0]}")
    if samples.shape[1] != len(channels):
        raise ValueError(f"{row['record_id']}: channel metadata does not match loaded data")
    if not np.isfinite(samples).all():
        raise ValueError(f"{row['record_id']}: non-finite signal value")
    metadata: dict[str, object] = dict(row)
    metadata.update(extra)
    return SignalRecord(
        samples=samples,
        channel_names=channels,
        sampling_rate_hz=float(row["sampling_rate_hz"]),
        metadata=metadata,
    )

