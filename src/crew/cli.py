"""Command-line application for a frozen CREW rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .sequential import FrozenRule, apply_crew


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply a frozen CREW rule to a chronological 1-D novelty-score array."
    )
    parser.add_argument("--scores", type=Path, required=True, help="Input .npy vector")
    parser.add_argument("--output", type=Path, required=True, help="Output .npz file")
    parser.add_argument("--commissioning-count", type=int, required=True)
    parser.add_argument("--scale", type=float, required=True, help="Train-fitted robust scale")
    parser.add_argument("--center", type=float, default=None, help="Optional fixed center")
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--persistence", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scores = np.load(args.scores, allow_pickle=False)
    rule = FrozenRule(args.alpha, args.threshold, args.persistence)
    result = apply_crew(
        scores,
        args.commissioning_count,
        args.scale,
        rule,
        center=args.center,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        log_score=result.log_score,
        health=result.health,
        point_alarm=result.point_alarm,
        latched_warning=result.latched_warning,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "center": result.center,
                "scale": result.scale,
                "evidence_start_index": result.evidence_start_index,
                "confirmation_index": result.confirmation_index,
            },
            indent=2,
        )
    )
    return 0
