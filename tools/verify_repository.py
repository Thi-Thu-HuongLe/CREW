"""Fast pre-publication checks for the compact CREW repository."""

from __future__ import annotations

import ast
import json
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "__pycache__", "Dataset", "data", "outputs", "continuous_features"}
FORBIDDEN_SUFFIXES = {".dat", ".rar", ".tar", ".zip", ".npy", ".npz", ".png", ".svg", ".pdf", ".docx"}
FORBIDDEN_CODE_NAMES = {"build_report.py", "build_manuscript.py"}
LOCAL_MARKERS = ("One" + "Drive", "C:" + "\\Users" + "\\")
MAX_FILE_BYTES = 5 * 1024 * 1024


def included_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file() and not any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts)
    )


def main() -> None:
    files = included_files()
    python_count = 0
    for path in files:
        relative = path.relative_to(ROOT)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            raise ValueError(f"generated/data artifact must not be committed: {relative}")
        if path.name in FORBIDDEN_CODE_NAMES or path.name.startswith("plot_"):
            raise ValueError(f"report/figure code is outside compact repository scope: {relative}")
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"file exceeds 5 MiB review limit: {relative}")
        if path.suffix.lower() in {".py", ".md", ".toml", ".cff", ".json", ".csv", ".txt", ".yml", ".yaml"}:
            text = path.read_text(encoding="utf-8-sig")
            if any(marker in text for marker in LOCAL_MARKERS):
                raise ValueError(f"local machine path found: {relative}")
        if path.suffix.lower() == ".py":
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(relative))
            python_count += 1
        elif path.suffix.lower() == ".json":
            json.loads(path.read_text(encoding="utf-8-sig"))

    tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "status": "PASS",
                "repository_files_checked": len(files),
                "python_files_checked": python_count,
                "paper_figure_code_present": False,
                "data_or_generated_artifacts_present": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
