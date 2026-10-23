# CREW

CREW stands for **Causal Robust EWMA Warning** and provides a leakage-safe
one-class early fault warning pipeline.

This is the compact public implementation accompanying the CREW motor-vibration
study. It contains the proposed method and the computational pipeline needed to
reproduce the primary numerical experiments. Paper-figure generation, HTML
reports, manuscript builders, exploratory development stages, raw datasets, and
generated results are intentionally excluded. Those materials remain preserved
in the private `CREW-Full` archive.

## Scope

CREW converts a train-fitted non-negative novelty score into a causal health
indicator, then applies a calibration-locked threshold and persistence rule.
It is an early-warning method, not a physical fault-onset estimator. Warning
lead in the study is measured to the documented terminal recording.

PNU-NOMV contains normal operation only. Its role is to measure nuisance-warning
burden and domain shift, not PNU-NOMV fault recall.

## Contents

```text
src/crew/                          Reusable CREW and feature API
experiments/Real-Fault-Benchmarks Public benchmark computation and validation
experiments/PNU-NOMV               PNU-NOMV computation and validation
tests/                             Deterministic unit/equivalence tests
examples/                          Minimal API demonstration
docs/                              Data layout and leakage controls
tools/verify_repository.py         Pre-publication repository audit
```

Not included:

- plotting and manuscript-figure scripts;
- HTML/report builders and integrated manuscript synthesis;
- PNU-NOMV exploratory Stages D1--D4 and optional window ablation;
- raw signals, feature arrays, model outputs, caches, or provider archives;
- frozen result tables already reported in the manuscript.

## Installation and tests

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e .
python -m unittest discover -s tests -v
python tools/verify_repository.py
```

The locked environment used Python 3.14.5 and NumPy 2.5.1. The package supports
Python 3.10 or newer and NumPy 2.x.

## Minimal API

```python
import numpy as np
from crew import FrozenRule, apply_crew

score = np.load("chronological_novelty_score.npy")
rule = FrozenRule(alpha=0.20, threshold=3.5, persistence=3)
result = apply_crew(
    score,
    commissioning_count=50,
    scale=0.42,       # fitted using training assets only
    rule=rule,        # selected using training/calibration assets only
)
print(result.confirmation_index)
```

The warning time is the persistence-confirmation index. The earlier
`evidence_start_index` is retrospective and must not be reported as a causal
warning time.

## Data

Place locally obtained datasets in the ignored `Dataset/` folder following
`docs/DATA_LAYOUT.md`. PNU-NOMV should be obtained from its DOI repository.
XJTU-SY, PRONOSTIA, and NASA-IMS must be obtained from their original providers.
No third-party dataset is redistributed here.

## Primary reproduction workflow

Run from the repository root after installation.

### Public run-to-failure benchmarks

```bash
python experiments/Real-Fault-Benchmarks/stage_d6a/build_benchmark_manifest.py
python experiments/Real-Fault-Benchmarks/stage_d6a/validate_d6a.py
python experiments/Real-Fault-Benchmarks/stage_d6b/build_split_manifest.py
python experiments/Real-Fault-Benchmarks/stage_d6b/extract_unified_features.py
python experiments/Real-Fault-Benchmarks/stage_d6b/fit_preprocessing.py
python experiments/Real-Fault-Benchmarks/stage_d6b/validate_d6b.py
python experiments/Real-Fault-Benchmarks/stage_d6c/build_health_indicator.py
python experiments/Real-Fault-Benchmarks/stage_d6c/validate_d6c.py
python experiments/Real-Fault-Benchmarks/stage_d6d/run_robustness_audit.py
python experiments/Real-Fault-Benchmarks/stage_d6d/validate_d6d.py
python experiments/Real-Fault-Benchmarks/stage_d6e/build_learned_baselines.py
python experiments/Real-Fault-Benchmarks/stage_d6e/validate_d6e.py
python experiments/Real-Fault-Benchmarks/stage_d8a/run_sequential_baselines.py
python experiments/Real-Fault-Benchmarks/stage_d8a/validate_d8a.py
```

### PNU-NOMV normal-only evaluation

```bash
python experiments/PNU-NOMV/stage_abc/prepare_pnu_inputs.py
python experiments/PNU-NOMV/stage_d5/build_continuous_features.py
python experiments/PNU-NOMV/stage_d5/run_continuous_evaluation.py
python experiments/PNU-NOMV/stage_d5/validate_stage_d5.py
python experiments/PNU-NOMV/stage_d6f/run_pnu_transfer_stress.py
python experiments/PNU-NOMV/stage_d6f/validate_d6f.py
python experiments/PNU-NOMV/stage_d8b/run_pnu_sequential_stress.py
python experiments/PNU-NOMV/stage_d8b/validate_d8b.py
```

The compact PNU-NOMV evaluator retains the primary `time19` representation and the
`all40` sensitivity model. Optional QC34 exploration and figure-only window
ablation remain in `CREW-Full`; this does not change the primary time19 transfer
results reported in the manuscript.

## Citation and license

Citation metadata are provided in `CITATION.cff`. Add the article DOI and
archival DOI after they are available. The CREW source code is released under
the Apache License 2.0. PNU-NOMV is a separate dataset released under CC BY 4.0.
