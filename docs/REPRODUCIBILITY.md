# Reproducibility and leakage controls

## Fixed data roles

- Split complete assets/runs before creating any model inputs.
- Fit representations, robust centers/scales, and train-selectable dynamics on
  training assets only.
- Select thresholds and persistence on calibration assets only.
- Open test trajectories only after the rule is frozen.
- Never randomly split overlapping windows from one PNU-NOMV motor recording across roles.

## Causal timing

CREW stores two indices when persistence is met. The evidence-start index is the
first member of the consecutive exceedance run and is retrospective. The actual
causal warning time is the later confirmation index. Reported warning decisions
and terminal lead use the confirmation index.

## Dataset-specific evidence

- XJTU-SY and PRONOSTIA provide known terminal outcomes for test trajectories.
- NASA-IMS known outcomes are mapped at bearing level; outcome-unknown assets are
  excluded from performance denominators.
- PNU-NOMV supplies independent normal-only exposure for nuisance-warning and
  domain-shift evaluation, not fault recall.

## Determinism

Scripts use deterministic chronological selection and fixed random seeds where
bootstrap sampling is required. Stage validation scripts verify array alignment,
hashes, split roles, causal state updates, and output consistency.

## Generated artifacts

Generated `outputs/`, feature arrays, plots, and reports are ignored by Git.
This compact repository contains computation only; frozen result tables and
paper-figure generation remain in the private `CREW-Full` archive.
