"""Small dependency-free statistical utilities used by the experiments."""

from __future__ import annotations

import math


def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
) -> tuple[float, float, float]:
    """Return estimate and two-sided Wilson interval for a binomial proportion."""

    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("require 0 <= successes <= total and total > 0")
    estimate = successes / total
    denominator = 1.0 + z * z / total
    center = (estimate + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(
        estimate * (1.0 - estimate) / total + z * z / (4.0 * total * total)
    ) / denominator
    return estimate, max(0.0, center - half), min(1.0, center + half)


def exact_mcnemar_two_sided(primary_only: int, comparator_only: int) -> float:
    """Return the exact two-sided McNemar p-value for discordant pairs."""

    if primary_only < 0 or comparator_only < 0:
        raise ValueError("counts must be non-negative")
    discordant = primary_only + comparator_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(min(primary_only, comparator_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)
