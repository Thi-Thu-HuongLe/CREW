"""Deterministic API demonstration; not a physical fault experiment."""

import numpy as np

from crew import FrozenRule, apply_crew, robust_scale


rng = np.random.default_rng(7)
normal_train = np.maximum(rng.lognormal(mean=0.0, sigma=0.15, size=500) - 0.8, 0.0)
stream = np.maximum(rng.lognormal(mean=0.0, sigma=0.15, size=300) - 0.8, 0.0)
stream[220:] += np.linspace(0.0, 8.0, 80)

train_log = np.log1p(normal_train)
scale = robust_scale(train_log - np.median(train_log))
rule = FrozenRule(alpha=0.2, threshold=3.0, persistence=3)
result = apply_crew(stream, commissioning_count=30, scale=scale, rule=rule)

print(f"CREW confirmation index: {result.confirmation_index}")
