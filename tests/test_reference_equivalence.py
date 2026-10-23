import importlib.util
import unittest
from pathlib import Path

import numpy as np

from crew.features import extract_features
from crew.sequential import causal_ewma, first_persistent, robust_scale


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReferenceEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.feature_reference = load_module(
            "feature_schema_reference",
            "experiments/Real-Fault-Benchmarks/stage_d6b/feature_schema.py",
        )
        cls.crew_reference = load_module(
            "crew_reference",
            "experiments/Real-Fault-Benchmarks/stage_d6c/build_health_indicator.py",
        )

    def test_feature_implementation_matches_frozen_experiment(self) -> None:
        rng = np.random.default_rng(20261023)
        samples = rng.normal(size=(2048, 3))
        actual_feature, actual_mask = extract_features(samples, 2048.0)
        expected_feature, expected_mask = self.feature_reference.extract_features(samples, 2048.0)
        np.testing.assert_array_equal(actual_mask, expected_mask)
        np.testing.assert_allclose(actual_feature, expected_feature, rtol=0.0, atol=0.0)

    def test_sequential_implementation_matches_frozen_experiment(self) -> None:
        values = np.linspace(0.0, 5.0, 100) ** 1.3
        self.assertEqual(robust_scale(values), self.crew_reference.robust_scale(values))
        actual = causal_ewma(values, center=0.7, scale=1.2, alpha=0.2)
        expected = self.crew_reference.causal_health(values, center=0.7, scale=1.2, alpha=0.2)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
        flags = actual > 1.0
        self.assertEqual(first_persistent(flags, 3), self.crew_reference.first_persistent(flags, 3))


if __name__ == "__main__":
    unittest.main()
