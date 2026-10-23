import unittest

import numpy as np

from crew.sequential import FrozenRule, apply_crew, causal_ewma, first_persistent, robust_scale


class SequentialTests(unittest.TestCase):
    def test_robust_scale_is_positive_for_constant_input(self) -> None:
        self.assertEqual(robust_scale(np.ones(10)), 1e-3)

    def test_causal_ewma_uses_no_future_values(self) -> None:
        values = np.array([0.0, 1.0, 100.0])
        full = causal_ewma(values, center=0.0, scale=1.0, alpha=0.5)
        prefix = causal_ewma(values[:2], center=0.0, scale=1.0, alpha=0.5)
        np.testing.assert_allclose(full[:2], prefix)

    def test_persistence_returns_start_and_confirms_later(self) -> None:
        flags = np.array([False, True, True, True, False])
        self.assertEqual(first_persistent(flags, 3), 1)

        score = np.array([0.0, 0.0, 10.0, 10.0, 10.0, 0.0])
        result = apply_crew(
            score,
            commissioning_count=2,
            scale=1.0,
            rule=FrozenRule(alpha=1.0, threshold=1.0, persistence=3),
        )
        self.assertEqual(result.evidence_start_index, 2)
        self.assertEqual(result.confirmation_index, 4)
        self.assertFalse(result.latched_warning[3])
        self.assertTrue(result.latched_warning[4])


if __name__ == "__main__":
    unittest.main()
