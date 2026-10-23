import unittest

import numpy as np

from crew.pnu_temporal import episode_ranges, persistence_flags, quantile_higher


class PnuTemporalTests(unittest.TestCase):
    def test_k_of_n_is_causal(self) -> None:
        flags = np.array([True, False, True, True, False])
        observed = persistence_flags(flags, k=2, n=3)
        np.testing.assert_array_equal(observed, [False, False, True, True, True])

    def test_episode_ranges_are_inclusive(self) -> None:
        flags = np.array([False, True, True, False, True])
        self.assertEqual(episode_ranges(flags), [(1, 2), (4, 4)])

    def test_higher_quantile_matches_locked_rule(self) -> None:
        self.assertEqual(quantile_higher(np.arange(1, 11), 0.9), 9.0)


if __name__ == "__main__":
    unittest.main()
