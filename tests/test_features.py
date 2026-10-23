import unittest

import numpy as np

from crew.features import FEATURE_NAMES, extract_features, feature_indices_for_axes


class FeatureTests(unittest.TestCase):
    def test_three_axis_feature_shape_and_mask(self) -> None:
        time = np.arange(1000, dtype=float) / 1000.0
        samples = np.column_stack(
            (
                np.sin(2 * np.pi * 50 * time),
                np.sin(2 * np.pi * 80 * time),
                np.sin(2 * np.pi * 120 * time),
            )
        )
        features, mask = extract_features(samples, sampling_rate_hz=1000.0)
        self.assertEqual(features.shape, (40,))
        self.assertEqual(mask.shape, (40,))
        self.assertTrue(mask.all())
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(len(FEATURE_NAMES), 40)

    def test_one_axis_preserves_unavailable_features(self) -> None:
        samples = np.arange(100, dtype=float)
        features, mask = extract_features(samples, sampling_rate_hz=1000.0)
        self.assertEqual(mask.sum(), 14)
        self.assertTrue(np.isnan(features[14:]).all())
        np.testing.assert_array_equal(feature_indices_for_axes(1), np.arange(14))


if __name__ == "__main__":
    unittest.main()
