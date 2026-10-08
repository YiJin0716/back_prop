import numpy as np
import unittest

from back_prop.model_v4.compat_radiomics64 import FEATURE_NAMES, extract_radiomics64


class Radiomics64Tests(unittest.TestCase):
    def test_feature_vector_has_published_64_dimensions_and_is_finite(self):
        rows, cols = np.ogrid[:41, :45]
        mask = ((rows - 20) / 9) ** 2 + ((cols - 22) / 12) ** 2 <= 1
        image = -850.0 + 18.0 * rows + 7.0 * cols
        features = extract_radiomics64(image, mask)
        self.assertEqual(tuple(features), FEATURE_NAMES)
        self.assertEqual(len(features), 64)
        self.assertTrue(np.isfinite(list(features.values())).all())


    def test_shape_and_size_relationships_for_disk_like_mask(self):
        rows, cols = np.ogrid[:33, :33]
        mask = (rows - 16) ** 2 + (cols - 16) ** 2 <= 8**2
        image = np.zeros(mask.shape)
        features = extract_radiomics64(image, mask)
        self.assertEqual(features["Area"], float(mask.sum()))
        self.assertLessEqual(features["Roughness"], 1.0 + 1e-12)
        self.assertGreater(features["Roughness"], 0)
        self.assertLessEqual(features["Solidity"], 1.0)
        self.assertGreater(features["Solidity"], 0)
        self.assertGreaterEqual(features["MajorAxisLength"], features["MinorAxisLength"])


if __name__ == "__main__":
    unittest.main()
