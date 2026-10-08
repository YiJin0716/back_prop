"""Regression checks for mask support and physical crop coordinates."""
import gzip
import unittest

import nibabel as nib
import numpy as np
from roivue import view as cv

from back_prop.evaluate.eval_package.comparisons import nodule_segmentation
from back_prop.evaluate.eval_package.types import (
    CaseEvaluation, GroundTruthNodule, MaskCrop, PredictedNodule,
)
from back_prop.evaluate.visualize_mask.roivue_helpers import (
    PREDICTION, REFERENCE, nodule_case, whole_scan_case,
)


def fixture():
    affine = np.array([[-2., 0., 0., 80.], [0., 1.5, 0., -30.],
                       [0., 0., 2.5, 10.], [0., 0., 0., 1.]])
    pred = MaskCrop(np.ones((4, 4, 4), bool), (10, 11, 12))
    gt = MaskCrop(np.ones((3, 3, 3), bool), (10, 11, 12))
    prediction = PredictedNodule(7, pred, pred, .9, {}, {}, .5)
    truth = GroundTruthNodule(101, gt, (10, 11), {})
    return CaseEvaluation('test__scan1', np.arange(40**3, dtype=np.float32).reshape((40,)*3),
                          affine, np.zeros((40,)*3, bool), [prediction], [truth], .5)


class RoivueGeometryTests(unittest.TestCase):
    def test_cropped_affine_and_image_retain_world_coordinates(self):
        result = fixture()
        case = nodule_case(result, nodule_segmentation(result)[0], min_mm=16, padding_mm=2)
        start = np.array(case.meta['crop_origin_xyz'])
        local = np.array([2, 3, 1])
        np.testing.assert_allclose(case.affine @ np.r_[local, 1],
                                   result.affine @ np.r_[start+local, 1])
        self.assertEqual(case.image[tuple(local)], result.image_hu[tuple(start+local)])
        np.testing.assert_array_equal(case.masks[PREDICTION],
                                      result.predictions[0].mask.on_grid(case.image.shape, start))

    def test_roivue_region_retains_prediction_outside_gt(self):
        result = fixture()
        case = nodule_case(result, nodule_segmentation(result)[0], min_mm=16)
        scene = cv.load(case).region_scene(0, min_mm=16)
        masks = {layer.name: nib.Nifti1Image.from_bytes(gzip.decompress(layer.overlay))
                 .get_fdata().astype(bool) for layer in scene.layers}
        self.assertEqual(int(masks[PREDICTION].sum()), 64)
        self.assertEqual(int(masks[REFERENCE].sum()), 27)
        self.assertEqual(int((masks[PREDICTION] & ~masks[REFERENCE]).sum()), 37)

    def test_missing_sides_and_empty_queries_are_explicit(self):
        result = fixture()
        result.predictions[0].mask = MaskCrop(np.ones((2, 2, 2), bool), (-1, 0, 0))
        rows = nodule_segmentation(result)
        self.assertEqual([r['status'] for r in rows], ['missed_gt', 'unmatched_prediction'])
        missed = nodule_case(result, rows[0], min_mm=12)
        self.assertEqual(set(missed.masks), {REFERENCE})
        unmatched = nodule_case(result, rows[1], min_mm=12)
        self.assertEqual(set(unmatched.masks), {PREDICTION})
        self.assertEqual(int(unmatched.masks[PREDICTION].sum()), 4)
        result.predictions[0].mask.values[:] = False
        rows = nodule_segmentation(result)
        self.assertEqual(len(rows), 2)
        self.assertIsNone(nodule_case(result, rows[1]))

    def test_whole_scan_uses_saved_grid_and_union_metrics(self):
        result = fixture()
        case, metrics = whole_scan_case(result)
        np.testing.assert_array_equal(case.affine, result.affine)
        self.assertEqual(metrics['dice'], 2*27/(64+27))
        self.assertEqual(int(case.masks[PREDICTION].sum()), 64)
        self.assertFalse(case.regions.any())


if __name__ == '__main__':
    unittest.main()
