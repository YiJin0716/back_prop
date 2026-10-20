"""Check that visual comparisons preserve matching, missingness and geometry."""
import unittest

import numpy as np

from back_prop.evaluate.eval_package.types import (
    CaseEvaluation, GroundTruthNodule, MaskCrop, PredictedNodule, SEMANTIC_NAMES,
)
from .pipeline_report import comparison_rows, region_case


def prediction(query, mask, confidence=.9):
    return PredictedNodule(query, mask, mask, confidence,
        {n: 2.5 for n in SEMANTIC_NAMES}, {n: [.2]*5 for n in SEMANTIC_NAMES}, .4)


def truth(index, mask, rating=3.):
    return GroundTruthNodule(index, mask, (10+index,),
                             {n: [rating] for n in (*SEMANTIC_NAMES, 'malignancy')})


class PipelineReportTests(unittest.TestCase):
    def test_reader_matching_not_consensus_and_missing_values(self):
        empty = MaskCrop(np.zeros((0, 0, 0), bool), (0, 0, 0))
        one = MaskCrop(np.ones((2, 2, 2), bool), (5, 6, 7))
        two = MaskCrop(np.ones((2, 2, 2), bool), (25, 26, 27))
        extra = MaskCrop(np.ones((2, 2, 2), bool), (35, 36, 37))
        result = CaseEvaluation('test', np.zeros((50,)*3), np.eye(4), np.zeros((50,)*3),
            [prediction(0, one), prediction(1, extra), prediction(2, two, .1)],
            [truth(1, empty), truth(2, two, 4)], .6, {'object_threshold': .5})
        rows = comparison_rows(result, {1: [one], 2: [two]})
        self.assertEqual([r['status'] for r in rows], ['Matched', 'Missed nodule', 'Unmatched prediction'])
        self.assertEqual(rows[0]['closest_annotation_id'], 11)
        self.assertEqual(rows[0]['overlap_closest']['iou'], 1)
        self.assertEqual(rows[0]['overlap_consensus']['iou'], 0)
        self.assertEqual(rows[0]['official_labels'], {'exclude_3': None, 'ge_3': 1, 'gt_3': 0})
        self.assertIsNone(rows[1]['risk'])
        self.assertIsNone(rows[1]['semantic'])
        self.assertIsNone(rows[2]['official_mean'])
        self.assertEqual([r['query_id'] for r in rows], [0, None, 1])

    def test_crop_keeps_prediction_and_each_reader_in_world_coordinates(self):
        shape = (100, 110, 120)
        affine = np.diag([.8, 1.2, 1.5, 1.]); affine[:3, 3] = [-90, 40, -60]
        p = MaskCrop(np.ones((4, 4, 4), bool), (40, 42, 44))
        gt = MaskCrop(np.ones((4, 4, 4), bool), (45, 42, 44))
        reader = MaskCrop(np.ones((4, 4, 4), bool), (50, 42, 44))
        result = CaseEvaluation('geometry', np.zeros(shape), affine, np.zeros(shape),
            [prediction(0, p)], [truth(1, gt)], .5)
        case = region_case(result, {'prediction_index': 0, 'gt_index': 0}, {1: [reader]})
        from roivue import view
        scene = view.load(case).region_scene(0)
        import nibabel as nib
        import gzip
        layers = {layer.name: nib.Nifti1Image.from_bytes(gzip.decompress(layer.overlay)) for layer in scene.layers}
        for name, crop in [('Predicted mask', p), ('Official consensus', gt), ('Official annotation 11', reader)]:
            image = layers[name]
            points = np.argwhere(np.asarray(image.dataobj) > 0)
            self.assertEqual(len(points), 64)
            np.testing.assert_allclose(image.affine @ np.r_[points.mean(0), 1],
                                       affine @ np.r_[np.asarray(crop.origin)+1.5, 1], atol=1e-5)


if __name__ == '__main__':
    unittest.main()
