"""Geometry, official provenance, missing counterparts, and real V3 forward."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import nibabel as nib
import numpy as np

from back_prop.evaluate.eval_package import (
    CaseEvaluation, GroundTruthNodule, MaskCrop, OfficialAnnotations, PredictedNodule,
    JointModelEvaluator, evaluate_case, load_joint_model, load_result, load_test_cases, malignancy,
    nodule_segmentation, plot_malignancy, plot_nodule_segmentation,
    plot_screening_segmentation, plot_semantic_features, save_result,
    screening_segmentation, semantic_features,
)
from back_prop.evaluate.eval_package.annotations import load_image
from back_prop.evaluate.eval_package.types import RATING_NAMES, SEMANTIC_NAMES


def ground_truth(nodule_id=1, origin=(1, 1, 1), ratings=(2, 4)):
    return GroundTruthNodule(nodule_id, MaskCrop(np.ones((2, 2, 2), bool), origin),
        (10, 11), {name: list(ratings) for name in RATING_NAMES})


def prediction(query_id=7, origin=(1, 1, 1)):
    mask = MaskCrop(np.ones((2, 2, 2), bool), origin)
    return PredictedNodule(query_id, mask, MaskCrop(mask.values.astype(float), origin),
        0.9, {name: 4.0 for name in SEMANTIC_NAMES},
        {name: [0, 0, 0, 1, 0] for name in SEMANTIC_NAMES}, 0.8, [0.7, 0.9])


def result_fixture():
    return CaseEvaluation("test__scan1", np.zeros((12, 12, 12), np.float32), np.eye(4),
        np.zeros((12, 12, 12), bool), [prediction(), prediction(12, (9, 9, 9))],
        [ground_truth(), ground_truth(2, (5, 5, 5), (4, 5))], 0.83)


class ComparisonTests(unittest.TestCase):
    def test_matching_preserves_misses_and_unmatched_predictions(self):
        rows = nodule_segmentation(result_fixture())
        self.assertEqual([r["status"] for r in rows], ["matched", "missed_gt", "unmatched_prediction"])
        self.assertEqual(rows[0]["query_id"], 7)
        self.assertEqual(rows[0]["metrics"]["dice"], 1)
        self.assertIsNone(rows[1]["prediction"])
        self.assertIsNone(rows[2]["ground_truth"])

    def test_one_to_one_no_zero_overlap_matches(self):
        r = result_fixture()
        r.predictions = [prediction(), prediction(8)]
        r.minimum_iou = 0
        rows = nodule_segmentation(r)
        self.assertEqual(sum(x["status"] == "matched" for x in rows), 1)
        self.assertEqual(sum(x["status"] == "missed_gt" for x in rows), 1)

    def test_negative_roi_paste_has_no_wraparound(self):
        crop = MaskCrop(np.ones((4, 3, 3), bool), (-2, 1, 1))
        grid = crop.on_grid((5, 5, 5))
        self.assertEqual(int(grid.sum()), 18)
        self.assertFalse(grid[-1].any())
        np.testing.assert_array_equal(crop.on_grid((2, 2, 2), (0, 1, 1)), np.ones((2, 2, 2)))

    def test_whole_ct_final_and_screening_differ(self):
        r = result_fixture()
        self.assertEqual(screening_segmentation(r)["metrics"]["predicted_voxels"], 16)
        self.assertEqual(screening_segmentation(r, stage="screening")["metrics"]["predicted_voxels"], 0)
        self.assertEqual(screening_segmentation(r)["metrics"]["gt_voxels"], 16)

    def test_ratings_and_unknown_binary_labels_are_retained(self):
        r = result_fixture()
        rows = semantic_features(r)
        self.assertEqual(rows[0]["gt_reader_ratings"]["margin"], [2, 4])
        self.assertEqual(rows[0]["gt_reader_histograms"]["margin"], [0, .5, 0, .5, 0])
        self.assertIsNone(malignancy(r)["nodules"][0]["ground_truth"])
        self.assertEqual(malignancy(r)["scan"]["ground_truth"], 1)
        r.ground_truth = r.ground_truth[:1]
        self.assertIsNone(malignancy(r)["scan"]["ground_truth"])
        self.assertIsNone(rows[1]["prediction"])
        self.assertIsNone(rows[2]["ground_truth"])

    def test_empty_predictions_gt_and_empty_consensus(self):
        r = result_fixture()
        r.predictions, r.ground_truth = [], []
        self.assertEqual(nodule_segmentation(r), [])
        self.assertEqual(screening_segmentation(r)["metrics"]["dice"], 1)
        self.assertEqual(malignancy(r)["scan"]["ground_truth"], 0)
        r.ground_truth = [ground_truth()]
        r.ground_truth[0].mask = MaskCrop(np.zeros((0, 0, 0), bool), (0, 0, 0))
        self.assertTrue(nodule_segmentation(r)[0]["gt_mask_empty"])

    def test_export_round_trip_and_all_four_plots(self):
        import matplotlib.pyplot as plt
        r = result_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            folder = save_result(r, temporary, nifti=True)
            restored = load_result(folder)
            np.testing.assert_array_equal(screening_segmentation(restored)["prediction"],
                                          screening_segmentation(r)["prediction"])
            self.assertEqual(malignancy(restored), malignancy(r))
            self.assertEqual(semantic_features(restored), semantic_features(r))
            for filename in ("screening.png", "final.png", "semantic_features.png", "malignancy.png",
                             "nodule_id_1.png", "nodule_id_2.png", "query_id_12.png"):
                self.assertGreater((folder / filename).stat().st_size, 100)
            np.testing.assert_array_equal(nib.load(folder / "official_gt.nii.gz").affine, r.affine)
        r.predictions, r.ground_truth = [], []
        for fn in (plot_screening_segmentation, plot_semantic_features, plot_malignancy):
            plt.close(fn(r))
        with self.assertRaises(ValueError):
            plot_nodule_segmentation(r, query_id=99)


class OfficialIOTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.case = dict(patient_id="test", scan_id="1", image=str(self.root / "ct.nii.gz"))
        self.affine = np.diag([-2., 1., 1., 1.])
        self.affine[0, 3] = 14
        nib.save(nib.Nifti1Image(np.zeros((8, 8, 8), np.float32), self.affine), self.case["image"])
        fields = ["patient_id", "scan_index", "annotation_index", "nodule_id", *RATING_NAMES]
        self.rows = [dict(patient_id="test", scan_index="1", annotation_index=str(i),
                         nodule_id="1", **{name: score for name in RATING_NAMES})
                     for i, score in ((10, 2), (11, 4))]
        for filename, rows in (("official.csv", self.rows),
                               ("identity.csv", [{**r, **{name: 1 for name in RATING_NAMES}} for r in self.rows])):
            with (self.root / filename).open("w", newline="") as f:
                writer = csv.DictWriter(f, fields); writer.writeheader(); writer.writerows(rows)
        for ann in (10, 11):
            mask = np.zeros((8, 8, 8), np.uint8)
            mask[2:4, 2:4, 2:4] = 1
            if ann == 10:
                mask[0, 0, 0] = 1  # only one reader: excluded by two-vote consensus
            nib.save(nib.Nifti1Image(mask, self.affine), self.root / f"test__scan1__ann{ann}_mask.nii.gz")

    def annotations(self):
        return OfficialAnnotations(self.root / "official.csv", self.root / "identity.csv",
                                   self.root, cases=[self.case])

    def test_official_scores_override_identity_csv_scores_and_align_masks(self):
        a = self.annotations()
        hu, affine, source, zoom = load_image(self.case)
        nodules = a.load_nodules("test__scan1", source, zoom)
        self.assertEqual(nodules[0].reader_ratings["margin"], [2, 4])
        self.assertEqual(int(nodules[0].mask.values.sum()), 16)
        self.assertEqual(hu.shape, (16, 8, 8))
        self.assertAlmostEqual(affine[0, 0], 14/15)
        self.assertEqual(nodules[0].mask.origin, (8, 2, 2))

    def test_missing_mask_is_an_error(self):
        (self.root / "test__scan1__ann10_mask.nii.gz").unlink()
        _, _, source, zoom = load_image(self.case)
        with patch("back_prop.common.base_data.time.sleep"):
            with self.assertRaises(FileNotFoundError):
                self.annotations().load_nodules("test__scan1", source, zoom)

    def test_misaligned_official_mask_is_an_error(self):
        _, _, source, zoom = load_image(self.case)
        path = self.root / "test__scan1__ann10_mask.nii.gz"
        nib.save(nib.Nifti1Image(np.ones((8, 8, 8), np.uint8), np.eye(4)), path)
        with self.assertRaisesRegex(ValueError, "geometry mismatch"):
            self.annotations().load_nodules("test__scan1", source, zoom)

    def test_missing_official_annotation_is_an_error(self):
        with (self.root / "official.csv").open() as f:
            reader = csv.DictReader(f); fields = reader.fieldnames; row = next(reader)
        with (self.root / "official.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fields); writer.writeheader(); writer.writerow(row)
        with self.assertRaisesRegex(ValueError, "absent official"):
            self.annotations()

    def test_test_split_membership_and_patient_leakage(self):
        manifest = self.root / "fold.json"
        manifest.write_text(json.dumps({"training": [], "testing": [self.case]}))
        evaluator = JointModelEvaluator(None, manifest, annotations=self.annotations())
        with self.assertRaisesRegex(ValueError, "not in"):
            evaluator.evaluate("other__scan2")
        with self.assertRaisesRegex(ValueError, "outside"):
            list(evaluator.evaluate_test_set(["other__scan2"]))
        manifest.write_text(json.dumps({"training": [self.case], "testing": [self.case]}))
        with self.assertRaisesRegex(ValueError, "overlap"):
            load_test_cases(manifest)

    def test_real_v3_v4_variants_forward_without_gt_and_without_bank_updates(self):
        import importlib.util
        import torch
        from back_prop.tests.model_fixtures import TinySegmenter
        from back_prop.tests.model_fixtures import TinyEncoder
        from back_prop.common.semantic_model import WholeCTJointModelV3
        if importlib.util.find_spec("back_prop.model_v3_hard") is None:
            WholeCTJointModelV3Hard = None
        else:
            from back_prop.model_v3_hard.model import WholeCTJointModelV3Hard
        from back_prop.common.annealed_model import WholeCTJointModelV3 as AnnealModel
        from back_prop.model_v4.model import WholeCTJointModelV4

        torch.set_num_threads(2)
        config = dict(window_size=(8,8,8), screen_shape=(8,8,8), roi_shape=(8,8,8),
                      num_queries=2, detr_hidden_dim=16, detr_coarse_shape=(4,4,4), detr_nheads=4,
                      detr_encoder_layers=1, detr_decoder_layers=1, vista_feature_dim=1,
                      context_channels=2, medicalnet_roi_size=8)
        for cls in (WholeCTJointModelV3, WholeCTJointModelV3Hard, AnnealModel, WholeCTJointModelV4):
            with self.subTest(model=cls.__module__ if cls is not None else "V3_hard"):
                if cls is None:
                    self.skipTest("The optional model_v3_hard experiment directory was removed")
                extra = {"baseline_l2": .017} if cls is WholeCTJointModelV4 else {}
                model = cls(**config, **extra, refiner_base_channels=2,
                    segmenter_factory=TinySegmenter, encoder_factory=TinyEncoder,
                    use_checkpoint=False)
                model.rashomon.count.fill_(1)
                if cls is WholeCTJointModelV4:
                    model.rashomon.baseline_count.fill_(1)
                if cls is AnnealModel:
                    model.set_temperature(.625)
                import importlib
                checkpoint = self.root / "checkpoint.pt"
                torch.save(dict(model=model.state_dict(), epoch=17,
                    architecture=importlib.import_module(cls.__module__).ARCHITECTURE,
                    config=dict(config, refiner_base=2, mask_temperature=.25,
                                manifest=self.root / "fold.json")), checkpoint)
                with patch("back_prop.evaluate.eval_package.inference._uninitialized_segmenter", TinySegmenter), \
                     patch("back_prop.evaluate.eval_package.inference._uninitialized_encoder", TinyEncoder):
                    model = load_joint_model(checkpoint, device="cpu")
                # Actual training checkpoints store argparse Path objects.
                # Export provenance must remain JSON serializable.
                json.dumps(model.evaluation_metadata, allow_nan=False)
                if cls is WholeCTJointModelV4:
                    self.assertEqual(model.rashomon.baseline_l2, .017)
                    self.assertEqual(int(model.rashomon.baseline_count), 1)
                if cls is AnnealModel:
                    self.assertEqual(float(model.mask_temperature), .625)
                if cls is WholeCTJointModelV3Hard:
                    self.assertEqual(float(model.radiomics.mask_temperature), .25)
                with patch.object(model, "forward", wraps=model.forward) as forward:
                    r = evaluate_case(model, self.case, annotations=self.annotations(), object_threshold=0)
                self.assertIsNone(forward.call_args.kwargs["batch"])
                self.assertFalse(forward.call_args.kwargs["update_bank"])
                self.assertEqual(len(r.predictions), 2)
                self.assertTrue(r.metadata["bank_unchanged"])
                self.assertEqual(r.ground_truth[0].means["malignancy"], 3)
                empty = evaluate_case(model, self.case, annotations=self.annotations(), object_threshold=1)
                self.assertEqual(empty.predictions, [])
                model.rashomon.allowed_cases = {"test__scan999"}
                with self.assertRaisesRegex(ValueError, "used to fit"):
                    evaluate_case(model, self.case, annotations=self.annotations())

    def test_v2_prediction_and_gt_adapter(self):
        import torch
        from back_prop.common.coarse_model import WholeCTJointModelV2
        from back_prop.tests.model_fixtures import TinySegmenter, TinyRadiomics, TinyOrdinal
        torch.set_num_threads(2)
        model = WholeCTJointModelV2(window_size=(8,8,8), screen_shape=(8,8,8), roi_shape=(8,8,8),
            num_queries=2, detr_hidden_dim=16, detr_coarse_shape=(4,4,4), detr_nheads=4,
            detr_encoder_layers=1, detr_decoder_layers=1, vista_feature_dim=1,
            context_channels=2, refiner_base_channels=2, segmenter_factory=TinySegmenter,
            radiomics_factory=TinyRadiomics, ordinal_heads_factory=TinyOrdinal, use_checkpoint=False)
        r = evaluate_case(model, self.case, annotations=self.annotations(), object_threshold=0)
        self.assertEqual(len(r.predictions), 2)
        self.assertIn("malignancy", r.predictions[0].semantic_features)
        self.assertEqual(r.ground_truth[0].means["malignancy"], 3)


if __name__ == "__main__":
    unittest.main()
