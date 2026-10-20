"""Independent edge cases for detection, labels, probabilities and geometry."""
import unittest
import numpy as np
import torch

from back_prop.common.rashomon import scan_logits_per_model
from back_prop.evaluate.eval_package.types import MaskCrop
from .metrics import (binary_metrics, semantic_metrics, match_predictions, froc,
                      malignancy_label, volume_label, scan_probability, diameter_mm)
from .run import reader_composites
from .report import size_analysis


class AnalysisTests(unittest.TestCase):
    def test_uncertainty_and_three_malignancy_modes(self):
        self.assertEqual([malignancy_label(3, m) for m in ('exclude_3','ge_3','gt_3')], [None,1,0])
        self.assertEqual(volume_label([None,1,0]),1)
        self.assertIsNone(volume_label([None,0]))
        self.assertEqual(volume_label([]),0)
        self.assertEqual(malignancy_label(3.25,'exclude_3'),1)

    def test_auc_requires_classes_but_accuracy_remains_defined(self):
        row=binary_metrics([1,1,None],[.8,.2,.9])
        self.assertIsNone(row['auc']);self.assertEqual(row['accuracy'],.5)
        self.assertEqual(row['excluded'],1);self.assertEqual(row['fn'],1)
        with self.assertRaises(ValueError):binary_metrics([.5],[.2])
        row=binary_metrics([0,1],[.1,.9]);self.assertEqual(row['auc'],1)

    def test_threshold_auc_uses_probability_tail_and_consensus_not_rounded(self):
        y=np.array([[1]*6,[2.5]*6,[4]*6])
        p=np.zeros((3,6,5));p[0,:,0]=1;p[1,:,1:3]=.5;p[2,:,3]=1
        result=semantic_metrics(y,p)[0]
        self.assertIsNone(result['thresholds'][0]['auc'])
        self.assertEqual(result['thresholds'][2]['positives'],1)
        self.assertEqual(result['thresholds'][2]['auc'],1)

    def test_matching_maximizes_count_before_overlap(self):
        pairs,fp=match_predictions([[.9,.11],[.11,0]],[.8,.7])
        self.assertEqual(set(pairs),{(0,1),(1,0)});self.assertEqual(fp,[])
        pairs,fp=match_predictions([[.8],[.7],[0]],[.9,.8,.7])
        self.assertEqual(pairs,[(0,0)]);self.assertEqual(fp,[1,2])
        pairs,fp=match_predictions([[0]],[1.],minimum_iou=0)
        self.assertEqual(pairs,[]);self.assertEqual(fp,[0])

    def test_froc_includes_missed_nodules_and_negative_scans(self):
        cases=[dict(predictions=[dict(object_probability=.9),dict(object_probability=.4)],
                    nodules=[{},{}],best_reader_iou=[[.8,0],[.5,0]]),
               dict(predictions=[dict(object_probability=.8)],nodules=[],best_reader_iou=[[]])]
        curve=froc(cases)
        self.assertEqual(curve['scans'],2);self.assertEqual(curve['nodules'],2)
        self.assertEqual(curve['points'][-1]['sensitivity'],.5)
        self.assertEqual(curve['points'][-1]['false_positives_per_scan'],1)

    def test_noisy_or_is_per_bank_member_not_mean_nodule_risk(self):
        ensembles=torch.tensor([[.9,.1],[.1,.9]])
        obj=torch.tensor([.8,.7])
        expected=float(scan_logits_per_model(torch.logit(ensembles),torch.logit(obj)).sigmoid().mean())
        rows=[dict(ensemble_probabilities=p.tolist(),object_probability=float(o)) for p,o in zip(ensembles,obj)]
        self.assertAlmostEqual(scan_probability(rows),expected,places=6)
        self.assertEqual(scan_probability([]),1e-6)
        self.assertGreater(abs(scan_probability(rows)-(1-(1-.8*.5)*(1-.7*.5))),.05)

    def test_reader_union_and_intersection_include_empty_reader(self):
        readers=[dict(mask=MaskCrop(np.ones((2,2,2),bool),(0,0,0))),
                 dict(mask=MaskCrop(np.ones((2,2,2),bool),(1,0,0)))]
        union,intersection=reader_composites(readers)
        self.assertEqual(union.values.sum(),12);self.assertEqual(intersection.values.sum(),4)
        readers.append(dict(mask=MaskCrop(np.zeros((0,0,0),bool),(0,0,0))))
        self.assertEqual(reader_composites(readers)[1].values.sum(),0)

    def test_physical_size_uses_affine_and_voxel_extent(self):
        mask=np.ones((2,3,4),bool)
        affine=np.diag([2,1,3,1])
        self.assertAlmostEqual(diameter_mm(mask,affine),13.)
        self.assertEqual(diameter_mm(np.zeros((2,2,2),bool),affine),0)

    def test_size_filter_keeps_missed_malignancies_in_denominator(self):
        predictions=[dict(object_probability=.9,diameter_mm=d,malignancy_probability=p,
                          ensemble_probabilities=[p,p]) for d,p in [(6.,.9),(8.,.2),(2.,.95)]]
        case=dict(predictions=predictions,object_threshold=.5,minimum_iou=.1,
            nodules=[dict(malignancy_mean=4.,diameter_mm=6.),dict(malignancy_mean=2.,diameter_mm=8.)],
            best_reader_iou=[[.8,0],[0,.8],[0,0]],
            matches=[dict(prediction_index=0,nodule_index=0),dict(prediction_index=1,nodule_index=1)])
        filtering,cumulative=size_analysis([case])
        rows={r['cutoff_mm']:r for r in filtering if r['mode']=='exclude_3'}
        self.assertEqual(rows[5]['removed_predictions'],1)
        self.assertEqual(rows[5]['removed_predicted_malignant'],1)
        self.assertEqual(rows[5]['additional_malignant_nodules_lost'],0)
        self.assertEqual(rows[6]['additional_malignant_nodules_lost'],1)
        self.assertEqual(rows[6]['positives'],1)
        self.assertEqual(rows[6]['sensitivity'],0.)
        self.assertEqual(rows[6]['specificity'],1.)
        small=next(r for r in cumulative if r['mode']=='exclude_3' and r['upper_diameter_mm']==5)
        self.assertEqual(small['n'],0);self.assertIsNone(small['sensitivity'])


if __name__=='__main__':unittest.main()
