"""Threshold labels, official inputs, full-mask crops and frozen risk aggregation."""
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from .data import crop_annotation
from .metrics import binary_auc, threshold_auc
from .run import infer_annotation, bank_fingerprint
from back_prop.common.features import SEMANTIC_NAMES, SoftRadiomics3D
from back_prop.model_v4.risk import ResidualRiskBank


class AnalysisTests(unittest.TestCase):
    def test_all_five_thresholds_and_single_class_endpoint(self):
        truth=np.arange(1,6)
        probability=np.eye(5)
        for t in range(1,6):
            stat,_=threshold_auc(truth,probability,t)
            self.assertEqual(stat['n_positive'],6-t)
            self.assertEqual(stat['n_negative'],t-1)
            self.assertEqual(stat['auc'],None if t==1 else 1.)
        stat,_=threshold_auc(truth,probability,5,'gt')
        self.assertIsNone(stat['auc'])

    def test_auc_uses_continuous_probabilities_and_keeps_ties(self):
        p=np.array([[1-s,0,0,0,s] for s in (.2,.3,.4,.49)])
        stat,_=threshold_auc([1,2,4,5],p,3)
        self.assertEqual(stat['auc'],1.)  # all argmax predictions are class 1
        tied,_=threshold_auc([1,2,4,5],np.full((4,5),.2),3)
        self.assertEqual(tied['auc'],.5)

    def test_rating_three_excluded_from_risk_only(self):
        stat,_=binary_auc([0,0,np.nan,1,1],[.1,.2,.99,.8,.9])
        self.assertEqual(stat['n_evaluated'],4)
        self.assertEqual(stat['n_excluded'],1)
        self.assertEqual(stat['auc'],1.)

    def test_crops_keep_entire_annotation_and_padding(self):
        image=np.arange(20*21*22,dtype=np.float32).reshape(20,21,22)
        mask=np.zeros_like(image,dtype=bool);mask[:12,:2,2:4]=True
        crop=crop_annotation(image,mask,(8,8,8))
        self.assertEqual(crop['shape'],[16,8,8])
        self.assertEqual(crop['mask_voxels'],int(mask.sum()))
        self.assertLess(crop['origin'][1],0)
        self.assertFalse((crop['mask'] & ~crop['valid']).any())
        self.assertIsNone(crop_annotation(image,np.zeros_like(mask),(8,8,8)))

    def test_official_semantics_exact_binary_radiomics_and_frozen_bank(self):
        torch.set_num_threads(1)
        class Semantics(nn.Module):
            def forward(self,ct,logits,valid):
                probability=torch.full((1,6,5),.2)
                return probability,torch.full((1,6),3.)
        bank=ResidualRiskBank(pool_size=2).eval()
        bank.count.fill_(2);bank.baseline_count.fill_(1)
        bank.radiomics_weights[0]=.01
        bank.weights[0,0]=.25;bank.weights[1,0]=.5
        bank.intercepts[1]=-1.
        mask=np.zeros((12,12,12),bool);mask[4:6,4:6,4:6]=True
        hu=np.full(mask.shape,-500.,np.float32)
        crop=crop_annotation(hu,mask,(8,8,8))
        row=dict(annotation_key='x',gt_malignancy=3,
                 **{'gt_'+name:2 for name in SEMANTIC_NAMES})
        before=bank_fingerprint(bank)
        parts=dict(semantics=Semantics(),risk=bank,radiomics=SoftRadiomics3D())
        predictions=infer_annotation(dict(row=row,crop=crop),{'V4':parts},'cpu','both')
        result=predictions[0]
        self.assertEqual(result['radiomics_volume_mm3'],8.)
        self.assertEqual(result['radiomics_mean_hu'],-500.)
        expected=torch.sigmoid(torch.tensor([.08+.25*2,.08+.5*2-1])).mean()
        self.assertAlmostEqual(result['malignancy_probability_official'],float(expected),places=6)
        self.assertNotEqual(result['malignancy_probability_official'],result['malignancy_probability_predicted'])
        self.assertEqual(before,bank_fingerprint(bank))


if __name__=='__main__':
    unittest.main()
