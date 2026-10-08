import copy
import math
import unittest
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from back_prop.common.features import SoftRadiomics3D, MedicalNetSemantics, FEATURE_NAMES, SEMANTIC_NAMES
from back_prop.common.rashomon import ContinuousRashomonBank, fit_continuous_pool, scan_logits_per_model
from back_prop.common.semantic_model import WholeCTJointModelV3
from back_prop.common.semantic_loss import WholeCTCriterionV3
from back_prop.tests.model_fixtures import TinySegmenter, make_batch


class TinyEncoder(nn.Module):
    output_dim = 4
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 4, 3, padding=1)
    def forward(self, x):
        return self.conv(x).tanh().mean((2, 3, 4))


class V3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_soft_radiomics_formulas_and_gradients(self):
        ct = torch.full((1,1,4,4,4), 100.)
        logits = torch.zeros_like(ct, requires_grad=True)
        valid = torch.ones_like(ct)
        valid[:,:,:1] = 0
        f = SoftRadiomics3D()(ct, logits, valid)
        self.assertEqual(f.shape, (1,18))
        torch.testing.assert_close(f[0,0], torch.tensor(24.))
        torch.testing.assert_close(f[0,8], torch.tensor(100.))
        torch.testing.assert_close(f[0,10], torch.tensor(0.))
        g = torch.autograd.grad(f[0,0], logits, retain_graph=True)[0]
        torch.testing.assert_close(g, valid*.25)
        f.sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        for val in (-100., 100.):
            x = torch.full_like(ct, val, requires_grad=True)
            o = SoftRadiomics3D()(ct, x, valid)
            o.sum().backward()
            self.assertTrue(torch.isfinite(o).all() and torch.isfinite(x.grad).all())

    def test_medicalnet_semantics_independent_of_radiomics(self):
        model = MedicalNetSemantics(encoder_factory=TinyEncoder, roi_size=8)
        model.train()
        ct = torch.rand(1,1,8,8,8)
        mask = torch.zeros_like(ct, requires_grad=True)
        prob, expected = model(ct, mask, torch.ones_like(ct))
        self.assertEqual(expected.shape, (1,6))
        self.assertNotIn('malignancy', SEMANTIC_NAMES)
        torch.testing.assert_close(prob.sum(-1), torch.ones(1,6))
        expected.sum().backward()
        self.assertGreater(float(mask.grad.abs().sum()), 0.)
        self.assertGreater(float(model.encoder.conv.weight.grad.abs().sum()), 0.)

    def test_fasterrisk_pool_and_input_gradients(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(100,6))
        x[:,5] = 7
        y = (x[:,0]+rng.normal(size=100) > 0).astype(float)
        result = fit_continuous_pool(x,y,sparsity=2,pool_size=8,beam=3,gap=.2)
        self.assertTrue((np.abs(result['weights']) <= 5+1e-6).all())
        self.assertTrue(((np.abs(result['weights'])>1e-9).sum(1)<=2).all())
        self.assertTrue((result['losses'] <= result['losses'].min()*1.2+1e-9).all())
        bank = ContinuousRashomonBank(tuple('abcdef'),sparsity=2,pool_size=8,beam=3,gap=.2,min_samples=8)
        bank.allowed_cases={'train'}
        features=torch.tensor(x,dtype=torch.float32,requires_grad=True)
        bank.observe_and_refit(features,torch.tensor(y),[('train',i) for i in range(100)])
        logits,indices=bank(features,all_models=True)
        losses=F.binary_cross_entropy_with_logits(logits,torch.tensor(y,dtype=torch.float32)[:,None].expand_as(logits),reduction='none').mean(0)
        np.testing.assert_allclose(losses.detach().numpy(),result['losses'],rtol=1e-5)
        self.assertEqual(len(list(bank.parameters())),0)
        losses.mean().backward()
        self.assertGreater(float(features.grad.abs().sum()),0.)
        sampled,idx=bank(features)
        self.assertEqual(len(idx),max(1,math.ceil(int(bank.count)*.3)))
        self.assertEqual(len(idx.unique()),len(idx))
        bank2=ContinuousRashomonBank(tuple('abcdef'),sparsity=2,pool_size=8,beam=3,gap=.2,min_samples=8)
        bank2.load_state_dict(copy.deepcopy(bank.state_dict()))
        bank.eval();bank2.eval()
        torch.testing.assert_close(bank(features)[0],bank2(features)[0])
        with self.assertRaises(RuntimeError):
            bank.observe_and_refit(features,torch.tensor(y),[('train',i) for i in range(100)])
        bank.train()
        with self.assertRaises(ValueError):
            bank.observe_and_refit(features[:1],torch.tensor(y[:1]),[('testing',0)])

    def test_average_losses_not_average_predictions(self):
        x=torch.tensor([[1.,2.]],requires_grad=True)
        w=torch.tensor([[3.,0.],[0.,-2.]])
        z=x@w.T
        target=torch.ones_like(z)
        loss=F.binary_cross_entropy_with_logits(z,target)
        loss.backward()
        expected=((z.detach().sigmoid()-target)@w)/2
        torch.testing.assert_close(x.grad,expected)
        self.assertFalse(torch.isclose(loss,F.binary_cross_entropy_with_logits(z.mean(1),torch.ones(1))))
        obj=torch.tensor([.3,.8],requires_grad=True)
        z=torch.tensor([[1.,-2.],[.5,.7]],requires_grad=True)
        scan=scan_logits_per_model(z,obj)
        direct=1-(1-z.sigmoid()*obj.sigmoid()[:,None]).prod(0)
        torch.testing.assert_close(scan.sigmoid(),direct)
        F.binary_cross_entropy_with_logits(scan,torch.ones_like(scan)).backward()
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_ablation_modes_and_frozen_refresh(self):
        x = torch.tensor([[.2, .7], [-.3, .1]], requires_grad=True)
        for mode, expected in [('sample', 2), ('single', 1), ('all', 4), ('best', 1)]:
            bank = ContinuousRashomonBank(('a', 'b'), sparsity=1, pool_size=4,
                                         mode=mode, refresh_every=0, min_samples=2)
            bank.allowed_cases = {'train'}
            with torch.no_grad():
                bank.count.fill_(4)
                bank.weights.copy_(torch.tensor([[1.,0.], [-1.,0.], [0.,1.], [0.,-1.]]))
            before = bank.weights.clone()
            bank.observe_and_refit(x, torch.tensor([0.,1.]), [('train',0),('train',1)])
            torch.testing.assert_close(bank.weights, before)
            self.assertEqual(int(bank.fit_count), 0)
            z, indices = bank(x)
            self.assertEqual(z.shape, (2, expected))
            # One sampled index vector is applied to both nodules in the CT.
            torch.testing.assert_close(z, x @ before[indices].T)
            bank.eval()
            a, _ = bank(x); b, _ = bank(x)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertEqual(a.shape[1], 1 if mode == 'best' else 4)

    def test_full_path_and_diagnostic_only_gradients(self):
        torch.manual_seed(9)
        m=WholeCTJointModelV3(window_size=(8,8,8), screen_shape=(8,8,8), roi_shape=(8,8,8),
            num_queries=4,detr_hidden_dim=16,detr_coarse_shape=(4,4,4),detr_nheads=4,
            detr_encoder_layers=1,detr_decoder_layers=1,vista_feature_dim=1,context_channels=2,
            refiner_base_channels=2,hard_negatives=1,teacher_full_epochs=1,teacher_zero_epoch=2,
            teacher_jitter=0,segmenter_factory=TinySegmenter,encoder_factory=TinyEncoder,
            medicalnet_roi_size=8,use_checkpoint=False)
        # Nonzero refiner head lets diagnostic gradients reach its encoder.
        nn.init.normal_(m.refiner.residual_head.weight,std=.01)
        b=make_batch(); b['semantic_histograms']=b['semantic_histograms'][:,[0,2,3,4,5,6]]
        b['case_id']='train';b['nodule_ids']=torch.tensor([1])
        b['image_hu']=b['image']*2048-1024
        m.train()
        # Controlled frozen sparse head includes one radiomic and one semantic.
        with torch.no_grad():
            m.rashomon.count.fill_(2)
            m.rashomon.weights[0,8]=.001
            m.rashomon.weights[0,18]=.5
            m.rashomon.weights[1,8]=-.001
            m.rashomon.weights[1,19]=.5
        o=m(b['image'],batch=b,epoch=0,update_bank=False)
        self.assertEqual(o.semantic_features.shape,(2,6))
        self.assertEqual(o.diagnostic_features.shape,(2,24))
        losses=WholeCTCriterionV3()(o,b)
        # Isolate malignancy; segmentation and semantic losses cannot mask a
        # broken diagnostic gradient path.
        (losses.nodule+losses.risk).backward()
        for name,p in [('segmenter',m.segmenter.scale),('detr',m.detr.mask_feature_projection[0].weight),
                       ('refiner',m.refiner.residual_head.weight),('medicalnet',m.semantics.encoder.conv.weight)]:
            self.assertIsNotNone(p.grad,name)
            self.assertTrue(torch.isfinite(p.grad).all(),name)
            self.assertGreater(float(p.grad.abs().sum()),0.,name)
        m.eval()
        o=m(b['image'],image_hu=b['image_hu'],object_threshold=0.,update_bank=False)
        self.assertEqual(o.ensemble_logits.shape[1],2)
        self.assertEqual(o.teacher_forced_count,0)


if __name__=='__main__':
    unittest.main(verbosity=2)
