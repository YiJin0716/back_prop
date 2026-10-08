import copy
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.special import expit
import torch
import torch.nn.functional as F

from back_prop.model_v4.risk import fit_radiomics, fit_offset_pool, ResidualRiskBank
from back_prop.model_v4.loss import missing_nodule_loss, WholeCTCriterionV4
from back_prop.model_v4.model import WholeCTJointModelV4
from back_prop.model_v4.schedule import LearningRateSchedule, make_lr_scheduler
from back_prop.tests.model_fixtures import TinyEncoder
from back_prop.tests.model_fixtures import TinySegmenter, make_batch


class V4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def sample(self):
        rng = np.random.default_rng(13)
        r, s = rng.normal(size=(180, 18)), rng.normal(size=(180, 6))
        y = (rng.random(180) < expit(2*r[:, 0] + 2*s[:, 0])).astype(float)
        return r, s, y

    def test_two_stage_fit_uses_baseline_and_improves_residual(self):
        r, s, y = self.sample()
        base = fit_radiomics(r, y)
        offset = np.clip((r-base['mean'])/base['scale'], -10, 10) @ base['weights'] + base['intercept']
        fit = fit_offset_pool(s, y, offset, sparsity=2, pool_size=8, beam=3, gap=.2)
        self.assertEqual(fit['weights'].shape[1], 6)
        self.assertLess(fit['losses'].min(), fit['baseline_loss']-.05)
        self.assertTrue((abs(fit['weights']) <= 5+1e-6).all())
        self.assertTrue(((abs(fit['weights']) > 1e-9).sum(1) <= 2).all())
        normalized = np.clip((s-fit['mean'])/fit['scale'], -10, 10)*fit['varying']
        direct = np.logaddexp(0, -(2*y-1)[:, None]*(offset[:, None]+normalized@fit['weights'].T+fit['intercepts'])).mean(0)
        np.testing.assert_allclose(direct, fit['losses'])
        no_offset = fit_offset_pool(s, y, np.zeros_like(y), sparsity=2, pool_size=8, beam=3, gap=.2)
        self.assertFalse(np.allclose(fit['weights'][0], no_offset['weights'][0]))

    def test_constant_semantics_returns_baseline(self):
        _, _, y = self.sample()
        offset = np.linspace(-2, 2, len(y))
        fit = fit_offset_pool(np.ones((len(y), 6)), y, offset)
        self.assertEqual(len(fit['weights']), 1)
        self.assertEqual(fit['varying_semantics'], 0)
        np.testing.assert_array_equal(fit['weights'], np.zeros((1, 6)))
        self.assertAlmostEqual(fit['losses'][0], fit['baseline_loss'])

    def test_cache_checkpoint_frozen_heads_live_gradients_and_no_leakage(self):
        r, s, y = self.sample()
        r, s, y = torch.tensor(r).float().requires_grad_(), torch.tensor(s).float().requires_grad_(), torch.tensor(y).float()
        bank = ResidualRiskBank(sparsity=2, pool_size=8, beam=3, min_samples=8)
        bank.allowed_cases = {'train'}
        keys = [('train', i) for i in range(len(y))]
        bank.observe_radiomics(r, y, keys)
        self.assertEqual(int(bank.baseline_count), 1)
        self.assertEqual(int(bank.count), 0)
        bank.observe_semantics(s, y, keys)
        z_base = bank.baseline_logits(r)
        z, _, correction = bank(s, z_base, all_models=True)
        torch.testing.assert_close(z, z_base.detach()[:, None] + correction)
        loss = F.binary_cross_entropy_with_logits(z, y[:, None].expand_as(z))
        loss.backward()
        self.assertIsNone(r.grad)  # semantic residual cannot move its baseline target
        self.assertGreater(float(s.grad.abs().sum()), 0)
        self.assertFalse(z_base.requires_grad)
        self.assertEqual(len(list(bank.parameters())), 0)
        restored = ResidualRiskBank(sparsity=2, pool_size=8, beam=3, min_samples=8)
        restored.load_state_dict(copy.deepcopy(bank.state_dict()))
        bank.eval(); restored.eval()
        torch.testing.assert_close(bank(s, bank.baseline_logits(r))[0],
                                   restored(s, restored.baseline_logits(r))[0], rtol=0, atol=0)
        with self.assertRaises(RuntimeError):
            restored.observe_radiomics(r, y, keys)
        restored.train()
        with self.assertRaises(ValueError):
            restored.observe_radiomics(r[:1], y[:1], [('test', 0)])
        self.assertEqual(len(restored.memory), len(y))

    def test_initial_semantic_gradient_is_negative_probability_residual(self):
        baseline = torch.tensor([-2., .7, 3.])
        y = torch.tensor([1., 0., 1.])
        correction = torch.zeros(3, requires_grad=True)
        F.binary_cross_entropy_with_logits(baseline+correction, y, reduction='sum').backward()
        torch.testing.assert_close(correction.grad, -(y-baseline.sigmoid()))

    def test_missing_penalty_has_gradients_for_severely_missed_gt(self):
        mask = torch.full((2, 2, 2, 2), -100., requires_grad=True)
        obj = torch.tensor([-100., 3.], requires_grad=True)
        gt = torch.zeros(1, 2, 2, 2); gt[0, 0, 0, 0] = 1
        out = SimpleNamespace(coarse_mask_logits=mask, object_logits=obj,
                              matched_query_indices=torch.tensor([0]), matched_target_indices=torch.tensor([0]))
        loss = missing_nodule_loss(out, {'target_masks': gt})
        self.assertAlmostEqual(float(loss), 200., places=4)
        loss.backward()
        self.assertLess(float(obj.grad[0]), -.99)
        self.assertLess(float(mask.grad[0, 0, 0, 0]), -.99)
        self.assertEqual(float(mask.grad[0, 1, 1, 1]), 0.)
        out.coarse_mask_logits = torch.full_like(mask, 10.)
        out.object_logits = torch.full_like(obj, 10.)
        self.assertLess(float(missing_nodule_loss(out, {'target_masks': gt})), .001)
        with self.assertRaises(ValueError):
            missing_nodule_loss(out, {'target_masks': gt.expand(3, -1, -1, -1)})

    def tiny(self):
        torch.manual_seed(9)
        model = WholeCTJointModelV4(window_size=(8,8,8), screen_shape=(8,8,8), roi_shape=(8,8,8),
            num_queries=4,detr_hidden_dim=16,detr_coarse_shape=(4,4,4),detr_nheads=4,
            detr_encoder_layers=1,detr_decoder_layers=1,vista_feature_dim=1,context_channels=2,
            refiner_base_channels=2,hard_negatives=1,teacher_full_epochs=1,teacher_zero_epoch=2,
            teacher_jitter=0,segmenter_factory=TinySegmenter,encoder_factory=TinyEncoder,
            medicalnet_roi_size=8,use_checkpoint=False)
        torch.nn.init.normal_(model.refiner.residual_head.weight, std=.01)
        b=make_batch(); b['semantic_histograms']=b['semantic_histograms'][:,[0,2,3,4,5,6]]
        b['case_id']='train'; b['nodule_ids']=torch.tensor([1]); b['image_hu']=b['image']*2048-1024
        with torch.no_grad():
            bank=model.rashomon
            bank.count.fill_(2); bank.baseline_count.fill_(1)
            bank.radiomics_weights[8]=.001
            bank.weights[0,0]=.5; bank.weights[1,1]=-.5
        return model,b

    def test_full_model_stage_order_losses_and_diagnostic_gradients(self):
        m,b=self.tiny(); m.train()
        order=[]
        handle=m.semantics.register_forward_pre_hook(lambda *_: order.append('semantics'))
        base=m.rashomon.baseline_logits
        m.rashomon.baseline_logits=lambda x: (order.append('baseline') or base(x))
        o=m(b['image'],batch=b,epoch=0,update_bank=False)
        handle.remove()
        self.assertEqual(order[0], 'baseline')
        self.assertEqual(o.semantic_features.shape, (2,6))
        self.assertEqual(float(m.mask_temperature), 1.)
        valid=(o.refined_target_indices>=0)&o.fine_supervision_valid
        torch.testing.assert_close(o.residual_targets[valid], b['malignancy_targets']-o.radiomics_logits[valid].detach().sigmoid())
        criterion=WholeCTCriterionV4(); losses=criterion(o,b)
        expected=sum(getattr(losses,k)*v for k,v in criterion.loss_weights.items())
        torch.testing.assert_close(losses.total, expected)
        self.assertFalse(hasattr(losses, 'object'))
        self.assertFalse(hasattr(losses, 'radiomics'))
        self.assertFalse(hasattr(losses, 'ordinal'))
        (losses.nodule+losses.risk).backward()
        for p in (m.segmenter.scale,m.detr.mask_feature_projection[0].weight,
                  m.refiner.residual_head.weight,m.semantics.encoder.conv.weight):
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(float(p.grad.abs().sum()), 0.)
        # Fine supervision failure must not erase the whole-scan miss penalty.
        o.fine_supervision_valid.zero_()
        skipped=criterion(o,b)
        self.assertEqual(float(skipped.fine_dice), 0.)
        torch.testing.assert_close(skipped.missing_nodule, losses.missing_nodule)
        m.eval()
        a=m(b['image'],image_hu=b['image_hu'],object_threshold=0.)
        c=m(b['image'],image_hu=b['image_hu'],object_threshold=0.)
        torch.testing.assert_close(a.nodule_logits, c.nodule_logits, rtol=0, atol=0)
        self.assertEqual(a.teacher_forced_count, 0)

    def test_geometry_matches_v3_with_revised_semantic_encoder(self):
        from back_prop.common.semantic_model import WholeCTJointModelV3
        m,b=self.tiny()
        torch.manual_seed(9)
        base=WholeCTJointModelV3(window_size=(8,8,8), screen_shape=(8,8,8), roi_shape=(8,8,8),
            num_queries=4,detr_hidden_dim=16,detr_coarse_shape=(4,4,4),detr_nheads=4,
            detr_encoder_layers=1,detr_decoder_layers=1,vista_feature_dim=1,context_channels=2,
            refiner_base_channels=2,hard_negatives=1,teacher_full_epochs=1,teacher_zero_epoch=2,
            teacher_jitter=0,segmenter_factory=TinySegmenter,encoder_factory=TinyEncoder,
            medicalnet_roi_size=8,use_checkpoint=False)
        shared={k:v for k,v in m.state_dict().items() if not k.startswith(('rashomon.','mask_temperature'))}
        incompatible=base.load_state_dict(shared,strict=False)
        self.assertFalse(incompatible.unexpected_keys)
        self.assertTrue(all(k.startswith('rashomon.') for k in incompatible.missing_keys))
        m.train(); base.train()
        torch.manual_seed(81)
        expected=base(b['image'],batch=b,compute_diagnostics=False,update_bank=False)
        torch.manual_seed(81)
        actual=m(b['image'],batch=b,compute_diagnostics=False,update_bank=False)
        for name in ('boxes','object_logits','coarse_mask_logits','fine_mask_logits',
                     'crop_origins','radiomics_features'):
            torch.testing.assert_close(getattr(actual,name),getattr(expected,name),rtol=0,atol=0)
        # V4's standardized MedicalNet filters deliberately change semantics,
        # while the geometry and radiomics paths remain identical to V3.
        self.assertFalse(torch.equal(actual.semantic_features,expected.semantic_features))

    def test_mining_bypasses_semantics_and_gt_without_fitted_risk(self):
        from unittest.mock import patch
        m,b=self.tiny(); m.eval()
        m.rashomon.count.zero_(); m.rashomon.baseline_count.zero_()
        with torch.inference_mode(), patch.object(m.semantics,'forward',side_effect=AssertionError('Unexpected semantics')):
            out=m(b['image'],image_hu=b['image_hu'],object_threshold=0.,segmentation_only=True)
        self.assertEqual(out.teacher_forced_count,0)
        self.assertEqual(out.fallback_count,0)
        self.assertEqual(len(out.matched_query_indices),0)
        self.assertEqual(len(out.fine_mask_logits),4)
        self.assertEqual(len(m.rashomon.memory),0)
        m.train()
        with self.assertRaises(ValueError):
            m(b['image'],batch=b,segmentation_only=True)

    def test_removing_object_and_radiomics_objectives_removes_their_gradients(self):
        m,b=self.tiny(); m.train()
        out=m(b['image'],batch=b,update_bank=False,compute_diagnostics=False)
        loss=WholeCTCriterionV4(missing_nodule_weight=0.)(out,b,diagnostic_scale=0.)
        grad=torch.autograd.grad(loss.total,out.object_logits,allow_unused=True)[0]
        self.assertTrue(grad is None or bool((grad==0).all()))
        self.assertFalse(out.radiomics_features.requires_grad)
        self.assertFalse(out.radiomics_logits.requires_grad)

    def test_lr_resume_continues_cosine_without_temperature_schedule(self):
        parameter=torch.nn.Parameter(torch.ones(1))
        optimizer=torch.optim.AdamW([{'params':[parameter],'lr':1e-4,'name':'main'}])
        schedule=LearningRateSchedule(); base={'main':1e-4}
        scheduler=make_lr_scheduler(optimizer,schedule,base)
        for _ in range(3):
            parameter.sum().backward(); optimizer.step(); optimizer.zero_grad(); scheduler.step()
        state=copy.deepcopy(optimizer.state_dict()); saved=copy.deepcopy(scheduler.state_dict())
        other=torch.optim.AdamW([{'params':[torch.nn.Parameter(torch.ones(1))],'lr':1e-4,'name':'main'}])
        other.load_state_dict(state)
        resumed=make_lr_scheduler(other,schedule,base,next_epoch=3,saved_state=saved)
        self.assertEqual(optimizer.param_groups[0]['lr'], other.param_groups[0]['lr'])
        self.assertEqual(resumed.last_epoch,3)
        self.assertAlmostEqual(schedule(17),.1)
        self.assertEqual(schedule(100), schedule(17))


if __name__=='__main__':
    unittest.main(verbosity=2)
