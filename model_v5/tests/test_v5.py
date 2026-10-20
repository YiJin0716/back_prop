import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch

from back_prop.model_v5.model import WholeCTJointModelV5, ARCHITECTURE
from back_prop.model_v5.loss import WholeCTCriterionV5, LossAccumulator, TERM_NAMES
from back_prop.model_v5.features import CNNSemantics, check_semantic_spread
from back_prop.tests.model_fixtures import TinySegmenter, make_batch


def tiny(detector_kind='legacy'):
    torch.manual_seed(9)
    with patch('back_prop.common.semantic_model.MedicalNetSemantics', side_effect=AssertionError('MedicalNet constructed')):
        model = WholeCTJointModelV5(window_size=(8,8,8), screen_shape=(8,8,8), roi_shape=(8,8,8),
            num_queries=4, detr_hidden_dim=16, detr_coarse_shape=(4,4,4), detr_nheads=4,
            detr_encoder_layers=1, detr_decoder_layers=1, vista_feature_dim=1, context_channels=2,
            refiner_base_channels=2, hard_negatives=1, teacher_jitter=0,
            segmenter_factory=TinySegmenter, semantic_roi_size=8, semantic_width=4, use_checkpoint=False,
            detector_kind=detector_kind)
    model.semantic_min_dice = 0.  # Existing tests isolate coverage, not mask quality.
    torch.nn.init.normal_(model.refiner.residual_head.weight, std=.01)
    batch=make_batch()
    for key in ('semantic_histograms','semantic_targets'):
        batch[key]=batch[key][:,[0,2,3,4,5,6]]
    batch.update(case_id='train', nodule_ids=torch.tensor([1]), image_hu=batch['image']*2048-1024)
    with torch.no_grad():
        model.rashomon.count.fill_(2); model.rashomon.baseline_count.fill_(1)
        model.rashomon.radiomics_weights[8]=.001
        model.rashomon.weights[0,0]=.5; model.rashomon.weights[1,1]=-.5
    return model,batch


class V5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_native_warmup_loads_for_segmentation_without_claiming_fitted_risk(self):
        from back_prop.evaluate.eval_package.inference import load_joint_model
        model,batch=tiny('native_seeded')
        model.eval()
        model.detr.confidence_residual_bound=2.
        with torch.no_grad():model.detr.native_prediction.bias[0]=3.
        model.rashomon.count.zero_();model.rashomon.baseline_count.zero_()
        config=dict(window_size=(8,8,8),screen_shape=(8,8,8),roi_shape=(8,8,8),
            num_queries=4,detr_hidden_dim=16,detr_coarse_shape=(4,4,4),detr_nheads=4,
            detr_encoder_layers=1,detr_decoder_layers=1,vista_feature_dim=1,context_channels=2,
            refiner_base=2,hard_negatives=1,teacher_jitter=0,semantic_roi_size=8,semantic_width=4,
            detector_kind='native_seeded',semantic_min_dice=0.,native_confidence_residual_bound=2.)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'geometry.pt'
            torch.save(dict(model=model.state_dict(),architecture=ARCHITECTURE,config=config,
                            epoch=-1,phase='independent_warmup'),path)
            with patch('back_prop.evaluate.eval_package.inference._uninitialized_segmenter',TinySegmenter):
                with self.assertRaisesRegex(ValueError,'no fitted Rashomon'):
                    load_joint_model(path,device='cpu')
                restored=load_joint_model(path,device='cpu',require_risk_bank=False)
            self.assertFalse(restored.evaluation_metadata['risk_bank_ready'])
            self.assertEqual(restored.detr.confidence_residual_bound,2.)
            self.assertEqual(restored.evaluation_metadata['phase'],'independent_warmup')
            kwargs=dict(image_hu=batch['image_hu'],segmentation_only=True,object_threshold=0.)
            with torch.no_grad():
                before=model(batch['image'],**kwargs)
                after=restored(batch['image'],**kwargs)
            for name in ('object_logits','boxes','fine_mask_logits'):
                torch.testing.assert_close(getattr(before,name),getattr(after,name),atol=0,rtol=0)
            with torch.no_grad(),self.assertRaisesRegex(RuntimeError,'Fit or load both'):
                restored(batch['image'],image_hu=batch['image_hu'])

    def test_cnn_mask_gradient_and_batch_independence(self):
        torch.manual_seed(3)
        m=CNNSemantics(roi_size=16,width=4,use_checkpoint=False)
        ct=torch.rand(4,1,16,16,16)
        mask=torch.randn_like(ct,requires_grad=True)
        p,s=m(ct,mask,torch.ones_like(ct))
        self.assertEqual(p.shape,(4,6,5))
        self.assertTrue((s.std(0)>1e-3).all())
        single=torch.cat([m(x,y,torch.ones_like(x))[1] for x,y in zip(ct.split(1),mask.split(1))])
        torch.testing.assert_close(single,s,rtol=1e-4,atol=1e-5)
        m.eval();torch.testing.assert_close(m(ct,mask,torch.ones_like(ct))[1],s)
        (-p[:,:,0].log().mean()).backward()
        self.assertGreater(float(mask.grad.abs().sum()),0)
        self.assertGreater(float(next(m.encoder.parameters()).grad.abs().sum()),0)
        self.assertFalse(any(isinstance(x,torch.nn.modules.batchnorm._BatchNorm) for x in m.modules()))

    def test_teacher_schedule_and_no_gt_forward_dependency(self):
        m,b=tiny()
        self.assertEqual([m.teacher_probability(i) for i in range(11)],
                         [1.,.9,.8,.7,.6,.5,.4,.3,.2,.1,0.])
        self.assertEqual(sum(m.teacher_probability(i)==0 for i in range(18)),8)
        m.eval()
        with torch.no_grad():
            a=m(b['image'],batch=b,epoch=10,object_threshold=0.,update_bank=False)
            c=m(b['image'],image_hu=b['image_hu'],object_threshold=0.,update_bank=False)
        for k in ('refined_query_indices','crop_origins','fine_mask_logits','semantic_features','risk_logit'):
            torch.testing.assert_close(getattr(a,k),getattr(c,k),rtol=0,atol=0)
        self.assertEqual(a.teacher_forced_count,0);self.assertEqual(a.fallback_count,0)
        # Also test training mode, with the same dropout RNG and bank frozen.
        m.train();torch.manual_seed(2)
        a=m(b['image'],batch=b,epoch=10,object_threshold=0.,update_bank=False)
        torch.manual_seed(2)
        c=m(b['image'],image_hu=b['image_hu'],object_threshold=0.,update_bank=False)
        torch.testing.assert_close(a.fine_mask_logits,c.fine_mask_logits,rtol=0,atol=0)
        altered=dict(b)
        altered['target_boxes']=b['target_boxes'].clone()
        altered['target_boxes'][:,:3]=.95
        altered['target_mask_origins']=torch.tensor([[9,9,9]])
        altered['target_masks']=b['target_masks'].flip(-1)
        torch.manual_seed(2)
        d=m(b['image'],batch=altered,epoch=10,object_threshold=0.,update_bank=False)
        for k in ('refined_query_indices','crop_origins','fine_mask_logits','semantic_features'):
            torch.testing.assert_close(getattr(c,k),getattr(d,k),rtol=0,atol=0)

    def test_no_coverage_triggered_teacher_fallback(self):
        m,b=tiny();m.train()
        b['target_mask_origins']=torch.tensor([[100,100,100]])
        o=m(b['image'],batch=b,teacher_probability=1e-20,update_bank=False)
        self.assertEqual(o.teacher_forced_count,0)
        self.assertEqual(o.fallback_count,0)
        self.assertFalse(o.fine_supervision_valid[o.refined_target_indices>=0].any())

    def test_empty_selection_preserves_missing_loss_and_records_no_supervision(self):
        m,b=tiny();m.train()
        o=m(b['image'],batch=b,epoch=10,object_threshold=1.,update_bank=False)
        self.assertEqual(len(o.refined_query_indices),0)
        self.assertFalse(o.scan_path_valid)
        loss=WholeCTCriterionV5()(o,b)
        self.assertEqual(loss.counts['semantic'],0)
        self.assertEqual(loss.counts['risk'],0)
        self.assertEqual(loss.counts['missing_nodule'],1)
        loss.total.backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in m.detr.parameters() if p.grad is not None),0)
        acc=LossAccumulator();acc.update(loss);r=acc.compute(WholeCTCriterionV5().loss_weights)
        self.assertIsNone(r['semantic']);self.assertIsNone(r['total'])
        self.assertIsNotNone(r['missing_nodule'])

    def test_loss_mean_ignores_invalid_rois_and_scans(self):
        m,b=tiny();m.train();criterion=WholeCTCriterionV5()
        o=m(b['image'],batch=b,teacher_probability=1.,update_bank=False)
        valid=criterion(o,b)
        first=LossAccumulator();first.update(valid)
        o.fine_supervision_valid.zero_();o.scan_path_valid=False
        invalid=criterion(o,b)
        acc=LossAccumulator();acc.update(valid)
        for _ in range(9): acc.update(invalid)
        a,c=first.compute(criterion.loss_weights),acc.compute(criterion.loss_weights)
        for name in ('fine_dice','fine_focal_positive','semantic','nodule','risk'):
            self.assertAlmostEqual(a[name],c[name],places=6)
            self.assertEqual(c['loss_counts'][name],1)
        self.assertEqual(c['loss_counts']['box_l1'],10)
        valid.total.backward()
        self.assertGreater(float(next(m.semantics.encoder.parameters()).grad.abs().sum()),0)

    def test_box_l1_uses_the_same_coordinate_sum_as_matching(self):
        m,b=tiny();m.eval()
        o=m(b['image'],batch=b,update_bank=False)
        q,t=o.matched_query_indices,o.matched_target_indices
        o.boxes=o.boxes.detach().clone()
        o.boxes[q]=b['target_boxes'][t]+.1
        self.assertAlmostEqual(WholeCTCriterionV5()(o,b).mean('box_l1'),.6,places=6)

    def test_unmatched_object_scores_are_suppressed(self):
        m,b=tiny();m.eval()
        o=m(b['image'],batch=b,update_bank=False)
        o.object_logits=o.object_logits.detach().requires_grad_(True)
        loss=WholeCTCriterionV5()(o,b)
        loss.numerators['object'].backward()
        q=o.matched_query_indices
        unmatched=torch.ones_like(o.object_logits,dtype=torch.bool);unmatched[q]=False
        self.assertEqual(loss.counts['object'],len(o.object_logits))
        self.assertTrue((o.object_logits.grad[unmatched]>0).all())
        self.assertTrue((o.object_logits.grad[q]<0).all())
        # An excluded/indeterminate annotation must not become background.
        i=int(unmatched.nonzero()[0])
        b['ignored_target_boxes']=o.boxes[i:i+1].detach().clone()
        b['ignored_target_masks']=(o.coarse_mask_logits[i:i+1].detach()>0).byte()
        o.object_logits.grad=None
        loss=WholeCTCriterionV5()(o,b)
        loss.numerators['object'].backward()
        self.assertEqual(float(o.object_logits.grad[i]),0.)

    def test_overlap_penalty_rejects_broad_masks_and_keeps_missed_nodule_gradients(self):
        from back_prop.model_v5.loss import nodule_overlap_loss
        target=torch.zeros(1,1024);target[:,0]=1.
        correct=torch.full_like(target,-20.);correct[:,0]=20.
        broad=torch.full_like(target,8.,requires_grad=True)
        self.assertLess(float(nodule_overlap_loss(correct,target)),1e-5)
        self.assertGreater(float(nodule_overlap_loss(broad,target).detach()),6.)
        nodule_overlap_loss(broad,target).sum().backward()
        self.assertGreater(float(broad.grad[0,1:].sum()),0.)
        missed=torch.full_like(target,-100.,requires_grad=True)
        loss=nodule_overlap_loss(missed,target)
        self.assertTrue(torch.isfinite(loss).all())
        loss.sum().backward()
        self.assertLess(float(missed.grad[0,0]),-.9)
        valid=target.bool()
        self.assertLess(float(nodule_overlap_loss(broad,target,valid=valid).detach()),.001)

    def test_coarse_supervision_excludes_ignored_regions(self):
        m,b=tiny();m.eval()
        o=m(b['image'],batch=b,update_bank=False)
        b['ignored_target_masks']=torch.ones_like(b['target_masks'])
        a=WholeCTCriterionV5()(o,b)
        q,t=o.matched_query_indices,o.matched_target_indices
        logits=o.coarse_mask_logits.detach().clone()
        for query,target in zip(q,t):
            logits[query][~b['target_masks'][target].bool()]=20.
        o.coarse_mask_logits=logits
        c=WholeCTCriterionV5()(o,b)
        for name in ('coarse_dice','coarse_focal'):
            self.assertAlmostEqual(a.mean(name),c.mean(name),places=6)

    def test_overlap_penalty_has_no_positive_only_object_gradient(self):
        m,b=tiny();m.eval()
        o=m(b['image'],batch=b,update_bank=False)
        a=WholeCTCriterionV5()(o,b).mean('missing_nodule')
        o.object_logits=torch.full_like(o.object_logits,-20.)
        self.assertAlmostEqual(a,WholeCTCriterionV5()(o,b).mean('missing_nodule'),places=6)

    def test_fixed_geometry_probe_reports_overlap_and_is_prediction_only(self):
        from types import SimpleNamespace
        from back_prop.model_v5.preparation import geometry_audit
        m,b=tiny()
        args=SimpleNamespace(seed=42,geometry_probe_cases=1,geometry_min_coverage=0.,
                             geometry_min_dice=0.,geometry_min_coarse_dice=0.,geometry_min_usable=0.)
        result=geometry_audit(m,[b],args,0,1)
        for name in ('coarse_dice','coarse_precision','coarse_recall','box_iou'):
            self.assertGreaterEqual(result[name],0.)
            self.assertLessEqual(result[name],1.)
        self.assertEqual(result['nodules'],1)
        self.assertFalse(m.training)
        args.geometry_min_coarse_dice=1.
        self.assertFalse(geometry_audit(m,[b],args,0,1)['passed'])

    def test_checkpoint_loader_uses_cnn_without_medicalnet(self):
        from back_prop.evaluate.eval_package.inference import load_joint_model
        m,b=tiny()
        m.set_temperature(.1)
        cfg=dict(window_size=(8,8,8),screen_shape=(8,8,8),roi_shape=(8,8,8),num_queries=4,
            detr_hidden_dim=16,detr_coarse_shape=(4,4,4),detr_nheads=4,detr_encoder_layers=1,
            detr_decoder_layers=1,vista_feature_dim=1,context_channels=2,refiner_base=2,
            hard_negatives=1,teacher_jitter=0,semantic_roi_size=8,semantic_width=4)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'v5.pt'
            torch.save(dict(model=m.state_dict(),config=cfg,architecture=ARCHITECTURE,epoch=5),path)
            with patch('back_prop.evaluate.eval_package.inference._uninitialized_segmenter',TinySegmenter), \
                 patch('back_prop.evaluate.eval_package.inference._uninitialized_encoder',side_effect=AssertionError('MedicalNet')):
                restored=load_joint_model(path,device='cpu')
            m.eval()
            with torch.no_grad():
                a=m(b['image'],image_hu=b['image_hu'],object_threshold=0.)
                c=restored(b['image'],image_hu=b['image_hu'],object_threshold=0.)
            torch.testing.assert_close(a.semantic_features,c.semantic_features,rtol=0,atol=0)
            self.assertEqual(float(restored.mask_temperature), float(m.mask_temperature))
            torch.testing.assert_close(a.fine_mask_probabilities,c.fine_mask_probabilities,rtol=0,atol=0)

    def test_collapse_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'Near-constant'):
            check_semantic_spread([0.]*6,torch.arange(64.)[:,None].expand(64,6))

    def test_epoch_mean_weights_by_nodules_not_steps(self):
        from back_prop.model_v5.loss import V5Loss
        criterion=WholeCTCriterionV5();acc=LossAccumulator()
        for n, mean in [(1,2.),(3,6.)]:
            sums={k:torch.tensor(mean*n) for k in TERM_NAMES}
            counts={k:n for k in TERM_NAMES}
            acc.update(V5Loss(torch.tensor(0.),sums,counts,n,n,n,True))
        r=acc.compute(criterion.loss_weights)
        self.assertEqual(r['semantic'],5.)
        self.assertEqual(r['loss_counts']['semantic'],4)

    def test_predicted_roi_collapse_guard(self):
        from back_prop.model_v5.diagnostics import SemanticAccumulator
        from types import SimpleNamespace
        out=SimpleNamespace(refined_target_indices=torch.arange(32), fine_supervision_valid=torch.ones(32,dtype=torch.bool),
            roi_loss_valid=torch.ones(32,2,2,2,dtype=torch.bool),semantic_features=torch.full((32,6),3.))
        acc=SemanticAccumulator('cpu')
        acc.update(out,{'semantic_targets':torch.linspace(1,5,32)[:,None].expand(32,6)})
        with self.assertRaisesRegex(RuntimeError,'supervised fine ROIs'):
            acc.compute()

    def test_quality_gate_preserves_segmentation_supervision(self):
        m,b=tiny();m.train();m.semantic_min_dice=1.
        out=m(b['image'],batch=b,teacher_probability=1.,update_bank=False)
        self.assertTrue(out.fine_supervision_valid[out.refined_target_indices>=0].all())
        self.assertFalse(out.semantic_supervision_valid.any())
        loss=WholeCTCriterionV5()(out,b)
        self.assertEqual(loss.counts['fine_dice'],1)
        self.assertEqual(loss.counts['semantic'],0)
        self.assertEqual(loss.counts['nodule'],0)
        self.assertEqual(loss.counts['risk'],0)
        loss.total.backward()
        self.assertGreater(float(m.refiner.residual_head.weight.grad.abs().sum()),0)

    def test_anchor_trains_semantics_when_no_roi_is_selected(self):
        m,b=tiny();m.train()
        out=m(b['image'],batch=b,teacher_probability=0.,object_threshold=1.,
              update_bank=False,semantic_anchor_roi=torch.rand(2,2,8,8,8))
        hist=b['semantic_histograms'][:1].expand(2,-1,-1)
        loss=WholeCTCriterionV5()(out,b,anchor=(out.anchor_probabilities,hist))
        self.assertEqual(loss.counts['semantic'],0)
        self.assertEqual(loss.counts['semantic_anchor'],2)
        loss.total.backward()
        self.assertGreater(float(next(m.semantics.encoder.parameters()).grad.abs().sum()),0)

    def test_geometry_only_does_not_execute_semantics_or_radiomics(self):
        m,b=tiny();m.train()
        with patch.object(m.semantics,'forward',side_effect=AssertionError('CNN used')), \
             patch.object(m.radiomics,'forward',side_effect=AssertionError('Radiomics used')):
            out=m(b['image'],batch=b,teacher_probability=1.,geometry_only=True)
            loss=WholeCTCriterionV5()(out,b,semantic_scale=0.,diagnostic_scale=0.)
        self.assertEqual(loss.counts['semantic'],0)
        self.assertGreater(loss.counts['fine_dice'],0)
        loss.total.backward()

    def test_translation_alignment_physical_units_and_zero_padding(self):
        from back_prop.model_v5.augmentation import translate_rois
        roi=torch.zeros(1,2,8,8,8);roi[0,:,4,4,4]=torch.tensor([2.,1.])
        shifted=translate_rois(roi,[[2.,0.,0.]],[2.,2.,2.])
        self.assertAlmostEqual(float(shifted[0,0,3,4,4]),2.,places=5)
        torch.testing.assert_close(shifted[:,0],2*shifted[:,1])
        self.assertEqual(float(translate_rois(roi,[[40.,0.,0.]],[2.,2.,2.]).sum()),0.)

    def test_partial_semantic_degradation_is_rejected(self):
        from back_prop.model_v5.semantic_metrics import semantic_failures
        metrics=dict(samples=64,target_std=[1.]*6,predicted_std=[.09]*6,
                     std_ratio=[.09]*6,correlation=[.5]*6,nll=1.,prior_nll=1.2)
        failures=semantic_failures({'predicted':metrics})
        self.assertTrue(any('spread' in f for f in failures))
        metrics.update(predicted_std=[.4]*6,std_ratio=[.4]*6)
        self.assertEqual(semantic_failures({'predicted':metrics}),[])
        metrics['correlation']=[0.]*6
        self.assertTrue(any('correlation' in f for f in semantic_failures({'predicted':metrics})))

    def test_extended_warmup_requires_minimum_and_sustained_readiness(self):
        from back_prop.model_v5.preparation import WarmupReadiness
        geometry=WarmupReadiness(15,3)
        decisions=[geometry.update(epoch,True)
                   for epoch in range(1,16)]
        self.assertFalse(any(decisions[:-1]))
        self.assertTrue(decisions[-1])
        # One passing epoch at the minimum cannot release the semantic stage.
        semantic=WarmupReadiness(20,3)
        for epoch in range(1,20):
            self.assertFalse(semantic.update(epoch,False))
        self.assertFalse(semantic.update(20,True))
        self.assertFalse(semantic.update(21,False))
        self.assertFalse(semantic.update(22,True))
        self.assertFalse(semantic.update(23,True))
        resumed=WarmupReadiness(20,3,semantic.consecutive_passes)
        self.assertTrue(resumed.update(24,True))

    def test_joint_guard_uses_regression_not_the_warmup_admission_cutoff(self):
        from copy import deepcopy
        from back_prop.model_v5.semantic_metrics import semantic_failures
        reference=dict(samples=128,target_std=[.76]*6,predicted_std=[.134]*6,
                       std_ratio=[.176]*6,correlation=[.5]*6,nll=1.09,prior_nll=1.18)
        current=deepcopy(reference)
        current.update(predicted_std=[.103]*6,std_ratio=[.135]*6)
        self.assertTrue(semantic_failures({'shifted':current}))
        self.assertEqual(semantic_failures({'shifted':current},reference={'shifted':reference}),[])
        # A real loss of half the signal remains a stop condition.
        current.update(predicted_std=[.06]*6,std_ratio=[.079]*6)
        self.assertTrue(any('half' in f for f in semantic_failures(
            {'shifted':current},reference={'shifted':reference})))

    def test_semantic_guard_rejects_nonfinite_and_attribute_regression(self):
        from copy import deepcopy
        from back_prop.model_v5.semantic_metrics import semantic_failures
        reference=dict(samples=128,target_std=[1.]*6,predicted_std=[.5]*6,
                       std_ratio=[.5]*6,correlation=[.5]*6,nll=1.,prior_nll=1.2,
                       nll_by_attribute=[1.]*6)
        current=deepcopy(reference);current['nll']=float('nan')
        self.assertTrue(any('nonfinite' in f for f in semantic_failures({'centered':current})))
        current=deepcopy(reference);current['nll_by_attribute'][2]=1.2
        self.assertTrue(any('sphericity: NLL' in f for f in semantic_failures(
            {'centered':current},reference={'centered':reference})))

    def test_official_jitter_crops_before_resizing(self):
        from back_prop.model_v5.augmentation import official_view
        import torch.nn.functional as F
        full=torch.zeros(1,2,8,8,8);full[:,:,3:5,3:5,3:5]=1.
        row=dict(compact=torch.ones(2,2,2,2),compact_origin=torch.tensor([3,3,3]),
                 crop_shape=(8,8,8),roi=F.interpolate(full,size=4,mode='trilinear',align_corners=False)[0])
        shifted=full.new_zeros(full.shape);shifted[:,:,2:4,3:5,3:5]=1.
        torch.testing.assert_close(official_view(row,[1,0,0]),
            F.interpolate(shifted,size=4,mode='trilinear',align_corners=False)[0])
        torch.testing.assert_close(official_view(row,[0,0,0]),row['roi'])
        torch.testing.assert_close(official_view(row,[20,0,0]),row['roi'])

    def test_cached_jitter_matches_direct_ct_crop(self):
        from back_prop.model_v5.warmup import official_row
        from back_prop.model_v5.augmentation import official_view
        from back_prop.common.roi import integer_crop, paste_compact_mask
        m,b=tiny();row=official_row(m,b,0)
        torch.testing.assert_close(official_view(row,[0,0,0]),row['roi'],rtol=0,atol=0)
        center=m._box_center_voxel(b['target_boxes'][0],b['image'].shape[-3:])+torch.tensor([1,0,0])
        ct,origin,valid=integer_crop(b['image'],center,row['crop_shape'])
        mask=paste_compact_mask(b['target_mask_crops'][0],b['target_mask_origins'][0],
                                origin,row['crop_shape']).float()[None,None]
        logits=torch.where(mask>0,torch.inf,-torch.inf)
        direct=m.semantics.prepare_roi(ct[None],logits,valid[None])[0]
        torch.testing.assert_close(official_view(row,[1,0,0]),direct,rtol=0,atol=0)


if __name__ == '__main__':
    unittest.main()
