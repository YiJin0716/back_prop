import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch

from back_prop.model_v5.model import WholeCTJointModelV5, ARCHITECTURE
from back_prop.model_v5.loss import WholeCTCriterionV5, LossAccumulator, TERM_NAMES
from back_prop.model_v5.features import CNNSemantics, check_semantic_spread
from back_prop.tests.model_fixtures import TinySegmenter, make_batch


def tiny():
    torch.manual_seed(9)
    with patch('back_prop.common.semantic_model.MedicalNetSemantics', side_effect=AssertionError('MedicalNet constructed')):
        model = WholeCTJointModelV5(window_size=(8,8,8), screen_shape=(8,8,8), roi_shape=(8,8,8),
            num_queries=4, detr_hidden_dim=16, detr_coarse_shape=(4,4,4), detr_nheads=4,
            detr_encoder_layers=1, detr_decoder_layers=1, vista_feature_dim=1, context_channels=2,
            refiner_base_channels=2, hard_negatives=1, teacher_jitter=0,
            segmenter_factory=TinySegmenter, semantic_roi_size=8, semantic_width=4, use_checkpoint=False)
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

    def test_checkpoint_loader_uses_cnn_without_medicalnet(self):
        from back_prop.evaluate.eval_package.inference import load_joint_model
        m,b=tiny()
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

    def test_geometry_teacher_reaches_prediction_only_before_readiness(self):
        from back_prop.model_v5.preparation import geometry_teacher
        self.assertEqual([geometry_teacher(i,5) for i in range(7)],[1.,.75,.5,.25,0.,0.,0.])

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
