import unittest

import torch

from back_prop.common.geometry import V3MaskDETR
from back_prop.model_v5.detector import MaskGuidedDETR,NativeSeededDETR,mask_geometry,select_mask_queries


class MaskGuidedDetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_predicted_masks_control_localization_and_local_features(self):
        logits=torch.full((1,2,12,10,8),-20.)
        logits[0,0,2:4,3:5,1:3]=20.
        logits[0,1,8:10,6:8,5:7]=20.
        coordinates=torch.stack(torch.meshgrid(*(
            (torch.arange(n)+.5)/n for n in (12,10,8)),indexing='ij'))[None]
        center,extent,pooled,stats=mask_geometry(logits,coordinates)
        wanted=torch.tensor([[[3/12,4/10,2/8],[9/12,7/10,6/8]]])
        torch.testing.assert_close(center,wanted,atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(pooled,wanted,atol=1e-6,rtol=1e-6)
        self.assertTrue((extent>0).all())
        self.assertTrue(torch.isfinite(stats).all())

    def test_empty_and_diffuse_masks_are_finite_with_geometry_gradients(self):
        torch.manual_seed(2)
        logits=(torch.randn(1,2,8,8,8)-1000).requires_grad_()
        features=torch.randn(1,4,8,8,8,requires_grad=True)
        center,extent,pooled,stats=mask_geometry(logits,features)
        loss=(center-torch.tensor([.2,.3,.4])).square().sum()+extent.sum()+pooled.square().sum()
        loss.backward()
        self.assertTrue(torch.isfinite(stats).all())
        for tensor in (logits,features):
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(float(tensor.grad.abs().sum()),0)

    def test_weight_transfer_preserves_masks_and_inference_is_batch_independent(self):
        torch.manual_seed(9)
        old=V3MaskDETR(num_queries=4,hidden_dim=16,coarse_shape=(3,3,3),
            nheads=4,encoder_layers=1,decoder_layers=1,input_channels=4,mask_dim=8).eval()
        new=MaskGuidedDETR.from_legacy(old).eval()
        x=torch.randn(2,4,8,8,8)
        with torch.no_grad():
            before=old(x);after=new(x)
            torch.testing.assert_close(before[2],after[2])
            torch.testing.assert_close(before[3],after[3])
            for field in range(4):
                separately=torch.cat([new(row[None])[field] for row in x])
                torch.testing.assert_close(after[field],separately,atol=1e-5,rtol=1e-5)
            self.assertTrue((after[1][...,:3]>=0).all() and (after[1][...,:3]<=1).all())
            self.assertTrue((after[1][...,3:]>0).all())

    def test_background_weight_keeps_actual_counts_and_uses_weighted_denominator(self):
        from back_prop.model_v5.tests.test_v5 import tiny
        from back_prop.model_v5.parallel_warmup import DetectorBranch,coarse_output
        from back_prop.model_v5.loss import WholeCTCriterionV5,LossAccumulator
        model,batch=tiny()
        output=coarse_output(DetectorBranch(model)(batch['image']),batch)
        criterion=WholeCTCriterionV5(object_background_weight=.1)
        result=criterion(output,batch,synchronize=False)
        labels=torch.zeros_like(output.object_logits)
        labels[output.matched_query_indices]=1
        weights=torch.where(labels>0,1.,.1)
        expected=(torch.nn.functional.binary_cross_entropy_with_logits(
            output.object_logits,labels,reduction='none')*weights).sum()/weights.sum()
        self.assertAlmostEqual(result.mean('object'),float(expected.detach()),places=6)
        self.assertEqual(result.counts['object'],4)
        self.assertAlmostEqual(result.normalizers['object'],1.3,places=6)
        acc=LossAccumulator();acc.update(result)
        metrics=acc.compute(criterion.loss_weights,synchronize=False)
        self.assertEqual(metrics['loss_counts']['object'],4)
        self.assertAlmostEqual(metrics['object'],float(expected.detach()),places=6)

    def test_duplicate_suppression_keeps_distinct_instances(self):
        masks=torch.full((4,8,8,8),-10.)
        masks[0,:3,:3,:3]=10.;masks[1]=masks[0]
        masks[2,5:,5:,5:]=10.
        scores=torch.logit(torch.tensor([.9,.8,.85,.1]))
        self.assertEqual(select_mask_queries(scores,masks).tolist(),[0,2])
        self.assertEqual(select_mask_queries(scores,masks,threshold=1.).numel(),0)

    def test_repaired_joint_forward_does_not_use_gt_to_choose_predictions(self):
        from back_prop.model_v5.tests.test_v5 import tiny
        model,batch=tiny(detector_kind='mask_guided');model.eval()
        with torch.no_grad():
            model.detr.prediction_head.layers[-1].bias[0]=1.
            a=model(batch['image'],batch=batch,geometry_only=True,update_bank=False)
            b=model(batch['image'],image_hu=batch['image_hu'],geometry_only=True,update_bank=False)
            self.assertGreater(len(a.refined_query_indices),0)
            for field in ('object_logits','boxes','coarse_mask_logits','refined_query_indices','crop_origins','fine_mask_logits'):
                torch.testing.assert_close(getattr(a,field),getattr(b,field),atol=0,rtol=0)
            changed=dict(batch,target_boxes=batch['target_boxes'].clone())
            changed['target_boxes'][:,:3]=.95
            c=model(batch['image'],batch=changed,geometry_only=True,update_bank=False)
            for field in ('refined_query_indices','crop_origins','fine_mask_logits'):
                torch.testing.assert_close(getattr(a,field),getattr(c,field),atol=0,rtol=0)
            empty=model(batch['image'],batch=batch,geometry_only=True,object_threshold=1.,update_bank=False)
            self.assertEqual(len(empty.refined_query_indices),0)

    def test_detection_metric_maximizes_true_matches_and_ignores_excluded_nodules(self):
        from back_prop.model_v5.detection_metrics import detection_counts
        centers=torch.tensor([[12.,16.,16.],[15.,16.,16.]])
        gt=torch.cat((centers/32,torch.full((2,3),6/32)),dim=1)
        predicted_centers=torch.tensor([[12.3,16.,16.],[12.9,16.+(.81-.09)**.5*3,16.],[26.,26.,26.]])
        boxes=torch.cat((predicted_centers/32,torch.full((3,3),6/32)),dim=1)
        targets=torch.zeros(2,8,8,8);targets[0,2:4,3:5,3:5]=1;targets[1,3:5,3:5,3:5]=1
        masks=torch.cat((targets,torch.zeros_like(targets[:1])))*20-10
        batch=dict(target_boxes=gt,target_masks=targets,isotropic_shape=(32,32,32),
            target_mask_crops=[torch.ones(6,6,6,dtype=torch.uint8),torch.ones(6,6,6,dtype=torch.uint8)],
            target_mask_origins=torch.tensor([[9,13,13],[12,13,13]]),
            ignored_target_boxes=boxes[2:].clone(),ignored_target_masks=torch.zeros(1,8,8,8))
        result=detection_counts(torch.ones(3),boxes,masks,batch,torch.arange(3))
        self.assertEqual(result['true_positives'],2)
        self.assertEqual(result['false_positives'],0)
        empty=detection_counts(torch.ones(3),boxes,masks,batch,torch.empty(0,dtype=torch.long))
        self.assertEqual(empty['true_positives'],0)
        self.assertEqual(empty['coverage'],0)

    def test_native_priors_preserve_fine_centers_and_have_trainable_predictions(self):
        from back_prop.model_v5.native_proposals import native_proposals
        old=MaskGuidedDETR(num_queries=4,hidden_dim=16,coarse_shape=(3,3,3),
            nheads=4,encoder_layers=1,decoder_layers=1,input_channels=4,mask_dim=8).eval()
        model=NativeSeededDETR.from_mask_guided(old).eval()
        p=torch.zeros(32,32,32);p[1:3,4:6,7:9]=.9;p[25:27,22:24,19:21]=.8
        priors=native_proposals(p,(8,8,8))
        screen=torch.randn(1,4,8,8,8)
        before=old(screen);after=model(screen,priors)
        torch.testing.assert_close(after[1][0,:2],priors.boxes)
        torch.testing.assert_close(after[0][0,:2],priors.object_logits)
        torch.testing.assert_close(after[2][0,:2],priors.coarse_logits)
        for a,b in zip(before,after):torch.testing.assert_close(a[:,2:],b[:,2:])
        # Native seeds initialize predictions, rather than bypassing learning.
        loss=after[0][0,:2].sum()+after[1][0,:2].sum()+after[2][0,:2].sum()
        loss.backward()
        self.assertGreater(float(model.native_prediction.weight.grad.abs().sum()),0)
        self.assertGreater(float(model.native_mask_embedding.weight.grad.abs().sum()),0)
        empty=native_proposals(torch.zeros_like(p),(8,8,8))
        with torch.no_grad():
            for a,b in zip(old(screen),model(screen,empty)):torch.testing.assert_close(a,b)
            from back_prop.model_v5.native_proposals import NativeProposals
            crowded=NativeProposals(priors.boxes.repeat(2,1),priors.object_logits.repeat(2),
                priors.coarse_logits.repeat(2,1,1,1),priors.origins*2,priors.probabilities*2)
            self.assertLess(model.seeded_count(crowded),model.num_queries)
            # Crowded priors must not remove all independently movable queries.
            for a,b in zip(old(screen),model(screen,crowded)):
                torch.testing.assert_close(a[:,-1:],b[:,-1:])
        with self.assertRaisesRegex(ValueError,'requires prediction-derived'):
            model(screen)

    def test_native_confidence_cannot_erase_strong_priors_but_can_learn(self):
        from back_prop.model_v5.native_proposals import native_proposals
        base=MaskGuidedDETR(num_queries=4,hidden_dim=16,coarse_shape=(3,3,3),
            nheads=4,encoder_layers=1,decoder_layers=1,input_channels=4,mask_dim=8).eval()
        model=NativeSeededDETR.from_mask_guided(base,confidence_residual_bound=2.).eval()
        p=torch.zeros(32,32,32);p[1:3,4:6,7:9]=.95
        priors=native_proposals(p,(8,8,8));screen=torch.randn(1,4,8,8,8)
        with torch.no_grad():
            model.native_prediction.bias[0]=-100.
            prediction=model(screen,priors)[0][0,0]
            self.assertGreater(float(prediction.sigmoid()),.5)
            self.assertLessEqual(float((prediction-priors.object_logits[0]).abs()),2.)
            legacy=NativeSeededDETR.from_mask_guided(base).eval()
            legacy.load_state_dict(model.state_dict(),strict=True)
            self.assertLess(float(legacy(screen,priors)[0][0,0].sigmoid()),.01)
            model.native_prediction.bias[0]=-.7
        model(screen,priors)[0][0,0].backward()
        self.assertGreater(float(model.native_prediction.weight.grad.abs().sum()),0.)


if __name__=='__main__':unittest.main()
