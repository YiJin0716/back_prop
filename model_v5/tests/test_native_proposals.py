import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from back_prop.common.semantic_model import WholeCTJointModelV3
from back_prop.model_v5.native_proposals import discover_native,native_proposals,native_roi
from back_prop.model_v5.tests.test_feature_cache import SmallSegmenter


class NativeProposalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_native_retention_preserves_stitched_features_and_projection_gradients(self):
        for shape in ((11,13,9),(5,6,3)):
            torch.manual_seed(7)
            segmenter=SmallSegmenter();segmenter.model.eval().requires_grad_(False)
            model=SimpleNamespace(segmenter=segmenter,device=torch.device('cpu'),
                amp_dtype=torch.bfloat16,window_size=(8,8,8),screen_shape=(7,6,5),
                vista_feature_dim=2,overlap=.25,training=False,use_checkpoint=False)
            def window(image,starts):
                patch=image[(...,*(slice(s,s+8) for s in starts))]
                return F.pad(patch,tuple(v for n in reversed(patch.shape[-3:]) for v in (0,8-n)))
            model._window=window
            model._vista_window_summary=lambda *a:WholeCTJointModelV3._vista_window_summary(model,*a)
            image=torch.randn(1,*shape)
            original=WholeCTJointModelV3.discover(model,image)
            screen,probability,native,windows=discover_native(model,image)
            torch.testing.assert_close(screen,original[0],atol=5e-7,rtol=2e-6)
            torch.testing.assert_close(probability,original[1],atol=5e-7,rtol=2e-6)
            self.assertEqual(windows,original[3])
            torch.testing.assert_close(native,segmenter.model(image[None])[0,0].sigmoid(),atol=1e-6,rtol=1e-6)
            self.assertFalse(native.requires_grad)
            weights=torch.randn_like(screen);params=tuple(segmenter.pyramid_projections.parameters())
            first=torch.autograd.grad((screen*weights).sum(),params)
            second=torch.autograd.grad((original[0]*weights).sum(),params)
            for a,b in zip(first,second):torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-5)

    def test_tiny_components_survive_coarse_projection_and_keep_native_centers(self):
        p=torch.zeros(32,32,32)
        p[1:3,4:6,7:9]=.9;p[25:27,22:24,19:21]=.8
        # An isolated noise voxel is below the fixed size requirement.
        p[15,15,15]=.99
        proposals=native_proposals(p,(4,4,4),margin=2)
        self.assertEqual(len(proposals.boxes),2)
        torch.testing.assert_close(proposals.boxes[:,:3]*32,torch.tensor([[2.,5.,8.],[26.,23.,20.]]))
        self.assertTrue((proposals.coarse_logits.sigmoid().flatten(1).amax(1)>.7).all())
        origin=torch.tensor([-2,1,4])
        roi=native_roi(proposals,0,origin,(8,8,8)).sigmoid()[0,0]
        torch.testing.assert_close(roi[3:5,3:5,3:5],torch.full((2,2,2),.9))
        self.assertLess(float(roi[:2].max()),2e-6)
        empty=native_proposals(torch.zeros_like(p),(4,4,4))
        self.assertEqual(empty.boxes.shape,(0,6))
        self.assertEqual(empty.coarse_logits.shape,(0,4,4,4))
        self.assertEqual(len(native_proposals(p,(4,4,4),maximum=1).boxes),1)

    def test_assembled_refiner_starts_in_its_warmup_input_domain_and_gains_can_learn(self):
        from back_prop.model_v5.tests.test_v5 import tiny
        model,batch=tiny(detector_kind='native_seeded');model.eval()
        with torch.no_grad():model.segmenter.scale.fill_(20.)
        observed=[]
        hook=model.refiner.register_forward_pre_hook(lambda _module,args:observed.append(args))
        result=model(batch['image'],batch=batch,object_threshold=0.,geometry_only=True,update_bank=False)
        hook.remove()
        self.assertGreater(len(observed),0)
        for value,query in observed:
            self.assertEqual(float(value[:,1].detach().abs().sum()),0.)
            self.assertEqual(float(value[:,4:].detach().abs().sum()),0.)
            # Zero-initialized FiLM preserves the neutral-query output while
            # keeping a live query input to learn its weights in the first step.
            torch.testing.assert_close(model.refiner(value,query),model.refiner(value,torch.zeros_like(query)))
        result.fine_mask_logits.mean().backward()
        self.assertIsNotNone(model.refiner_conditioning.grad)
        self.assertTrue(torch.isfinite(model.refiner_conditioning.grad).all())
        self.assertGreater(float(model.refiner_conditioning.grad[:2].abs().sum()),0.)
        self.assertGreater(float(model.refiner.encoder_1.film.weight.grad.abs().sum()),0.)
        with torch.no_grad():
            reference=model(batch['image'],batch=batch,object_threshold=0.,geometry_only=True,update_bank=False)
            inference=model(batch['image'],image_hu=batch['image_hu'],object_threshold=0.,
                geometry_only=True,update_bank=False)
            for field in ('object_logits','boxes','coarse_mask_logits','crop_origins','fine_mask_logits'):
                torch.testing.assert_close(getattr(reference,field),getattr(inference,field),rtol=0,atol=0)


if __name__=='__main__':unittest.main()
