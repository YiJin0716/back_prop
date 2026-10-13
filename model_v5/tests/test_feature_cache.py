"""Frozen-feature caching must retain projection learning and crop geometry."""
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from back_prop.common.semantic_model import WholeCTJointModelV3
from back_prop.model_v5.feature_cache import cached_screening, extract_raw_features


class SmallSegmenter(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Conv3d(1,3,1)
        self.pyramid_projections = nn.ModuleList([nn.Conv3d(3,2,1)])

    def forward_with_features(self, x):
        raw = self.model(x)
        return raw[:,:1], (self.pyramid_projections[0](raw),)


class CacheTests(unittest.TestCase):
    def test_overlapping_and_padded_windows_preserve_outputs_and_projection_gradients(self):
        torch.set_num_threads(2)
        for shape in ((11,13,9),(5,6,3)):
            with self.subTest(shape=shape):
                torch.manual_seed(7)
                segmenter = SmallSegmenter()
                segmenter.model.eval().requires_grad_(False)
                model = SimpleNamespace(segmenter=segmenter,device=torch.device('cpu'),
                    amp_dtype=torch.bfloat16,window_size=(8,8,8),screen_shape=(7,6,5),
                    vista_feature_dim=2,overlap=.25)
                def window(image, starts):
                    patch = image[(...,*(slice(s,s+8) for s in starts))]
                    return torch.nn.functional.pad(patch,tuple(v for n in reversed(patch.shape[-3:]) for v in (0,8-n)))
                model._window = window
                image = torch.randn(1,*shape)
                cache, reference = extract_raw_features(model,image,reference=True)
                got = cached_screening(segmenter,cache)
                torch.testing.assert_close(got,reference,atol=5e-7,rtol=2e-6)
                # Compare to the actual original discovery path, including its
                # derivative through each independently projected window.
                model.training = False
                model.use_checkpoint = False
                model._vista_window_summary = lambda *a: WholeCTJointModelV3._vista_window_summary(model,*a)
                original = WholeCTJointModelV3.discover(model,image)[0]
                torch.testing.assert_close(got,original,atol=5e-7,rtol=2e-6)
                weights = torch.randn_like(original)
                params = tuple(segmenter.pyramid_projections.parameters())
                grad_original = torch.autograd.grad((original*weights).sum(),params)
                grad_cached = torch.autograd.grad((got*weights).sum(),params)
                for a,b in zip(grad_original,grad_cached):
                    torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-5)
                self.assertTrue(all(p.grad is None for p in segmenter.model.parameters()))
                # Cache cannot accidentally hide a subsequently unfrozen backbone.
                segmenter.model.requires_grad_(True)
                with self.assertRaisesRegex(ValueError,'frozen backbone'):
                    cached_screening(segmenter,cache)


if __name__=='__main__': unittest.main()
