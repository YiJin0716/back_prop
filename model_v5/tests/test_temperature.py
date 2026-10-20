"""Check annealed forward masks, gradients, checkpoints and warmup reuse."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from back_prop.common.schedule import TemperatureSchedule
from back_prop.model_v5.loss import WholeCTCriterionV5
from back_prop.model_v5.parallel_warmup import validate_parallel_resume
from back_prop.model_v5.preparation import SCHEMA
from back_prop.model_v5.tests.test_v5 import tiny
from back_prop.model_v5.train import parse_args


class TemperatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_actual_forward_sharpens_masks_and_preserves_segmentation_objective(self):
        model, batch = tiny()
        model.eval()
        anchor = torch.rand(2, 2, 8, 8, 8)
        kwargs = dict(batch=batch, object_threshold=0., update_bank=False,
                      semantic_anchor_roi=anchor)
        semantic_inputs, radiomics_inputs = [], []
        handles = [model.semantics.register_forward_pre_hook(
            lambda _, inputs: semantic_inputs.append(inputs[1].detach().clone()) if len(inputs) == 3 else None),
            model.radiomics.register_forward_pre_hook(
                lambda _, inputs: radiomics_inputs.append(inputs[1].detach().clone()))]
        try:
            with torch.no_grad():
                soft = model(batch['image'], **kwargs)
                raw_semantic = semantic_inputs.copy()
                raw_radiomics = radiomics_inputs.copy()
                semantic_inputs.clear(); radiomics_inputs.clear()
                model.set_temperature(.1)
                sharp = model(batch['image'], **kwargs)
        finally:
            for handle in handles:
                handle.remove()
        torch.testing.assert_close(soft.fine_mask_logits, sharp.fine_mask_logits, rtol=0, atol=0)
        torch.testing.assert_close(soft.fine_mask_probabilities,
            soft.fine_mask_logits.sigmoid() * soft.roi_valid, rtol=0, atol=0)
        self.assertGreater(len(raw_semantic), 0)
        for old, new in zip(raw_semantic + raw_radiomics, semantic_inputs + radiomics_inputs):
            torch.testing.assert_close(new, old / model.mask_temperature)
        p, q = soft.fine_mask_probabilities, sharp.fine_mask_probabilities
        valid = soft.roi_valid
        self.assertTrue((abs(q[valid]-.5) >= abs(p[valid]-.5)-1e-7).all())
        self.assertLess(float((q*(1-q))[valid].mean()), float((p*(1-p))[valid].mean()))
        self.assertTrue(torch.equal(p >= .5, q >= .5))
        self.assertFalse(torch.allclose(soft.semantic_features, sharp.semantic_features))
        torch.testing.assert_close(soft.anchor_probabilities, sharp.anchor_probabilities, rtol=0, atol=0)
        self.assertTrue(torch.equal(soft.semantic_supervision_valid, sharp.semantic_supervision_valid))
        criterion = WholeCTCriterionV5()
        losses = [criterion(output, batch, synchronize=False) for output in (soft, sharp)]
        for name in ('fine_dice', 'fine_focal_positive', 'fine_focal_negative', 'coarse_dice'):
            torch.testing.assert_close(losses[0].numerators[name], losses[1].numerators[name], rtol=0, atol=0)

    def test_low_temperature_semantic_gradient_reaches_refiner(self):
        model, batch = tiny()
        model.eval()
        model.set_temperature(.1)
        output = model(batch['image'], batch=batch, object_threshold=0., update_bank=False)
        (-output.semantic_probabilities[:, :, 0].log().mean()).backward()
        gradient = model.refiner.residual_head.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0.)

    def test_schedule_resume_and_legacy_defaults(self):
        schedule = TemperatureSchedule(start=1., end=.1, epochs=18)
        values = [schedule(epoch) for epoch in range(18)]
        self.assertEqual(values[0], 1.)
        self.assertEqual(values[-1], .1)
        self.assertTrue(all(a > b for a, b in zip(values, values[1:])))
        self.assertEqual(schedule(30), .1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'checkpoint.pt'
            for config in ({}, dict(mask_temperature_start=1., mask_temperature_end=.1,
                                    mask_temperature_epochs=18)):
                torch.save(dict(config=config), path)
                args = parse_args(['--output-dir', directory, '--resume', str(path)])
                resumed = TemperatureSchedule(args.mask_temperature_start,
                    args.mask_temperature_end, args.mask_temperature_epochs)
                self.assertEqual(resumed(11), schedule(11) if config else 1.)

    def test_annealing_can_reuse_preexisting_warmup(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory)/'manifest.json'
            manifest.write_text('{}')
            args = SimpleNamespace(manifest=manifest, mask_temperature_start=1.,
                mask_temperature_end=.1, mask_temperature_epochs=18)
            saved = dict(preparation_schema=SCHEMA, world_size=4,
                manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                config=dict(manifest=str(manifest)))
            validate_parallel_resume(args, saved, 4)

    def test_temperature_validation(self):
        model, _ = tiny()
        for value in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                model.set_temperature(value)


if __name__ == '__main__':
    unittest.main()
