"""Convergence must require a plateau, lower learning rates, and usable models."""
import unittest
from back_prop.model_v5.convergence import PlateauStop


class ConvergenceTests(unittest.TestCase):
    def test_quality_gate_alone_cannot_stop_an_improving_loss(self):
        c = PlateauStop()
        c.baseline(1.)
        for i in range(100):
            result = c.update(.99**(i+1), .99**(i+1), True)
            self.assertFalse(result['converged'])

    def test_plateau_requires_additional_training_and_lr_reductions(self):
        c = PlateauStop()
        c.baseline(1.)
        for i in range(20):
            result = c.update(1., 1., True)
            self.assertEqual(result['converged'], i == 19)
        self.assertEqual(result['reductions'], 3)
        self.assertEqual(result['lr_ratio'], .125)

    def test_improving_training_loss_prevents_premature_probe_stop(self):
        c = PlateauStop()
        c.baseline(1.)
        for i in range(40):
            result = c.update(1., .98**i, True)
            self.assertFalse(result['converged'])

    def test_regression_and_failed_quality_do_not_count_as_convergence(self):
        for probe, passed in [(1.2, True), (1., False)]:
            c = PlateauStop()
            c.baseline(1.)
            for _ in range(40):
                self.assertFalse(c.update(probe, 1., passed)['converged'])

    def test_resumed_controller_has_identical_decisions(self):
        c = PlateauStop()
        c.baseline(1.)
        for i in range(12):
            c.update(1.-i*.0001, .9, True)
        resumed = PlateauStop(**c.state_dict())
        for i in range(20):
            self.assertEqual(c.update(.999, .9, True), resumed.update(.999, .9, True))

    def test_nonfinite_is_an_error(self):
        c = PlateauStop()
        with self.assertRaises(ValueError):
            c.update(float('nan'), 1., True)

    def test_preparation_export_requires_convergence_not_just_readiness(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        import torch
        from back_prop.model_v5.continue_warmup import export_preparation, CONTINUATION_SCHEMA
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'detr').mkdir()
            torch.save(dict(continuation_schema=CONTINUATION_SCHEMA, ready=True,
                convergence_complete=False), root/'detr/converged.pt')
            with self.assertRaisesRegex(ValueError, 'has not converged'):
                export_preparation(SimpleNamespace(output_dir=root, source_prepared=root/'old'))
            self.assertFalse((root/'prepared').exists())


if __name__ == '__main__':
    unittest.main()
