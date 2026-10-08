"""V4 must run without importing the retired 64-feature diagnostic branch."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


class V4DependencyTests(unittest.TestCase):
    def test_training_and_inference_do_not_load_legacy_radiomics(self):
        # A fresh process catches accidental imports even when other tests
        # have already loaded compatibility modules into sys.modules.
        script = textwrap.dedent('''
            import importlib.abc
            import sys
            blocked = (
                'back_prop.frozen_joint', 'back_prop.frozen_extrac', 'radiomics64',
                'back_prop.model_v4.compat_radiomics',
                'back_prop.model_v4.compat_radiomics64',
                'back_prop.common.radiomics', 'back_prop.common.ordinal',
            )
            class BlockLegacyImports(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if any(fullname == p or fullname.startswith(p + '.') for p in blocked):
                        raise AssertionError('Unexpected legacy dependency: ' + fullname)
            sys.meta_path.insert(0, BlockLegacyImports())
            import torch
            torch.set_num_threads(2)
            import back_prop.model_v4.train
            import back_prop.model_v4_oversample.train
            import back_prop.evaluate.eval_package.inference
            from back_prop.model_v4.tests.test_v4 import V4Tests
            from back_prop.model_v4.loss import WholeCTCriterionV4
            model, batch = V4Tests().tiny()
            output = model(batch['image'], batch=batch,
                           teacher_probability=1., update_bank=False)
            assert output.radiomics_features.shape[1] == 18
            assert output.semantic_features.shape[1] == 6
            loss = WholeCTCriterionV4()(output, batch).total
            loss.backward()
            assert torch.isfinite(loss)
            assert any(p.grad is not None for p in model.semantics.parameters())
            model.eval()
            with torch.no_grad():
                for threshold in (0., 1.):
                    result = model(batch['image'], image_hu=batch['image_hu'],
                                   object_threshold=threshold, update_bank=False)
                    assert result.radiomics_features.shape[1] == 18
                    if threshold == 1.:
                        assert result.radiomics_features.shape[0] == 0
        ''')
        root = Path(__file__).resolve().parents[3]
        env = {**os.environ, 'PYTHONPATH': str(root), 'OMP_NUM_THREADS': '2',
               'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1'}
        result = subprocess.run([sys.executable, '-c', script], cwd=root, env=env,
                                text=True, capture_output=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
