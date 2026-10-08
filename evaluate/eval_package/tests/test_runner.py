"""Notebook cache, GPU errors and Slurm job reuse without submitting test jobs."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from back_prop.evaluate.eval_package.runner import ensure_case_result

MODULE = 'back_prop.evaluate.eval_package.runner'


class RunnerTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix='eval runner ')
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.kwargs = dict(checkpoint=self.root/'checkpoint.pt', manifest=self.root/'fold.json',
                           python=self.root/'python', case_id='test__scan1',
                           output_root=self.root/'outputs', root=self.root)
        for name in ('checkpoint', 'manifest', 'python'):
            self.kwargs[name].touch()
        self.folder = self.kwargs['output_root'] / self.kwargs['case_id']
        self.folder.mkdir(parents=True)

    def test_valid_cache_needs_no_cuda_or_subprocess(self):
        metadata = dict(checkpoint=str(self.kwargs['checkpoint']), manifest=str(self.kwargs['manifest']),
                        split='testing', mask_threshold=.5)
        (self.folder/'result.json').write_text(json.dumps(dict(
            case_id='test__scan1', minimum_iou=.1, metadata=metadata)))
        (self.folder/'arrays.npz').touch()
        with patch(MODULE+'.subprocess.run') as run:
            self.assertEqual(ensure_case_result(**self.kwargs), self.folder)
            run.assert_not_called()

    def test_explicit_local_without_gpu_explains_the_problem(self):
        with patch(MODULE+'._run', return_value='0'):
            with self.assertRaisesRegex(RuntimeError, '本机没有可用 CUDA GPU'):
                ensure_case_result(**self.kwargs, execution='local')

    def test_slurm_failure_includes_underlying_stderr(self):
        (self.folder/'inference_123.err').write_text('Underlying inference error: missing checkpoint')
        with patch(MODULE+'.shutil.which', return_value='/bin/slurm'), \
             patch(MODULE+'._run', return_value='123'), \
             patch(MODULE+'._job_state', return_value='FAILED'):
            with self.assertRaisesRegex(RuntimeError, 'Underlying inference error: missing checkpoint'):
                ensure_case_result(**self.kwargs, execution='slurm')
        self.assertEqual(json.loads((self.folder/'inference_job.json').read_text())['job_id'], '123')

    def test_retry_waits_for_existing_job_instead_of_resubmitting(self):
        with patch(MODULE+'.shutil.which', return_value='/bin/slurm'), \
             patch(MODULE+'._run', return_value='123'), \
             patch(MODULE+'._job_state', return_value='PENDING'), \
             patch(MODULE+'.time.monotonic', side_effect=[0, 2]):
            with self.assertRaises(TimeoutError):
                ensure_case_result(**self.kwargs, execution='slurm', max_wait_seconds=1)
        with patch(MODULE+'.shutil.which', return_value='/bin/slurm'), \
             patch(MODULE+'._run') as submit, \
             patch(MODULE+'._job_state', side_effect=['RUNNING', 'FAILED']):
            with self.assertRaisesRegex(RuntimeError, 'job 123 failed'):
                ensure_case_result(**self.kwargs)
            submit.assert_not_called()


if __name__ == '__main__':
    unittest.main()
