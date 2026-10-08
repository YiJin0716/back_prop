"""Fault-injection coverage for shared-filesystem annotation mask reads."""
import errno
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np

from back_prop.common.base_data import _read_official_mask


class MaskReadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / 'mask.nii.gz'
        self.values = np.zeros((4, 5, 6), dtype=np.uint8)
        self.values[1:3, 2:4, 3:5] = 1
        self.reference = nib.Nifti1Image(self.values, np.eye(4))
        nib.save(self.reference, self.path)

    def test_success_matches_original_voxels(self):
        np.testing.assert_array_equal(_read_official_mask(self.path, self.reference),
                                      self.values > 0)

    def test_missing_then_stale_then_success(self):
        loaded = nib.load(self.path)
        with patch('back_prop.common.base_data.nib.load', side_effect=[
            FileNotFoundError('temporary invisible path'),
            OSError(errno.ESTALE, 'stale handle'), loaded,
        ]) as load, patch('back_prop.common.base_data.time.sleep') as sleep:
            np.testing.assert_array_equal(_read_official_mask(self.path, self.reference),
                                          self.values > 0)
            self.assertEqual(load.call_count, 3)
            self.assertEqual([c.args[0] for c in sleep.call_args_list], [1.0, 2.0])

    def test_proxy_read_is_retried(self):
        class BrokenProxy:
            def __array__(self, dtype=None):
                raise OSError(errno.EIO, 'data read failed after header load')
        class BrokenImage:
            shape = (4, 5, 6)
            affine = np.eye(4)
            dataobj = BrokenProxy()
        loaded = nib.load(self.path)
        with patch('back_prop.common.base_data.nib.load', side_effect=[BrokenImage(), loaded]), \
             patch('back_prop.common.base_data.nib.as_closest_canonical', side_effect=lambda x, **kw: x), \
             patch('back_prop.common.base_data.time.sleep') as sleep:
            np.testing.assert_array_equal(_read_official_mask(self.path, self.reference),
                                          self.values > 0)
            sleep.assert_called_once_with(1.0)

    def test_persistent_missing_is_fatal_and_bounded(self):
        with patch('back_prop.common.base_data.nib.load', side_effect=FileNotFoundError('missing')) as load, \
             patch('back_prop.common.base_data.time.sleep') as sleep:
            with self.assertRaises(FileNotFoundError):
                _read_official_mask(self.path, self.reference)
            self.assertEqual(load.call_count, 6)
            self.assertEqual([c.args[0] for c in sleep.call_args_list], [1, 2, 4, 8, 16])

    def test_permission_error_is_not_retried(self):
        with patch('back_prop.common.base_data.nib.load', side_effect=PermissionError(errno.EACCES, 'denied')), \
             patch('back_prop.common.base_data.time.sleep') as sleep:
            with self.assertRaises(PermissionError):
                _read_official_mask(self.path, self.reference)
            sleep.assert_not_called()

    def test_wrong_geometry_is_not_retried(self):
        reference = nib.Nifti1Image(np.zeros((3, 3, 3), dtype=np.uint8), np.eye(4))
        with patch('back_prop.common.base_data.time.sleep') as sleep:
            with self.assertRaisesRegex(ValueError, 'geometry mismatch'):
                _read_official_mask(self.path, reference)
            sleep.assert_not_called()


if __name__ == '__main__':
    unittest.main()
