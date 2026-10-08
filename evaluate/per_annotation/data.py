"""Reader-level cohort and official-mask ROIs on the training CT grid."""
from __future__ import annotations

from pathlib import Path
import hashlib

import numpy as np
from scipy.ndimage import zoom
import torch
from torch.utils.data import Dataset

from back_prop.evaluate.eval_package.annotations import (
    OfficialAnnotations, case_key, load_image, load_test_cases,
)
from back_prop.common.base_data import _read_official_mask
from back_prop.common.roi import integer_crop


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def build_cohort(manifest):
    cases = sorted(load_test_cases(manifest, 'testing'), key=case_key)
    official = OfficialAnnotations(cases=cases)
    cohort = []
    for case in cases:
        cid = case_key(case)
        annotations = []
        for nid, readers in sorted(official.grouped.get(cid, {}).items()):
            for aid, ratings in sorted(readers):
                annotations.append(dict(
                    annotation_key=f'{cid}__ann{aid}', case_id=cid,
                    patient_id=case['patient_id'], scan_id=int(case['scan_id']),
                    nodule_id=nid, annotation_id=aid,
                    mask_path=str(official.mask_dir/f'{cid}__ann{aid}_mask.nii.gz'),
                    image_path=case['image'], **{f'gt_{k}':int(v) for k,v in ratings.items()},
                    malignancy_target=None if ratings['malignancy']==3 else int(ratings['malignancy']>3),
                ))
        cohort.append(dict(case=case, case_id=cid, annotations=annotations))
    keys = [a['annotation_key'] for c in cohort for a in c['annotations']]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate reader annotations in the test cohort')
    return cohort, dict(
        test_cases=len(cases), test_patients=len({c['patient_id'] for c in cases}),
        physical_nodules=sum(len(nodes) for nodes in official.grouped.values()),
        annotations=len(keys), manifest=str(Path(manifest).resolve()),
        manifest_sha256=sha256(manifest),
        annotations_csv=str(official.annotations_csv),
        annotations_csv_sha256=sha256(official.annotations_csv),
        identities_csv=str(official.identity_csv),
        identities_csv_sha256=sha256(official.identity_csv),
        mask_directory=str(official.mask_dir),
    )


def crop_annotation(hu, mask, base_shape):
    """Binary GT crop: >=128^3, expanded if needed to retain the entire mask.

    Match official warmup's bounding-box center and integer routing. No
    consensus, connected-component filtering or malignancy-based selection.
    """
    positions = np.nonzero(mask)
    if not len(positions[0]):
        return None
    lower = np.array([p.min() for p in positions])
    upper = np.array([p.max()+1 for p in positions])
    shape = tuple(max(int(base), int(n)+4) for base,n in zip(base_shape, upper-lower))
    # The model converts normalized voxel-edge boxes back to voxel indices
    # with center * scan_shape - 0.5.
    center = (lower + upper)/2.0 - 0.5
    patch, origin, valid = integer_crop(torch.from_numpy(hu)[None], center, shape)
    mask_patch, _, _ = integer_crop(torch.from_numpy(mask)[None], center, shape)
    if int(mask_patch.sum()) != len(positions[0]):
        raise ValueError('ROI crop lost official mask voxels')
    return dict(hu=patch, mask=mask_patch, valid=valid,
                origin=origin.tolist(), shape=list(shape), mask_voxels=int(mask_patch.sum()))


class AnnotationCases(Dataset):
    """Load each CT once, with independent reader masks; workers are CPU only."""
    def __init__(self, cohort, roi_shape):
        self.cohort, self.roi_shape = cohort, roi_shape

    def __len__(self):
        return len(self.cohort)

    def __getitem__(self, index):
        item = self.cohort[index]
        if not item['annotations']:
            return dict(case_id=item['case_id'], annotations=[])
        hu, affine, source, factors = load_image(item['case'])
        annotations = []
        for entry in item['annotations']:
            native = _read_official_mask(Path(entry['mask_path']), source)
            native_voxels = int(native.sum())
            mask = zoom(native, factors, order=0, mode='nearest', prefilter=False)
            if mask.shape != hu.shape:
                raise ValueError('Reader mask and resampled CT grids differ')
            crop = crop_annotation(hu, mask, self.roi_shape)
            row = dict(entry, native_mask_voxels=native_voxels,
                       mask_sha256=sha256(entry['mask_path']),
                       grid_shape=list(hu.shape), grid_affine=affine.tolist())
            row['input_status'] = ('ok' if crop is not None else
                                   'empty_native_mask' if not native_voxels else 'empty_resampled_mask')
            annotations.append(dict(row=row, crop=crop))
        return dict(case_id=item['case_id'], annotations=annotations)


def identity(item):
    return item


def worker_init(_):
    torch.set_num_threads(1)
