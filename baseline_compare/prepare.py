"""Shared CT/consensus cache; no fitting and no detector inputs from annotations."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time

import cv2
import nibabel as nib
import numpy as np
from scipy import ndimage
import torch
import torchio as tio

from back_prop.common.base_data import WholeCTLIDCDataset
from .cohort import HERE, load, sha256

CACHE = Path('/usr/project/rudinlab/datasets/LIDC_IDRI/joint_model/baseline_compare/cache_v1')


def atomic_npy(path, value):
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:
        np.save(stream, value)
    temporary.replace(path)


def sybil_transform(raw, spacing, attention):
    """NIfTI RAS -> axial DICOM row/column order -> official Serie transforms.

    Preserve upstream's resize-before-spacing-resampling convention. The mask
    accompanies the exact image geometry and is used only for training loss.
    """
    axial = raw[::-1, ::-1, :].transpose(1, 0, 2)
    c, w = -600.5, 1499.0
    window = np.clip(((axial - c) / w + 0.5) * 65535, 0, 65535) // 256
    slices = np.stack([cv2.resize(window[:, :, z], (256, 256), interpolation=cv2.INTER_LINEAR)
                       for z in range(window.shape[2])], axis=-1)
    slices = (slices - 128.1722) / 87.1849
    mask = attention[::-1, ::-1, :].transpose(1, 0, 2)
    mask = np.stack([cv2.resize(mask[:, :, z], (256, 256), interpolation=cv2.INTER_NEAREST)
                     for z in range(mask.shape[2])], axis=-1)
    affine = np.diag([spacing[1], spacing[0], spacing[2], 1.0])
    subject = tio.Subject(image=tio.ScalarImage(tensor=slices[None], affine=affine),
                          attention=tio.LabelMap(tensor=mask[None], affine=affine))
    subject = tio.Resample((0.703125, 0.703125, 2.5))(subject)
    subject = tio.CropOrPad((256, 256, 200), padding_mode=0)(subject)
    return (subject.image.data[0].permute(2, 0, 1).numpy().astype(np.float16),
            subject.attention.data[0].permute(2, 0, 1).numpy().astype(np.uint8))


def prepare_case(row, cache_root, cohort_digest):
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    started = time.time()
    dest = Path(cache_root) / row['key']
    dest.mkdir(parents=True, exist_ok=True)
    metadata_path = dest / 'metadata.json'
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata['cohort_sha256'] != cohort_digest:
            raise ValueError(f'Stale cohort cache: {dest}')
        for name in ('hu_1mm.npy', 'sybil.npy', 'sybil_attention.npy'):
            array = np.load(dest / name, mmap_mode='r')
            assert list(array.shape) == metadata['array_shapes'][name]
        return {'key': row['key'], 'cached': True}
    image = nib.as_closest_canonical(nib.load(row['image']))
    raw = np.asarray(image.dataobj, dtype=np.float32)
    if not np.isfinite(raw).all():
        raise ValueError(f'Nonfinite CT: {row["key"]}')
    spacing = nib.affines.voxel_sizes(image.affine)
    volume = ndimage.zoom(raw, spacing, order=1, mode='nearest', prefilter=False)
    atomic_npy(dest / 'hu_1mm.npy', volume)
    shape = volume.shape
    del volume
    helper = WholeCTLIDCDataset([row])
    attention = np.zeros(raw.shape, dtype=np.uint8)
    nodules, skipped_empty = [], []
    for item in row['nodules']:
        consensus = helper._consensus_mask(row['patient_id'], str(int(row['scan_id'])),
                                          tuple(item['annotation_ids']), image)
        if not consensus.any():
            # V3 skips an empty reader-vote consensus before building targets.
            # Keep the scan and record the unusable annotation explicitly.
            skipped_empty.append(item['nodule_id'])
            continue
        if not item['ignored']:
            attention[consensus] = 1
        iso = ndimage.zoom(consensus.astype(np.uint8), spacing, order=0, mode='nearest', prefilter=False)
        locations = np.argwhere(iso)
        del iso, consensus
        if not len(locations):
            skipped_empty.append(item['nodule_id'])
            continue
        lower, upper = locations.min(0), locations.max(0) + 1
        # Geometry used for the LIDC adaptation of EDICNet semantic labels.
        diameter_xy = float(np.mean((upper - lower)[:2]))
        nodules.append({**item, 'box_xyzxyz': [*lower.tolist(), *upper.tolist()],
                        'center_xyz': ((lower + upper) / 2).tolist(),
                        'diameter_mm': float(max(upper - lower)),
                        'mean_axial_bbox_diameter_mm': diameter_xy,
                        'edicnet_semantic_targets': [int(diameter_xy > 10) + int(diameter_xy > 20),
                                                     int(item['semantic_means'][4] >= 4),
                                                     int(not (item['semantic_means'][1] >= 4
                                                              and item['semantic_means'][2] <= 2
                                                              and item['semantic_means'][3] <= 2))]})
    sybil, sybil_attention = sybil_transform(raw, spacing, attention)
    atomic_npy(dest / 'sybil.npy', sybil)
    atomic_npy(dest / 'sybil_attention.npy', sybil_attention)
    risk = max((n['target'] for n in nodules if not n['ignored']), default=0)
    risk_valid = bool(risk or not any(n['ignored'] for n in nodules))
    metadata = {**row, 'nodules': nodules, 'cohort_sha256': cohort_digest,
                'skipped_empty_consensus_nodules': skipped_empty,
                'risk_target': risk, 'risk_target_valid': risk_valid,
                'raw_shape_xyz': list(raw.shape), 'raw_spacing_xyz': spacing.tolist(),
                'raw_canonical_affine': image.affine.tolist(),
                'array_shapes': {'hu_1mm.npy': list(shape), 'sybil.npy': list(sybil.shape),
                                 'sybil_attention.npy': list(sybil_attention.shape)},
                'sybil_has_visible_annotation': bool(sybil_attention.any()),
                'seconds': time.time() - started}
    temporary = metadata_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(metadata, indent=2) + '\n')
    temporary.replace(metadata_path)
    return {'key': row['key'], 'seconds': metadata['seconds'],
            'sybil_has_visible_annotation': metadata['sybil_has_visible_annotation']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort', type=Path, default=HERE / 'cohort.json')
    parser.add_argument('--cache', type=Path, default=CACHE)
    parser.add_argument('--split', choices=['training', 'testing'], default='training')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    payload = load(args.cohort)
    rows = payload['splits'][args.split]
    if args.limit:
        rows = rows[:args.limit]
    digest = sha256(args.cohort)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks = [pool.submit(prepare_case, row, args.cache, digest) for row in rows]
        for i, task in enumerate(as_completed(tasks), 1):
            print(json.dumps({'done': i, 'total': len(rows), **task.result()}), flush=True)
    print(json.dumps({'status': 'complete', 'split': args.split, 'cases': len(rows),
                      'cohort_sha256': digest}), flush=True)
