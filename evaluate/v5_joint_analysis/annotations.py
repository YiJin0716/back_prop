"""Reader annotation inputs and frozen component inference for V5 evaluation."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from back_prop.common.features import SEMANTIC_NAMES, RADIOMICS_NAMES
from back_prop.common.roi import integer_crop
from back_prop.evaluate.eval_package.annotations import (
    OfficialAnnotations, case_key, load_test_cases,
)


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


def bank_fingerprint(bank):
    digest = hashlib.sha256()
    for name, value in sorted(bank.named_buffers()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    for memory in (bank.memory, bank.radiomics_memory):
        for key, (features, label) in sorted(memory.items()):
            digest.update(repr(key).encode())
            digest.update(np.asarray(features).tobytes())
            digest.update(repr(label).encode())
    digest.update(str(bank.updates).encode())
    return digest.hexdigest()


def infer_annotation(entry, models, device, semantic_input):
    row, crop = entry['row'], entry['crop']
    if crop is None:
        return [dict(row, model=name) for name in models]
    hu = crop['hu'][None].to(device).float()
    mask = crop['mask'][None].to(device).bool()
    valid = crop['valid'][None].to(device).float()
    logits = torch.where(mask, torch.inf, -torch.inf)
    ct = (hu.clamp(-1024, 1024)+1024)/2048
    official_semantic = torch.tensor([[row['gt_'+name] for name in SEMANTIC_NAMES]],
                                    device=device, dtype=torch.float32)
    base_row = dict(row, crop_origin=crop['origin'], crop_shape=crop['shape'],
                    mask_voxels=crop['mask_voxels'])
    output = []
    with torch.inference_mode():
        # Shared fixed radiomics features from the individual official mask.
        radio = next(iter(models.values()))['radiomics'](hu, logits, valid)
        if not torch.isfinite(radio).all():
            raise ValueError('Nonfinite radiomics features')
        base_row.update({f'radiomics_{key}':float(value)
                         for key,value in zip(RADIOMICS_NAMES,radio[0].cpu())})
        for name, parts in models.items():
            probability, expected = parts['semantics'](ct, logits, valid)
            if not torch.isfinite(probability).all() or not torch.isfinite(expected).all():
                raise ValueError(f'{name}: nonfinite semantic predictions')
            p = probability[0].cpu().numpy()
            prediction = dict(base_row, model=name)
            for j, feature in enumerate(SEMANTIC_NAMES):
                prediction['pred_'+feature] = float(expected[0,j])
                prediction.update({f'{feature}_p{k+1}':float(p[j,k]) for k in range(5)})
            baseline = parts['risk'].baseline_logits(radio)
            prediction['radiomics_malignancy_probability'] = float(baseline.sigmoid()[0])
            inputs = dict(official=official_semantic, predicted=expected)
            for source in ('official','predicted') if semantic_input=='both' else (semantic_input,):
                ensemble, indices, correction = parts['risk'](inputs[source], baseline)
                score = ensemble.sigmoid().mean(1).clamp(1e-6,1-1e-6)
                if not torch.isfinite(score).all():
                    raise ValueError(f'{name}: nonfinite malignancy prediction')
                prediction[f'malignancy_probability_{source}'] = float(score[0])
                prediction[f'ensemble_probabilities_{source}'] = ensemble.sigmoid()[0].cpu().tolist()
                prediction[f'semantic_correction_logits_{source}'] = correction[0].cpu().tolist()
                prediction['bank_model_indices'] = indices.cpu().tolist()
            output.append(prediction)
    return output
