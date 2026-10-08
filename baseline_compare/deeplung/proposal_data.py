"""Detector-centered DPN classification samples, matching the det2cls workflow."""
import json
from pathlib import Path
import numpy as np

from ..cohort import HERE, sha256
from ..nodule_data import NoduleDataset, crop_cube


def add_proposals(dataset: NoduleDataset, rows, cache, path, smoke=False, cohort_path=HERE / 'cohort.json'):
    payload = json.loads(Path(path).read_text())
    assert payload['cohort_sha256'] == sha256(cohort_path)
    assert payload['split'] == 'training'
    by_key = {case['key']: case for case in payload['cases']}
    assert len(by_key) == len(payload['cases'])
    expected = {row['key'] for row in rows}
    if not smoke:
        assert not payload['smoke_subset'] and set(by_key) == expected
    else:
        assert expected <= set(by_key)
    # Preserve all original physical-nodule samples so detector misses cannot
    # silently remove any of the 638 V3 CTs from training.
    patches, added, ignored, negatives = [], [], 0, 0
    for row in rows:
        directory = Path(cache) / row['key']
        meta = json.loads((directory / 'metadata.json').read_text())
        volume = np.load(directory / 'hu_1mm.npy', mmap_mode='r')
        nodules = meta['nodules']
        centers = np.asarray([n['center_xyz'] for n in nodules])
        radii = np.asarray([max(16., n['diameter_mm'] / 2) for n in nodules])
        for number, proposal in enumerate(by_key[row['key']]['candidates']):
            probability, x, y, z, diameter = proposal
            center = np.asarray([x, y, z])
            distance = np.linalg.norm(centers - center, axis=1)
            matches = np.flatnonzero(distance < radii)
            if len(matches):
                nearest = matches[np.argmin(distance[matches])]
                if nodules[nearest]['ignored']:
                    ignored += 1
                    continue
                label = nodules[nearest]['target']
            else:
                label = 0
                negatives += 1
            patch, _ = crop_cube(volume, center, 32)
            patch = (np.clip((patch + 1200) / 1800, 0, 1) * 255).astype(np.uint8).astype(np.float32)
            patches.append(patch)
            added.append({'case_key': row['key'], 'target': label, 'center_xyz': center.tolist(),
                          'diameter_mm': diameter, 'source': 'detector_proposal',
                          'proposal_index': number, 'detection_probability': probability,
                          'edicnet_semantic_targets': [-1, -1, -1]})
    if not added:
        raise ValueError('No eligible detected proposals for classifier adaptation')
    dataset.patches = np.concatenate([dataset.patches, np.stack(patches)], axis=0)
    dataset.items.extend(added)
    dataset.mean = float(dataset.patches.mean(dtype=np.float64))
    dataset.std = float(dataset.patches.std(dtype=np.float64))
    assert dataset.std > 0 and {item['case_key'] for item in dataset.items} == expected
    return {'proposals_added': len(added), 'unmatched_negative_proposals': negatives,
            'ignored_proposals': ignored, 'gt_samples_retained': len(dataset) - len(added),
            'proposal_manifest_sha256': sha256(path), 'detector_sha256': payload['detector_sha256']}
