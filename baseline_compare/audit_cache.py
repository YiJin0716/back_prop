"""Audit every cached training case against the frozen V3 case/nodule ledger."""
import json
from pathlib import Path
import numpy as np

from .cohort import HERE, load, sha256
from .prepare import CACHE


def audit():
    cohort = load(); digest = sha256(HERE / 'cohort.json')
    rows = cohort['splits']['training']
    result = {'cohort_sha256': digest, 'training_case_keys_sha256': cohort['training_case_keys_sha256'],
              'cases': len(rows), 'patients': len({r['patient_id'] for r in rows}), 'retained_nodules': 0,
              'ignored_nodules_with_masks': 0, 'valid_scan_labels': 0, 'slices': 0,
              'empty_consensus_annotations': [], 'effective_scan_label_changes': [],
              'sybil_cases_without_visible_attention': []}
    seen = []
    for row in rows:
        directory = CACHE / row['key']
        meta = json.loads((directory / 'metadata.json').read_text())
        assert meta['key'] == row['key'] and meta['image'] == row['image']
        assert meta['cohort_sha256'] == digest
        declared = {n['nodule_id']: n for n in row['nodules']}
        observed = {n['nodule_id']: n for n in meta['nodules']}
        skipped = set(meta.get('skipped_empty_consensus_nodules', []))
        assert set(observed) | skipped == set(declared)
        assert not (set(observed) & skipped) and len(observed) == len(meta['nodules'])
        for ident, nodule in observed.items():
            assert nodule['ignored'] == declared[ident]['ignored']
            assert nodule['target'] == declared[ident]['target']
            assert nodule['annotation_ids'] == declared[ident]['annotation_ids']
            box = np.asarray(nodule['box_xyzxyz'])
            shape = np.asarray(meta['array_shapes']['hu_1mm.npy'])
            assert (box[:3] >= 0).all() and (box[3:] <= shape).all() and (box[3:] > box[:3]).all()
            result['ignored_nodules_with_masks' if nodule['ignored'] else 'retained_nodules'] += 1
        for ident in sorted(skipped):
            result['empty_consensus_annotations'].append({'case': row['key'], 'nodule_id': ident,
                                                         'ignored': declared[ident]['ignored']})
        for filename, expected_shape in meta['array_shapes'].items():
            array = np.load(directory / filename, mmap_mode='r')
            assert list(array.shape) == expected_shape
        for field in ('risk_target', 'risk_target_valid'):
            if meta[field] != row[field]:
                assert skipped
                result['effective_scan_label_changes'].append({'case': row['key'], 'field': field,
                                                               'csv_declared': row[field], 'v3_effective': meta[field]})
        result['valid_scan_labels'] += int(meta['risk_target_valid'])
        if not meta['sybil_has_visible_annotation']:
            result['sybil_cases_without_visible_attention'].append(row['key'])
        slices = json.loads((directory / 'edicnet_slices.json').read_text())
        assert slices['key'] == row['key'] and slices['cohort_sha256'] == digest
        array = np.load(directory / 'edicnet_slices.npy', mmap_mode='r')
        assert list(array.shape) == slices['shape'] and len(array) == len(slices['slices'])
        result['slices'] += len(array)
        seen.append(row['key'])
    assert len(seen) == len(set(seen)) == 638
    result['audit_passed'] = True
    return result


if __name__ == '__main__':
    result = audit()
    destination = HERE / 'cache_audit.json'
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
