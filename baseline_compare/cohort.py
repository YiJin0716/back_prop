"""Freeze the actual V3 cohort, with physical-nodule identities and label policy."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from back_prop.common.base_data import (
    DEFAULT_ANNOTATIONS, WholeCTLIDCDataset, load_cases, _case_key,
)
from back_prop.lidc_policy import is_indeterminate_malignancy

HERE = Path(__file__).resolve().parent
V3_CONFIG = Path('/usr/project/rudinlab/datasets/LIDC_IDRI/joint_model/model_v3_runs/12616464/config.json')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def build(config_path=V3_CONFIG):
    config = json.loads(Path(config_path).read_text())
    assert config['max_cases'] is None
    assert config['malignancy_policy']['exclude_indeterminate']
    result = {'schema': 1, 'v3_config': str(config_path),
              'v3_config_sha256': sha256(config_path),
              'manifest': config['manifest'], 'manifest_sha256': sha256(config['manifest']),
              'annotations_csv': str(DEFAULT_ANNOTATIONS),
              'annotations_sha256': sha256(DEFAULT_ANNOTATIONS), 'splits': {}}
    for split in ('training', 'validation', 'testing'):
        dataset = WholeCTLIDCDataset(load_cases(config['manifest'], split))
        if split == 'training':
            for field in ('retained_cases', 'retained_physical_nodules', 'excluded_indeterminate_nodules'):
                assert dataset.policy_summary[field] == config['malignancy_policy'][field], field
        rows = []
        for case in dataset.cases:
            patient, scan = _case_key(case)
            nodules = []
            for nodule in dataset.annotations[(patient, scan)]:
                ignored = is_indeterminate_malignancy(nodule['malignancy'])
                nodules.append({
                    'nodule_id': nodule['nodule_id'], 'annotation_ids': list(nodule['annotation_ids']),
                    'reader_mean_malignancy': nodule['malignancy'], 'ignored': ignored,
                    'target': None if ignored else int(nodule['malignancy'] > 3),
                    'semantic_means': nodule['features'].tolist(),
                })
            risk = max(n['target'] for n in nodules if not n['ignored'])
            rows.append({**case, 'key': f'{patient}__scan{scan}', 'nodules': nodules,
                         'risk_target': risk,
                         'risk_target_valid': bool(risk or not any(n['ignored'] for n in nodules))})
        result['splits'][split] = rows
    train_patients = {r['patient_id'] for r in result['splits']['training']}
    test_patients = {r['patient_id'] for r in result['splits']['testing']}
    assert not (train_patients & test_patients), 'Train/test patient leakage'
    overlap = sorted(train_patients & {r['patient_id'] for r in result['splits']['validation']})
    result['validation_train_overlap_patients'] = overlap
    result['validation_usable_for_selection'] = not bool(overlap)
    result['policy'] = config['malignancy_policy']
    result['training_case_keys_sha256'] = hashlib.sha256(
        '\n'.join(sorted(r['key'] for r in result['splits']['training'])).encode()).hexdigest()
    return result


def load(path=HERE / 'cohort.json', verify=True):
    result = json.loads(Path(path).read_text())
    if verify:
        for name in ('v3_config', 'manifest', 'annotations'):
            source = result['annotations_csv' if name == 'annotations' else name]
            assert sha256(source) == result[name + '_sha256'], f'Changed source: {source}'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--v3-config', type=Path, default=V3_CONFIG)
    parser.add_argument('--output', type=Path, default=HERE / 'cohort.json')
    args = parser.parse_args()
    payload = build(args.v3_config)
    if args.output.exists():
        assert json.loads(args.output.read_text()) == payload, 'Refusing to replace a different frozen cohort'
    else:
        args.output.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({
        'cohort': str(args.output),
        'training_case_keys_sha256': payload['training_case_keys_sha256'],
        'splits': {s: {'scans': len(rows), 'patients': len({r['patient_id'] for r in rows}),
                      'nodules': sum(not n['ignored'] for r in rows for n in r['nodules']),
                      'valid_scan_labels': sum(r['risk_target_valid'] for r in rows)}
                   for s, rows in payload['splits'].items()},
        'validation_usable_for_selection': payload['validation_usable_for_selection'],
    }, indent=2))
