"""Compare cached targets against the independent original V3 input pipeline."""
import argparse
import json
import numpy as np

from back_prop.common.base_data import WholeCTLIDCDataset
from .cohort import load
from .prepare import CACHE


def verify(row):
    original = WholeCTLIDCDataset([row])[0]
    meta = json.loads((CACHE / row['key'] / 'metadata.json').read_text())
    shape = np.asarray(meta['array_shapes']['hu_1mm.npy'])
    for ignored, prefix in ((False, ''), (True, 'ignored_')):
        nodules = [n for n in meta['nodules'] if n['ignored'] == ignored]
        expected_ids = original[prefix + 'nodule_ids'].tolist()
        assert [n['nodule_id'] for n in nodules] == expected_ids
        boxes = np.asarray([n['box_xyzxyz'] for n in nodules], dtype=np.float32).reshape(-1, 6)
        normalized = np.concatenate(((boxes[:, :3] + boxes[:, 3:]) / 2 / shape,
                                     (boxes[:, 3:] - boxes[:, :3]) / shape), -1)
        np.testing.assert_allclose(normalized, original[prefix + 'target_boxes'].numpy(), rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal([n['target'] for n in meta['nodules'] if not n['ignored']],
                                  original['malignancy_targets'].numpy())
    assert meta['risk_target'] == float(original['risk_target'])
    assert meta['risk_target_valid'] == bool(original['risk_target_valid'])
    hu = np.load(CACHE / row['key'] / 'hu_1mm.npy', mmap_mode='r')
    expected_image = ((np.clip(hu.reshape(-1)[::100003], -1024, 1024) + 1024) / 2048)
    np.testing.assert_allclose(expected_image, original['image'].numpy().reshape(-1)[::100003], atol=1e-6)
    return {'case': row['key'], 'passed': True,
            'retained_nodules': sum(not n['ignored'] for n in meta['nodules']),
            'risk_target_valid': meta['risk_target_valid']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=3)
    parser.add_argument('--case-key')
    args = parser.parse_args()
    rows = load()['splits']['training']
    if args.case_key:
        print(json.dumps(verify(next(row for row in rows if row['key'] == args.case_key))), flush=True)
        raise SystemExit(0)
    selected = [rows[0], next(row for row in rows if not row['risk_target_valid'])]
    selected += [row for row in rows if row not in selected][:max(0, args.count - 2)]
    for row in selected[:args.count]:
        print(json.dumps(verify(row)), flush=True)
