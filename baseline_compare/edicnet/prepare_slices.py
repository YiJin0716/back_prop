"""Create native-CT slice targets for RetinaNet from the frozen V3 cohort."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import numpy as np
import nibabel as nib
import torch

from back_prop.common.base_data import WholeCTLIDCDataset
from ..cohort import HERE, load, sha256
from ..prepare import CACHE, atomic_npy


def prepare(row, cache, digest):
    torch.set_num_threads(1)
    dest = Path(cache) / row['key']
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / 'edicnet_slices.json'
    if path.exists():
        meta = json.loads(path.read_text())
        assert meta['cohort_sha256'] == digest
        array = np.load(dest / 'edicnet_slices.npy', mmap_mode='r')
        assert len(array) == len(meta['slices'])
        return {'key': row['key'], 'slices': len(array), 'cached': True}
    image = nib.as_closest_canonical(nib.load(row['image']))
    helper = WholeCTLIDCDataset([row])
    boxes_by_slice = {}
    for nodule in row['nodules']:
        mask = helper._consensus_mask(row['patient_id'], str(int(row['scan_id'])),
                                      tuple(nodule['annotation_ids']), image)
        occupied = np.flatnonzero(mask.any(axis=(0, 1)))
        for z in occupied:
            points = np.argwhere(mask[:, :, z])
            lo, hi = points.min(0), points.max(0) + 1
            # Native RAS array: x is image row, y is image column.
            box = [int(lo[1]), int(lo[0]), int(hi[1]), int(hi[0]), -2 if nodule['ignored'] else 0]
            boxes_by_slice.setdefault(int(z), []).append(box)
    # Paper trains on nodule-containing slices. Every retained CT contributes.
    slices = [{'z': z, 'boxes': boxes} for z, boxes in sorted(boxes_by_slice.items())
              if any(box[-1] == 0 for box in boxes)]
    assert slices, row['key']
    raw = np.asarray(image.dataobj, dtype=np.float32)
    data = np.stack([raw[:, :, item['z']] for item in slices])
    data = np.clip((data + 1000) / 1500, 0, 1).astype(np.float16)
    atomic_npy(dest / 'edicnet_slices.npy', data)
    meta = {'key': row['key'], 'cohort_sha256': digest, 'shape': list(data.shape), 'slices': slices}
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(meta) + '\n'); temporary.replace(path)
    return {'key': row['key'], 'slices': len(slices)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    rows = load()['splits']['training']
    if args.limit:
        rows = rows[:args.limit]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks = [pool.submit(prepare, row, CACHE, sha256(HERE / 'cohort.json')) for row in rows]
        for i, result in enumerate(as_completed(tasks), 1):
            print(json.dumps({'done': i, 'total': len(rows), **result.result()}), flush=True)
    print(json.dumps({'status': 'complete', 'cases': len(rows)}), flush=True)
