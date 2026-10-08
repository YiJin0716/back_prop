"""Run the fine-tuned detector on training CTs for official det2cls adaptation."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from ..cohort import HERE, load, sha256
from ..prepare import CACHE
from ..predict import deeplung_detect


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--cache', type=Path, default=CACHE)
    parser.add_argument('--cohort', type=Path, default=HERE / 'cohort.json')
    args = parser.parse_args()
    torch.set_num_threads(2)
    cohort = load(args.cohort); digest = sha256(args.cohort)
    rows = cohort['splits']['training']
    if args.limit:
        rows = rows[:args.limit]
    elif not (args.checkpoint.parent / 'complete.json').exists():
        raise ValueError('Formal proposal generation requires finished detector fine-tuning')
    weight_digest = sha256(args.checkpoint)
    args.output.mkdir(parents=True, exist_ok=True)
    cases = []
    for index, row in enumerate(rows):
        path = args.output / f'{row["key"]}.json'
        if path.exists():
            case = json.loads(path.read_text())
            assert case['detector_sha256'] == weight_digest and case['cohort_sha256'] == digest
        else:
            volume = np.load(args.cache / row['key'] / 'hu_1mm.npy', mmap_mode='r')
            predictions, saved = deeplung_detect(volume, args.checkpoint, torch.device('cuda'), max_candidates=20)
            assert saved['cohort_sha256'] == digest
            case = {'key': row['key'], 'cohort_sha256': digest, 'detector_sha256': weight_digest,
                    'candidates': predictions.tolist()}
            temporary = path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(case, allow_nan=False) + '\n'); temporary.replace(path)
        cases.append(case)
        print(json.dumps({'done': index + 1, 'total': len(rows), 'key': row['key'],
                          'proposals': len(case['candidates'])}), flush=True)
    manifest = {'cohort_sha256': digest, 'detector_sha256': weight_digest,
                'detector_checkpoint': str(args.checkpoint), 'cases': cases,
                'split': 'training', 'max_candidates': 20, 'smoke_subset': bool(args.limit)}
    temporary = args.output / 'proposals.json.tmp'
    temporary.write_text(json.dumps(manifest, indent=2) + '\n'); temporary.replace(args.output / 'proposals.json')
