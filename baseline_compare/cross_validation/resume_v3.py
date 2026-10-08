"""Explicit, audited continuation of an interrupted CV V3 training run."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import torch
from back_prop.common.semantic_data import WholeCTLIDCDataset, load_cases
from back_prop.common.semantic_model import ARCHITECTURE
from back_prop.legacy_training.train_semantic import parse_args


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    here = Path(__file__).resolve().parent
    plan = json.loads((here / 'plan.json').read_text())
    item = next(f for f in plan['folds'] if f['fold'] == args.fold)
    output = Path(item['directory']) / 'v3'
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    config = saved['config']
    completed = int(saved['epoch']) + 1
    assert saved['architecture'] == ARCHITECTURE
    assert 0 < completed < config['epochs'] == 18
    assert config['world_size'] == 4 and len(saved['rng_states']) == 4
    assert saved['optimizer']['state'] and config['init_v2'] is None
    assert Path(config['output_dir']).resolve() == output.resolve()
    assert sha256(item['cohort']) == item['cohort_sha256']
    assert sha256(config['manifest']) == item['manifest_sha256']
    assert sha256(config['vista_checkpoint']) == item['vista_checkpoint_sha256']
    metrics = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
    assert [r['epoch'] for r in metrics] == list(range(1, completed + 1))
    assert all(math.isfinite(r['loss']) and r['benchmark_steps'] is None for r in metrics)
    assert not (output / 'epoch_018.pt').exists(), 'Final checkpoint already exists'
    dataset = WholeCTLIDCDataset(load_cases(config['manifest'], 'training'),
                                target_mask_shape=tuple(config['screen_shape']))
    assert dataset.policy_summary == config['malignancy_policy']
    paths = set()
    for case in dataset.cases:
        key = (case['patient_id'], str(int(case['scan_id'])))
        for nodule in dataset.annotations.get(key, []):
            paths.update(dataset.official_mask_dir /
                         f'{key[0]}__scan{key[1]}__ann{a}_mask.nii.gz'
                         for a in nodule['annotation_ids'])
    # A real read of every required header, not just a directory existence check.
    # Full original preprocessing is additionally exercised for both failed CTs.
    start = time.perf_counter()
    for path in sorted(paths):
        with gzip.open(path, 'rb') as stream:
            header = stream.read(352)
        assert len(header) == 352 and header[344:348] == b'n+1\x00', str(path)
    failed_cases = {'LIDC-IDRI-0704__scan951', 'LIDC-IDRI-0673__scan982'}
    exercised = []
    for index, case in enumerate(dataset.cases):
        key = f"{case['patient_id']}__scan{int(case['scan_id'])}"
        if key in failed_cases:
            batch = dataset[index]
            assert batch['case_id'] == key
            assert torch.isfinite(batch['image']).all() and torch.isfinite(batch['image_hu']).all()
            exercised.append(key)
            print(json.dumps(dict(event='failed_case_preflight_pass', case_id=key,
                                  nodules=len(batch['nodule_ids']))), flush=True)
            del batch
    if args.fold == 3:
        assert set(exercised) == failed_cases
    audit = dict(fold=args.fold, checkpoint=str(args.checkpoint.resolve()),
                 checkpoint_sha256=sha256(args.checkpoint), completed_epochs=completed,
                 target_epochs=18, world_size=4, training_cases=len(dataset),
                 mask_headers_read=len(paths), failed_cases_reprocessed=exercised,
                 preflight_seconds=time.perf_counter()-start,
                 optimizer_state_entries=len(saved['optimizer']['state']),
                 cohort_sha256=item['cohort_sha256'], source_job='12660732',
                 mask_io_source_sha256=sha256(here.parents[1] / 'common/base_data.py'))
    (args.checkpoint.parent / 'preflight.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(dict(event='resume_preflight_pass', **audit)), flush=True)
    previous_argv = sys.argv
    try:
        sys.argv = ['train', '--output-dir', str(output)]
        names = list(vars(parse_args()))
    finally:
        sys.argv = previous_argv
    config = dict(config, resume=str(args.checkpoint.resolve()))
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
               '--nproc_per_node=4', '-m', 'back_prop.legacy_training.train_semantic']
    for name in names:
        value = config[name]
        if value is None or value is False:
            continue
        command.append('--' + name.replace('_', '-'))
        if value is not True:
            command.extend(map(str, value if isinstance(value, (list, tuple)) else [value]))
    del saved, dataset
    print(json.dumps(dict(event='resume_command', command=command)), flush=True)
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
