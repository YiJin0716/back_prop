"""Verify completed runs using saved weights, optimizer/BN counters and coverage.

The run registry and completion marker alone are not accepted as proof. Running
or missing runs are reported as unfinished; --require-all exits nonzero for them.
"""
import argparse
import json
import math
from pathlib import Path
import pickle

import numpy as np
import torch

from .cohort import HERE, load, sha256


def verify_run(name, run, cohort, cache_audit, cohort_path=HERE / 'cohort.json'):
    directory = Path(run['output'])
    if not (directory / 'complete.json').exists():
        return {'verified': False, 'reason': 'configured training has not finished', 'job_id': run['job_id']}
    config = json.loads((directory / 'config.json').read_text())
    checkpoint = torch.load(directory / 'last.pt', map_location='cpu', weights_only=False)
    complete = json.loads((directory / 'complete.json').read_text())
    metrics = [json.loads(line) for line in (directory / 'epochs.jsonl').read_text().splitlines()]
    expected_epochs = run['epochs']
    digest = sha256(cohort_path)
    scans = len(cohort['splits']['training'])
    nodules = cache_audit['retained_nodules']
    assert not config.get('smoke'), 'Smoke artifact cannot be a final model'
    assert checkpoint['epoch'] == complete['epochs'] == config['epochs'] == expected_epochs
    assert checkpoint['cohort_sha256'] == complete['cohort_sha256'] == config['cohort_sha256'] == digest
    assert config['training_scans'] == scans
    assert config['training_case_keys_sha256'] == cohort['training_case_keys_sha256']
    assert [item['epoch'] for item in metrics] == list(range(1, expected_epochs + 1))
    assert all(math.isfinite(item['loss']) for item in metrics)
    if name == 'sybil':
        samples, batch_size, coverage = scans, 1, 'scans_seen'
        assert sorted(checkpoint['seen_case_keys']) == sorted(r['key'] for r in cohort['splits']['training'])
    elif name == 'edicnet_detector':
        samples, batch_size, coverage = cache_audit['slices'], 2, 'slices_seen'
        assert config['training_slices'] == samples
    elif name == 'deeplung_detector':
        samples, batch_size, coverage = nodules + int(nodules * .3 / .7), 4, 'samples_seen'
        assert config['samples_per_epoch'] == samples
        assert all(item['scans_seen'] == scans for item in metrics)
        assert sorted(checkpoint['seen_cases']) == sorted(r['key'] for r in cohort['splits']['training'])
    else:
        samples = config.get('training_samples', cache_audit['retained_nodules'])
        batch_size = 6 if name == 'edicnet_classifier' else 16
        coverage = 'nodules_seen'
        assert config['physical_nodules'] == nodules
    assert all(item[coverage] == samples for item in metrics)
    expected_steps = math.ceil(samples / batch_size) * expected_epochs
    weights = checkpoint['model']
    assert all(torch.isfinite(tensor).all() for tensor in weights.values() if tensor.is_floating_point())
    optimizer_states = checkpoint['optimizer']['state'].values()
    assert optimizer_states
    assert all(torch.isfinite(value).all() for state in optimizer_states for value in state.values()
               if isinstance(value, torch.Tensor) and value.is_floating_point())
    evidence = {'expected_optimizer_steps': expected_steps}
    adam_steps = [int(state['step']) for state in optimizer_states if 'step' in state]
    if adam_steps:
        assert min(adam_steps) == max(adam_steps) == expected_steps
        evidence['adam_step_counter'] = expected_steps
    if name in ('deeplung_detector', 'deeplung_initial_classifier', 'deeplung_final_classifier', 'edicnet_classifier'):
        bn_key = 'features.1.num_batches_tracked' if name == 'edicnet_classifier' else 'bn1.num_batches_tracked'
        initial_counter = 0
        if name == 'deeplung_detector':
            initial = torch.load(config['checkpoint'], map_location='cpu', weights_only=False)['state_dict']
            initial_counter = int(initial.get(bn_key, 0))
            assert not torch.equal(initial['output.2.weight'], weights['output.2.weight'])
        elif name == 'deeplung_final_classifier':
            initial = torch.load(config['init'], map_location='cpu', weights_only=False)['model']
            initial_counter = int(initial[bn_key])
            assert not torch.equal(initial['linear.weight'], weights['linear.weight'])
        assert int(weights[bn_key]) == initial_counter + expected_steps
        evidence['batchnorm_training_forward_count'] = int(weights[bn_key])
    if name.startswith('deeplung_') and 'classifier' in name:
        with (directory / 'gbm.pkl').open('rb') as stream:
            gbm = pickle.load(stream)
        assert gbm['cohort_sha256'] == digest and gbm['fitting_samples'] == samples
        assert gbm['model'].n_features_in_ == 2560 + 17 ** 3 + 1
        assert list(gbm['model'].classes_) == [0, 1]
        assert gbm['model'].max_depth == (2 if name == 'deeplung_final_classifier' else 1)
        assert gbm['pixel_mean'] == config['pixel_mean'] and gbm['pixel_std'] == config['pixel_std'] > 0
        assert math.isfinite(gbm['max_diameter']) and gbm['max_diameter'] > 0
        assert all(np.isfinite(tree.tree_.value).all() for tree in gbm['model'].estimators_.ravel())
        evidence['gbm_fitting_samples'] = samples
    if name == 'deeplung_final_classifier':
        proposals = json.loads(Path(config['proposals']).read_text())
        assert proposals['cohort_sha256'] == digest and not proposals['smoke_subset']
        assert sorted(c['key'] for c in proposals['cases']) == sorted(r['key'] for r in cohort['splits']['training'])
        assert sha256(config['proposals']) == config['proposal_summary']['proposal_manifest_sha256']
        assert sha256(proposals['detector_checkpoint']) == proposals['detector_sha256']
        evidence['training_proposal_cases'] = len(proposals['cases'])
    report = {'verified': True, 'job_id': run['job_id'], 'epochs': expected_epochs,
              'training_scans': scans, 'samples_per_epoch': samples, 'cohort_sha256': digest,
              'finite_weights_and_optimizer': True, 'checkpoint_sha256': sha256(directory / 'last.pt'),
              'final_loss': metrics[-1]['loss'], **evidence}
    (directory / 'completion_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--only')
    parser.add_argument('--require-all', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    cohort = load()
    cache_audit = json.loads((HERE / 'cache_audit.json').read_text())
    assert cache_audit['audit_passed'] and cache_audit['cohort_sha256'] == sha256(HERE / 'cohort.json')
    registry = json.loads((HERE / 'run_registry.json').read_text())
    runs = registry['runs']
    if args.only:
        runs = {args.only: runs[args.only]}
    reports = {name: verify_run(name, run, cohort, cache_audit) for name, run in runs.items()}
    print(json.dumps(reports, indent=2))
    if args.require_all and not all(report['verified'] for report in reports.values()):
        raise SystemExit(2)
