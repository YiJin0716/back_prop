"""Audit finished training and publish usable, verified model artifact links."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch

from .audit_training import verify_run
from .cohort import HERE, load, sha256


def link_artifact(destination, source):
    if destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        raise ValueError(f'Existing link points to a different model: {destination}')
    if destination.exists():
        raise ValueError(f'Refusing to replace an existing artifact: {destination}')
    destination.symlink_to(source)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['sybil', 'edicnet', 'deeplung'])
    parser.add_argument('--reuse-verification', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    cohort = load()
    registry = json.loads((HERE / 'run_registry.json').read_text())
    cache_audit = json.loads((HERE / 'cache_audit.json').read_text())
    assert cache_audit['audit_passed']
    selected = {'sybil': ['sybil'], 'edicnet': ['edicnet_detector', 'edicnet_classifier'],
                'deeplung': ['deeplung_detector', 'deeplung_initial_classifier', 'deeplung_final_classifier']}[args.model]
    reports = {name: verify_run(name, registry['runs'][name], cohort, cache_audit) for name in selected}
    assert all(report['verified'] for report in reports.values()), 'Training is still incomplete'
    classifier_name = {'sybil': 'sybil', 'edicnet': 'edicnet_classifier', 'deeplung': 'deeplung_final_classifier'}[args.model]
    classifier_directory = Path(registry['runs'][classifier_name]['output'])
    classifier = classifier_directory / 'last.pt'
    command = [sys.executable, '-m', 'back_prop.baseline_compare.predict', '--model', args.model,
               '--image', cohort['splits']['training'][0]['image'], '--checkpoint', str(classifier)]
    artifacts = {'classifier': classifier}
    if args.model != 'sybil':
        detector = Path(registry['runs'][args.model + '_detector']['output']) / 'last.pt'
        command += ['--detector-checkpoint', str(detector)]
        artifacts['detector'] = detector
    if args.model == 'deeplung':
        gbm = classifier_directory / 'gbm.pkl'
        command += ['--gbm', str(gbm)]
        artifacts['gbm'] = gbm
    output = classifier_directory / 'whole_ct_verification.json'
    if args.reuse_verification:
        output = args.reuse_verification
    else:
        subprocess.run([*command, '--output', str(output)], check=True)
    prediction = json.loads(output.read_text())
    assert prediction['model'] == args.model and prediction['annotation_inputs'] is False
    assert Path(prediction['checkpoint']).resolve() == classifier.resolve()
    assert prediction['checkpoint_epoch'] == registry['runs'][classifier_name]['epochs']
    assert 0 <= prediction['risk'] <= 1
    verifications = {'whole_ct': str(output)}
    if args.model != 'sybil':
        assert Path(prediction['detector_checkpoint']).resolve() == artifacts['detector'].resolve()
        assert prediction['detector_epoch'] == registry['runs'][args.model + '_detector']['epochs']
    if args.model == 'edicnet':
        # Prespecified whole-scan extension as well as the paper's primary ROI.
        alternative = classifier_directory / 'whole_ct_max_risk_verification.json'
        subprocess.run([*command, '--max-candidates', '20', '--output', str(alternative)], check=True)
        result = json.loads(alternative.read_text())
        assert result['annotation_inputs'] is False and 0 <= result['risk'] <= 1
        assert result['checkpoint_epoch'] == prediction['checkpoint_epoch']
        assert result['detector_epoch'] == prediction['detector_epoch']
        verifications['max_risk_over_up_to_20_detected_rois'] = str(alternative)
    module = HERE / args.model
    links = {'classifier': module / ('finetuned.pt' if args.model == 'sybil' else 'classifier.pt'),
             'detector': module / 'detector.pt', 'gbm': module / 'gbm.pkl'}
    for name, source in artifacts.items():
        link_artifact(links[name], source)
    report = {'training_and_inference_verified': True, 'model': args.model,
              'cohort_sha256': sha256(HERE / 'cohort.json'), 'training_scans': 638,
              'endpoint': 'LIDC reader-derived current malignancy', 'held_out_performance_evaluated': False,
              'training_audits': reports, 'inference_verifications': verifications,
              'artifacts': {name: {'path': str(source), 'link': str(links[name]), 'sha256': sha256(source)}
                            for name, source in artifacts.items()}}
    (module / 'final_artifacts.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'model': args.model, 'training_and_inference_verified': True,
                      'manifest': str(module / 'final_artifacts.json')}), flush=True)


if __name__ == '__main__':
    main()
