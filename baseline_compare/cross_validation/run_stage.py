"""Execute one frozen fold stage; Slurm dependencies provide stage ordering."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from .build import HERE_CV, RUN, PLOTS
from ..cohort import load, sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['cache', 'v3', 'sybil', 'detector', 'classifier', 'proposals', 'adapt', 'evaluate', 'aggregate'])
    p.add_argument('fold', nargs='?', type=int)
    args = p.parse_args()
    plan = json.loads((HERE_CV / 'plan.json').read_text())
    if args.stage in ('cache', 'aggregate'):
        module = 'cache' if args.stage == 'cache' else 'aggregate'
        subprocess.run([sys.executable, '-m', f'back_prop.baseline_compare.cross_validation.{module}'], check=True)
        return
    assert args.fold in range(5)
    if args.fold in plan.get('reused_folds', []):
        assert args.stage == 'evaluate', 'This fold is already trained; reuse its completed results'
        from .reuse_fold0 import main as reuse
        reuse()
        return
    item = plan['folds'][args.fold]; directory = Path(item['directory'])
    assert sha256(item['cohort']) == item['cohort_sha256']
    cohort = load(item['cohort'])
    assert sha256(cohort['manifest']) == item['manifest_sha256']
    output = directory / args.stage
    common = ['--cohort', item['cohort'], '--cache', str(directory / 'cache')]
    prefix = [sys.executable, '-m']
    if args.stage == 'v3':
        assert sha256(item['vista_checkpoint']) == item['vista_checkpoint_sha256']
        config = json.loads((directory / 'planned_v3_config.json').read_text())
        from back_prop.legacy_training.train_semantic import parse_args
        old_argv = sys.argv
        try:
            sys.argv = ['train', '--output-dir', str(output)]
            defaults = vars(parse_args())
        finally:
            sys.argv = old_argv
        command = prefix + ['torch.distributed.run', '--standalone', '--nproc_per_node=4', '-m', 'back_prop.legacy_training.train_semantic']
        for key in defaults:
            value = config[key]
            if value is None or value is False:
                continue
            command.append('--' + key.replace('_', '-'))
            if value is not True:
                command.extend(map(str, value if isinstance(value, list) else [value]))
        # Continuations are explicit separate submissions, never automatic restarts.
        assert not (output / 'metrics.jsonl').exists(), 'Refusing to overwrite an existing V3 run'
    elif args.stage == 'evaluate':
        command = prefix + ['back_prop.baseline_compare.cross_validation.evaluate', '--fold', str(args.fold)]
    else:
        audit = json.loads((directory / 'cache_audit.json').read_text())
        assert audit['cohort_sha256'] == item['cohort_sha256']
        assert not (output / 'epochs.jsonl').exists(), 'Refusing to overwrite existing training'
        if args.stage == 'sybil':
            command = prefix + ['back_prop.baseline_compare.sybil.train', *common, '--output', str(output), '--epochs', '3']
        elif args.stage == 'detector':
            assert sha256(plan['deeplung_public_checkpoint']) == plan['deeplung_public_sha256']
            command = prefix + ['back_prop.baseline_compare.deeplung.train_detector', *common, '--output', str(output),
                                '--checkpoint', plan['deeplung_public_checkpoint'], '--epochs', '5']
        elif args.stage == 'proposals':
            command = prefix + ['back_prop.baseline_compare.deeplung.prepare_proposals', *common,
                                '--output', str(output), '--checkpoint', str(directory / 'detector/last.pt')]
        else:
            command = prefix + ['back_prop.baseline_compare.deeplung.train_classifier', *common,
                                '--output', str(output), '--epochs', '105' if args.stage == 'adapt' else '700']
            if args.stage == 'adapt':
                assert json.loads((directory / 'classifier/complete.json').read_text())['epochs'] == 700
                command += ['--init', str(directory / 'classifier/last.pt'), '--proposals', str(directory / 'proposals/proposals.json')]
    print(json.dumps({'fold': args.fold, 'stage': args.stage, 'command': command}), flush=True)
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
