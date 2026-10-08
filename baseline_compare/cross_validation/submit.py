"""Idempotently submit the frozen dependency graph, persisting every job ID."""
import argparse
import json
import subprocess
from .build import HERE_CV, PLOTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--initial-only', action='store_true')
    args = parser.parse_args()
    registry_path = HERE_CV / 'jobs.json'
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {}
    def submit(key, stage, fold=None, dependencies=(), v3=False, cpu=False):
        if key in registry:
            return registry[key]['job_id']
        command = ['sbatch', '--parsable', '--job-name=cv-' + key,
                   '--output=' + str(HERE_CV / f'logs/{key}_%j.out'),
                   '--error=' + str(HERE_CV / f'logs/{key}_%j.err')]
        if cpu:
            command += ['--partition=compsci', '--cpus-per-task=8', '--mem=128G', '--time=1-00:00:00']
        elif v3:
            command += ['--gres=gpu:a5000:4', '--cpus-per-task=16', '--mem=192G', '--time=7-00:00:00']
        else:
            command += ['--gres=gpu:a5000:1']
        if dependencies:
            command += ['--dependency=afterok:' + ':'.join(dependencies), '--kill-on-invalid-dep=yes']
        command += [str(HERE_CV / 'job.sbatch'), stage]
        if fold is not None:
            command += [str(fold)]
        result = subprocess.run(command, text=True, capture_output=True, check=True)
        job = result.stdout.strip().split(';')[0]
        assert job.isdigit(), result
        registry[key] = {'job_id': job, 'stage': stage, 'fold': fold, 'dependencies': list(dependencies), 'command': command}
        temp = registry_path.with_suffix('.json.tmp')
        temp.write_text(json.dumps(registry, indent=2) + '\n'); temp.replace(registry_path)
        print(json.dumps({'key': key, 'job_id': job, 'dependencies': dependencies}), flush=True)
        return job
    plan = json.loads((HERE_CV / 'plan.json').read_text())
    folds = plan.get('new_training_folds', list(range(5)))
    for fold in plan.get('reused_folds', []):
        assert json.loads((PLOTS / f'fold_{fold}' / 'complete.json').read_text())['reused_existing_evaluation']
    cache = submit('cache', 'cache', cpu=True)
    v3 = {i: submit(f'f{i}-v3', 'v3', i, v3=True) for i in folds}
    if args.initial_only:
        return
    evaluation = []
    for fold in folds:
        sybil = submit(f'f{fold}-sybil', 'sybil', fold, [cache])
        detector = submit(f'f{fold}-detector', 'detector', fold, [cache])
        classifier = submit(f'f{fold}-classifier', 'classifier', fold, [cache])
        proposals = submit(f'f{fold}-proposals', 'proposals', fold, [detector])
        adapt = submit(f'f{fold}-adapt', 'adapt', fold, [classifier, proposals])
        evaluation.append(submit(f'f{fold}-evaluate', 'evaluate', fold, [v3[fold], sybil, adapt]))
    submit('aggregate', 'aggregate', dependencies=evaluation, cpu=True)


if __name__ == '__main__':
    main()
