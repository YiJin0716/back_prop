"""Generate one offline HTML per test CT, loading the checkpoint once per shard."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import traceback

from back_prop.evaluate.eval_package.annotations import load_test_cases, case_key

ROOT = Path(__file__).resolve().parents[3]


def source_fingerprint():
    """Pin the exporter and its JS/CSS for resumable output validation."""
    root = Path(__file__).parent
    paths = [root/'pipeline_report.py', *sorted((root/'pipeline_report_assets').iterdir())]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def checkpoint_signature(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def validate_plan(plan):
    cases = load_test_cases(plan['manifest'], split='testing')
    keys = [case_key(c) for c in cases]
    if keys != plan['case_ids']:
        raise ValueError('The testing cohort differs from the submitted plan')
    if hashlib.sha256(Path(plan['manifest']).read_bytes()).hexdigest() != plan['manifest_sha256']:
        raise ValueError('Manifest changed after submission')
    if checkpoint_signature(plan['checkpoint']) != plan['checkpoint_signature']:
        raise ValueError('Checkpoint changed after submission')
    if source_fingerprint() != plan['exporter_sha256']:
        raise ValueError('Exporter changed after submission')
    shards = plan['shards']
    if shards < 1 or len(keys) != len(set(keys)):
        raise ValueError('Invalid shard count or duplicate cases')
    return [keys[index::shards] for index in range(shards)]


def completed_html(path, case_id, plan):
    """An atomic HTML is complete only if its embedded case/model/layout agree."""
    if not path.is_file():
        return False
    page = path.read_text()
    prefix = '<script>const DATA = '
    if prefix not in page or not page.rstrip().endswith('</html>'):
        raise ValueError(f'Incomplete existing HTML: {path}')
    start = page.index(prefix)+len(prefix)
    data = json.loads(page[start:page.index(';</script>', start)])
    report = data['report']
    if (report['case_id'] != case_id or report['epoch'] != plan['epoch'] or
            Path(report['checkpoint']).resolve() != Path(plan['checkpoint']).resolve() or
            report.get('exporter_sha256') != plan['exporter_sha256']):
        raise ValueError(f'Existing HTML belongs to a different run: {path}')
    if 'scan' not in data['scenes'] or not data['scenes']['scan']['image']:
        raise ValueError(f'HTML has no CT scene: {path}')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--shard', type=int, default=int(os.environ.get('SLURM_ARRAY_TASK_ID', '0')))
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    groups = validate_plan(plan)
    if not 0 <= args.shard < len(groups):
        parser.error('Invalid shard index')
    if args.validate_only:
        print(json.dumps(dict(cases=len(plan['case_ids']), shards=[len(g) for g in groups],
                              output=plan['output'], checkpoint=plan['checkpoint'])), flush=True)
        return
    import torch
    from back_prop.evaluate.eval_package import JointModelEvaluator, save_result

    torch.set_num_threads(2)
    torch.manual_seed(42)
    output = Path(plan['output'])
    output.mkdir(parents=True, exist_ok=True)
    logs = output/'_job'
    logs.mkdir(exist_ok=True)
    assigned = groups[args.shard]
    evaluator = None
    finished, failed = [], {}
    progress = logs/f'shard_{args.shard}.json'

    def record(state):
        write_json(progress, dict(state=state, job_id=os.environ.get('SLURM_JOB_ID'),
            assigned=assigned, completed=finished, failed=failed, updated_unix=time.time()))

    record('running')
    for ordinal, key in enumerate(assigned):
        destination = output/f'{key}.html'
        try:
            if completed_html(destination, key, plan):
                finished.append(key)
                record('running')
                print(json.dumps(dict(event='cached', case_id=key)), flush=True)
                continue
        except Exception:
            failed[key] = traceback.format_exc()
            record('running')
            continue
        if evaluator is None:
            evaluator = JointModelEvaluator.from_checkpoint(plan['checkpoint'], plan['manifest'], device='cuda')
            if evaluator.model.evaluation_metadata['epoch'] != plan['epoch']:
                raise ValueError('Unexpected checkpoint epoch')
            evaluator.model.requires_grad_(False)
        started = time.monotonic()
        for attempt in range(1, 3):
            result = None
            try:
                print(json.dumps(dict(event='case_start', case_id=key, ordinal=ordinal+1,
                                      cases=len(assigned), attempt=attempt)), flush=True)
                # Temporary CT/mask caches stay on the compute node and are
                # removed after every attempt, including unsuccessful exports.
                scratch = os.environ.get('SLURM_TMPDIR')
                with tempfile.TemporaryDirectory(prefix=f'v5-html-{key}-', dir=scratch) as work:
                    result = evaluator.evaluate(key)
                    result.metadata['exporter_sha256'] = plan['exporter_sha256']
                    save_result(result, work, plots=False, nifti=False)
                    del result
                    result = None
                    gc.collect()
                    subprocess.run([plan['visual_python'], '-m',
                        'back_prop.evaluate.every_CT.pipeline_report',
                        '--result-dir', work, '--output', str(destination),
                        '--model-label', plan['model_label'], '--scan-step', str(plan['scan_step'])],
                        cwd=ROOT, check=True, timeout=1800)
                if not completed_html(destination, key, plan):
                    raise RuntimeError('Exporter returned without a complete HTML')
                finished.append(key)
                failed.pop(key, None)
                print(json.dumps(dict(event='case_complete', case_id=key,
                    seconds=round(time.monotonic()-started, 1), bytes=destination.stat().st_size)), flush=True)
                break
            except Exception:
                failed[key] = traceback.format_exc()
                print(failed[key], flush=True)
            finally:
                del result
                gc.collect()
                torch.cuda.empty_cache()
        record('running')
    record('failed' if failed else 'completed')
    if failed:
        raise RuntimeError(f'{len(failed)} cases failed; see {progress}')
    if set(finished) != set(assigned):
        raise RuntimeError('Shard did not cover every assigned CT')
    print(json.dumps(dict(event='shard_complete', shard=args.shard, cases=len(finished))), flush=True)


if __name__ == '__main__':
    main()
