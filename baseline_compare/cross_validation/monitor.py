"""Read-only status checks with timestamped evidence; never infer completion from IDs."""
import datetime
import json
import math
from pathlib import Path
import subprocess
from .build import HERE_CV, RUN


def json_lines(path):
    rows = []
    if path.exists():
        for line in path.read_text(errors='replace').splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # Warnings and an unfinished final log line are not metric records.
    return rows


def main():
    jobs = json.loads((HERE_CV / 'jobs.json').read_text())
    ids = ','.join(v['job_id'] for v in jobs.values())
    result = subprocess.run(['sacct', '-n', '-X', '-P', '-j', ids,
                             '--format=JobID,State,ExitCode,Elapsed,NodeList'], text=True, capture_output=True, check=True)
    states = {line.split('|')[0]: line.split('|')[1:5] for line in result.stdout.splitlines() if line}
    snapshot = {'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'jobs': {}}
    failures = []
    for key, job in jobs.items():
        state = states.get(job['job_id'], ['UNKNOWN', '', '', ''])
        out = HERE_CV / f'logs/{key}_{job["job_id"]}.out'
        err = HERE_CV / f'logs/{key}_{job["job_id"]}.err'
        entries = json_lines(out)
        progress = [r for r in entries if 'loss' in r or 'done' in r or 'prepared' in r]
        invalid = [r for r in progress if any(isinstance(r.get(k), (int, float)) and not math.isfinite(r[k])
                                              for k in ('loss', 'grad_norm')) or r.get('optimizer_step_skipped')]
        errors = [line for line in err.read_text(errors='replace').splitlines()
                  if any(word in line for word in ('Traceback (most recent call last)', 'OutOfMemoryError', 'FloatingPointError', 'AssertionError'))] if err.exists() else []
        record = {'job_id': job['job_id'], 'state': state[0], 'exit_code': state[1], 'elapsed': state[2],
                  'nodes': state[3], 'progress_records': len(progress), 'invalid_steps': len(invalid), 'errors': errors[-5:]}
        if progress:
            record['last_progress'] = progress[-1]
        if job['fold'] is not None and job['stage'] in ('v3', 'sybil', 'detector', 'classifier', 'adapt'):
            name = 'metrics.jsonl' if job['stage'] == 'v3' else 'epochs.jsonl'
            metrics = json_lines(RUN / f'fold_{job["fold"]}' / job['stage'] / name)
            record['completed_epochs'] = len(metrics)
            if metrics:
                record['last_epoch'] = metrics[-1]
        snapshot['jobs'][key] = record
        if invalid or errors or any(s in state[0] for s in ('FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL')):
            failures.append(key)
        suffix = ''
        if progress:
            r = progress[-1]
            suffix = f" epoch={r.get('epoch', '-')} step={r.get('step', '-')} loss={r.get('loss', '-')} prepared={r.get('prepared', '-')}"
        print(f"{key:15} {job['job_id']} {state[0]:12} {state[2]:10} {suffix}")
    snapshot['failures'] = failures
    evidence = HERE_CV / 'monitoring'; evidence.mkdir(exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    (evidence / f'{stamp}.json').write_text(json.dumps(snapshot, indent=2) + '\n')
    (HERE_CV / 'latest_status.json').write_text(json.dumps(snapshot, indent=2) + '\n')
    if failures:
        raise SystemExit('Investigate: ' + ', '.join(failures))


if __name__ == '__main__':
    main()
