"""Run a notebook case locally or on Slurm without requiring a GPU kernel."""
from __future__ import annotations

import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time


def _run(command, *, cwd=None):
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {shlex.join(map(str, command))}\n"
            f"{completed.stdout}\n{completed.stderr}")
    return completed.stdout.strip()


def _job_state(job_id):
    try:
        state = _run(['squeue', '--jobs', job_id, '--noheader', '--format=%T'])
    except RuntimeError as exc:
        # Finished jobs eventually disappear from the controller but remain
        # in accounting. This is normal when revisiting a saved notebook.
        if 'Invalid job id specified' not in str(exc):
            raise
        state = ''
    if state:
        return state.splitlines()[0].strip()
    rows = _run(['sacct', '--jobs', job_id, '--noheader', '--parsable2',
                 '--format=JobIDRaw,State'])
    for row in rows.splitlines():
        fields = row.split('|')
        if fields[0] == job_id:
            return fields[1].split()[0].rstrip('+')
    return 'ACCOUNTING_PENDING'


def _logs(folder, job_id):
    parts = []
    for suffix in ('out', 'err'):
        path = folder / f'inference_{job_id}.{suffix}'
        if path.exists():
            parts.append(f'{path}\n{path.read_text(errors="replace")[-16000:]}')
    return '\n'.join(parts)


def ensure_case_result(*, checkpoint, manifest, case_id, output_root, python,
                       root=None, force=False, mask_threshold=0.5, minimum_iou=0.1,
                       execution='auto', partition='rudin', account='rudin',
                       gres='gpu:a6000:1', poll_seconds=10, max_wait_seconds=3600):
    """Ensure one cached case exists, returning its directory.

    In auto mode, probe CUDA in the inference Python environment. If that
    process has no GPU, submit one Slurm job and wait with progress messages.
    Existing active jobs are reattached on cell reruns. A keyboard interrupt
    leaves the job running; its ID is printed and saved in inference_job.json.
    Errors include the child process or Slurm logs, rather than a bare
    CalledProcessError. Cached predictions never require torch or CUDA.
    """
    if execution not in ('auto', 'local', 'slurm'):
        raise ValueError("execution must be 'auto', 'local', or 'slurm'")
    if not 0 < mask_threshold < 1 or not 0 <= minimum_iou <= 1:
        raise ValueError('Invalid mask or matching threshold')
    if poll_seconds <= 0 or max_wait_seconds <= 0:
        raise ValueError('Polling and timeout values must be positive')
    root = Path(root or Path(__file__).resolve().parents[3]).resolve()
    checkpoint, manifest, python = (Path(p).resolve() for p in (checkpoint, manifest, python))
    output_root = Path(output_root).resolve()
    folder = output_root / case_id
    report_path = folder / 'result.json'

    def cache_ready():
        return report_path.is_file() and (folder / 'arrays.npz').is_file()

    def check_cache():
        report = json.loads(report_path.read_text())
        metadata = report['metadata']
        matches = (report['case_id'] == case_id
                   and Path(metadata['checkpoint']).resolve() == checkpoint
                   and Path(metadata['manifest']).resolve() == manifest
                   and metadata['split'] == 'testing'
                   and metadata['mask_threshold'] == mask_threshold
                   and report['minimum_iou'] == minimum_iou)
        if not matches:
            raise ValueError('Cached case/configuration differs; set FORCE_INFERENCE=True to rerun')

    if cache_ready() and not force:
        check_cache()
        print(f'读取已完成的推理结果: {folder}', flush=True)
        return folder
    for path in (python, checkpoint, manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    command = [str(python), '-m', 'back_prop.evaluate.eval_package',
               '--checkpoint', str(checkpoint), '--manifest', str(manifest),
               '--case-id', case_id, '--output', str(output_root),
               '--device', 'cuda', '--threads', '2', '--no-plots', '--nifti',
               '--mask-threshold', str(mask_threshold), '--minimum-iou', str(minimum_iou)]
    folder.mkdir(parents=True, exist_ok=True)
    job_path = folder / 'inference_job.json'
    active = {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED',
              'REQUEUED', 'RESIZING', 'ACCOUNTING_PENDING'}
    job_id = None
    if job_path.is_file() and shutil.which('squeue'):
        saved = json.loads(job_path.read_text())
        if saved.get('command') == command and _job_state(saved['job_id']) in active:
            job_id = saved['job_id']
            print(f'继续等待已有 GPU 作业 {job_id}', flush=True)

    use_slurm = execution == 'slurm' or job_id is not None
    if not use_slurm:
        probe = _run([str(python), '-c',
                      'import torch; print(int(torch.cuda.is_available()))'], cwd=root)
        has_cuda = probe.splitlines()[-1] == '1'
        if not has_cuda and execution == 'local':
            raise RuntimeError("本机没有可用 CUDA GPU；使用 execution='auto' 或 'slurm'")
        use_slurm = not has_cuda
    if not use_slurm:
        print('使用本机已分配的 CUDA GPU 推理', flush=True)
        completed = subprocess.run(command, cwd=root, capture_output=True, text=True)
        logs = completed.stdout + '\n' + completed.stderr
        (folder / 'inference_local.log').write_text(logs)
        if completed.returncode:
            raise RuntimeError(f'GPU inference failed ({completed.returncode}):\n{logs}')
    else:
        if not all(shutil.which(name) for name in ('sbatch', 'squeue', 'sacct')):
            raise RuntimeError('本机没有 CUDA GPU，也找不到 Slurm 命令；请在 GPU 节点运行新病例推理')
        if job_id is None:
            script = folder / 'inference.sbatch'
            script.write_text(
                '#!/usr/bin/env bash\nset -euo pipefail\n'
                f'cd {shlex.quote(str(root))}\n'
                'export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1\n'
                'export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128\n'
                f'{shlex.join(command)}\n')
            submitted = _run([
                'sbatch', '--parsable', '--job-name=joint-eval-notebook',
                f'--partition={partition}', f'--account={account}', f'--gres={gres}',
                '--cpus-per-task=4', '--mem=64G', '--time=00:20:00',
                f'--output={folder}/inference_%j.out', f'--error={folder}/inference_%j.err', str(script)])
            job_id = submitted.split(';')[0].strip()
            if not re.fullmatch(r'\d+', job_id):
                raise RuntimeError(f'Unexpected sbatch job ID: {submitted!r}')
            job_path.write_text(json.dumps({'job_id': job_id, 'command': command}, indent=2) + '\n')
            print(f'已提交单卡 GPU 作业 {job_id}，病例 {case_id}；日志目录: {folder}', flush=True)
        started = time.monotonic()
        last_state, last_notice = None, 0.0
        try:
            while True:
                state = _job_state(job_id)
                elapsed = time.monotonic() - started
                if state != last_state or elapsed - last_notice >= 30:
                    print(f'Slurm {job_id}: {state}，已等待 {elapsed:.0f} 秒', flush=True)
                    last_state, last_notice = state, elapsed
                if state == 'COMPLETED':
                    break
                if state not in active:
                    raise RuntimeError(f'Slurm job {job_id} failed: {state}\n{_logs(folder, job_id)}')
                if elapsed >= max_wait_seconds:
                    raise TimeoutError(f'作业 {job_id} 仍为 {state}；重新运行此单元格可继续等待同一作业')
                time.sleep(poll_seconds)
        except KeyboardInterrupt:
            print(f'停止等待；GPU 作业 {job_id} 仍在运行。重跑此单元格可继续等待；取消命令: scancel {job_id}', flush=True)
            raise
    if not cache_ready():
        detail = _logs(folder, job_id) if job_id is not None else (folder / 'inference_local.log').read_text()
        raise RuntimeError(f'Inference finished without complete result files in {folder}\n{detail}')
    check_cache()
    print(f'推理完成: {folder}', flush=True)
    return folder
