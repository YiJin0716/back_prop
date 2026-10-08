#!/usr/bin/env bash
#SBATCH --partition=compsci-gpu
#SBATCH --account=rudin
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
set -euo pipefail
ROOT=/home/users/yj255/code/imaging_feature
cd "$ROOT"
export PYTHONPATH="$ROOT/vista3D/VISTA/vista3d:$ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
exec "$ROOT/back_prop/baseline_compare/.venv/bin/python" -m back_prop.baseline_compare.cross_validation.run_stage "$@"
