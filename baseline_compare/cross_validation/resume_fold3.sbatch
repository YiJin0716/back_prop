#!/usr/bin/env bash
#SBATCH --job-name=cv-f3-v3-resume
#SBATCH --partition=rudin
#SBATCH --account=rudin
#SBATCH --gres=gpu:6000_ada:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/users/yj255/code/imaging_feature/back_prop/baseline_compare/cross_validation/logs/f3-v3_%j.out
#SBATCH --error=/home/users/yj255/code/imaging_feature/back_prop/baseline_compare/cross_validation/logs/f3-v3_%j.err
set -euo pipefail
ROOT=/home/users/yj255/code/imaging_feature
cd "$ROOT"
export PYTHONPATH="$ROOT/vista3D/VISTA/vista3d:$ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
PYTHON="$ROOT/back_prop/baseline_compare/.venv/bin/python"
"$PYTHON" -m unittest back_prop.tests.test_mask_io -v
exec "$PYTHON" -m back_prop.baseline_compare.cross_validation.resume_v3 --fold 3 --checkpoint "$1"
