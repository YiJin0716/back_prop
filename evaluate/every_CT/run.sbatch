#!/usr/bin/env bash
#SBATCH --job-name=v5-html-12903757
#SBATCH --partition=rudin,compsci-gpu
#SBATCH --account=rudin
#SBATCH --gres=gpu:a6000:1
#SBATCH --array=0-3%4
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/users/yj255/code/imaging_feature/back_prop/evaluate/every_CT/V5_12903757/_job/%A_%a.out
#SBATCH --error=/home/users/yj255/code/imaging_feature/back_prop/evaluate/every_CT/V5_12903757/_job/%A_%a.err
set -euo pipefail
ROOT=/home/users/yj255/code/imaging_feature
cd "$ROOT"
export PYTHONPATH="$ROOT/vista3D/VISTA/vista3d:$ROOT:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export MPLBACKEND=Agg
/home/users/yj255/.local/miniforge3/envs/pro6000/bin/python -m back_prop.evaluate.every_CT.run \
  --plan "${1:-$ROOT/back_prop/evaluate/every_CT/V5_12903757/_job/plan.json}"
