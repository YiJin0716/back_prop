#!/usr/bin/env bash
#SBATCH --job-name=v5-joint-evaluation
#SBATCH --partition=rudin,compsci-gpu
#SBATCH --account=rudin
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
set -euo pipefail
ROOT=/home/users/yj255/code/imaging_feature
cd "$ROOT"
export PYTHONPATH="$ROOT/vista3D/VISTA/vista3d:$ROOT:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
/home/users/yj255/.local/miniforge3/envs/pro6000/bin/python -m back_prop.evaluate.v5_joint_analysis.run "$@"
