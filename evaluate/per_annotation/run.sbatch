#!/usr/bin/env bash
#SBATCH --job-name=v4-per-annotation
#SBATCH --partition=rudin
#SBATCH --account=rudin
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/users/yj255/code/imaging_feature/back_prop/evaluate/per_annotation/logs/%j.out
#SBATCH --error=/home/users/yj255/code/imaging_feature/back_prop/evaluate/per_annotation/logs/%j.err
set -euo pipefail
cd /home/users/yj255/code/imaging_feature
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
exec /home/users/yj255/.local/miniforge3/envs/pro6000/bin/python -u \
    -m back_prop.evaluate.per_annotation.run --device cuda --workers 4 "$@"
