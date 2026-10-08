#!/usr/bin/env bash
#SBATCH --job-name=baseline-finalize
#SBATCH --partition=compsci-gpu
#SBATCH --account=rudin
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/users/yj255/code/imaging_feature/back_prop/baseline_compare/logs/finalize_%j.out
#SBATCH --error=/home/users/yj255/code/imaging_feature/back_prop/baseline_compare/logs/finalize_%j.err
set -euo pipefail
cd /home/users/yj255/code/imaging_feature
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
back_prop/baseline_compare/.venv/bin/python -m back_prop.baseline_compare.finalize "$@"
