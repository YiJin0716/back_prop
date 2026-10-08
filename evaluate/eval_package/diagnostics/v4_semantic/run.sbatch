#!/usr/bin/env bash
#SBATCH --job-name=v4-semantic-audit
#SBATCH --partition=rudin
#SBATCH --account=rudin
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/users/yj255/code/imaging_feature/back_prop/evaluate/eval_package/diagnostics/v4_semantic/%j.out
#SBATCH --error=/home/users/yj255/code/imaging_feature/back_prop/evaluate/eval_package/diagnostics/v4_semantic/%j.err
set -euo pipefail
cd /home/users/yj255/code/imaging_feature
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
exec /home/users/yj255/.local/miniforge3/envs/pro6000/bin/python -u -m back_prop.evaluate.eval_package.diagnostics.v4_semantic.probe
