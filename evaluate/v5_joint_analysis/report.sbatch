#!/usr/bin/env bash
#SBATCH --job-name=v5-evaluation-report
#SBATCH --partition=compsci
#SBATCH --account=rudin
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
ROOT=/home/users/yj255/code/imaging_feature
cd "$ROOT"
export PYTHONPATH="$ROOT/vista3D/VISTA/vista3d:$ROOT:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
/home/users/yj255/.local/miniforge3/envs/pro6000/bin/python -m back_prop.evaluate.v5_joint_analysis.report "$@"
