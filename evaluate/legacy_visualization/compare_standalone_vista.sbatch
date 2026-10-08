#!/usr/bin/env bash
#SBATCH --job-name=vista-seg-example
#SBATCH --partition=compsci-gpu
#SBATCH --account=rudin
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home/users/yj255/code/imaging_feature/back_prop/baseline_compare/logs/vista_seg_example_%j.out
#SBATCH --error=/home/users/yj255/code/imaging_feature/back_prop/baseline_compare/logs/vista_seg_example_%j.err
set -euo pipefail
cd /home/users/yj255/code/imaging_feature
export PYTHONPATH="$PWD/vista3D/VISTA/vista3d:$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
/home/users/yj255/.local/miniforge3/envs/vista3d/bin/python -m back_prop.evaluate.legacy_visualization.compare_standalone_vista "$@"
/home/users/yj255/.local/miniforge3/envs/visual/bin/python -m back_prop.evaluate.legacy_visualization.compare_standalone_vista --render-only "$@"
