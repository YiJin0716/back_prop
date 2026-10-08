#!/usr/bin/env bash
#SBATCH --job-name=v3-seg-example
#SBATCH --partition=compsci-gpu
#SBATCH --account=rudin
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home/users/yj255/code/imaging_feature/back_prop/baseline_compare/logs/v3_seg_example_%j.out
#SBATCH --error=/home/users/yj255/code/imaging_feature/back_prop/baseline_compare/logs/v3_seg_example_%j.err
set -euo pipefail
cd /home/users/yj255/code/imaging_feature
export PYTHONPATH="$PWD/vista3D/VISTA/vista3d:$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
back_prop/baseline_compare/.venv/bin/python -m back_prop.evaluate.legacy_visualization.visualize_test_ct "$@"
/home/users/yj255/.local/miniforge3/envs/visual/bin/python -m back_prop.evaluate.legacy_visualization.visualize_test_ct --render-only "$@"
