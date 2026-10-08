#!/usr/bin/env bash
#SBATCH --job-name=cv-v3-late-smoke
#SBATCH --partition=compsci-gpu
#SBATCH --account=rudin
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/home/users/yj255/code/imaging_feature/back_prop/baseline_compare/cross_validation/logs/smoke_v3_%j.out
#SBATCH --error=/home/users/yj255/code/imaging_feature/back_prop/baseline_compare/cross_validation/logs/smoke_v3_%j.err
set -euo pipefail
ROOT=/home/users/yj255/code/imaging_feature
cd "$ROOT"
export PYTHONPATH="$ROOT/vista3D/VISTA/vista3d:$ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
exec "$ROOT/back_prop/baseline_compare/.venv/bin/python" -m back_prop.legacy_training.train_semantic \
 --manifest "$ROOT/vista3D/data/folds/fold_1.json" \
 --vista-checkpoint /usr/project/rudinlab/datasets/LIDC_IDRI/vista3D_workspace/results/fold_1/checkpoints/best_metric_model.pt \
 --output-dir "/usr/project/rudinlab/datasets/LIDC_IDRI/joint_model/cross_validation_20260921/smoke_v3_${SLURM_JOB_ID}" \
 --epochs 1 --max-cases 4 --window-size 128 128 128 --amp-dtype bfloat16 \
 --bank-min-samples 2 --smoke-curriculum-epoch 17 --teacher-full-epochs 18 --teacher-zero-epoch 19 --skip-save
