#!/usr/bin/env bash
#SBATCH --job-name=sybil-lidc
#SBATCH --partition=compsci-gpu
#SBATCH --account=rudin
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/users/yj255/code/imaging_feature/back_prop/baseline_compare/logs/sybil_%j.out
#SBATCH --error=/home/users/yj255/code/imaging_feature/back_prop/baseline_compare/logs/sybil_%j.err
set -euo pipefail
cd /home/users/yj255/code/imaging_feature
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
back_prop/baseline_compare/.venv/bin/python -m back_prop.baseline_compare.sybil.train \
  --output /usr/project/rudinlab/datasets/LIDC_IDRI/joint_model/baseline_compare/sybil/${SLURM_JOB_ID} "$@"
