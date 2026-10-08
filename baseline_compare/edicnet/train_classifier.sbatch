#!/usr/bin/env bash
#SBATCH --job-name=edicnet-hscnn
#SBATCH --partition=compsci-gpu
#SBATCH --account=rudin
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/users/yj255/code/imaging_feature/back_prop/baseline_compare/logs/edicnet_hscnn_%j.out
#SBATCH --error=/home/users/yj255/code/imaging_feature/back_prop/baseline_compare/logs/edicnet_hscnn_%j.err
set -euo pipefail
cd /home/users/yj255/code/imaging_feature
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
back_prop/baseline_compare/.venv/bin/python -m back_prop.baseline_compare.edicnet.train_classifier \
  --output /usr/project/rudinlab/datasets/LIDC_IDRI/joint_model/baseline_compare/edicnet/hscnn/${SLURM_JOB_ID} "$@"
