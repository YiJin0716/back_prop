#!/usr/bin/env bash
#SBATCH --job-name=lidc-v4-smoke
#SBATCH --partition=compsci-gpu
#SBATCH --account=rudin
#SBATCH --gres=gpu:rtx_pro_6000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=03:00:00
#SBATCH --output=/home/users/yj255/code/imaging_feature/back_prop/model_v4/logs/smoke_%j.out
#SBATCH --error=/home/users/yj255/code/imaging_feature/back_prop/model_v4/logs/smoke_%j.err
set -euo pipefail
ROOT=/home/users/yj255/code/imaging_feature
PYTHON=/home/users/yj255/.local/miniforge3/envs/pro6000/bin/python
cd "$ROOT"
export PYTHONPATH="$ROOT/vista3D/VISTA/vista3d:$ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
OUTPUT=/usr/project/rudinlab/datasets/LIDC_IDRI/joint_model/model_v4_smoke/${SLURM_JOB_ID}
INITIAL=/usr/project/rudinlab/datasets/LIDC_IDRI/joint_model/model_v2_runs/12596089/latest.pt
LAUNCH=("$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 -m back_prop.model_v4.train)
COMMON=(--max-cases 8 --window-size 128 128 128 --bank-min-samples 4 --warmup-epochs 1 --warmup-batch-size 2 --wandb-mode online)
"$PYTHON" -m unittest back_prop.model_v4.tests.test_v4 back_prop.model_v4.tests.test_revision -v
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 -m back_prop.model_v4.tests.ddp_bank
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 -m back_prop.model_v4.tests.ddp_output
"${LAUNCH[@]}" "${COMMON[@]}" --output-dir "$OUTPUT/first" --epochs 18 \
    --stop-after-epoch 1 --init-v2 "$INITIAL"
"${LAUNCH[@]}" "${COMMON[@]}" --output-dir "$OUTPUT/resumed" --epochs 18 \
    --stop-after-epoch 2 --resume "$OUTPUT/first/latest.pt"
"${LAUNCH[@]}" "${COMMON[@]}" --output-dir "$OUTPUT/diagnostic" --epochs 1 \
    --smoke-curriculum-epoch 15 --init-v2 "$INITIAL"
OVER=("$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 -m back_prop.model_v4_oversample.train)
"${OVER[@]}" "${COMMON[@]}" --output-dir "$OUTPUT/oversample" --epochs 18 \
    --stop-after-epoch 2 --mining-start-epoch 1 --mining-every 1 --init-v2 "$INITIAL"
"${OVER[@]}" "${COMMON[@]}" --output-dir "$OUTPUT/oversample_resumed" --epochs 18 \
    --stop-after-epoch 3 --mining-start-epoch 1 --mining-every 1 --resume "$OUTPUT/oversample/latest.pt"
"$PYTHON" -m back_prop.model_v4.tests.verify_smoke "$OUTPUT"
