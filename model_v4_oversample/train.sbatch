#!/usr/bin/env bash
#SBATCH --job-name=lidc-v4-gn-over
#SBATCH --partition=compsci-gpu
#SBATCH --account=rudin
#SBATCH --gres=gpu:rtx_pro_6000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/users/yj255/code/imaging_feature/back_prop/model_v4_oversample/logs/train_%j.out
#SBATCH --error=/home/users/yj255/code/imaging_feature/back_prop/model_v4_oversample/logs/train_%j.err
set -euo pipefail
# Optional afternotok recovery: a manual cancellation must not restart training.
if [[ -n "${V4_RECOVER_FAILED_JOB_ID:-}" ]]; then
    PREVIOUS_STATE=$(sacct -X -n -P -j "$V4_RECOVER_FAILED_JOB_ID" --format=State)
    case "$PREVIOUS_STATE" in
        FAILED|OUT_OF_MEMORY|NODE_FAIL|TIMEOUT) ;;
        *) printf 'Recovery skipped: job %s is %s\n' "$V4_RECOVER_FAILED_JOB_ID" "$PREVIOUS_STATE"; exit 0 ;;
    esac
    if [[ $# -ne 1 || ! -s "$1" ]]; then
        printf 'Recovery requires the previous job checkpoint\n' >&2
        exit 1
    fi
fi
ROOT=/home/users/yj255/code/imaging_feature
PYTHON=/home/users/yj255/.local/miniforge3/envs/pro6000/bin/python
cd "$ROOT"
export PYTHONPATH="$ROOT/vista3D/VISTA/vista3d:$ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
OUTPUT=/usr/project/rudinlab/datasets/LIDC_IDRI/joint_model/model_v4_oversample_runs/${SLURM_JOB_ID}
INITIAL=(--init-v2 /usr/project/rudinlab/datasets/LIDC_IDRI/joint_model/model_v2_runs/12596089/latest.pt)
if [[ $# -gt 0 ]]; then INITIAL=(--resume "$1"); fi
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 -m back_prop.model_v4_oversample.train \
    --output-dir "$OUTPUT" --epochs 18 --window-size 128 128 128 \
    --amp-dtype bfloat16 --medicalnet-roi-size 64 --roi-shape 128 128 128 \
    --sparsity 5 --coefficient-bound 5 --pool-size 30 --beam 10 --gap 0.05 \
    --sample-fraction 0.3 --bank-mode sample --bank-min-samples 32 --bank-refresh-every 1 \
    --missing-nodule-weight 2 --baseline-l2 0.001 \
    --oversample-fraction 0.25 --oversample-max-extra 2 --mining-start-epoch 2 --mining-every 2 \
    --warmup-epochs 3 --warmup-batch-size 4 \
    --wandb-mode online --wandb-name "v4-gn-early-oversample-${SLURM_JOB_ID}" \
    --lr-min-ratio 0.1 --lr-schedule-epochs 18 "${INITIAL[@]}"
