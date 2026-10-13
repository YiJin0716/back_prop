#!/usr/bin/env bash
#SBATCH --job-name=v5-converge-joint
#SBATCH --partition=rudin
#SBATCH --account=rudin
#SBATCH --nodelist=rudin-01
#SBATCH --gres=gpu:a6000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/users/yj255/code/imaging_feature/back_prop/model_v5/logs/converge_%j.out
#SBATCH --error=/home/users/yj255/code/imaging_feature/back_prop/model_v5/logs/converge_%j.err
set -euo pipefail
ROOT=/home/users/yj255/code/imaging_feature
PYTHON=/home/users/yj255/.local/miniforge3/envs/pro6000/bin/python
RUNS=/usr/project/rudinlab/datasets/LIDC_IDRI/joint_model/model_v5_runs
CACHE=/usr/project/rudinlab/datasets/LIDC_IDRI/joint_model/v5_convergence_cache
OUTPUT=${V5_CONTINUE_OUTPUT:-$RUNS/${SLURM_JOB_ID}}
SOURCE=$RUNS/12876587/prepared
cd "$ROOT"
export PYTHONPATH="$ROOT/vista3D/VISTA/vista3d:$ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
# The default NCCL route hangs even on a one-element all-reduce on rudin-01.
# This configuration passed the same two-GPU communication check on that node.
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
mkdir -p "$OUTPUT"
# Node-local detector input copies avoid repeatedly streaming 32 GB per epoch.
LOCAL_CACHE=/tmp/yj255-v5-convergence-$(basename "$OUTPUT")
mkdir -p "$LOCAL_CACHE/projected" "$LOCAL_CACHE/native_training"
rsync -rt --no-group --no-owner "$CACHE/projected/" "$LOCAL_CACHE/projected/"
rsync -rt --no-group --no-owner "$CACHE/native_training/" "$LOCAL_CACHE/native_training/"
mkdir -p "$LOCAL_CACHE/roi/official_warmup" "$LOCAL_CACHE/roi/refiner_official"
rsync -rt --no-group --no-owner "$SOURCE/official_warmup/" "$LOCAL_CACHE/roi/official_warmup/"
rsync -rt --no-group --no-owner "$SOURCE/refiner_official/" "$LOCAL_CACHE/roi/refiner_official/"
cp "$SOURCE/official_cache_complete.json" "$LOCAL_CACHE/roi/official_cache_complete.json"
IFS=',' read -ra V5_GPUS <<< "${CUDA_VISIBLE_DEVICES:?A four-GPU allocation is required}"
if [[ ${#V5_GPUS[@]} != 4 ]]; then
    echo 'Expected exactly four visible GPUs' >&2
    exit 1
fi
COMMON=(--source-prepared "$SOURCE" --output-dir "$OUTPUT" --wandb-mode online --roi-cache-root "$LOCAL_CACHE/roi"
    --maximum-additional-epochs "${V5_CONTINUE_MAXIMUM_EPOCHS:-200}")
PIDS=()
if [[ ! -f "$OUTPUT/detr/converged.pt" ]]; then
    CUDA_VISIBLE_DEVICES="${V5_GPUS[0]},${V5_GPUS[1]}" "$PYTHON" -m torch.distributed.run \
        --standalone --nproc_per_node=2 -m back_prop.model_v5.continue_warmup \
        --branch detr --detector-checkpoint "$RUNS/12876587/latest.pt" \
        --projected-cache "$LOCAL_CACHE/projected" --native-cache "$LOCAL_CACHE/native_training" \
        "${COMMON[@]}" >> "$OUTPUT/detr.log" 2>&1 &
    PIDS+=("$!")
fi
if [[ ! -f "$OUTPUT/refiner/converged.pt" ]]; then
    CUDA_VISIBLE_DEVICES="${V5_GPUS[2]}" "$PYTHON" -m back_prop.model_v5.continue_warmup \
        --branch refiner "${COMMON[@]}" >> "$OUTPUT/refiner.log" 2>&1 &
    PIDS+=("$!")
fi
if [[ ! -f "$OUTPUT/semantic/converged.pt" ]]; then
    CUDA_VISIBLE_DEVICES="${V5_GPUS[3]}" "$PYTHON" -m back_prop.model_v5.continue_warmup \
        --branch semantic "${COMMON[@]}" >> "$OUTPUT/semantic.log" 2>&1 &
    PIDS+=("$!")
fi
# Finish the unaffected branches if one fails; preserve all resumable states.
FAILED=0
for PID in "${PIDS[@]}"; do
    wait "$PID" || FAILED=1
done
if [[ $FAILED != 0 ]]; then
    echo 'A branch failed or reached its epoch cap; joint training will not start.' >&2
    exit 1
fi
"$PYTHON" -m back_prop.model_v5.continue_warmup --branch export "${COMMON[@]}"
echo 'All three branches converged. Starting a separate joint training run.'
bash back_prop/model_v5/train.sbatch "$OUTPUT/prepared/preparation.pt" \
    --output-dir "$OUTPUT/joint" --wandb-name "v5-converged-joint-${SLURM_JOB_ID}"
