#!/usr/bin/env bash
#SBATCH --job-name=lidc-v5-cnn
#SBATCH --partition=compsci-gpu
#SBATCH --account=rudin
#SBATCH --gres=gpu:a6000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/users/yj255/code/imaging_feature/back_prop/model_v5/logs/train_%j.out
#SBATCH --error=/home/users/yj255/code/imaging_feature/back_prop/model_v5/logs/train_%j.err
set -euo pipefail
ROOT=/home/users/yj255/code/imaging_feature
PYTHON=/home/users/yj255/.local/miniforge3/envs/pro6000/bin/python
cd "$ROOT"
export PYTHONPATH="$ROOT/vista3D/VISTA/vista3d:$ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
OUTPUT=/usr/project/rudinlab/datasets/LIDC_IDRI/joint_model/model_v5_runs/${SLURM_JOB_ID}
# A separately submitted joint run can supersede the continuation's automatic
# launch while its warmup processes finish normally.
if [[ -n ${V5_CONTINUE_OUTPUT:-} && "${1:-}" == "$V5_CONTINUE_OUTPUT/prepared/preparation.pt" && -f "$V5_CONTINUE_OUTPUT/joint_handoff.json" ]]; then
    "$PYTHON" - "$V5_CONTINUE_OUTPUT/joint_handoff.json" "${SLURM_JOB_ID}" <<'PY'
import json, sys
from pathlib import Path
handoff = json.loads(Path(sys.argv[1]).read_text())
assert str(handoff['continuation_job_id']) == sys.argv[2], 'Joint handoff belongs to another job'
assert str(handoff['joint_job_id']).isdigit() and str(handoff['joint_job_id']) != sys.argv[2]
assert Path(handoff['preparation_checkpoint']).is_file(), 'Joint handoff checkpoint is missing'
print(f"Separate joint job {handoff['joint_job_id']} was submitted; automatic duplicate launch skipped.", flush=True)
PY
    exit 0
fi
INITIAL=()
if [[ $# -gt 0 && "$1" != --* ]]; then
    RESUME_KIND=$("$PYTHON" -c 'import sys,torch; s=torch.load(sys.argv[1],map_location="cpu",weights_only=False); print("--prepare-resume" if "preparation_schema" in s else "--resume")' "$1")
    INITIAL=("$RESUME_KIND" "$1")
    shift
fi
# Explicit initialization flags replace positional resume.
for argument in "$@"; do
    case "$argument" in
        --restart-from|--init-v2|--resume|--prepare-resume|--restart-from=*|--init-v2=*|--resume=*|--prepare-resume=*) INITIAL=() ;;
    esac
done
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
    --log-dir "$OUTPUT/torchrun" -m back_prop.model_v5.train \
    --output-dir "$OUTPUT" --epochs 18 --window-size 128 128 128 \
    --amp-dtype bfloat16 --semantic-roi-size 64 --roi-shape 128 128 128 \
    --sparsity 5 --coefficient-bound 5 --pool-size 30 --beam 10 --gap 0.05 \
    --sample-fraction 0.3 --bank-mode sample --bank-min-samples 32 --bank-refresh-every 1 \
    --missing-nodule-weight 2 --baseline-l2 0.001 \
    --geometry-min-epochs 15 --geometry-max-epochs 30 \
    --refiner-min-epochs 15 --refiner-max-epochs 30 \
    --warmup-ready-epochs 3 --geometry-probe-cases 32 \
    --geometry-min-coverage 0.75 --geometry-min-dice 0.25 --geometry-min-usable 0.5 \
    --semantic-min-epochs 20 --warmup-epochs 40 --warmup-batch-size 8 \
    --semantic-min-dice 0.3 --semantic-anchor-weight 0.5 --semantic-anchor-batch-size 4 \
    --wandb-mode online --wandb-name "v5-cnn-${SLURM_JOB_ID}" \
    --teacher-full-epochs 1 --teacher-zero-epoch 11 \
    --lr-semantic-warmup 0.0003 --lr-semantic 0.00003 \
    --lr-min-ratio 0.1 --lr-schedule-epochs 18 "${INITIAL[@]}" "$@"
