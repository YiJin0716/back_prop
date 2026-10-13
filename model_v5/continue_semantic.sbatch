#!/usr/bin/env bash
#SBATCH --job-name=v5-semantic-continue
#SBATCH --partition=rudin,compsci-gpu
#SBATCH --account=rudin
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/users/yj255/code/imaging_feature/back_prop/model_v5/logs/semantic_continue_%j.out
#SBATCH --error=/home/users/yj255/code/imaging_feature/back_prop/model_v5/logs/semantic_continue_%j.err
set -euo pipefail
ROOT=/home/users/yj255/code/imaging_feature
PYTHON=/home/users/yj255/.local/miniforge3/envs/pro6000/bin/python
OUTPUT=${1:?Pass the continued warmup output directory}
SOURCE=/usr/project/rudinlab/datasets/LIDC_IDRI/joint_model/model_v5_runs/12876587/prepared
cd "$ROOT"
export PYTHONPATH="$ROOT/vista3D/VISTA/vista3d:$ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
test -f "$OUTPUT/semantic/latest.pt"
test ! -e "$OUTPUT/semantic/converged.pt"
MAXIMUM=$("$PYTHON" - "$OUTPUT/semantic/extension.json" <<'PY'
import json, sys
with open(sys.argv[1]) as stream:
    settings = json.load(stream)
maximum = int(settings['maximum_additional_epochs'])
assert maximum >= 20
print(maximum)
PY
)
LOCAL_CACHE=/tmp/yj255-v5-semantic-${SLURM_JOB_ID:?}
mkdir -p "$LOCAL_CACHE/official_warmup"
rsync -rt --no-group --no-owner "$SOURCE/official_warmup/" "$LOCAL_CACHE/official_warmup/"
cp "$SOURCE/official_cache_complete.json" "$LOCAL_CACHE/official_cache_complete.json"
exec "$PYTHON" -m back_prop.model_v5.continue_warmup \
    --branch semantic --source-prepared "$SOURCE" --output-dir "$OUTPUT" \
    --roi-cache-root "$LOCAL_CACHE" --maximum-additional-epochs "$MAXIMUM" \
    --wandb-mode online
