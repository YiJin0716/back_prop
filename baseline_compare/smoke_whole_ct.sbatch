#!/usr/bin/env bash
#SBATCH --job-name=baseline-whole-smoke
#SBATCH --partition=compsci-gpu
#SBATCH --account=rudin
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/users/yj255/code/imaging_feature/back_prop/baseline_compare/logs/whole_smoke_%j.out
#SBATCH --error=/home/users/yj255/code/imaging_feature/back_prop/baseline_compare/logs/whole_smoke_%j.err
set -euo pipefail
cd /home/users/yj255/code/imaging_feature
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
PYTHON=back_prop/baseline_compare/.venv/bin/python
RUNROOT=/usr/project/rudinlab/datasets/LIDC_IDRI/joint_model/baseline_compare
SMOKE_OUTPUT=$RUNROOT/whole_smoke/${SLURM_JOB_ID}
export SMOKE_OUTPUT
mkdir -p "$SMOKE_OUTPUT"
# This tiny GBM tests connectivity only; it is never a formal trained artifact.
"$PYTHON" - <<'PY'
import json, os
from pathlib import Path
import torch
from back_prop.baseline_compare.cohort import load
from back_prop.baseline_compare.prepare import CACHE
from back_prop.baseline_compare.nodule_data import NoduleDataset
from back_prop.baseline_compare.deeplung.ported.classifier import DPN92_3D
from back_prop.baseline_compare.deeplung.train_classifier import fit_gbm
root = Path('/usr/project/rudinlab/datasets/LIDC_IDRI/joint_model/baseline_compare')
data = NoduleDataset(load()['splits']['training'][:10], CACHE, 'deeplung', augment=False)
assert len({item['target'] for item in data.items}) == 2
model = DPN92_3D().cuda()
saved = torch.load(root/'deeplung/classifier/12622704/last.pt',map_location='cpu',weights_only=False)
model.load_state_dict(saved['model'])
data.mean, data.std = saved['config']['pixel_mean'], saved['config']['pixel_std']
fit_gbm(model, data, Path(os.environ['SMOKE_OUTPUT']))
print(json.dumps({'test_only_gbm': True, 'nodules': len(data)}), flush=True)
PY
IMAGE=/usr/project/rudinlab/datasets/LIDC_IDRI/original_LIDC/NIFTI_ver/LIDC-IDRI-0001__scan12.nii.gz
"$PYTHON" -m back_prop.baseline_compare.predict --model sybil --image "$IMAGE" \
  --checkpoint "$RUNROOT/sybil/12620983/last.pt" --output "$SMOKE_OUTPUT/sybil.json"
"$PYTHON" -m back_prop.baseline_compare.predict --model deeplung --image "$IMAGE" \
  --checkpoint "$RUNROOT/deeplung/classifier/12622704/last.pt" \
  --detector-checkpoint "$RUNROOT/deeplung/detector/12622312/last.pt" \
  --gbm "$SMOKE_OUTPUT/gbm.pkl" --output "$SMOKE_OUTPUT/deeplung.json"
"$PYTHON" -m back_prop.baseline_compare.predict --model edicnet --image "$IMAGE" \
  --checkpoint "$RUNROOT/edicnet/hscnn/12620988/last.pt" \
  --detector-checkpoint "$RUNROOT/edicnet/retinanet/12620996/last.pt" \
  --output "$SMOKE_OUTPUT/edicnet.json"
