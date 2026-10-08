#!/usr/bin/env bash
#SBATCH --job-name=baseline-a6000-bench
#SBATCH --partition=compsci-gpu
#SBATCH --account=rudin
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=/home/users/yj255/code/imaging_feature/back_prop/baseline_compare/logs/a6000_bench_%j.out
#SBATCH --error=/home/users/yj255/code/imaging_feature/back_prop/baseline_compare/logs/a6000_bench_%j.err
set -euo pipefail
cd /home/users/yj255/code/imaging_feature
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
back_prop/baseline_compare/.venv/bin/python - <<'PY'
import json,time,numpy as np,torch
from back_prop.baseline_compare.cohort import load
from back_prop.baseline_compare.prepare import CACHE
from back_prop.baseline_compare.nodule_data import NoduleDataset
from back_prop.baseline_compare.deeplung.ported.classifier import DPN92_3D
torch.set_num_threads(2); torch.backends.cudnn.benchmark=True
data=NoduleDataset(load()['splits']['training'][:10],CACHE,'deeplung')
x=torch.stack([data[i][0] for i in range(16)]).cuda()
y=torch.tensor([data.items[i]['target'] for i in range(16)]).cuda()
model=DPN92_3D().cuda().train()
optimizer=torch.optim.SGD(model.parameters(),lr=.1,momentum=.9,weight_decay=5e-4)
times=[]
for i in range(8):
 torch.cuda.synchronize(); start=time.monotonic()
 optimizer.zero_grad(set_to_none=True)
 with torch.autocast('cuda',dtype=torch.bfloat16):
  logits,_=model(x)
 loss=torch.nn.functional.cross_entropy(logits.float(),y)
 loss.backward(); optimizer.step(); torch.cuda.synchronize()
 if i>=3:times.append(time.monotonic()-start)
assert torch.isfinite(loss)
print(json.dumps({'gpu':torch.cuda.get_device_name(),'batch_size':16,'median_step_seconds':float(np.median(times)),
 'peak_allocated_gb':torch.cuda.max_memory_allocated()/2**30,'estimated_700_epoch_compute_hours':float(np.median(times))*100*700/3600,
 'note':'Compute estimate excludes data loading and checkpoint writes; benchmark weights are discarded'}),flush=True)
PY
