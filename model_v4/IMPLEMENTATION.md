# V4 implementation and training

`README.md` is the user specification. This document records implementation choices.

The 2026-09-30 user revision replaces MedicalNet BatchNorm with GroupNorm and
enables both diagnostic losses at full weight from joint-training epoch 1.
It supersedes the earlier live-BN and delayed-risk details in the original specification.

## Initialization and official-annotation warmup

VISTA3D/DETR/refiner geometry is initialized from the established V2 epoch-5 checkpoint;
MedicalNet starts from its pretrained ResNet50 weights. Before whole-CT joint training,
all retained **training-only** physical nodules are cropped using official consensus masks.
Masks are grouped by nodule ID, resampled once to 1 mm, and kept binary for this warmup.
The unchanged exclusion policy ignores reader-mean malignancy exactly 3.

1. Extract the existing 18 radiomics descriptors from original HU and official masks,
   including volume and diameter/spatial-extent descriptors. Fit the radiomics logistic baseline.
2. Fit the six-feature continuous FasterRisk pool using official reader-mean semantic
   features and the radiomics logit as a frozen offset. The probability residual is
   `y - sigmoid(baseline_logit)`; it is modeled through the logistic offset likelihood,
   not by treating a signed residual as a binary class.
3. Train MedicalNet against the official reader histograms for 3 warmup epochs
   (`--warmup-epochs`, default 3), with batches of 4 GT-masked ROIs and LR 1e-5.
   Save `warmup.pt` and `warmup.json`, including encoder updates, the number of GN layers,
   feature variation, and fixed-weight singleton/batched/train/eval consistency checks.
4. Begin the existing 18-epoch whole-CT curriculum. Warmup optimizer state is intentionally
   separate; joint training starts a fresh optimizer while retaining warmed weights and risk fits.

All 53 MedicalNet BatchNorm layers are converted to GroupNorm after loading pretrained
weights, including normalization in residual shortcuts. Convolution weights and the
per-channel affine weights/biases are retained; BN running means/variances are discarded.
For each layer, use the largest divisor of its channel count up to
`min(32, max(1, channels // 8))`: at least eight channels per group for normal MedicalNet
layers. This avoids one group per channel. No BN calibration, running statistics,
cross-rank BN averaging, or special recomputation update suppression is needed.
Warmup and joint training use the same per-sample normalization. The number of input
ROIs can differ without changing the mathematical prediction for an individual ROI.

## Objective

`5 box_l1 + 2 box_giou + coarse_dice + coarse_focal + 2 fine_dice + fine_focal
+ 2 missing_nodule + semantic + 0.5 nodule + 0.5 risk`

`diagnostic_scale` is always 1 from the first joint-training update. Both fitted stages
are initialized during official-annotation warmup and refreshed on predicted features
from that first update. There are no five diagnostic-free epochs or ten-epoch ramp.

There is **no independent object loss and no radiomics loss**. The DETR object-probability
output remains necessary for detection and the missing-nodule penalty. Radiomics is detached
from backpropagation and fitted on the deduplicated training cache solely as a baseline.
Semantic residual supervision cannot change the baseline through gradients.

`semantic` is the previous ordinal reader-histogram loss under its new name.
`detr_loss = 5 box_l1 + 2 box_giou + coarse_dice + coarse_focal`.
`finetuner_loss = 2 fine_dice + fine_focal`.
Fine focal still includes the V3 hard-negative fine-mask term. Matching and ignored-GT behavior
remain unchanged. `coarse_dict` in the specification is interpreted as `coarse_dice`.

Epochs 1–15 train segmentation, semantics, nodule malignancy and scan risk together;
epochs 16–18 additionally unfreeze the last VISTA automatic decoder block/class head.
The cosine LR schedule ends at 0.1 times each initial LR. Mask temperature stays exactly 1.

## Monitoring and restart

W&B receives one aggregate per warmup epoch and one per joint-training epoch, only on rank 0.
All raw joint losses use the same per-CT denominator, including zeros for ineligible cases;
weighted contributions reconstruct the total. Local SLURM logs retain batch diagnostics.
Component plots precede the ordered final group:
`missing_nodule, semantic, nodule, risk, detr_loss, finetuner_loss`.
`workspace.py` creates an explicit ordered comparison view and records its URL.

Checkpoints contain weights, both fitted risk stages and their deduplicated caches, optimizer,
scaler, LR schedule, all-rank RNG, warmup provenance, W&B identity and (for the sibling variant)
hard-case sampling history. Old V4 checkpoints use a different architecture ID and are rejected.

Submit with `sbatch back_prop/model_v4/train.sbatch`. Resume by adding the new checkpoint path.
`smoke.sbatch` exercises official warmup, 4-GPU joint training, resume, late VISTA unfreezing,
real training-only mining, oversampling and sampler resume, followed by held-out GT-free inference.

The 2026-10-02 DDP fix registers `WholeCTOutputV4` as a PyTorch pytree. PyTorch 2.8
traverses dataclasses during unused-parameter discovery, but its output backward
sink needs pytree registration to handle returned tensors not used by the loss.
When every matched ROI on a rank fails coverage, semantic/nodule losses and scan
risk are skipped. Without registration, MedicalNet is incorrectly left awaiting
gradient reduction and the next iteration fails. This is especially exposed at
epoch 18, when teacher routing is zero. The fix changes no loss, weights or
checkpoint format, and existing GroupNorm checkpoints can resume directly.
`tests/ddp_output.py` checks active/inactive semantic branches across ranks and
across iterations, including all-rank inactivity, against serial-reference gradients.
