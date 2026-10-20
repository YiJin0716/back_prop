# V5

V5 combines whole-CT DETR localization, residual mask refinement, a six-attribute
3D semantic CNN, fixed radiomics, and a residual malignancy model.

## Detector variants

`--detector-kind legacy` preserves historical checkpoints. `mask_guided` derives
box geometry and local confidence features from each predicted query mask.
The `native_seeded` repair additionally preserves VISTA probabilities on the
original 1-mm grid, before whole-CT compression. Connected predictions at a
probability threshold of 0.1, with at least eight voxels, initialize up to 20
of the 24 queries. Learned heads adjust their confidence, boxes and masks; at
least four queries continue the original global search. Ground truth only supplies matching and
loss targets. The final object-confidence threshold remains 0.5.

Native priors retain their full-resolution soft masks for the refiner, with a
learned DETR mask residual added to them. Native proposal extraction is detached;
the learned residual and context still provide gradient paths. VISTA and context
conditioning gains start at zero to preserve the independently warmed refiner's
input domain, then learn during joint training. Query FiLM weights are already
zero initialized and retain live query inputs so their weights can learn.

Use `--object-background-weight 0.1` with the repaired detectors. Weighted object
loss uses a weighted denominator while logging the actual supervised query count.
`repair_detector.py` trains from fixed projected inputs and accepts a matching
native-prior cache through `--native-cache`; it retains the completed refiner and
semantic branches. Ordinary native-seeded warmup computes priors directly from
each CT. A raw-feature-only cache is insufficient for that variant. These repair
options require detection and assembled-pipeline evaluation before promotion;
passing a training probe alone does not establish held-out performance.

Resume and restart infer the detector variant, proposal/NMS thresholds and object
background weight from the selected checkpoint unless explicitly overridden.
Historical checkpoints without these fields retain the legacy defaults.

A full warmup `geometry.pt` can be loaded with
`load_joint_model(path, require_risk_bank=False)` from `evaluate.eval_package`.
Call the V5 model with `segmentation_only=True` for masks, or
`compute_diagnostics=False` for masks and semantic features, supplying the CT
and original `image_hu` without annotations. Malignancy evaluation still requires
both risk stages to be fitted on training predictions.

`--native-confidence-residual-bound 2` limits the learned correction to native
confidence to two log-odds units in either direction, using a smooth tanh bound.
This keeps a coarse confidence head from arbitrarily suppressing strong native
evidence for small nodules. The option is stored in checkpoints; absent values
preserve the original unbounded repair. It must be evaluated after retraining.

## Independent concurrent warmup

The four-GPU launcher runs three branches **at the same time**:

| Branch | GPUs | Training inputs and targets | Epochs |
|---|---:|---|---:|
| DETR | 2 | CT through its frozen VISTA backbone and trainable projections; official boxes and coarse masks supervise all queries | 15–30 |
| Refiner | 1 | Official CT crops with up to 8 mm position jitter; corrupted official masks as coarse priors, original official masks as targets | 15–30 |
| Semantic CNN | 1 | CT masked by official masks, with fresh crop translations; official reader-rating histograms as targets | 20–40 |

Each branch has its own optimizer, data iterator, epoch counter, readiness check,
and checkpoint. A finished branch waits while the others continue. No warmup
branch consumes another branch's prediction or sends gradients to it. DETR's
frozen VISTA backbone is part of its CT encoder; it is not a source of Refiner
or semantic training crops. Only the manifest's training patients are used.

Refiner priors use random translations, dilation/erosion, downsampling and
occasional removal. The target mask remains unchanged. Its VISTA probability,
DETR query and learned context input slots are zero during warmup; their normal
conditioning is introduced during joint training. The context compressor starts
training in that joint phase. This means standalone Refiner readiness does not
establish the quality of the assembled prediction pipeline.

Semantic inputs use the exact official binary mask on the original 1-mm grid,
then resize once to the CNN's 64-cube. Eighty percent receive random translations
up to 8 mm per axis; translations losing over 10% of foreground are rejected.
No predicted mask is used for semantic warmup. The CNN has ordinary 3D residual
convolutions, GroupNorm, and six five-class heads; it starts from random weights.

All branches require their minimum duration and three consecutive passing audits:

- DETR: a fixed 32-CT training probe must reach 75% crop coverage using selected
  predicted boxes and mean coarse Dice of 0.1. The refiner is not called.
- Refiner: a fixed 64-nodule probe with reproducible corruptions must reach mean
  soft Dice of 0.5 and at least match its corrupted-prior Dice.
- Semantics: centered and shifted official-mask probes, up to 128 nodules each,
  must beat a constant-distribution NLL by 0.01, achieve mean rating correlation
  of 0.25, and prediction/target standard-deviation ratios of 0.15 for varying
  attributes. These are training prerequisites, not held-out accuracy claims.

An exhausted epoch limit stops training with branch checkpoints preserved.
Readiness bypasses are allowed only in the explicitly limited smoke test.

After all branches finish, their weights are assembled on every GPU. Usable
prediction-only crops then initialize the malignancy bank. This step is separate
from the three warmups. If too few usable predictions exist, the job stops with
the independently warmed models preserved. Joint training does not start with
an invalid risk bank.

## Joint training and monitoring

For continued warmup, `continue_warmup.sbatch` restores the three previous
component checkpoints and Adam states in a separate run. DETR retains the repaired
detector's frozen projected inputs; refiner and semantic inputs remain independent.
`continue_warmup.py` requires at least 20 additional epochs, a 12-epoch fixed-probe
plateau (0.2% relative improvement tolerance), at least two learning-rate reductions,
a plateau in the last ten training losses, and passing quality checks. Probes use
128 training CTs, 256 refiner crops, and 512 centered/shifted semantic crops. The
manifest's validation patients overlap its training patients, so these checks
measure optimization convergence, not independent validation performance.
The best qualifying weights are assembled only after all three branches converge;
the epoch cap raises an error and cannot start joint training. The launcher then
starts a separate joint run with fresh prediction crops and fitted risk models.
Existing runs and source checkpoints are never overwritten.
To resume a stopped continuation, set `V5_CONTINUE_OUTPUT` to its output directory.
`V5_CONTINUE_MAXIMUM_EPOCHS` can extend the total additional-epoch cap (default 200)
without resetting optimizer states, learning rates, or the convergence history.

Joint training uses all four GPUs. Official semantic retention loss has weight
0.5 and four augmented official ROIs per GPU per step. Predicted-mask semantic
and malignancy supervision requires both at least 80% GT crop coverage and
soft Dice of at least 0.3. Fine segmentation still learns from poor predictions.
Risk and nodule weights are active from the first joint epoch.

Joint mask annealing is optional. Add `--mask-temperature-start 1.0
--mask-temperature-end 0.1 --mask-temperature-epochs 18` to `train.sbatch` to
decrease temperature by a cosine schedule from epoch 1 through epoch 18.
The final probability is `sigmoid(fine_mask_logits / temperature)`, shared by
the semantic inputs, radiomics, and evaluation exports. Segmentation losses
and the mask-quality supervision gate still use raw logits. Official-mask
warmup and semantic anchors are unchanged. The default remains constant 1.0.
Temperature is saved with the model, restored at inference, and logged to
`00_training/mask_temperature`; joint resume inherits and validates the schedule.

Joint teacher probabilities are 1.0, 0.9, …, 0.1 for epochs 1–10, then zero.
At zero, candidate selection and crop positions come only from predictions;
there is no coverage-triggered GT fallback. Ground truth remains available for
matching and loss targets. The VISTA backbone remains frozen until joint epoch
16, when its last automatic decoder block and class head are unfrozen.

Loss schema `v5_localization_v2` supervises object confidence on matched and
unmatched queries, ignores excluded annotations, uses the six-coordinate L1
sum, and defines missing-nodule loss as negative log soft Dice. Loss averages
use the actual supervised population, including across DDP ranks. Missing
supervision is N/A, not zero. `active_total` sums available terms; the comparable
`total` is N/A when a required term has no supervision.

W&B sections `detr_warmup`, `refiner_warmup`, and `semantic_warmup` each use that
branch's own epoch axis. They contain its losses and readiness metrics. Faster
branches may accumulate many epochs while DETR is still in its first epoch.
Rank zero relays their metrics between DETR steps. Joint losses are in
`01_components`, their weighted contributions in `00_training`, effective
sample counts in `02_objectives`, semantic probes in `03_semantic`, and the
assembled prediction-only geometry probe in `04_geometry`.

During joint training, fixed official-mask semantic probes compare with their
accepted warmup reference. NLL worsening by over 0.1, per-attribute spread
falling below half its reference, correlation decreasing by over 0.2, or nearly
constant predictions are flagged. Two consecutive failed audits stop the job
after saving. Current predicted-ROI semantic spread is logged separately.

## Running and resuming

From the repository's parent directory:

```bash
sbatch back_prop/model_v5/train.sbatch
sbatch back_prop/model_v5/train.sbatch /path/to/preparation.pt
sbatch back_prop/model_v5/train.sbatch /path/to/latest.pt
```

A fresh run loads only the existing standalone VISTA backbone checkpoint. DETR, Refiner and the
semantic CNN do not load previous warmup or joint weights by default.
Preparation schema is `v5_parallel_official_v3`; older preparations cannot resume.
Explicit initialization flags remain available for deliberately separate
experiments, but are not used by the new default run.

During preparation, `preparation.pt` points to the preparation directory.
Keep its `detr_warmup.pt`, `refiner_warmup.pt`, `semantic_warmup.pt`, official
caches and readiness files together. Unfinished branch checkpoints preserve
optimizer, AMP scaler, RNG and readiness streak. Completed branches restore
weights only; a detector repair can therefore retain the completed refiner and
semantic branches without reusing a different detector optimizer. Pass the
exported `prepared/preparation.pt` to the same launcher to continue a repaired run.
Resume reruns only unfinished branches.
Maximum epoch limits can be extended without changing the training policy.
Joint checkpoints are self-contained for inference, while joint resume also
requires the official semantic cache for retention training.

Legacy and mask-guided DETR warmup can reuse frozen VISTA features with
`--detr-feature-cache /path/to/cache`.
Build this optional cache on the training node using
`torchrun --standalone --nproc_per_node=4 -m back_prop.model_v5.feature_cache`
with `--config /path/to/config.json`, `--detr-checkpoint /path/to/detr_warmup.pt`
and `--output-dir /local/disk/cache`. It needs about 460 GiB for the full cohort.
Raw backbone maps are stored in FP32; the four projection layers still train.
Projection after window blending introduces small AMP rounding differences,
so this is not a bitwise-identical replay. Labels, batch size, optimizer state,
epoch requirements and readiness thresholds stay unchanged. A cache is accepted
only when its backbone, cohort, annotations and preprocessing configuration match.
It is used only during DETR warmup; joint training always reads full CTs.
Node-local caches require resume jobs to run on the same node and can be removed
after DETR warmup has finished.

Old warmup artifacts and warmup-only test runs were removed when switching to
this policy. Retained historical joint checkpoints remain usable for inference;
they do not represent the new training procedure. The full outcome must be
assessed from the new run.
