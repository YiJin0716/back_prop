# V5

## Semantic degradation and repair

The first V5 run (`12837027`) was stopped after semantic predictions became
compressed during joint training. At epoch 7, the predicted standard deviations
of margin, sphericity, and texture on 1,599 fixed official ROIs were 0.123, 0.091,
and 0.086, compared with 0.588, 0.339, and 0.650 after semantic warmup.

Controlled experiments identified an input mismatch. Semantic warmup used
centered official masks, whereas joint training immediately introduced shifted
crops and predicted soft masks. Translating otherwise unchanged official ROIs
by up to 8 mm increased warmup-model NLL from 0.985 to 1.326 on 256 ROIs.
Geometry initialization already included five V2 warmup epochs with crop jitter,
but its predictions were still unreliable in the V5 pipeline: 78 of 160 sampled
nodules had predicted-mask IoU below 0.1. Crop coverage alone did not establish
mask quality. Semantic-only training reproduced degradation, so malignancy
gradients were not necessary for the failure. These are diagnostic training-cohort
experiments, not a held-out clinical performance evaluation.

The repaired preparation sequence is:

1. Train detection and mask refinement in the current V5 pipeline, with jittered
   teacher crops followed by predicted crops. Assess prediction-only coverage
   and segmentation quality before advancing; keep the semantic CNN frozen.
2. Freeze geometry and train the semantic CNN on randomly shifted official ROIs
   and sufficiently overlapping predicted masks. Check both centered and shifted
   inputs with a fixed model, using NLL, correlation, and relative output spread.
3. Start joint training with an official-ROI semantic retention objective.
   Apply a separate predicted-mask quality condition to semantic/malignancy
   supervision, while preserving segmentation losses on poorly predicted masks.
   Ground truth never changes inference candidate selection or crop positions.

The original standard-deviation cutoff of 0.01 missed partial degradation.
The new checks compare fixed-checkpoint accuracy and per-attribute spread with
the post-warmup reference, rather than treating any nonzero variation as success.
The stopped run is retained for comparison; new training uses a new output directory.

### Preparation settings

Geometry preparation runs for at least 5 and at most 10 epochs. Its teacher
probabilities are 1, 0.75, 0.5, 0.25, then 0. A fixed, prediction-only probe of
32 training CTs must reach 75% GT crop coverage, mean soft Dice of 0.25 over
**all** GT nodules (uncovered nodules contribute zero to this quality metric),
and a usable-mask fraction of 50%. A usable mask has crop coverage at least
0.8 and soft Dice at least 0.3. These configurable readiness thresholds are
minimum training prerequisites, not clinical accuracy claims.

Semantic preparation freezes geometry and uses both official and usable
prediction-only ROIs. Official inputs receive fresh physical translations up
to 8 mm per axis on the original 1-mm grid before the single CNN resize,
with 20% left centered; translations that remove more than
10% of foreground mass are rejected. Predicted inputs retain the actual
localization and mask errors. The CNN trains for 8–20 epochs with learning
rate `3e-4`. Fixed probes use up to 128 nodules in each of three domains:
centered official, shifted official, and predicted masks. Each domain must
beat its constant-distribution NLL by 0.01, have mean rating correlation at
least 0.25 across varying attributes, and retain prediction/target standard
deviation ratios of at least 0.15. At least 32 probe nodules are required.

The risk bank is then fitted using the current CNN outputs and radiomics on
usable predicted training masks, so its initial inputs match the model rather
than reader scores. Joint training uses CNN learning rate `3e-5` and an
additional official-ROI semantic loss, weight 0.5, with 4 augmented official
ROIs per GPU per step. This objective has its own nodule denominator. The
same mask quality rule gates predicted-ROI semantic/nodule losses and risk
cache updates; scan risk requires usable ROIs for all retained GT nodules.
Fine segmentation and DETR supervision remain active on poor masks.

After every joint epoch, fixed-model checks also compare against the accepted
warmup reference: NLL increases over 0.1, per-attribute spread dropping below
half its reference, or correlation decreases over 0.2 are flagged. Two
consecutive failed audits stop training after saving the checkpoint. Probe
metrics and the separate effective sample counts are logged to W&B. The
prediction cache is a fixed reference; held-out evaluation still uses current
geometry and remains necessary to assess generalization.

If a preparation stage exhausts its epoch limit without passing readiness,
it stops with `preparation.pt` preserved; it does not silently enter joint
training. The small-cohort `--preparation-smoke` mode bypasses readiness only
to test the pipeline, requires limited cases and steps, and cannot be resumed
as production training.

V5 keeps V4's whole-CT detection, mask refinement, radiomics baseline, and
semantic residual malignancy model. It changes three parts of training:

- **Semantics:** a randomly initialized residual 3D CNN replaces MedicalNet.
  Its two input channels are masked CT and the soft mask. Standard convolutions,
  GroupNorm, and six five-class heads predict the six expected reader ratings.
  The CNN receives augmented semantic preparation after geometry preparation,
  with 8 ROIs per GPU per batch. Fixed-model probes check both predictive
  accuracy and output spread before and during joint training.
- **Loss reduction:** each nodule loss uses the number of nodules actually
  supervised by that loss. Scan risk uses the number of eligible scans.
  Positive and negative fine-mask focal losses have separate denominators.
  DDP pools valid counts across ranks; epoch logging pools numerators and
  counts across all steps. Missing supervision is `null`/N/A, not a zero loss.
  A missing required objective also makes the comparable epoch total N/A;
  `active_total` reports only the available objectives. Counts accompany all
  curves. Coarse geometry and the missing-nodule penalty still supervise GT
  nodules that never reach a valid fine ROI.
- **Teacher withdrawal:** joint epochs 1–10 linearly decrease teacher probability
  from 1.0 to 0.1 in steps of 0.1. Epochs 11–18 use 0. The automatic GT
  coverage fallback is removed. At probability 0, both candidate selection and
  crop centers use predictions, exactly as at inference. GT remains available
  for matching and loss targets. Normal runs require at least three epochs
  with teacher probability 0.

From this repository's parent directory:

```bash
mkdir -p back_prop/model_v5/logs
sbatch back_prop/model_v5/train.sbatch
```

The launcher requests four A6000 GPUs. It uses the existing V2 checkpoint for
geometry initialization, performs fresh geometry preparation, and trains a new
CNN. To resume preparation, pass `preparation.pt`; to resume joint training,
pass `latest.pt`. Retain both `official_warmup/` and `predicted_warmup/` in the
original preparation directory. Preparation checkpoints preserve stage,
optimizer, RNG, and geometry weights. Epoch limits can be extended explicitly:

```bash
sbatch back_prop/model_v5/train.sbatch /path/to/preparation.pt --geometry-max-epochs 15
```

V4 and pre-repair V5 checkpoints cannot resume this new training procedure.
Existing V5 checkpoints remain loadable for inference through
`evaluate/eval_package`, with their original prediction behavior.

The repair has pipeline and gradient tests. Its full training outcome must be
assessed from the new run; the old-run diagnosis does not establish the repaired
model's final accuracy.
