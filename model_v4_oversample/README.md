# V4 with missing-nodule oversampling

This package shares the complete updated V4 model, official-annotation warmup, loss,
optimizer and curriculum. Only whole-CT training sampling differs.

The 2026-09-30 revision uses GroupNorm throughout MedicalNet and enables `nodule`
and `risk` at weight 0.5 each from joint-training epoch 1, after official-mask warmup.
New runs start from the original initialization; old BN runs/checkpoints are preserved.

Each epoch contains one ordinary pass over every retained training CT. Starting after
joint-training epoch 2, and every 2 epochs thereafter, the current model performs true
GT-free inference on the **training** cohort. The model stays in evaluation mode,
GT locations never route crops, risk caches are not updated, and RNG is preserved.
One-to-one matching compares predicted fine masks to the official physical-nodule masks.
Defaults: object threshold 0.5, mask threshold 0.5, mask IoU >= 0.1 with positive overlap.
These are the current evaluation conventions, not a claim of an official LUNA detection metric.

The next epoch adds up to 25% extra CT presentations, drawn from cases still containing a miss.
Sampling weights use the largest recent miss frequency among currently missed nodules in a CT,
over the last 3 mining passes. A CT receives at most 2 deliberate extra presentations per epoch;
if the eligible pool is too small, fewer repeats are used. Recovered cases stop being repeated.
GT rows are never duplicated inside a CT, and all other nodules/background remain supervised.

All ranks deterministically construct the same global schedule and take disjoint rank slices.
Before hard repeats begin, ordering/padding exactly match V4's ordinary DistributedSampler
for the same seed and epoch, so the initial joint-training passes share the same examples.
Up to `world_size-1` uniformly drawn padding presentations keep DDP iteration counts equal;
padding is reported separately from deliberate repeats. Bank caches continue to use one row
per physical nodule. Validation/testing data never enter the mining history or fitting caches.

`mining_XXX.json` records each nodule's misses and thresholds. Epoch logs record repeats,
padding, schedule hashes and update counts. Checkpoints restore the complete mining history.
W&B has epoch-only losses plus sampling counts. The shared ordered workspace compares both runs.

Submit: `sbatch back_prop/model_v4_oversample/train.sbatch`.
Defaults add compute (about 25% more training presentations, plus periodic inference passes).
Therefore these 18-epoch runs are a practical first comparison, **not** an equal-step ablation;
use an equal-optimizer-step control before attributing any improvement solely to hard-case selection.
