# V5 checkpoint evaluation

This analysis evaluates V5 checkpoints on all 178 held-out CTs (540 physical
nodules and 1,423 reader annotations), with separate directories for each run.
`results_12877415_epoch18/` contains the completed evaluation of run 12877415.
`results_12903757_epoch18/` is scheduled after Sybil and DeepLung fine-tuning;
its `_data/jobs.json` records the job chain. Both use the same evaluation protocol.
The 12903757 result directory links to project storage at
`/usr/project/rudinlab/datasets/LIDC_IDRI/joint_model/model_v5_runs/12903757/evaluation/`
to avoid the home-directory file-count quota; use the result path here as usual.

`results_12955332_epoch18/` evaluates the V5 hard/temperature-annealed checkpoint
(epoch 18, mask temperature 0.1). Its `_data/jobs.json` records two inference
shards followed by report generation and the existing DeepLung/Sybil comparison.
The matching and label definitions are the same as the other V5 evaluations.

`annotations.py` provides the reader-level cohort, full-mask CT crops, frozen
semantic/risk inference, and risk-bank fingerprint checks used by `run.py`.

## Single-CT inference

Run from `/home/users/yj255/code/imaging_feature` using the existing `pro6000` environment and an allocated GPU:

```python
from back_prop.evaluate.eval_package import JointModelEvaluator, save_result

evaluator = JointModelEvaluator.from_checkpoint(
    "/usr/project/rudinlab/datasets/LIDC_IDRI/joint_model/model_v5_runs/12877415/epoch_018.pt",
    manifest="vista3D/data/folds/fold_0.json",
    device="cuda",
)
result = evaluator.evaluate("LIDC-IDRI-0005__scan16")
print(result.scan_probability)
save_result(result, "back_prop/evaluate/my_ct_result", nifti=True)
```

This exports screening and refined segmentations, per-nodule mask probabilities, six semantic features and five-class distributions, nodule malignancy, ensemble probabilities, final CT risk, official references, plots, JSON, NPZ and NIfTI. `load_result` reopens the saved output on CPU. Inference receives only CT; official annotations are attached afterward. Existing manifest-based and Slurm/notebook entry points are documented in `../eval_package/README.md`.

## Analysis outputs

Each `results_<run>_epoch18/` directory is organized by evaluation level, then input → output point:

- `annotation_lvl/`: annotator segmentation → annotator LIDC features; annotator LIDC features → annotator malignancy.
- `volume_lvl/`: exclude unmatched segmentations; FROC; exclusion metrics; volume-level AUC/accuracy (any positive reader diagnosis).
- `nodule_lvl/`: predicted segmentation → overlap (DICE/IoU); predicted segmentation → LIDC features; predicted segmentation → LIDC features → annotator malignancy.
- `v5_joint_evaluation.pdf`: the complete combined report.
- `README.md`: links to every evaluation point. Each point contains its own PNG/PDF figures, CSV tables and input/target definitions.

For example, per-nodule **Predicted Segmentation → LIDC Feature** threshold AUC
is in `nodule_lvl/predicted_segmentation_to_lidc_feature/threshold_auc.png`.
Its two heatmaps compare with the closest reader and reader mean; annotation-level
results have their own figure under `annotation_lvl/`. All semantic threshold AUC
heatmaps show six feature rows and thresholds 1–5, with a shared 0–1 color scale
and cell values. Undefined AUCs are shown as N/A.

The overlap point has `vs_closest/`, `vs_union/`, and `vs_intersection/`.
The feature point has `closest_annotation/` and `consensus_annotation/`.
The nodule malignancy point has `exclude_3/`, `ge_3/`, and `gt_3/`.
Size analyses are under
`volume_lvl/exclusion_metrics/malignancy_vs_nodule_size/`, matching the screenshot's
placement within exclusion metrics. Its `filter_small_predictions/` subdirectory
contains the 5-mm cutoff analysis; `small_nodule_performance/` contains cumulative
results for small official nodules.

`_data/` retains `cases/*.json`, `summary.json`, inference protocols and completion
markers. These contain all predictions and overlap metrics needed to regenerate
the plots. Unused raw tensor archives, the example CT export, duplicate combined
PNGs/tables and old logs were removed. New evaluations save only the required JSON
records; the public `eval_package.save_result` API still supports explicit mask exports.
The additional annotation-mask → predicted-feature → malignancy comparison is
retained in `_data/supplementary/official_segmentation_to_malignancy/`.

To reorganize and redraw point-specific figures from existing outputs without
running inference, use `python -m back_prop.evaluate.v5_joint_analysis.organize`.
The inference and report entry points both understand the `_data/` layout;
report generation also writes the point-specific figures and tables.

DeepLung and Sybil fine-tuning, whole-CT evaluation and automatic ROC updates are
managed by `back_prop.baseline_compare.v5_comparison`. Its frozen plan and Slurm
job IDs are in `../../baseline_compare/v5_12877415/`. The comparison uses all 178
test CTs and the same three reader-label rules as V5. Once evaluation finishes,
`volume_lvl/volume_auc_accuracy/malignancy_roc.png` and `.pdf` contain all three
models, with paired predictions and metrics beside them. Redrawing the ordinary
report preserves the baseline curves.

For a subsequent checkpoint, use `run --checkpoint CHECKPOINT --output RESULT_DIR`
and `report --output RESULT_DIR`. `run --baseline-plan BASELINE_PLAN` verifies
both completed baseline training markers before starting inference. After the
new report and baseline inference finish, `back_prop.baseline_compare.v5_comparison
compare --output RESULT_DIR` adds the same fitted baselines to its volume ROC,
checking that training cases, test cases and reader-label rules agree.

Two inference shards share the same frozen checkpoint and disjoint CT subsets. The first CT on each shard checks that selecting default-threshold candidates from the complete candidate pass reproduces an independent ordinary forward, including final risk. Completion checks enforce full cohort coverage and an unchanged risk bank.

## Definitions

- **Reader level:** each reader's segmentation predicts their semantic ratings. Their six official semantic ratings, with the radiomics baseline from the same mask, separately test the malignancy model. An additional table uses predicted semantics.
- **Nodule level:** match physical nodules one-to-one using the best overlap with any reader mask, at IoU ≥ 0.1. “Closest” means highest IoU, with annotation ID breaking ties. Report Dice/IoU against that reader, union, intersection and two-vote consensus. Matched semantic metrics use closest-reader scores or reader means. Accuracy uses the highest-probability class; consensus classes round means half-up. Threshold labels use unrounded means.
- **Unmatched predictions:** exclude from paired nodule metrics, retain in deployed CT risk and count as FROC false positives. Record total, predicted-malignant exclusions and scans where an exclusion has the highest risk (including ties).
- **Malignancy:** evaluate all three rules: omit score 3; score ≥3 positive; score >3 positive. Apply these to individual reader scores or nodule means. Labels are reader assessments, as explicitly requested; no pathology labels are used. This checkpoint's training excluded physical nodules with mean malignancy = 3; the two inclusion rules also evaluate those nodules in the test cohort.
- **Whole CT:** report both any positive individual reader (the “any positive diagnosis” interpretation, used in ROC plots) and any positive mean nodule score (consensus). Under the omit-3 rule, a CT without positive labels is unknown if any label is indeterminate. CT risk follows the checkpoint's object-weighted noisy-OR within each bank member and subsequent ensemble mean. Secondary tables include oracle matched-only CT risk. Volume columns in the size-filter table use nodule-consensus labels.
- **Detection and misses:** FROC sweeps the full post-NMS candidate set; positives may match any reader at IoU ≥ 0.1. Missed GT stays in detection denominators. All-nodule malignancy metrics assign missed nodules risk 0; matched-only metrics are separately named.
- **Size:** maximum 3D Feret diameter in physical millimetres; GT averages reader-mask diameters. One analysis removes predictions of size ≤d and rematches while retaining all GT. Another evaluates GT nodules of size ≤d without filtering. Both report sensitivity/specificity at malignancy probability 0.5, including the 5-mm cutoff.

The inference module is `back_prop.evaluate.v5_joint_analysis.run`; aggregation is `back_prop.evaluate.v5_joint_analysis.report`. Report generation refuses incomplete or mixed-protocol results. The dedicated tests cover matching, missed-positive denominators, single-class AUC, threshold probabilities, uncertainty, risk pooling and physical geometry.
