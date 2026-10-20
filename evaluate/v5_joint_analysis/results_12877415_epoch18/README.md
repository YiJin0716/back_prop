# V5 evaluation results

Run **12877415**, epoch **18**. Existing inference and metric values are preserved.

[Complete report](v5_joint_evaluation.pdf)

Choose the evaluation level, then the input → output point. PNG figures, PDF figures and CSV tables are stored together.

## annotation_lvl

- [Annotator Segmentation → Annotator LIDC Feature](annotation_lvl/annotator_segmentation_to_annotator_lidc_feature/): Each individual reader mask is used to predict that same reader’s six feature ratings.
- [Annotator LIDC Feature → Annotator LIDC Malignancy](annotation_lvl/annotator_lidc_feature_to_annotator_lidc_malignancy/): Input: the same reader’s six official ratings and radiomics from their mask.

## volume_lvl

- [Exclude Segmentations without an Annotation Match](volume_lvl/exclude_unmatched_segmentations/): Exclude unmatched predicted segmentations from paired per-nodule evaluations. The retained matches and excluded predictions are listed here. Excluded predictions remain in whole-CT risk and count as FROC false positives.
- [Whole CT → Nodule Detection (FROC)](volume_lvl/froc/): Sweep objectness over all post-NMS candidates. Every official physical nodule remains in the sensitivity denominator.
- [Capture Metrics about Exclusions](volume_lvl/exclusion_metrics/): Total exclusions, exclusions predicted malignant and exclusions tied for the highest-risk nodule. The malignancy_vs_nodule_size subdirectory contains sensitivity/specificity curves and the 5-mm cutoff analysis.
- [Volume-level AUC / Accuracy](volume_lvl/volume_auc_accuracy/): The primary target is any positive reader diagnosis, using the previously agreed reader-score labels. Additional mean-nodule and matched-only comparisons are retained separately.

## nodule_lvl

- [Predicted Segmentation → Overlap (DICE / IoU)](nodule_lvl/predicted_segmentation_to_overlap/): Compare predicted masks with the closest annotation, reader union and reader intersection. Additional two-vote consensus metrics remain in the CSV. All-GT metrics assign zero overlap to misses.
- [Predicted Segmentation → LIDC Feature](nodule_lvl/predicted_segmentation_to_lidc_feature/): Matched physical nodules only. Compare predicted features with the closest reader’s ratings and the mean reader ratings.
- [Predicted Segmentation → LIDC Feature → LIDC Annotator Malignancy](nodule_lvl/predicted_segmentation_to_lidc_feature_to_annotator_malignancy/): Use predicted features and radiomics from predicted masks. Matched comparisons use closest-reader, all-reader and mean-reader labels; the separate all-nodule curve assigns risk zero to misses.

## Supporting data

`_data/` retains case JSON predictions, summary metrics and inference protocols needed to redraw these results.
Additional annotation-mask → predicted-feature → malignancy results are in `_data/supplementary/official_segmentation_to_malignancy/`.
The root PDF is the original combined report. Point-specific figures contain only their corresponding evaluation level.
