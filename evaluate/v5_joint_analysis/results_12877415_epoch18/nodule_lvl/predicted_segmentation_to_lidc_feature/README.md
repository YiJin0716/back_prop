# Predicted Segmentation → LIDC Feature

Matched physical nodules only. Compare predicted features with the closest reader’s ratings and the mean reader ratings.

Threshold AUC uses the target rating ≥ t and predicted probability P(rating ≥ t). The heatmap shows t = 1–5; AUC at t = 1 is undefined (N/A). Exact-class accuracy uses the most probable class; reader means are rounded half-up only for accuracy.

- [accuracy_and_mae.pdf](accuracy_and_mae.pdf)
- [accuracy_and_mae.png](accuracy_and_mae.png)
- [closest_annotation](closest_annotation)
- [consensus_annotation](consensus_annotation)
- [nodule_predictions.csv](nodule_predictions.csv)
- [semantic_metrics.csv](semantic_metrics.csv)
- [threshold_auc.pdf](threshold_auc.pdf)
- [threshold_auc.png](threshold_auc.png)
- [threshold_metrics.csv](threshold_metrics.csv)

[All evaluation points](../../README.md)
