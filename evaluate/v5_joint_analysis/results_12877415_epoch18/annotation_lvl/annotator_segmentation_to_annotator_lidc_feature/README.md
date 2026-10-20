# Annotator Segmentation → Annotator LIDC Feature

Each individual reader mask is used to predict that same reader’s six feature ratings.

Threshold AUC uses the target rating ≥ t and predicted probability P(rating ≥ t). The heatmap shows t = 1–5; AUC at t = 1 is undefined (N/A). Exact-class accuracy uses the most probable class; reader means are rounded half-up only for accuracy.

- [accuracy_and_mae.pdf](accuracy_and_mae.pdf)
- [accuracy_and_mae.png](accuracy_and_mae.png)
- [annotations.csv](annotations.csv)
- [semantic_metrics.csv](semantic_metrics.csv)
- [threshold_auc.pdf](threshold_auc.pdf)
- [threshold_auc.png](threshold_auc.png)
- [threshold_metrics.csv](threshold_metrics.csv)

[All evaluation points](../../README.md)
