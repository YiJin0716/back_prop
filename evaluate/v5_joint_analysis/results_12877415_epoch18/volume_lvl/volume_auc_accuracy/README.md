# Volume-level AUC / Accuracy

The primary target is any positive reader diagnosis, using the previously agreed reader-score labels. Additional mean-nodule and matched-only comparisons are retained separately.

Label rules: exclude rating 3; rating ≥ 3 positive; rating > 3 positive. Labels come from LIDC readers.

- [additional_comparisons.csv](additional_comparisons.csv)
- [malignancy_metrics.csv](malignancy_metrics.csv)
- [malignancy_roc.pdf](malignancy_roc.pdf)
- [malignancy_roc.png](malignancy_roc.png)
- [predictions.csv](predictions.csv)

[All evaluation points](../../README.md)

## Fine-tuned baselines

V5, DeepLung and Sybil use identical test CTs and reader-derived labels.

[Paired predictions](baseline_predictions.csv) · [AUC/accuracy](baseline_metrics.csv) · [Training protocol](baseline_protocol.json)
