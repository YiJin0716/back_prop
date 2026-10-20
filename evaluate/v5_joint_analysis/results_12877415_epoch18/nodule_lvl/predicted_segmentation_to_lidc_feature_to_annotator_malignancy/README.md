# Predicted Segmentation → LIDC Feature → LIDC Annotator Malignancy

Use predicted features and radiomics from predicted masks. Matched comparisons use closest-reader, all-reader and mean-reader labels; the separate all-nodule curve assigns risk zero to misses.

Label rules: exclude rating 3; rating ≥ 3 positive; rating > 3 positive. Labels come from LIDC readers.

- [exclude_3](exclude_3)
- [ge_3](ge_3)
- [gt_3](gt_3)
- [malignancy_metrics.csv](malignancy_metrics.csv)
- [malignancy_roc.pdf](malignancy_roc.pdf)
- [malignancy_roc.png](malignancy_roc.png)
- [malignancy_roc_all_nodules.pdf](malignancy_roc_all_nodules.pdf)
- [malignancy_roc_all_nodules.png](malignancy_roc_all_nodules.png)
- [predictions.csv](predictions.csv)

[All evaluation points](../../README.md)
