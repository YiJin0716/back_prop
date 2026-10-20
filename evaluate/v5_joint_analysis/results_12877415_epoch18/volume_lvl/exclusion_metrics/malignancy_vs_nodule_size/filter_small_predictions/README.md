# Filter Small Predicted Nodules

Remove predictions with diameter ≤ d, then rematch while retaining all official nodules. Includes the 5-mm cutoff and whole-CT metrics.

Label rules: exclude rating 3; rating ≥ 3 positive; rating > 3 positive. Labels come from LIDC readers.

- [cutoff_5mm.csv](cutoff_5mm.csv)
- [nodule_classification.pdf](nodule_classification.pdf)
- [nodule_classification.png](nodule_classification.png)
- [size_filtering.csv](size_filtering.csv)
- [volume_classification.pdf](volume_classification.pdf)
- [volume_classification.png](volume_classification.png)

[All evaluation points](../../../../README.md)
