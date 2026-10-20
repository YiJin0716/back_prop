# Annotator LIDC Feature → Annotator LIDC Malignancy

Input: the same reader’s six official ratings and radiomics from their mask.

Label rules: exclude rating 3; rating ≥ 3 positive; rating > 3 positive. Labels come from LIDC readers.

- [malignancy_metrics.csv](malignancy_metrics.csv)
- [malignancy_roc.pdf](malignancy_roc.pdf)
- [malignancy_roc.png](malignancy_roc.png)
- [predictions.csv](predictions.csv)

[All evaluation points](../../README.md)
