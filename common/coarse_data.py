"""Training data contract for the coarse-to-fine model v2.

The base loader performs the single canonical 1-mm resampling, groups reader
annotations by physical nodule, creates consensus masks, and applies the
workspace-wide indeterminate-malignancy policy.  This module adds empirical
reader distributions for the seven ordinal LIDC attributes.
"""

from __future__ import annotations

import numpy as np
import torch

from back_prop.common.base_data import (
    DEFAULT_ANNOTATIONS,
    DEFAULT_MANIFEST,
    DEFAULT_OFFICIAL_MASK_DIR,
    WholeCTLIDCDataset as _BaseDataset,
    load_cases,
    whole_ct_collate,
)


LABELS = (
    "lobulation",
    "malignancy",
    "margin",
    "sphericity",
    "spiculation",
    "subtlety",
    "texture",
)


class WholeCTLIDCDataset(_BaseDataset):
    """Return v1 geometry plus seven reader-level ordinal distributions."""

    @staticmethod
    def _reader_matrix(nodule: dict[str, object]) -> np.ndarray:
        # Base feature order: sphericity, margin, lobulation, spiculation,
        # texture, subtlety.  Keep every reader opinion for ordinal NLL.
        feature = np.asarray(nodule["reader_features"], dtype=np.float32)
        malignancy = np.asarray(nodule["reader_malignancy"], dtype=np.float32)
        return np.stack(
            (
                feature[:, 2],
                malignancy,
                feature[:, 1],
                feature[:, 0],
                feature[:, 3],
                feature[:, 5],
                feature[:, 4],
            ),
            axis=1,
        )

    def __getitem__(self, index: int) -> dict[str, object]:
        item = super().__getitem__(index)
        case = self.cases[index]
        key = (case["patient_id"], str(int(float(case["scan_id"]))))
        by_id = {
            int(row["nodule_id"]): row for row in self.annotations.get(key, [])
        }
        means: list[torch.Tensor] = []
        histograms: list[torch.Tensor] = []
        reader_counts: list[torch.Tensor] = []
        for nodule_id in item["nodule_ids"].tolist():
            readers = self._reader_matrix(by_id[int(nodule_id)])
            if not np.isfinite(readers).all() or not np.all((readers >= 1) & (readers <= 5)):
                raise ValueError(f"Invalid reader ratings for {key}, nodule {nodule_id}")
            means.append(torch.from_numpy(readers.mean(axis=0).astype(np.float32)))
            columns = []
            for column in range(len(LABELS)):
                counts = np.bincount(
                    readers[:, column].astype(np.int64), minlength=6
                )[1:6].astype(np.float32)
                columns.append(torch.from_numpy(counts / counts.sum()))
            histograms.append(torch.stack(columns))
            reader_counts.append(
                torch.full((len(LABELS),), readers.shape[0], dtype=torch.int16)
            )
        item["semantic_targets"] = (
            torch.stack(means)
            if means else torch.empty((0, len(LABELS)), dtype=torch.float32)
        )
        item["semantic_histograms"] = (
            torch.stack(histograms)
            if histograms
            else torch.empty((0, len(LABELS), 5), dtype=torch.float32)
        )
        item["semantic_reader_counts"] = (
            torch.stack(reader_counts)
            if reader_counts
            else torch.empty((0, len(LABELS)), dtype=torch.int16)
        )
        return item


__all__ = (
    "DEFAULT_ANNOTATIONS",
    "DEFAULT_MANIFEST",
    "DEFAULT_OFFICIAL_MASK_DIR",
    "LABELS",
    "WholeCTLIDCDataset",
    "load_cases",
    "whole_ct_collate",
)
