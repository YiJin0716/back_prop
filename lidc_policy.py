"""Shared LIDC-IDRI malignancy inclusion policy.

Binary malignancy experiments in this workspace operate on physical nodules,
not individual reader annotations.  A physical nodule is indeterminate when
the arithmetic mean of its available reader malignancy ratings is exactly 3.
Such nodules are excluded from supervised targets and evaluation cohorts.

Whole-CT images still contain excluded nodules.  Dataset/model code must carry
them as ignore regions so a detector is not trained to call a real but
indeterminate nodule background.
"""

from __future__ import annotations

import math
from typing import Iterable


INDETERMINATE_MALIGNANCY = 3.0
MALIGNANCY_ATOL = 1e-6
POLICY_NAME = "exclude_physical_nodule_reader_mean_malignancy_equal_3"


def reader_mean_malignancy(ratings: Iterable[float]) -> float:
    """Return a finite arithmetic reader mean for one physical nodule."""
    values = [float(value) for value in ratings if math.isfinite(float(value))]
    if not values:
        raise ValueError("A physical nodule has no finite malignancy ratings")
    return sum(values) / len(values)


def is_indeterminate_malignancy(value: float) -> bool:
    """Whether a physical-nodule reader mean is the indeterminate value 3."""
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"Non-finite malignancy value: {value!r}")
    return math.isclose(
        numeric, INDETERMINATE_MALIGNANCY, rel_tol=0.0, abs_tol=MALIGNANCY_ATOL
    )


def binary_malignancy_target(value: float) -> float:
    """Map a definitive physical-nodule mean to {0,1}; reject mean==3."""
    numeric = float(value)
    if is_indeterminate_malignancy(numeric):
        raise ValueError("Indeterminate physical nodules do not have a binary target")
    return float(numeric > INDETERMINATE_MALIGNANCY)


__all__ = (
    "INDETERMINATE_MALIGNANCY",
    "MALIGNANCY_ATOL",
    "POLICY_NAME",
    "binary_malignancy_target",
    "is_indeterminate_malignancy",
    "reader_mean_malignancy",
)
