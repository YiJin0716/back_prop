"""CPU result objects; instance masks stay compact instead of N whole CTs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

SEMANTIC_NAMES = ("lobulation", "margin", "sphericity", "spiculation", "subtlety", "texture")
RATING_NAMES = SEMANTIC_NAMES + ("malignancy",)


@dataclass
class MaskCrop:
    """An XYZ array and its integer origin on the resampled CT grid."""

    values: np.ndarray
    origin: tuple[int, int, int]

    def __post_init__(self):
        self.values = np.asarray(self.values)
        if self.values.ndim != 3 or len(self.origin) != 3:
            raise ValueError("MaskCrop requires a 3D array and an XYZ origin")
        self.origin = tuple(int(x) for x in self.origin)

    def on_grid(self, shape, origin=(0, 0, 0)) -> np.ndarray:
        """Paste with clipping, including negative/padded ROI origins."""
        result = np.zeros(tuple(shape), dtype=self.values.dtype)
        start = np.asarray(self.origin) - origin
        lower = np.maximum(start, 0)
        upper = np.minimum(start + self.values.shape, shape)
        if np.all(upper > lower):
            dst = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))
            src = tuple(slice(int(a), int(b)) for a, b in zip(lower-start, upper-start))
            result[dst] = self.values[src]
        return result


@dataclass
class GroundTruthNodule:
    nodule_id: int
    mask: MaskCrop
    annotation_ids: tuple[int, ...]
    reader_ratings: dict[str, list[float]]
    mask_paths: tuple[str, ...] = ()

    @property
    def means(self):
        return {name: float(np.mean(values)) if values else None
                for name, values in self.reader_ratings.items()}

    @property
    def binary_malignancy(self):
        from back_prop.lidc_policy import binary_malignancy_target, is_indeterminate_malignancy
        mean = self.means.get("malignancy")
        if mean is None or is_indeterminate_malignancy(mean):
            return None
        return int(binary_malignancy_target(mean))


@dataclass
class PredictedNodule:
    query_id: int
    mask: MaskCrop
    probability: MaskCrop
    object_probability: float
    semantic_features: dict[str, float]
    semantic_probabilities: dict[str, list[float]]
    malignancy_probability: float
    ensemble_probabilities: list[float] = field(default_factory=list)


@dataclass
class CaseEvaluation:
    case_id: str
    image_hu: np.ndarray
    affine: np.ndarray
    screening_mask: np.ndarray
    predictions: list[PredictedNodule]
    ground_truth: list[GroundTruthNodule]
    scan_probability: float
    metadata: dict[str, Any] = field(default_factory=dict)
    minimum_iou: float = 0.1

    @property
    def shape(self):
        return self.image_hu.shape

    @property
    def scan_target(self):
        labels = [n.binary_malignancy for n in self.ground_truth]
        if 1 in labels:
            return 1
        return None if None in labels else 0


def mask_union(crops, shape):
    result = np.zeros(shape, dtype=bool)
    for crop in crops:
        # Only materialize the clipped ROI, not another full CT.
        lower = np.maximum(crop.origin, 0)
        upper = np.minimum(np.asarray(crop.origin) + crop.values.shape, shape)
        if np.all(upper > lower):
            dst = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))
            src = tuple(slice(int(a), int(b)) for a, b in
                        zip(lower - crop.origin, upper - crop.origin))
            result[dst] |= crop.values[src].astype(bool)
    return result


def compact_mask(mask):
    positions = np.where(mask)
    if not len(positions[0]):
        return MaskCrop(np.zeros((0, 0, 0), dtype=bool), (0, 0, 0))
    lower = tuple(int(x.min()) for x in positions)
    upper = tuple(int(x.max()) + 1 for x in positions)
    return MaskCrop(mask[tuple(slice(a, b) for a, b in zip(lower, upper))].copy(), lower)
