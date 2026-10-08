"""Four paired prediction/official-GT views, with explicit missed detections."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .types import mask_union


def overlap(prediction, ground_truth):
    p, g = np.asarray(prediction, dtype=bool), np.asarray(ground_truth, dtype=bool)
    if p.shape != g.shape:
        raise ValueError("Prediction and GT grids differ")
    intersection = int(np.count_nonzero(p & g))
    return _metrics(int(p.sum()), int(g.sum()), intersection)


def _metrics(p, g, intersection):
    union = p + g - intersection
    return {"predicted_voxels": p, "gt_voxels": g, "intersection_voxels": intersection,
            "dice": 2 * intersection / (p + g) if p + g else 1.0,
            "iou": intersection / union if union else 1.0}


def screening_segmentation(result, *, stage="final"):
    """Return whole-CT prediction, official consensus GT and voxel metrics.

    stage='screening' is the model's VISTA discovered_mask; 'final' is the
    union of refined, automatically selected nodule masks. Both use XYZ CT
    geometry and include mean-malignancy=3 nodules in the GT display/metrics.
    """
    if stage not in ("final", "screening"):
        raise ValueError("stage must be 'final' or 'screening'")
    prediction = (result.screening_mask if stage == "screening" else
                  mask_union((n.mask for n in result.predictions), result.shape))
    gt = mask_union((n.mask for n in result.ground_truth), result.shape)
    return {"case_id": result.case_id, "stage": stage, "prediction": prediction,
            "ground_truth": gt, "affine": result.affine, "metrics": overlap(prediction, gt)}


def _crop_overlap(first, second):
    p, g = int(first.values.sum()), int(second.values.sum())
    lower = np.maximum(first.origin, second.origin)
    upper = np.minimum(np.asarray(first.origin) + first.values.shape,
                       np.asarray(second.origin) + second.values.shape)
    intersection = 0
    if np.all(upper > lower):
        one = tuple(slice(int(a), int(b)) for a, b in zip(lower-first.origin, upper-first.origin))
        two = tuple(slice(int(a), int(b)) for a, b in zip(lower-second.origin, upper-second.origin))
        intersection = int(np.count_nonzero(first.values[one] & second.values[two]))
    return _metrics(p, g, intersection)


def nodule_segmentation(result):
    """One row per GT plus unmatched prediction, with paired compact XYZ masks.

    Match first maximizes the number of pairs meeting minimum_iou, then total
    IoU. Zero overlap and empty masks never match, even at minimum_iou=0.
    A missing side is None, never a fabricated prediction or label.
    """
    predicted, targets = result.predictions, result.ground_truth
    scores = np.zeros((len(predicted), len(targets)))
    for i, p in enumerate(predicted):
        for j, g in enumerate(targets):
            if p.mask.values.any() and g.mask.values.any():
                scores[i, j] = _crop_overlap(p.mask, g.mask)["iou"]
    matches = {}
    if scores.size:
        valid = (scores > 0) & (scores >= result.minimum_iou)
        bonus = min(scores.shape) + 1
        rows, columns = linear_sum_assignment(-(valid * bonus + scores))
        matches = {int(j): int(i) for i, j in zip(rows, columns) if valid[i, j]}
    pairs = [(matches.get(j), j) for j in range(len(targets))]
    pairs += [(i, None) for i in range(len(predicted)) if i not in matches.values()]
    result_rows = []
    for i, j in pairs:
        p = None if i is None else predicted[i]
        g = None if j is None else targets[j]
        metrics = (_crop_overlap(p.mask, g.mask) if p is not None and g is not None else
                   _metrics(int(p.mask.values.sum()) if p is not None else 0,
                            int(g.mask.values.sum()) if g is not None else 0, 0))
        result_rows.append({
            "query_id": None if p is None else p.query_id,
            "nodule_id": None if g is None else g.nodule_id,
            "status": "matched" if i is not None and j is not None else (
                "missed_gt" if i is None else "unmatched_prediction"),
            "prediction": None if p is None else p.mask,
            "ground_truth": None if g is None else g.mask,
            "prediction_index": i, "gt_index": j, "metrics": metrics,
            "gt_mask_empty": bool(g is not None and not g.mask.values.any()),
        })
    return result_rows


def semantic_features(result):
    """Paired model expected ratings/probabilities and official reader scores."""
    rows = []
    for match in nodule_segmentation(result):
        i, j = match["prediction_index"], match["gt_index"]
        p = None if i is None else result.predictions[i]
        g = None if j is None else result.ground_truth[j]
        row = {key: match[key] for key in ("query_id", "nodule_id", "status")}
        row.update(
            prediction=None if p is None else p.semantic_features,
            prediction_probabilities=None if p is None else p.semantic_probabilities,
            ground_truth=None if g is None else g.means,
            gt_reader_ratings=None if g is None else g.reader_ratings,
            gt_annotation_ids=None if g is None else list(g.annotation_ids),
            gt_reader_histograms=None if g is None else {
                name: (np.bincount(np.asarray(values, dtype=int), minlength=6)[1:6] / len(values)).tolist()
                if values else None for name, values in g.reader_ratings.items()},
        )
        rows.append(row)
    return rows


def malignancy(result):
    """Nodule and scan predicted risk alongside reader-derived official GT.

    Mean reader score <3 -> 0, >3 -> 1; mean==3 has no binary target.
    Scan GT is positive if any definitive nodule is malignant, unknown if
    otherwise any nodule is indeterminate, and negative otherwise.
    """
    rows = []
    for match in nodule_segmentation(result):
        i, j = match["prediction_index"], match["gt_index"]
        p = None if i is None else result.predictions[i]
        g = None if j is None else result.ground_truth[j]
        row = {key: match[key] for key in ("query_id", "nodule_id", "status")}
        row.update(prediction=None if p is None else p.malignancy_probability,
                   ground_truth=None if g is None else g.binary_malignancy,
                   gt_reader_mean=None if g is None else g.means["malignancy"],
                   gt_reader_ratings=None if g is None else g.reader_ratings["malignancy"],
                   gt_indeterminate=None if g is None else g.binary_malignancy is None,
                   object_probability=None if p is None else p.object_probability,
                   ensemble_probabilities=None if p is None else p.ensemble_probabilities)
        rows.append(row)
    return {"case_id": result.case_id,
            "scan": {"prediction": result.scan_probability, "ground_truth": result.scan_target,
                     "gt_valid": result.scan_target is not None}, "nodules": rows}
