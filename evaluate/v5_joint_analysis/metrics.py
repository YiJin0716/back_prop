"""Explicit denominators, one-to-one detection matching and frozen thresholds."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull, distance
from sklearn.metrics import roc_auc_score, roc_curve

MODES = ('exclude_3', 'ge_3', 'gt_3')


def malignancy_label(rating, mode):
    if rating is None:
        return None
    if mode == 'exclude_3':
        return None if abs(rating-3.) < 1e-8 else int(rating > 3)
    if mode == 'ge_3':
        return int(rating >= 3)
    if mode == 'gt_3':
        return int(rating > 3)
    raise ValueError(mode)


def volume_label(labels):
    """Any known positive wins; otherwise an unknown nodule makes scan unknown."""
    return 1 if 1 in labels else None if None in labels else 0


def binary_metrics(labels, scores, threshold=.5):
    y = np.asarray([np.nan if v is None else v for v in labels], float)
    p = np.asarray([np.nan if v is None else v for v in scores], float)
    if y.shape != p.shape:
        raise ValueError('Labels and scores differ in length')
    keep = np.isfinite(y) & np.isfinite(p)
    y, p = y[keep], p[keep]
    if not np.isin(y, [0, 1]).all() or not ((-1e-6 <= p) & (p <= 1+1e-6)).all():
        raise ValueError('Expected binary labels and probabilities')
    y, p = y.astype(int), np.clip(p, 0, 1)
    pred = p >= threshold
    tp, tn = int(((y == 1) & pred).sum()), int(((y == 0) & ~pred).sum())
    fp, fn = int(((y == 0) & pred).sum()), int(((y == 1) & ~pred).sum())
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None
    return dict(n_total=len(labels), n=len(y), excluded=int((~keep).sum()),
                positives=int((y == 1).sum()), negatives=int((y == 0).sum()),
                auc=auc, accuracy=(tp+tn)/len(y) if len(y) else None,
                sensitivity=tp/(tp+fn) if tp+fn else None,
                specificity=tn/(tn+fp) if tn+fp else None,
                tp=tp, tn=tn, fp=fp, fn=fn, probability_threshold=threshold)


def semantic_metrics(ratings, probability):
    """Six attributes; consensus means remain continuous for threshold labels."""
    y = np.asarray(ratings, float).reshape(-1, 6)
    p = np.asarray(probability, float).reshape(-1, 6, 5)
    if len(y) != len(p) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Invalid semantic evaluation rows')
    if len(p) and (not np.allclose(p.sum(-1), 1, atol=1e-5) or (p < 0).any()):
        raise ValueError('Invalid class probabilities')
    result = []
    for i in range(6):
        expected = p[:, i] @ np.arange(1, 6)
        truth_class = np.floor(y[:, i]+.5).astype(int)
        thresholds = []
        for t in range(1, 6):
            thresholds.append(dict(threshold=t, **binary_metrics(
                (y[:, i] >= t).astype(int), p[:, i, t-1:].sum(-1))))
        result.append(dict(n=len(y), accuracy=float((p[:, i].argmax(-1)+1 == truth_class).mean()) if len(y) else None,
            mae=float(np.abs(expected-y[:, i]).mean()) if len(y) else None,
            correlation=float(np.corrcoef(expected, y[:, i])[0, 1]) if len(y)>1 and expected.std()>0 and y[:, i].std()>0 else None,
            thresholds=thresholds))
    return result


def match_predictions(iou, scores, threshold=.5, minimum_iou=.1):
    """Maximize valid matches, then total best-reader IoU; duplicates are FP."""
    iou = np.asarray(iou, float)
    scores = np.asarray(scores, float)
    if iou.ndim != 2 or len(iou) != len(scores):
        raise ValueError('Expected prediction x physical-nodule IoU matrix')
    selected = np.flatnonzero(scores >= threshold)
    pairs = []
    if len(selected) and iou.shape[1]:
        sub = iou[selected]
        valid = (sub > 0) & (sub >= minimum_iou)
        a, b = linear_sum_assignment(-(valid*(min(sub.shape)+1)+sub))
        pairs = [(int(selected[i]), int(j)) for i, j in zip(a, b) if valid[i, j]]
    return pairs, [int(i) for i in selected if i not in {a for a, _ in pairs}]


def froc(cases, minimum_iou=.1):
    """Empirical curve over all post-NMS objectness thresholds; no extrapolation."""
    confidence = sorted({float(p['object_probability']) for c in cases for p in c['predictions']}, reverse=True)
    thresholds = [1.000001] + confidence
    nodules = sum(len(c['nodules']) for c in cases)
    points = []
    for threshold in thresholds:
        tp = fp = 0
        for c in cases:
            scores = [p['object_probability'] for p in c['predictions']]
            iou = np.asarray(c['best_reader_iou']).reshape(len(scores), len(c['nodules']))
            pairs, unmatched = match_predictions(iou, scores, threshold, minimum_iou)
            tp += len(pairs); fp += len(unmatched)
        points.append(dict(threshold=threshold, true_positives=tp, false_positives=fp,
                           sensitivity=tp/nodules if nodules else None,
                           false_positives_per_scan=fp/len(cases)))
    operating = []
    for budget in (.125, .25, .5, 1, 2, 4, 8):
        feasible = [p for p in points if p['false_positives_per_scan'] <= budget]
        best = max(feasible, key=lambda p: p['sensitivity'])
        operating.append(dict(fp_budget=budget, sensitivity=best['sensitivity'],
                              achieved_fp_per_scan=best['false_positives_per_scan'], threshold=best['threshold']))
    return dict(scans=len(cases), nodules=nodules, points=points, operating_points=operating,
                mean_sensitivity=float(np.mean([p['sensitivity'] for p in operating])))


def scan_probability(predictions):
    """Match model's per-ensemble object-weighted noisy-OR, then average."""
    if not predictions:
        return 1e-6
    p = np.asarray([row['ensemble_probabilities'] for row in predictions], float)
    obj = np.asarray([row['object_probability'] for row in predictions], float)
    event = np.minimum(p*obj[:, None], 1-1e-6)
    per_model = np.clip(-np.expm1(np.log1p(-event).sum(0)), 1e-6, 1-1e-6)
    return float(per_model.mean())


def diameter_mm(mask, affine):
    """Maximum 3D Feret diameter of occupied voxel cells, in physical mm."""
    from scipy.ndimage import binary_erosion
    mask = np.asarray(mask, bool)
    if not mask.any():
        return 0.
    xyz = np.argwhere(mask & ~binary_erosion(mask))
    if len(xyz) >= 4 and np.linalg.matrix_rank(xyz-xyz[0]) == 3:
        xyz = xyz[ConvexHull(xyz).vertices]
    offsets = np.asarray(np.meshgrid(*[[-.5, .5]]*3)).reshape(3, -1).T
    xyz = (xyz[:, None]+offsets).reshape(-1, 3) @ np.asarray(affine)[:3, :3].T
    xyz = xyz[ConvexHull(xyz).vertices]
    return float(distance.pdist(xyz).max())
