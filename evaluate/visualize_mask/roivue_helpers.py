"""Geometry-preserving adapters for v4_segmentation_roivue.ipynb."""
from __future__ import annotations

import numpy as np
from roivue import Case

from back_prop.evaluate.eval_package.comparisons import screening_segmentation

PREDICTION = "V4 prediction"
REFERENCE = "Official consensus"
COLORS = {PREDICTION: "red", REFERENCE: "blue"}


def whole_scan_case(result, *, window=(-1000, 400)):
    """Use the saved inference grid for both CT and masks, without reorientation."""
    comparison = screening_segmentation(result, stage="final")
    case = Case.from_arrays(
        result.image_hu, affine=result.affine,
        masks={PREDICTION: comparison["prediction"], REFERENCE: comparison["ground_truth"]},
        # Whole-scan navigation only. Nodule pairs have their own independent
        # Cases below, so overlapping queries cannot truncate one another.
        regions=np.zeros(result.shape, dtype=np.uint8),
        display={"window": window, "colors": COLORS},
        meta={"id": result.case_id, "checkpoint": result.metadata.get("checkpoint"),
              "stage": "final refined masks", "dice": comparison["metrics"]["dice"],
              "iou": comparison["metrics"]["iou"],
              "consensus_min_votes": result.metadata.get("consensus_min_votes", 2)},
    )
    return case, comparison["metrics"]


def _support_bounds(crop, shape):
    """Nonempty support clipped to the actual CT; padded ROI voxels stay out."""
    if crop is None:
        return None
    origin = np.asarray(crop.origin)
    lower = np.maximum(origin, 0)
    upper = np.minimum(origin + crop.values.shape, shape)
    if np.any(upper <= lower):
        return None
    slices = tuple(slice(int(a), int(b)) for a, b in zip(lower-origin, upper-origin))
    points = np.nonzero(crop.values[slices])
    if not len(points[0]):
        return None
    return (lower + [p.min() for p in points], lower + [p.max()+1 for p in points])


def nodule_case(result, row, *, window=(-1000, 400), min_mm=60., padding_mm=8.):
    """Crop a matched pair, missed GT or unmatched prediction in world coordinates.

    The region is the UNION of both masks. roivue clips region overlays to
    its region label, so using GT alone would incorrectly hide oversegmentation.
    Returns None for an empty pair; such rows remain in the notebook's table.
    """
    if min_mm <= 0 or padding_mm < 0:
        raise ValueError("Require min_mm > 0 and padding_mm >= 0")
    crops = (row["prediction"], row["ground_truth"])
    bounds = [b for crop in crops if (b := _support_bounds(crop, result.shape)) is not None]
    if not bounds:
        return None
    lower = np.min([b[0] for b in bounds], axis=0)
    upper = np.max([b[1] for b in bounds], axis=0)
    spacing = np.linalg.norm(result.affine[:3, :3], axis=0)
    padding = np.ceil(padding_mm / spacing).astype(int)
    extent = np.maximum(upper-lower + 2*padding, np.ceil(min_mm/spacing).astype(int))
    start = np.floor((lower + upper - extent) / 2).astype(int)
    # Shift a crop that reaches the CT boundary instead of unnecessarily
    # throwing away available context on its opposite side.
    start = np.maximum(0, np.minimum(start, np.maximum(np.asarray(result.shape)-extent, 0)))
    stop = np.minimum(start + extent, result.shape)
    slices = tuple(slice(int(a), int(b)) for a, b in zip(start, stop))
    shape = tuple((stop-start).tolist())
    masks = {name: np.zeros(shape, dtype=bool) if crop is None else
             crop.on_grid(shape, tuple(start)).astype(bool)
             for name, crop in zip((PREDICTION, REFERENCE), crops)}
    affine = result.affine.copy()
    affine[:3, 3] = (result.affine @ np.r_[start, 1])[:3]
    union = masks[PREDICTION] | masks[REFERENCE]
    # Empty layers are omitted from roivue, and the missing side is explicit
    # in the selection label, table and metadata. No dummy ROI is created.
    present = {name: mask for name, mask in masks.items() if mask.any()}
    return Case.from_arrays(
        result.image_hu[slices], affine=affine, masks=present,
        regions=union.astype(np.uint8),
        display={"window": window, "colors": {name: COLORS[name] for name in present}},
        meta={"id": result.case_id, "nodule_id": row["nodule_id"],
              "query_id": row["query_id"], "status": row["status"],
              "dice": row["metrics"]["dice"], "iou": row["metrics"]["iou"],
              "crop_origin_xyz": start.tolist(),
              "absent_masks": [name for name in masks if name not in present]},
    )


def selection_label(index, row):
    gt = "none" if row["nodule_id"] is None else str(row["nodule_id"])
    pred = "none" if row["query_id"] is None else str(row["query_id"])
    return (f'{index}: {row["status"]} | GT {gt} / query {pred} | '
            f'Dice {row["metrics"]["dice"]:.3f}')


def close_widgets(widget):
    """Release previous WebGL viewers when a selection/cell is replaced."""
    if widget is None:
        return
    seen = set()

    def close(item):
        if id(item) in seen or not hasattr(item, "close"):
            return
        seen.add(id(item))
        for field in ("children", "volumes", "meshes"):
            for child in getattr(item, field, ()):
                close(child)
        item.close()

    close(widget)
