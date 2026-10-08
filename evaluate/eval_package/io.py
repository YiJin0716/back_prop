"""Portable, pickle-free results and optional NIfTI/PNG exports."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .comparisons import malignancy, nodule_segmentation, screening_segmentation, semantic_features
from .types import CaseEvaluation, GroundTruthNodule, MaskCrop, PredictedNodule


def save_result(result, output_dir, *, plots=True, nifti=False):
    """Save all predictions AND GT, plus figures. Return the output directory.

    result.json contains paired per-nodule scores, reader ratings, identities,
    matching and metrics. arrays.npz stores CT, masks, probabilities and affine
    without pickle, and can be reopened with load_result without torch/GPU.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    arrays = {"image_hu": result.image_hu, "affine": result.affine,
              "screening_mask": result.screening_mask}
    predictions, truth = [], []
    for i, p in enumerate(result.predictions):
        arrays[f"prediction_{i}"] = p.mask.values
        arrays[f"probability_{i}"] = p.probability.values
        predictions.append(dict(query_id=p.query_id, origin=list(p.mask.origin),
            object_probability=p.object_probability, semantic_features=p.semantic_features,
            semantic_probabilities=p.semantic_probabilities,
            malignancy_probability=p.malignancy_probability, ensemble_probabilities=p.ensemble_probabilities))
    for i, g in enumerate(result.ground_truth):
        arrays[f"gt_{i}"] = g.mask.values
        truth.append(dict(nodule_id=g.nodule_id, origin=list(g.mask.origin),
                          annotation_ids=list(g.annotation_ids), reader_ratings=g.reader_ratings,
                          mask_paths=list(g.mask_paths)))
    metrics = {}
    for stage in ("screening", "final"):
        view = screening_segmentation(result, stage=stage)
        metrics[stage] = view["metrics"]
        if nifti:
            import nibabel as nib
            for name, array in ((f"{stage}_prediction", view["prediction"]),
                                ("official_gt", view["ground_truth"])):
                image = nib.Nifti1Image(array.astype(np.uint8), result.affine)
                image.header.set_xyzt_units("mm")
                nib.save(image, output / f"{name}.nii.gz")
    report = dict(format_version=1, case_id=result.case_id, metadata=result.metadata,
        minimum_iou=result.minimum_iou, scan_probability=result.scan_probability,
        predictions=predictions, ground_truth=truth, segmentation_metrics=metrics,
        nodule_segmentation=[{k: v for k, v in row.items() if k not in ("prediction", "ground_truth")}
                             for row in nodule_segmentation(result)],
        semantic_features=semantic_features(result), malignancy=malignancy(result))
    # Serialize JSON first, so invalid numeric outputs do not silently become NaN.
    encoded = json.dumps(report, indent=2, allow_nan=False)
    np.savez_compressed(output / "arrays.npz", **arrays)
    (output / "result.json").write_text(encoded + "\n")
    if plots:
        from .plots import (plot_malignancy, plot_nodule_segmentation,
                            plot_screening_segmentation, plot_semantic_features)
        import matplotlib.pyplot as plt
        for stage in ("screening", "final"):
            plt.close(plot_screening_segmentation(result, stage=stage, save_path=output / f"{stage}.png"))
        plt.close(plot_semantic_features(result, save_path=output / "semantic_features.png"))
        plt.close(plot_malignancy(result, save_path=output / "malignancy.png"))
        for row in nodule_segmentation(result):
            selector = ({"nodule_id": row["nodule_id"]} if row["nodule_id"] is not None
                        else {"query_id": row["query_id"]})
            stem = "_".join(f"{k}_{v}" for k, v in selector.items())
            plt.close(plot_nodule_segmentation(result, **selector, save_path=output / f"{stem}.png"))
    return output


def load_result(output_dir):
    """Load paired model/GT results on CPU, without model code or checkpoints."""
    output = Path(output_dir)
    report = json.loads((output / "result.json").read_text())
    if report["format_version"] != 1:
        raise ValueError("Unsupported evaluation result format")
    with np.load(output / "arrays.npz", allow_pickle=False) as arrays:
        predictions = [PredictedNodule(
            p["query_id"], MaskCrop(arrays[f"prediction_{i}"], p["origin"]),
            MaskCrop(arrays[f"probability_{i}"], p["origin"]),
            p["object_probability"], p["semantic_features"], p["semantic_probabilities"],
            p["malignancy_probability"], p["ensemble_probabilities"])
            for i, p in enumerate(report["predictions"])]
        truth = [GroundTruthNodule(
            g["nodule_id"], MaskCrop(arrays[f"gt_{i}"], g["origin"]),
            tuple(g["annotation_ids"]), g["reader_ratings"], tuple(g["mask_paths"]))
            for i, g in enumerate(report["ground_truth"])]
        return CaseEvaluation(report["case_id"], arrays["image_hu"], arrays["affine"],
            arrays["screening_mask"], predictions, truth, report["scan_probability"],
            report["metadata"], report["minimum_iou"])
