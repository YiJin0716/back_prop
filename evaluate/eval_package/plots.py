"""Paired Matplotlib and interactive Plotly views of predictions and official GT."""
from __future__ import annotations

import numpy as np

from .comparisons import malignancy, nodule_segmentation, screening_segmentation, semantic_features
from .types import SEMANTIC_NAMES


def _save(fig, save_path):
    if save_path is not None:
        from pathlib import Path
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig


def _slice_plot(image, prediction, gt, *, axis, index, title, window, save_path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0 (sagittal), 1 (coronal), or 2 (axial)")
    if index is None:
        reference = gt if gt.any() else prediction
        areas = reference.sum(axis=tuple(a for a in range(3) if a != axis))
        index = int(areas.argmax()) if areas.any() else image.shape[axis] // 2
    if not 0 <= index < image.shape[axis]:
        raise ValueError(f"Slice {index} is outside axis {axis}")
    ct, p, g = (np.take(a, index, axis=axis).T for a in (image, prediction, gt))
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, label, masks in zip(axes, ("CT (HU)", "Prediction", "Official GT", "Overlay"),
                                ((), ((p, "#ee5533"),), ((g, "#22bbee"),),
                                 ((g, "#22bbee"), (p, "#ee5533")))):
        ax.imshow(ct, cmap="gray", vmin=window[0], vmax=window[1], origin="lower")
        for mask, color in masks:
            if mask.any():
                ax.contour(np.arange(-1, mask.shape[1]+1), np.arange(-1, mask.shape[0]+1),
                           np.pad(mask, 1), levels=[0.5], colors=[color], linewidths=1.1)
        ax.set_title(label)
        ax.set_xlim(-0.5, ct.shape[1]-0.5)
        ax.set_ylim(-0.5, ct.shape[0]-0.5)
        ax.set_axis_off()
    axes[-1].legend(handles=[Patch(color="#ee5533", label="Prediction"),
                            Patch(color="#22bbee", label="Official GT")], fontsize=8)
    fig.suptitle(f"{title} | XYZ axis={axis}, slice={index}")
    fig.tight_layout()
    return _save(fig, save_path)


def plot_screening_segmentation(result, *, stage="final", axis=2, slice_index=None,
                                window=(-1000, 400), save_path=None):
    """Whole-screening CT, prediction, official consensus, and overlay."""
    view = screening_segmentation(result, stage=stage)
    return _slice_plot(result.image_hu, view["prediction"], view["ground_truth"],
                       axis=axis, index=slice_index, window=window, save_path=save_path,
                       title=f"{result.case_id} | {stage} | Dice={view['metrics']['dice']:.3f}")


def plot_screening_segmentation_3d(result, *, stage="final", save_path=None):
    """Rotatable whole-screening prediction/GT surfaces, as in visual.ipynb.

    Extract every foreground component at full resolution. Tight crops and a
    zero border avoid allocating a full float CT and close surfaces touching
    the volume boundary. Apply the complete CT affine to get RAS millimetres.
    Empty masks get an explicit legend entry and annotation, without a mesh.
    Returns a Plotly Figure; save_path optionally writes standalone HTML.
    """
    from pathlib import Path
    import plotly.graph_objects as go
    from skimage.measure import marching_cubes

    path = None if save_path is None else Path(save_path)
    if path is not None and path.suffix.lower() not in (".html", ".htm"):
        raise ValueError("The interactive 3D plot must be saved as .html or .htm")
    view = screening_segmentation(result, stage=stage)
    fig = go.Figure()
    empty = []
    for key, label, color in (("ground_truth", "Official GT", "blue"),
                              ("prediction", "Prediction", "crimson")):
        mask = view[key]
        if not mask.any():
            empty.append(label)
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None], mode="markers",
                marker=dict(color=color, size=8), name=f"{label} (empty)",
                showlegend=True, hoverinfo="skip"))
            continue
        # Axis projections find bounds without storing all positive coordinates.
        bounds = [np.flatnonzero(mask.any(axis=tuple(a for a in range(3) if a != axis)))
                  for axis in range(3)]
        lower = np.array([indices[0] for indices in bounds])
        upper = np.array([indices[-1]+1 for indices in bounds])
        slices = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))
        crop = np.pad(mask[slices], 1).astype(np.float32)
        vertices, faces, _, _ = marching_cubes(crop, level=0.5)
        vertices += lower - 1  # undo the crop offset and the padded border
        vertices = vertices @ result.affine[:3, :3].T + result.affine[:3, 3]
        fig.add_trace(go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color=color, opacity=0.5, name=label, showlegend=True))
    if empty:
        fig.add_annotation(
            text="Empty mask: " + ", ".join(empty), x=0, y=1,
            xref="paper", yref="paper", xanchor="left", showarrow=False)
    fig.update_layout(
        title=(f"3D segmentation — {result.case_id} | {stage}<br>"
               f"Dice={view['metrics']['dice']:.4f} | IoU={view['metrics']['iou']:.4f}"),
        scene=dict(xaxis_title="R (mm)", yaxis_title="A (mm)", zaxis_title="S (mm)",
                   aspectmode="data"),
        width=900, height=700, legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, b=0, t=85),
    )
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(path, include_plotlyjs=True, full_html=True, auto_open=False)
    return fig


def plot_nodule_segmentation(result, *, nodule_id=None, query_id=None, axis=2,
                             slice_index=None, padding=12, window=(-1000, 400), save_path=None):
    """Select a GT nodule ID or predicted query ID, including misses/extra queries.

    slice_index refers to the whole CT grid, not the displayed crop.
    """
    if (nodule_id is None) == (query_id is None):
        raise ValueError("Provide exactly one of nodule_id and query_id")
    if axis not in (0, 1, 2) or padding < 0:
        raise ValueError("Require axis in (0,1,2) and padding >= 0")
    key, value = ("nodule_id", nodule_id) if nodule_id is not None else ("query_id", query_id)
    rows = [row for row in nodule_segmentation(result) if row[key] == value]
    if len(rows) != 1:
        raise ValueError(f"Unknown {key}: {value}")
    row = rows[0]
    crops = [row[k] for k in ("prediction", "ground_truth")
             if row[k] is not None and row[k].values.size]
    if not crops:
        lower, upper = np.zeros(3, dtype=int), np.asarray(result.shape)
    else:
        lower = np.maximum(np.min([c.origin for c in crops], axis=0)-padding, 0)
        upper = np.minimum(np.max([np.asarray(c.origin)+c.values.shape for c in crops], axis=0)+padding,
                           result.shape)
    shape = upper-lower
    masks = [np.zeros(tuple(shape), dtype=bool) if row[k] is None else row[k].on_grid(shape, lower)
             for k in ("prediction", "ground_truth")]
    slices = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))
    title = (f"{result.case_id} | GT={row['nodule_id']} / query={row['query_id']} | {row['status']}"
             f" | crop origin={lower.tolist()} | Dice={row['metrics']['dice']:.3f}")
    return _slice_plot(result.image_hu[slices], *masks, axis=axis,
                       index=None if slice_index is None else slice_index-int(lower[axis]),
                       title=title, window=window, save_path=save_path)


def _label(row):
    return f"GT {row['nodule_id'] if row['nodule_id'] is not None else '-'} / query {row['query_id'] if row['query_id'] is not None else '-'}"


def plot_semantic_features(result, *, save_path=None):
    """Six semantic expected scores vs official reader means, for every match/miss."""
    import matplotlib.pyplot as plt

    rows = semantic_features(result)
    fig, axes = plt.subplots(1, 2, figsize=(14, max(3.5, len(rows)*0.45+1.5)))
    for ax, key, title in zip(axes, ("prediction", "ground_truth"),
                               ("Model expected rating", "Official reader mean")):
        data = np.array([[np.nan if row[key] is None else row[key].get(name, np.nan)
                          for name in SEMANTIC_NAMES] for row in rows], dtype=float).reshape(-1, 6)
        if rows:
            ax.imshow(np.ma.masked_invalid(data), vmin=1, vmax=5, cmap="viridis", aspect="auto")
            for (y, x), value in np.ndenumerate(data):
                ax.text(x, y, "--" if not np.isfinite(value) else f"{value:.2f}",
                        ha="center", va="center", color="black" if not np.isfinite(value) or value > 3 else "white")
        else:
            ax.text(0.5, 0.5, "No predicted or GT nodules", transform=ax.transAxes, ha="center")
        ax.set_xticks(range(6), SEMANTIC_NAMES, rotation=30, ha="right")
        ax.set_yticks(range(len(rows)), [_label(row) for row in rows])
        ax.set_title(title + " (1–5)")
    fig.suptitle(f"{result.case_id} | Semantic features | -- = missing counterpart")
    fig.tight_layout()
    return _save(fig, save_path)


def plot_malignancy(result, *, save_path=None):
    """Predicted probabilities vs binary GT; show original reader mean in labels."""
    import matplotlib.pyplot as plt

    view = malignancy(result)
    rows = [view["scan"]] + view["nodules"]
    labels = ["Whole scan"] + [
        _label(row) + (f" (reader mean={row['gt_reader_mean']:.2f})"
                      if row["gt_reader_mean"] is not None else " (no GT)")
        for row in view["nodules"]]
    fig, ax = plt.subplots(figsize=(11, max(3, len(rows)*0.55+1)))
    positions = np.arange(len(rows))
    for offset, key, color, label in ((-0.18, "prediction", "#ee5533", "Predicted probability"),
                                      (0.18, "ground_truth", "#22bbee", "Official binary GT")):
        values = [np.nan if row[key] is None else row[key] for row in rows]
        ax.barh(positions+offset, values, height=0.34, color=color, label=label)
        for y, value in zip(positions+offset, values):
            ax.text(0.02 if np.isnan(value) else value+0.02, y,
                    "unavailable" if np.isnan(value) else f"{value:.3f}", va="center", fontsize=9)
    ax.set_yticks(positions, labels)
    ax.set_xlim(0, 1.2)
    # Missing bars are NaN and do not affect Matplotlib's autoscale bounds.
    # Reserve both bar positions even when the final counterpart is missing.
    ax.set_ylim(len(rows)-0.5, -0.5)
    ax.set_xlabel("Probability / binary target; reader mean = 3 has no binary GT")
    ax.set_title(f"{result.case_id} | LIDC reader-derived malignancy")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return _save(fig, save_path)
