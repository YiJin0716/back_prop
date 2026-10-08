"""Final V3 whole-CT segmentation and official-mask visualization.

Inference uses the baseline CUDA environment; rendering uses the visual kernel.
No annotation, box, or target is passed to model.forward. The official reference
uses the exact V3 reader-vote consensus, including indeterminate nodules for display.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os

import numpy as np
import nibabel as nib

ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT = Path('/usr/project/rudinlab/datasets/LIDC_IDRI/joint_model/model_v3_runs/12616464/epoch_018.pt')
DEFAULT_CASE = 'LIDC-IDRI-0005__scan16'  # First frozen test case, selected before inference.
DEFAULT_OUTPUT = ROOT / 'back_prop/baseline_compare/plots/v3_segmentation_example'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def paste_union(destination, patch, origin):
    """Paste a possibly padded XYZ ROI without wrapping negative indices."""
    origin = np.asarray(origin, dtype=int)
    lower = np.maximum(origin, 0)
    upper = np.minimum(origin + np.asarray(patch.shape), destination.shape)
    if np.any(upper <= lower):
        return
    dst = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))
    src = tuple(slice(int(a), int(b)) for a, b in zip(lower-origin, upper-origin))
    destination[dst] |= patch[src].astype(bool)


def overlap_metrics(prediction, reference, valid=None):
    if valid is not None:
        prediction, reference = prediction[valid], reference[valid]
    p, g = int(prediction.sum()), int(reference.sum())
    intersection = int(np.count_nonzero(prediction & reference))
    return dict(predicted_voxels=p, official_voxels=g, intersection=intersection,
                dice=2*intersection/(p+g) if p+g else 1.,
                iou=intersection/(p+g-intersection) if p+g-intersection else 1.)


def infer(case_key=DEFAULT_CASE, checkpoint=CHECKPOINT, output=DEFAULT_OUTPUT):
    import torch
    from back_prop.baseline_compare.cohort import load, HERE
    from back_prop.baseline_compare.evaluate_roc import build_v3, write_json
    from back_prop.common.semantic_data import WholeCTLIDCDataset

    if not torch.cuda.is_available():
        raise RuntimeError('V3 inference needs an allocated CUDA GPU. Cached results can be viewed on CPU.')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    output, checkpoint = Path(output), Path(checkpoint)
    output.mkdir(parents=True, exist_ok=True)
    cohort = load()
    rows = [r for r in cohort['splits']['testing'] if r['key'] == case_key]
    if len(rows) != 1:
        raise ValueError(f'{case_key} is not uniquely present in the frozen test set')
    assert rows[0]['patient_id'] not in {r['patient_id'] for r in cohort['splits']['training']}
    model, config, epoch = build_v3(checkpoint, cohort)
    dataset = WholeCTLIDCDataset(rows, target_mask_shape=tuple(config['screen_shape']))
    batch = dataset[0]
    image, hu = batch['image'], batch['image_hu']
    before = {k: v.detach().cpu().clone() for k, v in model.rashomon.named_buffers()}
    fits, updates = int(model.rashomon.fit_count), model.rashomon.updates
    with torch.inference_mode(), torch.autocast('cuda', dtype=getattr(torch, config['amp_dtype'])):
        result = model(image, image_hu=hu, batch=None, teacher_probability=0., update_bank=False)
    assert result.teacher_forced_count == result.fallback_count == 0
    assert (result.refined_target_indices == -1).all()
    assert fits == int(model.rashomon.fit_count) and updates == model.rashomon.updates
    assert all(torch.equal(v.detach().cpu(), before[k]) for k, v in model.rashomon.named_buffers())
    assert torch.isfinite(result.fine_mask_logits).all()
    shape = tuple(image.shape[-3:])
    prediction = np.zeros(shape, dtype=bool)
    for logits, valid, origin in zip(result.fine_mask_logits, result.roi_valid, result.crop_origins):
        mask = ((logits >= 0) & valid).cpu().numpy()
        paste_union(prediction, mask, origin.cpu().numpy())
    official = np.zeros(shape, dtype=bool)
    ignored = np.zeros(shape, dtype=bool)
    nodules = []
    for prefix, is_ignored in [('', False), ('ignored_', True)]:
        for mask, origin, nodule_id in zip(batch[prefix+'target_mask_crops'],
                                          batch[prefix+'target_mask_origins'],
                                          batch[prefix+'nodule_ids']):
            patch, start = mask.numpy().astype(bool), origin.numpy()
            paste_union(ignored if is_ignored else official, patch, start)
            nodules.append(dict(nodule_id=int(nodule_id), origin=start.tolist(),
                                shape=list(patch.shape), ignored_for_malignancy=is_ignored))
    retained = official.copy()
    official |= ignored
    if not official.any():
        raise ValueError('Selected CT has no nonempty official consensus mask')
    nodules.sort(key=lambda n: n['nodule_id'])
    # Fixed first physical nodule and maximal official cross-section; no success-based selection.
    focus = nodules[0]
    lower = np.maximum(np.array(focus['origin']) - 12, 0)
    upper = np.minimum(np.array(focus['origin']) + focus['shape'] + 12, shape)
    focus_slices = tuple(slice(a, b) for a, b in zip(lower, upper))
    z = int(lower[2] + np.argmax(official[focus_slices].sum(axis=(0, 1))))
    xy = tuple(slice(a, b) for a, b in zip(lower[:2], upper[:2]))
    np.savez_compressed(output/'axial_slice.npz', ct_hu=hu[0].numpy()[xy+(z,)],
                        official=official[xy+(z,)], prediction=prediction[xy+(z,)],
                        z=z, lower=lower, upper=upper)
    # scipy.ndimage.zoom(grid_mode=False) aligns voxel-center endpoints. Retain
    # that exact geometry in exports rather than attaching the original affine.
    source = nib.as_closest_canonical(nib.load(rows[0]['image']))
    transform = np.eye(4)
    transform[:3, :3] = np.diag((np.array(source.shape)-1)/(np.array(shape)-1))
    affine = source.affine @ transform
    for name, array in [('v3_seg', prediction), ('official_seg', official)]:
        nii = nib.Nifti1Image(array.astype(np.uint8), affine)
        nii.header.set_xyzt_units('mm')
        nib.save(nii, output/f'{name}.nii.gz')
    official_hashes = {}
    for n in rows[0]['nodules']:
        for annotation in n['annotation_ids']:
            p = dataset.official_mask_dir/f'{case_key}__ann{annotation}_mask.nii.gz'
            official_hashes[str(p)] = digest(p)
    report = dict(case_key=case_key, split='testing', checkpoint=str(checkpoint),
        checkpoint_sha256=digest(checkpoint), epoch=epoch, image=rows[0]['image'],
        image_sha256=digest(rows[0]['image']), cohort_sha256=digest(HERE/'cohort.json'),
        helper_sha256=digest(__file__), official_annotation_sha256=official_hashes,
        source_sha256={str(p.relative_to(ROOT)): digest(p)
                       for folder in ('common', 'legacy_training')
                       for p in (ROOT/'back_prop'/folder).glob('*.py')},
        shape=list(shape), world_affine=affine.tolist(),
        segmentation='union of final refined masks from automatically selected V3 queries',
        object_threshold=model.inference_object_threshold, mask_threshold=.5,
        predicted_queries=int(len(result.crop_origins)),
        object_probabilities=result.object_logits.sigmoid().cpu().tolist(),
        official_policy=f'per physical nodule: at least min({dataset.min_votes}, reader count) reader votes; union across nodules',
        official_display_includes_indeterminate=True, nodules=nodules,
        focus_nodule_id=focus['nodule_id'], focus_lower=lower.tolist(), focus_upper=upper.tolist(),
        all_nodule_metrics=overlap_metrics(prediction, official),
        retained_nodule_metrics=overlap_metrics(prediction, retained, ~ignored),
        ground_truth_used_for_prediction=False, bank_unchanged=True,
        slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    report['artifact_sha256'] = {name: digest(output/name)
                               for name in ['v3_seg.nii.gz', 'official_seg.nii.gz', 'axial_slice.npz']}
    write_json(output/'metadata.json', report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def load_result(output=DEFAULT_OUTPUT):
    output = Path(output)
    meta = json.loads((output/'metadata.json').read_text())
    for name, sha in meta['artifact_sha256'].items():
        assert digest(output/name) == sha, f'Changed artifact: {name}'
    pred = nib.load(output/'v3_seg.nii.gz')
    gt = nib.load(output/'official_seg.nii.gz')
    assert pred.shape == gt.shape and np.allclose(pred.affine, gt.affine)
    return meta, np.asarray(pred.dataobj).astype(bool), np.asarray(gt.dataobj).astype(bool), pred.affine


def mesh(mask, affine):
    from skimage.measure import marching_cubes
    if not mask.any():
        return None
    # Crop before meshing; pad to close surfaces that touch the array boundary.
    positions = np.where(mask)
    lower = np.array([v.min() for v in positions])
    upper = np.array([v.max()+1 for v in positions])
    crop = mask[tuple(slice(a, b) for a, b in zip(lower, upper))]
    vertices, faces, _, _ = marching_cubes(np.pad(crop, 1).astype(np.float32), .5)
    vertices = nib.affines.apply_affine(affine, vertices + lower - 1)
    return vertices, faces


def render(output=DEFAULT_OUTPUT):
    import plotly.graph_objects as go
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    output = Path(output)
    meta, prediction, official, affine = load_result(output)
    surfaces = [(mesh(official, affine), 'Official consensus', 'blue', .45),
                (mesh(prediction, affine), 'V3 final segmentation', 'crimson', .5)]
    traces = []
    for geometry, name, color, opacity in surfaces:
        if geometry is not None:
            v, f = geometry
            traces.append(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2],
                i=f[:, 0], j=f[:, 1], k=f[:, 2], color=color, opacity=opacity,
                name=name, showlegend=True))
    suffix = ' | V3 predicted an empty mask' if not prediction.any() else ''
    title = f"V3 vs Official — {meta['case_key']} (test, epoch {meta['epoch']})"
    fig = go.Figure(traces)
    fig.update_layout(title=title+suffix,
        scene=dict(xaxis_title='R (mm)', yaxis_title='A (mm)', zaxis_title='S (mm)', aspectmode='data'),
        width=900, height=700, legend=dict(itemsizing='constant'))
    fig.write_html(output/'v3_vs_official_3d.html', include_plotlyjs=True)
    # Static companion for notebook viewers without Plotly support.
    static = plt.figure(figsize=(9, 7))
    ax = static.add_subplot(projection='3d')
    vertices_all = []
    for geometry, name, color, opacity in surfaces:
        if geometry is not None:
            v, f = geometry
            ax.add_collection3d(Poly3DCollection(v[f], facecolor=color, alpha=opacity, edgecolor='none'))
            vertices_all.append(v)
    vertices = np.concatenate(vertices_all)
    lo, hi = vertices.min(0), vertices.max(0)
    for setter, a, b in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), lo, hi):
        setter(a-2, b+2)
    ax.set_box_aspect(np.maximum(hi-lo, 1))
    ax.set(xlabel='R (mm)', ylabel='A (mm)', zlabel='S (mm)')
    ax.legend(handles=[Patch(color=color, label=name, alpha=opacity)
                       for _, name, color, opacity in surfaces], loc='upper left')
    ax.set_title(title+'\n'+suffix.lstrip(' |'))
    static.tight_layout()
    static.savefig(output/'v3_vs_official_3d.png', dpi=170)
    plt.close(static)
    with np.load(output/'axial_slice.npz') as data:
        hu, gt, pred = data['ct_hu'], data['official'], data['prediction']
    panels, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, label, masks in zip(axes, ['CT', 'Official consensus', 'V3 + Official'],
                               [[], [(gt, 'blue')], [(gt, 'blue'), (pred, 'crimson')]]):
        ax.imshow(hu.T, cmap='gray', vmin=-1000, vmax=400, origin='lower')
        for mask, color in masks:
            if mask.any():
                # Pad contours so lesions touching a crop boundary still have edges.
                ax.contour(np.pad(mask.T, 1), levels=[.5], colors=[color],
                           linewidths=1.3, extent=(-1, mask.shape[0], -1, mask.shape[1]))
        ax.set_title(label)
        ax.axis('off')
    panels.suptitle(f"{meta['case_key']} | fixed first official nodule {meta['focus_nodule_id']} | blue=official, red=V3")
    panels.tight_layout()
    panels.savefig(output/'v3_vs_official_axial.png', dpi=170)
    plt.close(panels)
    return fig, meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-key', default=DEFAULT_CASE)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    if args.render_only:
        render(args.output)
    else:
        infer(args.case_key, args.checkpoint, args.output)
