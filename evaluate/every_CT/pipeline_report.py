"""Export a complete CT / nodule / semantic / risk comparison as offline HTML.

Run in the visual environment after saving a CaseEvaluation with eval_package.
The renderer reuses roivue's scene serializer and HTML viewer (private APIs in
the local roivue checkout); no GPU or Python server is needed to open the HTML.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

from roivue import Case, export, view
from back_prop.evaluate.eval_package.io import load_result
from back_prop.evaluate.eval_package.comparisons import _crop_overlap
from back_prop.evaluate.eval_package.types import compact_mask, mask_union, SEMANTIC_NAMES
from back_prop.evaluate.v5_joint_analysis.metrics import match_predictions, malignancy_label, volume_label
from back_prop.evaluate.visualize_mask.roivue_helpers import _support_bounds

PREDICTED = "Predicted mask"
CONSENSUS = "Official consensus"
UNION = "Official reader union"
SCREENING = "Screening mask"


def load_reader_masks(result):
    """Apply the same canonical XYZ / endpoint-aligned zoom as evaluation."""
    source = nib.as_closest_canonical(nib.load(result.metadata['image']))
    factors = nib.affines.voxel_sizes(source.affine)
    expected_shape = tuple(np.round(np.asarray(source.shape) * factors).astype(int))
    ratio = np.divide(np.asarray(source.shape)-1, np.asarray(expected_shape)-1,
                      out=np.ones(3), where=np.asarray(expected_shape) > 1)
    if result.shape != expected_shape or not np.allclose(
            result.affine, source.affine @ np.diag([*ratio, 1.])):
        raise ValueError('Saved CT grid differs from official-mask resampling grid')
    masks = {}
    for gt in result.ground_truth:
        if not gt.annotation_ids or len(gt.mask_paths) != len(gt.annotation_ids):
            raise ValueError(f'Missing reader masks for nodule {gt.nodule_id}')
        readers = []
        for path in gt.mask_paths:
            image = nib.as_closest_canonical(nib.load(path))
            if image.shape != source.shape or not np.allclose(image.affine, source.affine):
                raise ValueError(f'Official mask / CT geometry mismatch: {path}')
            native = np.asarray(image.dataobj) > 0
            readers.append(compact_mask(zoom(native.astype(np.uint8), factors, order=0,
                                             mode='nearest', prefilter=False).astype(bool)))
        masks[gt.nodule_id] = readers
    return masks


def comparison_rows(result, reader_masks):
    """Use the V5 evaluation protocol: one-to-one best-reader IoU >= 0.1.

    Missing predictions / labels remain null; a missed nodule never receives a
    fabricated semantic score or a zero risk. Reader IDs are annotation IDs.
    """
    matrix = np.zeros((len(result.predictions), len(result.ground_truth)))
    overlaps = {}
    for i, prediction in enumerate(result.predictions):
        for j, gt in enumerate(result.ground_truth):
            values = [_crop_overlap(prediction.mask, mask) for mask in reader_masks[gt.nodule_id]]
            if not values:
                raise ValueError('Best-reader matching requires individual masks')
            closest = max(range(len(values)), key=lambda k: values[k]['iou'])
            matrix[i, j] = values[closest]['iou']
            overlaps[i, j] = (closest, values[closest])
    threshold = result.metadata.get('object_threshold', .5)
    matches, extra = match_predictions(matrix, [p.object_probability for p in result.predictions],
                                       threshold, result.minimum_iou)
    by_gt = {j: i for i, j in matches}
    pairs = [(by_gt.get(j), j) for j in range(len(result.ground_truth))]
    pairs += [(i, None) for i in extra]
    rows = []
    for index, (i, j) in enumerate(pairs):
        p = None if i is None else result.predictions[i]
        g = None if j is None else result.ground_truth[j]
        readers = [] if g is None else [dict(
            annotation_id=ann, ratings={n: g.reader_ratings[n][k] for n in SEMANTIC_NAMES},
            malignancy=g.reader_ratings['malignancy'][k]) for k, ann in enumerate(g.annotation_ids)]
        matched = p is not None and g is not None
        closest, metrics = overlaps[i, j] if matched else (None, None)
        rows.append(dict(
            index=index, id=f'r{index}', prediction_index=i, gt_index=j,
            query_id=None if p is None else p.query_id,
            nodule_id=None if g is None else g.nodule_id,
            status='Matched' if matched else 'Missed nodule' if g is not None else 'Unmatched prediction',
            confidence=None if p is None else p.object_probability,
            semantic=None if p is None else p.semantic_features,
            semantic_probabilities=None if p is None else p.semantic_probabilities,
            risk=None if p is None else p.malignancy_probability,
            official_mean=None if g is None else g.means,
            official_labels=None if g is None else {
                mode: malignancy_label(g.means['malignancy'], mode)
                for mode in ('exclude_3', 'ge_3', 'gt_3')},
            readers=readers,
            closest_annotation_id=None if closest is None else g.annotation_ids[closest],
            overlap_closest=metrics,
            overlap_consensus=_crop_overlap(p.mask, g.mask) if matched else None,
        ))
    return rows


def region_case(result, row, reader_masks, min_mm=60.):
    p = None if row['prediction_index'] is None else result.predictions[row['prediction_index']]
    g = None if row['gt_index'] is None else result.ground_truth[row['gt_index']]
    crops = {}
    colors = {PREDICTED: 'red', CONSENSUS: 'blue'}
    if p is not None:
        crops[PREDICTED] = p.mask
    if g is not None:
        crops[CONSENSUS] = g.mask
        for k, (ann, mask) in enumerate(zip(g.annotation_ids, reader_masks[g.nodule_id])):
            name = f'Official annotation {ann}'
            crops[name] = mask
            colors[name] = ('green', 'violet', 'cool', 'warm')[k % 4]
    bounds = [b for crop in crops.values() if (b := _support_bounds(crop, result.shape)) is not None]
    if not bounds:
        return None
    lower = np.min([b[0] for b in bounds], axis=0)
    upper = np.max([b[1] for b in bounds], axis=0)
    spacing = np.linalg.norm(result.affine[:3, :3], axis=0)
    extent = np.maximum(upper-lower+2*np.ceil(8/spacing).astype(int), np.ceil(min_mm/spacing).astype(int))
    start = np.floor((lower+upper-extent)/2).astype(int)
    start = np.maximum(0, np.minimum(start, np.maximum(np.asarray(result.shape)-extent, 0)))
    stop = np.minimum(start+extent, result.shape)
    slices = tuple(slice(int(a), int(b)) for a, b in zip(start, stop))
    masks = {name: crop.on_grid(stop-start, start).astype(bool) for name, crop in crops.items()}
    masks = {name: mask for name, mask in masks.items() if mask.any()}
    affine = result.affine.copy()
    affine[:3, 3] = (result.affine @ np.r_[start, 1])[:3]
    # Include every reader AND the prediction: otherwise roivue clips away
    # disagreement / oversegmentation outside the supplied region label.
    support = np.logical_or.reduce(list(masks.values()))
    return Case.from_arrays(result.image_hu[slices], affine=affine, masks=masks,
        regions=support.astype(np.uint8), display=dict(window=(-1000, 400),
        colors={name: colors[name] for name in masks},
        hidden=[name for name in masks if name.startswith('Official annotation')]))


def write_report(result, output, *, model_label='V5', scan_step=2, reader_masks=None):
    reader_masks = load_reader_masks(result) if reader_masks is None else reader_masks
    rows = comparison_rows(result, reader_masks)
    selected = [p for p in result.predictions if p.object_probability >= result.metadata.get('object_threshold', .5)]
    whole = Case.from_arrays(result.image_hu, affine=result.affine,
        masks={PREDICTED: mask_union([p.mask for p in selected], result.shape),
               UNION: mask_union([m for masks in reader_masks.values() for m in masks], result.shape),
               SCREENING: result.screening_mask.astype(bool)},
        regions=np.zeros(result.shape, np.uint8),
        display=dict(window=(-1000, 400), colors={PREDICTED: 'red', UNION: 'blue', SCREENING: 'green'},
                     hidden=[SCREENING]))
    overview = view.load(whole)
    scan_scene = overview.scan_scene(scan_step)
    scenes = {'scan': export._scene_json(scan_scene)}
    for row in rows:
        case = region_case(result, row, reader_masks)
        row['has_scene'] = case is not None
        if case is None:
            continue
        scene = view.load(case).region_scene(0)
        scene.index = row['index']
        scene.title = row['status'] + (f" · nodule {row['nodule_id']}" if row['nodule_id'] is not None
                                       else f" · query {row['query_id']}")
        scenes[row['id']] = export._scene_json(scene)
        row['center_mm'] = scenes[row['id']]['center_mm']
    title = f'{model_label} · {result.case_id}'
    data = dict(title=title, meta={}, summary={}, rows=rows, scenes=scenes,
        tiles=[list(t) for t in view.TILES], css=view.PALETTE,
        radiological=overview.radiological, opacity=.5, gap=1.,
        report=dict(case_id=result.case_id, model=model_label, scan_risk=result.scan_probability,
            official_scan_labels={mode: volume_label([
                malignancy_label(rating, mode) for gt in result.ground_truth
                for rating in gt.reader_ratings['malignancy']])
                for mode in ('exclude_3', 'ge_3', 'gt_3')},
            checkpoint=result.metadata.get('checkpoint'), epoch=result.metadata.get('epoch'),
            exporter_sha256=result.metadata.get('exporter_sha256'),
            object_threshold=result.metadata.get('object_threshold', .5),
            mask_threshold=result.metadata.get('mask_threshold', .5), minimum_iou=result.minimum_iou,
            mask_temperature=result.metadata.get('diagnostic_mask_temperature', 1.),
            consensus_votes=result.metadata.get('consensus_min_votes', 2),
            scan_step=scan_scene.step, semantic_names=list(SEMANTIC_NAMES),
            predicted=len(selected), official=len(result.ground_truth)))
    output = Path(output).with_suffix('.html')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.html.tmp')
    try:
        temporary.write_text(render_html(data), encoding='utf-8')
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output, data


def render_html(data):
    """Render saved scene data with the current layout, without new inference."""
    assets = Path(__file__).with_name('pipeline_report_assets')
    template = export._TEMPLATE
    replacements = {
        '</style>': '\n' + (assets/'report.css').read_text() + '\n</style>',
        '<div id="main">': '<div id="main">' + (assets/'header.html').read_text() + '<div id="workspace"><section id="viewer">',
        '</div>\n<script>__NIIVUE__</script>': '</section><aside id="report-panel"></aside></div></div>\n<script>__NIIVUE__</script>',
        'current = id;': 'current = id; renderReport(id);',
        'menu(scene, [scanNV], null, null);':
            'applyLayers(scene.layers.map(L => L.on), [scanNV], null);\n  menu(scene, [scanNV], null, null);',
        '(async () => {\n  navigation();': (assets/'report.js').read_text() + '\n(async () => {\n  reportNavigation();',
    }
    for old, new in replacements.items():
        if template.count(old) != 1:
            raise RuntimeError('Local roivue HTML API changed: ' + old[:60])
        template = template.replace(old, new)
    return (template.replace('__TITLE__', html.escape(data['title']))
            .replace('<script>__NIIVUE__</script>', export._niivue_tag(None))
            .replace('__DATA__', json.dumps(data, default=export._plain, allow_nan=False).replace('</', '<\\/')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-label', default='V5')
    parser.add_argument('--scan-step', type=int, default=2)
    args = parser.parse_args()
    if args.scan_step < 1:
        parser.error('--scan-step must be positive')
    output, data = write_report(load_result(args.result_dir), args.output,
                               model_label=args.model_label, scan_step=args.scan_step)
    print(json.dumps(dict(output=str(output), size_mb=round(output.stat().st_size/1e6, 2),
                         nodules=[{k:r[k] for k in ('query_id', 'nodule_id', 'status')}
                                  for r in data['rows']]), indent=2))


if __name__ == '__main__':
    main()
