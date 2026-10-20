"""Full held-out V5 inference; reuse the unchanged eval_package API."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
import torch

from back_prop.evaluate.eval_package import evaluate_case, load_joint_model
from back_prop.evaluate.eval_package.annotations import OfficialAnnotations, DEFAULT_MANIFEST
from back_prop.evaluate.eval_package.comparisons import _crop_overlap
from back_prop.evaluate.eval_package.types import MaskCrop, compact_mask, mask_union, SEMANTIC_NAMES
from .annotations import build_cohort, crop_annotation, sha256, bank_fingerprint, infer_annotation
from back_prop.common.base_data import _read_official_mask
from back_prop.common.features import RADIOMICS_NAMES
from .metrics import diameter_mm, match_predictions, scan_probability

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT = Path('/usr/project/rudinlab/datasets/LIDC_IDRI/joint_model/model_v5_runs/12877415/epoch_018.pt')
DEFAULT_OUTPUT = Path(__file__).parent/'results_12877415_epoch18'


def storage_root(output):
    return Path(output)/'_data'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def reader_composites(readers):
    crops = [r['mask'] for r in readers]
    nonempty = [c for c in crops if c.values.any()]
    if not nonempty:
        empty = MaskCrop(np.zeros((0, 0, 0), bool), (0, 0, 0))
        return empty, empty
    lower = np.min([c.origin for c in nonempty], axis=0)
    upper = np.max([np.array(c.origin)+c.values.shape for c in nonempty], axis=0)
    shape = upper-lower
    arrays = [c.on_grid(shape, lower).astype(bool) for c in crops]
    return MaskCrop(np.logical_or.reduce(arrays), lower), MaskCrop(np.logical_and.reduce(arrays), lower)


def prediction_row(prediction, output, index, affine):
    return dict(query_id=prediction.query_id, object_probability=prediction.object_probability,
        malignancy_probability=prediction.malignancy_probability,
        ensemble_probabilities=prediction.ensemble_probabilities,
        semantic_features=[prediction.semantic_features[n] for n in SEMANTIC_NAMES],
        semantic_probabilities=[prediction.semantic_probabilities[n] for n in SEMANTIC_NAMES],
        diameter_mm=diameter_mm(prediction.mask.values, affine), mask_voxels=int(prediction.mask.values.sum()),
        origin=list(prediction.mask.origin),
        radiomics_features=output.radiomics_features[index].float().cpu().tolist(),
        radiomics_probability=float(output.radiomics_logits[index].float().sigmoid()),
        semantic_correction_logits=output.semantic_correction_logits[index].float().cpu().tolist(),
        box_normalized_cxcyczwhd=output.boxes[prediction.query_id].float().cpu().tolist())


def analyze_case(model, item, official, output_dir, fingerprint, *, manifest=DEFAULT_MANIFEST,
                 verify_default=False):
    started = time.monotonic()
    captured = []
    handle = model.register_forward_hook(lambda module, inputs, output: captured.append(output))
    try:
        # Lowering objectness gives a full post-NMS FROC candidate set. GT is
        # attached only after the unchanged evaluator has finished forwarding.
        result = evaluate_case(model, item['case'], annotations=official, object_threshold=0.)
    finally:
        handle.remove()
    raw = captured.pop()
    default_threshold = float(model.inference_object_threshold)
    selected = [i for i, p in enumerate(result.predictions) if p.object_probability >= default_threshold]
    actual_queries = model.select_detection_queries(raw.object_logits, raw.coarse_mask_logits)
    assert set(actual_queries.tolist()) == {result.predictions[i].query_id for i in selected}
    predictions = [prediction_row(p, raw, i, result.affine) for i, p in enumerate(result.predictions)]
    default_predictions = [predictions[i] for i in selected]
    risk = scan_probability(default_predictions)
    result_default = replace(result, predictions=[result.predictions[i] for i in selected], scan_probability=risk,
        metadata=dict(result.metadata, object_threshold=default_threshold, manifest=str(manifest),
                      split='testing', derived_from_full_candidate_forward=True))
    equivalence = None
    if verify_default:
        direct = evaluate_case(model, item['case'], annotations=official)
        assert {p.query_id for p in direct.predictions} == {p.query_id for p in result_default.predictions}
        actual = {p.query_id:p for p in direct.predictions}
        for p in result_default.predictions:
            q = actual[p.query_id]
            np.testing.assert_array_equal(p.mask.values, q.mask.values)
            np.testing.assert_allclose(list(p.semantic_features.values()), list(q.semantic_features.values()), rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(p.malignancy_probability, q.malignancy_probability, atol=1e-6)
        np.testing.assert_allclose(risk, direct.scan_probability, atol=2e-6)
        equivalence = dict(direct_risk=direct.scan_probability, reconstructed_risk=risk, passed=True)
        del direct
    del raw
    gc.collect()
    source = nib.as_closest_canonical(nib.load(item['case']['image']))
    factors = nib.affines.voxel_sizes(source.affine)
    groups = {}
    annotation_rows = []
    parts = {'V5':dict(semantics=model.semantics, radiomics=model.radiomics, risk=model.rashomon)}
    for entry in item['annotations']:
        native = _read_official_mask(Path(entry['mask_path']), source)
        mask = zoom(native, factors, order=0, mode='nearest', prefilter=False).astype(bool)
        assert mask.shape == result.shape
        crop = crop_annotation(result.image_hu, mask, model.roi_shape)
        row = dict(entry, input_status='ok' if crop is not None else 'empty_mask')
        predicted = infer_annotation(dict(row=row, crop=crop), parts, model.device, 'both')[0]
        annotation_rows.append(predicted)
        compact = compact_mask(mask)
        reader = dict(annotation_id=entry['annotation_id'], mask=compact,
                      ratings=[entry['gt_'+n] for n in SEMANTIC_NAMES], malignancy=entry['gt_malignancy'],
                      diameter_mm=diameter_mm(compact.values, result.affine))
        groups.setdefault(entry['nodule_id'], []).append(reader)
        del native, mask, crop
    nodules = []
    matrix = np.zeros((len(predictions), len(result.ground_truth)))
    pair_details = []
    for j, gt in enumerate(result.ground_truth):
        readers = sorted(groups[gt.nodule_id], key=lambda r:r['annotation_id'])
        union, intersection = reader_composites(readers)
        nodules.append(dict(nodule_id=gt.nodule_id, readers=[{k:v for k,v in r.items() if k!='mask'} for r in readers],
            consensus_ratings=[gt.means[n] for n in SEMANTIC_NAMES], malignancy_mean=gt.means['malignancy'],
            diameter_mm=float(np.mean([r['diameter_mm'] for r in readers])),
            consensus_mask_voxels=int(gt.mask.values.sum()), intersection_voxels=int(intersection.values.sum()),
            union_voxels=int(union.values.sum())))
        pairs = []
        for i, pred in enumerate(result.predictions):
            overlaps = [_crop_overlap(pred.mask, r['mask']) for r in readers]
            closest = max(range(len(readers)), key=lambda k:overlaps[k]['iou'])
            matrix[i, j] = overlaps[closest]['iou']
            pairs.append(dict(closest_annotation_id=readers[closest]['annotation_id'],
                closest_ratings=readers[closest]['ratings'], closest_malignancy=readers[closest]['malignancy'],
                closest=overlaps[closest], union=_crop_overlap(pred.mask, union),
                intersection=_crop_overlap(pred.mask, intersection), consensus=_crop_overlap(pred.mask, gt.mask)))
        pair_details.append(pairs)
    matches, unmatched = match_predictions(matrix, [p['object_probability'] for p in predictions], default_threshold)
    max_risk_index = max(selected, key=lambda i: predictions[i]['malignancy_probability']) if selected else None
    max_event_index = max(selected, key=lambda i: predictions[i]['malignancy_probability']*predictions[i]['object_probability']) if selected else None
    exclusions = []
    for i in unmatched:
        eligible = bool(len(nodules) and np.any(matrix[i] >= .1))
        exclusions.append(dict(prediction_index=i, query_id=predictions[i]['query_id'],
            reason='duplicate_or_competing_match' if eligible else 'empty_mask' if predictions[i]['mask_voxels']==0 else 'no_reader_match',
            predicted_malignant=predictions[i]['malignancy_probability'] >= .5,
            maximum_risk_nodule=i==max_risk_index, maximum_object_weighted_risk=i==max_event_index))
    record = dict(case_id=item['case_id'], patient_id=item['case']['patient_id'], fingerprint=fingerprint,
        object_threshold=default_threshold, minimum_iou=.1, predictions=predictions, nodules=nodules,
        best_reader_iou=matrix.tolist(), pair_details=pair_details,
        matches=[dict(prediction_index=i, nodule_index=j) for i,j in matches], exclusions=exclusions,
        annotations=annotation_rows, scan_probability=risk,
        matched_only_scan_probability=scan_probability([predictions[i] for i,_ in matches]),
        default_forward_equivalence=equivalence, metadata=result_default.metadata,
        seconds=time.monotonic()-started)
    destination = output_dir/'cases'/item['case_id']
    write_json(destination.with_suffix('.json'), record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--max-cases', type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--baseline-plan', type=Path,
                        help='Require completed Sybil and DeepLung fine-tuning before inference')
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        parser.error('Invalid shard')
    if args.baseline_plan:
        from back_prop.baseline_compare.v5_comparison import read_plan, completed
        baseline_plan = read_plan(args.baseline_plan)
        for stage in ('sybil', 'adapt'):
            completed(baseline_plan, stage)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    cohort, census = build_cohort(args.manifest)
    all_cohort = cohort
    cohort = cohort[args.shard::args.shards]
    if args.max_cases is not None:
        cohort = cohort[:args.max_cases]
    model = load_joint_model(args.checkpoint, device=args.device).eval()
    model.requires_grad_(False)
    assert model.evaluation_metadata['architecture'] == 'whole_ct_cnn3d_valid_supervision_v5'
    patients = {c['case']['patient_id'] for c in all_cohort}
    training_patients = {key.rsplit('__scan',1)[0] for key in model.rashomon.allowed_cases}
    assert training_patients and not patients & training_patients
    before = bank_fingerprint(model.rashomon)
    protocol = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha256(args.checkpoint),
        epoch=model.evaluation_metadata['epoch'], cohort=census, schema='v5_joint_analysis_v1',
        diagnostic_mask_temperature=float(model.mask_temperature),
        reader_semantics='individual official reader mask; CT*mask and mask; exact binary input',
        reader_risk='same reader official semantics + radiomics from their mask; frozen risk bank',
        candidate_inference='object threshold 0, checkpoint NMS; operating threshold from checkpoint',
        matching='one-to-one maximum-cardinality, then maximum best-reader mask IoU; IoU >= 0.1',
        consensus_semantics='mean of reader scores; half-up rounding for five-class accuracy only',
        malignancy_modes=['exclude_3','ge_3','gt_3'], biopsy_labels_available=False,
        label_source='official reader ratings, explicitly selected by user',
        endpoint='LIDC reader-rated malignancy; no pathology outcomes in supplied annotation files',
        diameter='maximum 3D Feret diameter of voxel cells; GT averages reader diameters',
        semantic_names=list(SEMANTIC_NAMES), radiomics_names=list(RADIOMICS_NAMES),
        bank_fingerprint=before, held_out_patients=len(patients), training_patient_overlap=0,
        source_hashes={f.name:sha256(f) for f in (
            Path(__file__), Path(__file__).with_name('metrics.py'),
            Path(__file__).with_name('annotations.py'))})
    fingerprint = hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    storage = storage_root(args.output)
    (storage/'cases').mkdir(parents=True, exist_ok=True)
    protocol_path = storage/f'protocol_shard_{args.shard}.json'
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text())['fingerprint']==fingerprint, 'Output protocol changed'
    write_json(protocol_path, dict(protocol, fingerprint=fingerprint,
        expected_cases=[c['case_id'] for c in all_cohort], shard_cases=[c['case_id'] for c in cohort],
        shard=args.shard, shards=args.shards, started_at=datetime.now(timezone.utc).isoformat()))
    official = OfficialAnnotations(cases=[c['case'] for c in all_cohort])
    for index, item in enumerate(cohort):
        destination = storage/'cases'/f"{item['case_id']}.json"
        if destination.exists():
            cached = json.loads(destination.read_text())
            assert cached['fingerprint']==fingerprint
            print(json.dumps(dict(event='cached',case_id=item['case_id'])),flush=True)
            continue
        result = analyze_case(model, item, official, storage, fingerprint, manifest=args.manifest,
                              verify_default=index==0)
        print(json.dumps(dict(event='case_complete',case_id=item['case_id'], shard=args.shard,
            ordinal=index+1, cases=len(cohort), seconds=result['seconds'],
            candidates=len(result['predictions']), matched=len(result['matches']),
            annotations=len(result['annotations']))),flush=True)
        del result
        gc.collect()
    after = bank_fingerprint(model.rashomon)
    assert before==after, 'Risk bank mutated during evaluation'
    write_json(storage/f'complete_shard_{args.shard}.json', dict(fingerprint=fingerprint,
        cases=[c['case_id'] for c in cohort], bank_unchanged=True, completed_at=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    main()
