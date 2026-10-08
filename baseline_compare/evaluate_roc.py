"""Fixed, paired whole-CT test evaluation of V3 and the three trained baselines.

GT masks are used only by the unchanged V3 loader to determine valid labels.
Prediction receives image tensors alone. No fitting, calibration or selection
uses test data. Per-case results are resumable only under identical provenance.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch

from .cohort import HERE, V3_CONFIG, load, sha256
from .predict import sybil_predict, deeplung_predict, edicnet_predict

MODELS = ('v3', 'sybil', 'deeplung', 'edicnet')
DISPLAY = {'v3': 'V3', 'sybil': 'Sybil', 'deeplung': 'DeepLung', 'edicnet': 'EDICNet'}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def build_v3(path, cohort, smoke=False):
    from back_prop.common.semantic_model import ARCHITECTURE, WholeCTJointModelV3
    saved = torch.load(path, map_location='cpu', weights_only=False)
    config = saved['config']
    assert saved['architecture'] == ARCHITECTURE
    if not smoke:
        assert saved['epoch'] + 1 == config['epochs'] == 18, 'Require final V3 epoch'
        assert config['max_cases'] is None and config['benchmark_steps'] is None
    names = ('window_size', 'overlap', 'screen_shape', 'roi_shape', 'num_queries',
             'detr_hidden_dim', 'detr_coarse_shape', 'detr_nheads', 'detr_encoder_layers',
             'detr_decoder_layers', 'context_channels', 'hard_negatives', 'teacher_full_epochs',
             'teacher_zero_epoch', 'teacher_jitter', 'minimum_roi_coverage',
             'vista_checkpoint', 'medicalnet_checkpoint', 'medicalnet_roi_size')
    bank = dict(sparsity=config['sparsity'], bound=config['coefficient_bound'], beam=config['beam'],
                pool_size=config['pool_size'], gap=config['gap'], fraction=config['sample_fraction'],
                min_samples=config['bank_min_samples'], refresh_every=config['bank_refresh_every'],
                mode=config['bank_mode'])
    model = WholeCTJointModelV3(**{name: config[name] for name in names},
        refiner_base_channels=config['refiner_base'], amp_dtype=getattr(torch, config['amp_dtype']),
        bank_config=bank, use_checkpoint=False)
    model.load_state_dict(saved['model'], strict=True)
    assert model.rashomon.allowed_cases == {r['key'] for r in cohort['splits']['training']}
    assert int(model.rashomon.count) > 0
    assert bank['mode'] != 'best', 'Expected full-bank inference, not best-model selection'
    epoch = saved['epoch'] + 1
    del saved
    return model.cuda().eval(), config, epoch


def baseline_artifacts(cohort_hash):
    result = {}
    for name in MODELS[1:]:
        manifest = json.loads((HERE / name / 'final_artifacts.json').read_text())
        assert manifest['training_and_inference_verified']
        assert manifest['cohort_sha256'] == cohort_hash
        result[name] = {}
        for component, artifact in manifest['artifacts'].items():
            assert sha256(artifact['path']) == artifact['sha256'], (name, component)
            result[name][component] = artifact
    return result


def summarize(records, expected_keys, repetitions=2000):
    """Common evaluation rows; paired bootstrap with patients as sampling units."""
    from sklearn.metrics import roc_auc_score, roc_curve
    assert len(records) == len(expected_keys)
    assert {r['key'] for r in records} == set(expected_keys)
    assert len({r['key'] for r in records}) == len(records)
    included = [r for r in records if r['included']]
    excluded = [r for r in records if not r['included']]
    labels = np.asarray([r['label'] for r in included])
    assert set(labels.tolist()) == {0, 1}, 'ROC needs both classes'
    scores = {name: np.asarray([r['scores'][name] for r in included]) for name in MODELS}
    assert all(np.isfinite(s).all() and ((s >= 0) & (s <= 1)).all() for s in scores.values())
    patient_ids = sorted({r['patient_id'] for r in included})
    groups = [np.asarray([i for i, r in enumerate(included) if r['patient_id'] == p]) for p in patient_ids]
    rng = np.random.default_rng(42)
    bootstrap = {name: [] for name in MODELS}
    for _ in range(repetitions):
        indices = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        if len(np.unique(labels[indices])) < 2:
            continue
        for name in MODELS:
            bootstrap[name].append(float(roc_auc_score(labels[indices], scores[name][indices])))
    assert len(bootstrap['v3']) >= repetitions * .9, 'Too few valid bootstrap replicates'
    metrics, curves = {}, {}
    for name in MODELS:
        fpr, tpr, thresholds = roc_curve(labels, scores[name], drop_intermediate=False)
        curves[name] = (fpr, tpr, thresholds)
        metrics[name] = {'auc': float(roc_auc_score(labels, scores[name])),
                         'auc_ci95': np.percentile(bootstrap[name], [2.5, 97.5]).tolist(),
                         'bootstrap_valid_repetitions': len(bootstrap[name])}
    return {'test_scans': len(records), 'evaluated_scans': len(included),
            'evaluated_patients': len(patient_ids), 'positive_scans': int(labels.sum()),
            'negative_scans': int((labels == 0).sum()), 'excluded_case_keys': [r['key'] for r in excluded],
            'ci_method': 'paired patient-cluster percentile bootstrap, seed 42',
            'bootstrap_repetitions': repetitions, 'models': metrics}, curves


def plot_results(report, curves, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 7))
    colors = ('#2166ac', '#d6604d', '#1b9e77', '#984ea3')
    for name, color in zip(MODELS, colors):
        metric = report['models'][name]
        lo, hi = metric['auc_ci95']
        label = f"{DISPLAY[name]}: AUC {metric['auc']:.3f} (95% CI {lo:.3f}–{hi:.3f})"
        fpr, tpr, _ = curves[name]
        ax.plot(fpr, tpr, color=color, linewidth=2, label=label)
    ax.plot([0, 1], [0, 1], '--', color='0.6', linewidth=1)
    ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel='False positive rate', ylabel='True positive rate',
           title=f"Whole-CT test ROC · {report['evaluated_scans']} scans")
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=.15)
    fig.text(.5, .035, 'Endpoint: LIDC reader-derived malignancy. EDICNet: primary detected ROI.',
             ha='center', fontsize=8)
    fig.text(.5, .014, 'DeepLung public pretraining overlap with the test cohort has not been excluded.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, 1))
    for extension in ('png', 'pdf'):
        fig.savefig(output / f'test_roc.{extension}', dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--v3-checkpoint', type=Path, default=V3_CONFIG.parent / 'epoch_018.pt')
    parser.add_argument('--output', type=Path, default=HERE / 'plots')
    parser.add_argument('--smoke', action='store_true', help='One training CT; never produces a test ROC')
    args = parser.parse_args()
    if args.smoke and args.output.resolve() == (HERE / 'plots').resolve():
        parser.error('Smoke outputs require a separate directory')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'cases').mkdir(exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(42); np.random.seed(42)
    cohort = load()
    cohort_hash = sha256(HERE / 'cohort.json')
    assert not ({r['patient_id'] for r in cohort['splits']['training']} &
                {r['patient_id'] for r in cohort['splits']['testing']})
    artifacts = baseline_artifacts(cohort_hash)
    model, config, epoch = build_v3(args.v3_checkpoint, cohort, args.smoke)
    # Include source hashes so an interrupted evaluation cannot mix implementations.
    source_hashes = {str(p.relative_to(HERE.parent.parent)): sha256(p)
                     for folder in ('common', 'legacy_training', 'baseline_compare')
                     for p in (HERE.parent / folder).glob('*.py')}
    provenance = {'cohort_sha256': cohort_hash, 'v3_checkpoint': str(args.v3_checkpoint),
                  'v3_checkpoint_sha256': sha256(args.v3_checkpoint), 'v3_epoch': epoch,
                  'baselines': artifacts, 'source_sha256': source_hashes, 'smoke': args.smoke,
                  'v3_aggregation': 'object-weighted noisy-OR per model; mean probability over full frozen bank',
                  'v3_object_threshold': model.inference_object_threshold,
                  'deeplung_aggregation': 'maximum malignancy over up to 20 detected nodules',
                  'edicnet_aggregation': 'malignancy of single highest-detection-score ROI',
                  'split': 'training_smoke' if args.smoke else 'testing',
                  'endpoint': 'LIDC reader-derived current malignancy',
                  'label_policy': 'original V3 mask-aware risk_target and risk_target_valid',
                  'test_fitting_or_selection': False,
                  'pretraining_limitation': 'DeepLung LUNA16 pretraining/test overlap unresolved'}
    fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    manifest_path = args.output / 'evaluation_manifest.json'
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text())['fingerprint'] == fingerprint, 'Different evaluation already exists'
    write_json(manifest_path, {**provenance, 'fingerprint': fingerprint})
    from back_prop.common.semantic_data import WholeCTLIDCDataset
    rows = cohort['splits']['training'][:1] if args.smoke else cohort['splits']['testing']
    dataset = WholeCTLIDCDataset(rows, target_mask_shape=tuple(config['screen_shape']))
    assert len(dataset) == len(rows), 'Test cohort unexpectedly changed'
    assert [f"{r['patient_id']}__scan{int(r['scan_id'])}" for r in dataset.cases] == [r['key'] for r in rows]
    fits, updates = int(model.rashomon.fit_count), model.rashomon.updates
    bank_before = {name: value.detach().cpu().clone() for name, value in model.rashomon.named_buffers()}
    records = []
    for index, row in enumerate(rows):
        started = time.time()
        path = args.output / 'cases' / f"{row['key']}.json"
        if path.exists():
            record = json.loads(path.read_text())
            assert record['fingerprint'] == fingerprint and record['key'] == row['key']
            records.append(record)
            print(json.dumps({'done': index + 1, 'total': len(rows), 'key': row['key'], 'resumed': True}), flush=True)
            continue
        batch = dataset[index]
        assert batch['case_id'] == row['key']
        record = {'key': row['key'], 'patient_id': row['patient_id'], 'image': row['image'],
                  'fingerprint': fingerprint, 'label': int(batch['risk_target']),
                  'included': bool(batch['risk_target_valid']), 'scores': {}}
        # Exclusions are determined solely by GT-label ambiguity before any model call.
        if record['included'] or args.smoke:
            image, image_hu = batch['image'], batch['image_hu']
            del batch  # No GT boxes/masks/labels are passed to any prediction function.
            with torch.inference_mode(), torch.autocast('cuda', dtype=getattr(torch, config['amp_dtype'])):
                result = model(image, image_hu=image_hu, batch=None, teacher_probability=0., update_bank=False)
            assert result.teacher_forced_count == result.fallback_count == 0
            assert result.ensemble_logits.shape[1] == int(model.rashomon.count)
            assert torch.isfinite(result.scan_logits).all()
            record['scores']['v3'] = float(result.risk_logit.float().sigmoid().item())
            record['v3_candidates'] = len(result.nodule_logits)
            del result, image
            nii = nib.as_closest_canonical(nib.load(row['image']))
            raw = np.asarray(nii.dataobj, dtype=np.float32)
            spacing = nib.affines.voxel_sizes(nii.affine)
            hu = image_hu[0].numpy()
            device = torch.device('cuda')
            for name in MODELS[1:]:
                files = {k: Path(v['path']) for k, v in artifacts[name].items()}
                if name == 'sybil':
                    result = sybil_predict(raw, spacing, files['classifier'], device)
                elif name == 'deeplung':
                    result = deeplung_predict(hu, files['classifier'], files['detector'], files['gbm'], device, 4, 20)
                else:
                    result = edicnet_predict(raw, spacing, hu, files['classifier'], files['detector'], device, 4, 1)
                record['scores'][name] = result['risk']
                record[name + '_candidates'] = len(result['candidates'])
                del result
            assert set(record['scores']) == set(MODELS)
            assert all(math.isfinite(s) and 0 <= s <= 1 for s in record['scores'].values())
            del raw, hu, image_hu, nii
        else:
            record['exclusion_reason'] = 'ambiguous scan label under V3 policy'
            del batch
        assert fits == int(model.rashomon.fit_count) and updates == model.rashomon.updates
        assert all(torch.equal(value.detach().cpu(), bank_before[name])
                   for name, value in model.rashomon.named_buffers())
        record['seconds'] = time.time() - started
        write_json(path, record)
        records.append(record)
        print(json.dumps({'done': index + 1, 'total': len(rows), 'key': row['key'],
                          'included': record['included'], 'scores': record['scores'],
                          'seconds': record['seconds']}), flush=True)
    if args.smoke:
        write_json(args.output / 'smoke_complete.json', {'passed': True, 'scores': records[0]['scores'],
                                                       'bank_unchanged': True, 'v3_epoch': epoch})
        return
    report, curves = summarize(records, [r['key'] for r in rows])
    report.update(fingerprint=fingerprint, label_only_exclusions=True, bank_unchanged=True,
                  slurm_job_id=os.environ.get('SLURM_JOB_ID'), provenance=provenance)
    with (args.output / 'test_predictions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['case_key', 'patient_id', 'included', 'label', *MODELS])
        writer.writeheader()
        for r in records:
            writer.writerow({'case_key': r['key'], 'patient_id': r['patient_id'],
                             'included': r['included'], 'label': r['label'], **r['scores']})
    with (args.output / 'test_roc_points.csv').open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['model', 'fpr', 'tpr', 'threshold'])
        for name, values in curves.items():
            writer.writerows((name, *point) for point in zip(*values))
    write_json(args.output / 'test_metrics.json', report)
    plot_results(report, curves, args.output)
    write_json(args.output / 'complete.json', {'fingerprint': fingerprint, 'models': list(MODELS),
                                              'evaluated_scans': report['evaluated_scans'],
                                              'plot_sha256': sha256(args.output / 'test_roc.png')})
    print(json.dumps({'evaluation_complete': True, 'output': str(args.output), 'metrics': report['models']}), flush=True)


if __name__ == '__main__':
    main()
