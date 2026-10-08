"""Audit final models, then evaluate the same held-out CTs without GT inputs."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import nibabel as nib
import numpy as np
import torch
from .build import HERE_CV, PLOTS
from .statistics import MODELS, save_results
from ..cohort import HERE, load, sha256
from ..audit_training import verify_run
from ..evaluate_roc import build_v3, write_json
from ..predict import sybil_predict, deeplung_predict


def audit_baselines(item, jobs):
    directory = Path(item['directory']); cohort = load(item['cohort'])
    cache_audit = json.loads((directory / 'cache_audit.json').read_text())
    assert cache_audit['cohort_sha256'] == item['cohort_sha256']
    names = {'sybil': ('sybil', 3), 'detector': ('deeplung_detector', 5),
             'classifier': ('deeplung_initial_classifier', 700), 'adapt': ('deeplung_final_classifier', 105)}
    reports = {}
    for stage, (name, epochs) in names.items():
        run = {'output': str(directory / stage), 'epochs': epochs,
               'job_id': jobs[f'f{item["fold"]}-{stage}']['job_id']}
        report = verify_run(name, run, cohort, cache_audit, cohort_path=item['cohort'])
        assert report['verified'], (stage, report)
        reports[stage] = report
    return reports


def main():
    p = argparse.ArgumentParser(); p.add_argument('--fold', type=int, required=True)
    args = p.parse_args(); assert args.fold in range(5)
    torch.set_num_threads(2); torch.manual_seed(42); np.random.seed(42)
    plan = json.loads((HERE_CV / 'plan.json').read_text()); item = plan['folds'][args.fold]
    if args.fold in plan.get('reused_folds', []):
        from .reuse_fold0 import main as reuse
        reuse()
        return
    jobs = json.loads((HERE_CV / 'jobs.json').read_text())
    directory = Path(item['directory']); output = PLOTS / f'fold_{args.fold}'
    (output / 'cases').mkdir(parents=True, exist_ok=True)
    assert sha256(item['cohort']) == item['cohort_sha256']
    cohort = load(item['cohort'])
    reports = audit_baselines(item, jobs)
    weights = {'v3': directory / 'v3/epoch_018.pt', 'sybil': directory / 'sybil/last.pt',
               'classifier': directory / 'adapt/last.pt', 'detector': directory / 'detector/last.pt',
               'gbm': directory / 'adapt/gbm.pkl'}
    epochs = [json.loads(line) for line in (directory / 'v3/metrics.jsonl').read_text().splitlines()]
    assert [r['epoch'] for r in epochs] == list(range(1, 19))
    assert all(math.isfinite(r['loss']) and r['benchmark_steps'] is None for r in epochs)
    model, config, epoch = build_v3(weights['v3'], cohort)
    assert sha256(config['manifest']) == item['manifest_sha256']
    assert config['vista_checkpoint'] == item['vista_checkpoint']
    assert config['resume'] is None and config['init_v2'] is None
    assert config['malignancy_policy'] == cohort['policy']
    assert config['world_size'] == 4 and config['seed'] == 42
    sources = {str(p.relative_to(HERE.parents[1])): sha256(p)
               for folder in ('common', 'legacy_training', 'baseline_compare')
               for p in (HERE.parent / folder).glob('*.py')}
    sources.update({str(p.relative_to(HERE.parents[1])): sha256(p)
                    for sub in ('sybil', 'deeplung', 'deeplung/ported', 'cross_validation')
                    for p in (HERE / sub).glob('*.py')})
    provenance = {'fold': args.fold, 'cohort_sha256': item['cohort_sha256'], 'v3_epoch': epoch,
                  'weights': {k: {'path': str(p), 'sha256': sha256(p)} for k, p in weights.items()},
                  'sources': sources, 'baseline_audits': reports,
                  'endpoint': 'LIDC reader-derived current malignancy', 'label_policy': 'V3 mask-aware risk label',
                  'prediction_inputs': 'CT only; no GT boxes, masks, features, or labels',
                  'v3_aggregation': 'mean probability over full frozen continuous Rashomon bank',
                  'deeplung_aggregation': 'maximum predicted nodule malignancy; cap 20; risk 0 if none',
                  'test_fitting_or_selection': False, 'limitations': plan['limitations']}
    fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    manifest_path = output / 'evaluation_manifest.json'
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text())['fingerprint'] == fingerprint
    write_json(manifest_path, {**provenance, 'fingerprint': fingerprint})
    from back_prop.common.semantic_data import WholeCTLIDCDataset
    rows = cohort['splits']['testing']
    dataset = WholeCTLIDCDataset(rows, target_mask_shape=tuple(config['screen_shape']))
    assert [f"{r['patient_id']}__scan{int(r['scan_id'])}" for r in dataset.cases] == [r['key'] for r in rows]
    fits, updates = int(model.rashomon.fit_count), model.rashomon.updates
    before = {name: value.detach().cpu().clone() for name, value in model.rashomon.named_buffers()}
    records = []
    for index, row in enumerate(rows):
        path = output / 'cases' / (row['key'] + '.json'); started = time.time()
        if path.exists():
            record = json.loads(path.read_text())
            assert record['fingerprint'] == fingerprint and record['key'] == row['key']
            records.append(record); continue
        batch = dataset[index]; assert batch['case_id'] == row['key']
        record = {'fold': args.fold, 'key': row['key'], 'patient_id': row['patient_id'],
                  'fingerprint': fingerprint, 'label': int(batch['risk_target']),
                  'included': bool(batch['risk_target_valid']), 'scores': {}}
        if record['included']:
            image, image_hu = batch['image'], batch['image_hu']; del batch
            with torch.inference_mode(), torch.autocast('cuda', dtype=getattr(torch, config['amp_dtype'])):
                result = model(image, image_hu=image_hu, batch=None, teacher_probability=0., update_bank=False)
            assert result.teacher_forced_count == result.fallback_count == 0
            assert result.ensemble_logits.shape[1] == int(model.rashomon.count)
            record['scores']['v3'] = float(result.risk_logit.float().sigmoid().item())
            del image, result
            nii = nib.as_closest_canonical(nib.load(row['image']))
            raw = np.asarray(nii.dataobj, dtype=np.float32); spacing = nib.affines.voxel_sizes(nii.affine)
            device = torch.device('cuda')
            record['scores']['sybil'] = sybil_predict(raw, spacing, weights['sybil'], device)['risk']
            record['scores']['deeplung'] = deeplung_predict(image_hu[0].numpy(), weights['classifier'],
                weights['detector'], weights['gbm'], device, 4, 20)['risk']
            assert set(record['scores']) == set(MODELS)
            assert all(math.isfinite(s) and 0 <= s <= 1 for s in record['scores'].values())
            del raw, nii, image_hu
        else:
            record['exclusion_reason'] = 'ambiguous scan label under V3 policy'; del batch
        assert fits == int(model.rashomon.fit_count) and updates == model.rashomon.updates
        assert all(torch.equal(v.detach().cpu(), before[k]) for k, v in model.rashomon.named_buffers())
        record['seconds'] = time.time() - started; write_json(path, record); records.append(record)
        print(json.dumps({'done': index + 1, 'total': len(rows), **record}), flush=True)
    report = save_results(records, [r['key'] for r in rows], output, f'Fold {args.fold} held-out ROC')
    write_json(output / 'complete.json', {'fingerprint': fingerprint, 'fold': args.fold, 'models': list(MODELS),
               'cohort_sha256': item['cohort_sha256'], 'evaluated_scans': report['evaluated_scans'],
               'bank_unchanged': True, 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
               'plot_sha256': sha256(output / 'test_roc.png')})


if __name__ == '__main__':
    main()
