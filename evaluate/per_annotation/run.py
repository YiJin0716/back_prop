"""Run both epoch-18 models on every test reader annotation.

python -m back_prop.evaluate.per_annotation.run --output ... --device cuda
Completed cases are checked against checkpoint/protocol hashes and reused.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from back_prop.evaluate.eval_package.annotations import DEFAULT_MANIFEST
from back_prop.evaluate.eval_package.inference import load_joint_model
from back_prop.common.features import SEMANTIC_NAMES, RADIOMICS_NAMES
from .data import AnnotationCases, build_cohort, identity, sha256, worker_init

ROOT = Path(__file__).resolve().parents[3]
RUN_ROOT = Path('/usr/project/rudinlab/datasets/LIDC_IDRI/joint_model')
DEFAULTS = {
    'V4': RUN_ROOT/'model_v4_runs/12748637/epoch_018.pt',
    'V4oversample': RUN_ROOT/'model_v4_oversample_runs/12748663/epoch_018.pt',
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def bank_fingerprint(bank):
    digest = hashlib.sha256()
    for name, value in sorted(bank.named_buffers()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    for memory in (bank.memory, bank.radiomics_memory):
        for key, (features, label) in sorted(memory.items()):
            digest.update(repr(key).encode())
            digest.update(np.asarray(features).tobytes())
            digest.update(repr(label).encode())
    digest.update(str(bank.updates).encode())
    return digest.hexdigest()


def restore_components(path, name, device, cohort_info, test_patients):
    # Check saved configuration before constructing the full model. Extract
    # the already strictly restored branches; the detector is never forwarded.
    saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    config = saved['config']
    expected = 'v4' if name == 'V4' else 'v4_oversample'
    if config.get('variant') != expected:
        raise ValueError(f'{name}: checkpoint variant is {config.get("variant")!r}')
    manifest_hash = config.get('manifest_sha256')
    if manifest_hash != cohort_info['manifest_sha256']:
        raise ValueError(f'{name}: current test manifest differs from training manifest')
    config_metadata = dict(variant=expected, epoch=int(saved['epoch'])+1,
                           roi_shape=list(config['roi_shape']),
                           medicalnet_roi_size=config['medicalnet_roi_size'])
    del saved
    model = load_joint_model(path, device='cpu')
    training_patients = {key.rsplit('__scan',1)[0] for key in model.rashomon.allowed_cases}
    if not training_patients or training_patients & test_patients:
        raise ValueError(f'{name}: absent training provenance or test-patient leakage')
    parts = dict(semantics=model.semantics.to(device).eval(),
                 radiomics=model.radiomics.to(device).eval(),
                 risk=model.rashomon.to(device).eval())
    for part in parts.values():
        part.requires_grad_(False)
    config_metadata.update(checkpoint=str(path.resolve()), checkpoint_sha256=sha256(path),
        architecture=model.evaluation_metadata['architecture'],
        bank_mode=model.rashomon.mode, bank_models=int(model.rashomon.count),
        baseline_fits=int(model.rashomon.baseline_count),
        bank_training_patients=len(training_patients), test_patient_overlap=0,
        bank_fingerprint_before=bank_fingerprint(model.rashomon))
    del model
    gc.collect()
    return parts, config_metadata


def infer_annotation(entry, models, device, semantic_input):
    row, crop = entry['row'], entry['crop']
    if crop is None:
        return [dict(row, model=name) for name in models]
    hu = crop['hu'][None].to(device).float()
    mask = crop['mask'][None].to(device).bool()
    valid = crop['valid'][None].to(device).float()
    logits = torch.where(mask, torch.inf, -torch.inf)
    ct = (hu.clamp(-1024, 1024)+1024)/2048
    official_semantic = torch.tensor([[row['gt_'+name] for name in SEMANTIC_NAMES]],
                                    device=device, dtype=torch.float32)
    base_row = dict(row, crop_origin=crop['origin'], crop_shape=crop['shape'],
                    mask_voxels=crop['mask_voxels'])
    output = []
    with torch.inference_mode():
        # Both V4 variants use the same fixed SoftRadiomics3D formula.
        radio = next(iter(models.values()))['radiomics'](hu, logits, valid)
        if not torch.isfinite(radio).all():
            raise ValueError('Nonfinite radiomics features')
        base_row.update({f'radiomics_{key}':float(value)
                         for key,value in zip(RADIOMICS_NAMES,radio[0].cpu())})
        for name, parts in models.items():
            probability, expected = parts['semantics'](ct, logits, valid)
            if not torch.isfinite(probability).all() or not torch.isfinite(expected).all():
                raise ValueError(f'{name}: nonfinite semantic predictions')
            p = probability[0].cpu().numpy()
            prediction = dict(base_row, model=name)
            for j, feature in enumerate(SEMANTIC_NAMES):
                prediction['pred_'+feature] = float(expected[0,j])
                prediction.update({f'{feature}_p{k+1}':float(p[j,k]) for k in range(5)})
            baseline = parts['risk'].baseline_logits(radio)
            prediction['radiomics_malignancy_probability'] = float(baseline.sigmoid()[0])
            inputs = dict(official=official_semantic, predicted=expected)
            for source in ('official','predicted') if semantic_input=='both' else (semantic_input,):
                ensemble, indices, correction = parts['risk'](inputs[source], baseline)
                score = ensemble.sigmoid().mean(1).clamp(1e-6,1-1e-6)
                if not torch.isfinite(score).all():
                    raise ValueError(f'{name}: nonfinite malignancy prediction')
                prediction[f'malignancy_probability_{source}'] = float(score[0])
                prediction[f'ensemble_probabilities_{source}'] = ensemble.sigmoid()[0].cpu().tolist()
                prediction[f'semantic_correction_logits_{source}'] = correction[0].cpu().tolist()
                prediction['bank_model_indices'] = indices.cpu().tolist()
            output.append(prediction)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--v4', type=Path, default=DEFAULTS['V4'])
    parser.add_argument('--v4-oversample', type=Path, default=DEFAULTS['V4oversample'])
    parser.add_argument('--output', type=Path, default=Path(__file__).parent/'results')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--max-cases', type=int, help='Smoke run only; mark report as a subset')
    parser.add_argument('--threshold-rule', choices=('ge','gt'), default='ge')
    parser.add_argument('--semantic-input', choices=('official','predicted','both'), default='official')
    args = parser.parse_args()
    if args.workers < 0 or args.threads < 1 or (args.max_cases is not None and args.max_cases<1):
        parser.error('Invalid workers, threads or max-cases')
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    device = torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('This run requires an allocated CUDA GPU')
    cohort, info = build_cohort(args.manifest)
    full_count = len(cohort)
    if args.max_cases is not None:
        cohort = cohort[:args.max_cases]
    entries = [a for c in cohort for a in c['annotations']]
    test_patients = {c['case']['patient_id'] for c in cohort}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'cases').mkdir(exist_ok=True)
    models, metadata = {}, {}
    for name,path in (('V4',args.v4),('V4oversample',args.v4_oversample)):
        print(f'Loading {name}: {path}',flush=True)
        models[name], metadata[name] = restore_components(path,name,device,info,test_patients)
    if metadata['V4']['roi_shape'] != metadata['V4oversample']['roi_shape']:
        raise ValueError('Model ROI shapes differ; prepare independent inputs')
    protocol = dict(
        unit='one reader annotation, no consensus or reader averaging',
        semantic_threshold_rule=args.threshold_rule, thresholds=[1,2,3,4,5],
        semantic_auc_score='predicted probability mass above the requested rating threshold',
        malignancy_semantic_input=args.semantic_input,
        malignancy_label='reader rating 1/2 negative; 4/5 positive; 3 excluded from ROC only',
        malignancy_aggregation='mean of sigmoid(baseline logit + semantic correction), checkpoint eval mode',
        radiomics='checkpoint SoftRadiomics3D on original HU and exact binary reader mask',
        radiomics_units='model nominal 1-mm training grid',
        mask_input='individual official reader mask; sigmoid(+/- infinity) gives exact 0/1',
        crop='annotation bounding-box voxel center, checkpoint ROI shape enlarged to contain full mask',
        crop_rounding='integer_crop floor(center + 0.5)',
        ct_input='canonical XYZ; scipy zoom linear CT / nearest mask; normalized clip(HU,-1024,1024)',
        training_or_refitting=False, weights_frozen=True,
        semantic_names=list(SEMANTIC_NAMES), radiomics_names=list(RADIOMICS_NAMES),
        source_sha256={p.name:sha256(p) for p in (Path(__file__),Path(__file__).with_name('data.py'))},
    )
    fingerprint = hashlib.sha256(json.dumps(dict(cohort=info,models=metadata,protocol=protocol),
                                            sort_keys=True).encode()).hexdigest()
    audit = dict(cohort=info, models=metadata, protocol=protocol, fingerprint=fingerprint,
                 full_test_set=len(cohort)==full_count, evaluated_cases=len(cohort),
                 expected_annotations=len(entries), expected_prediction_rows=2*len(entries),
                 device=str(device), torch=torch.__version__, started_at=datetime.now(timezone.utc).isoformat())
    # A conflicting run never overwrites or combines existing predictions.
    provenance = args.output/'provenance.json'
    if provenance.exists() and json.loads(provenance.read_text())['fingerprint'] != fingerprint:
        raise ValueError('Output belongs to another checkpoint/protocol; choose a new --output')
    write_json(provenance,audit)
    pd.DataFrame(entries).to_csv(args.output/'annotations.csv',index=False)
    pending, results = [], []
    for item in cohort:
        path=args.output/'cases'/f"{item['case_id']}.json"
        if path.exists():
            cached=json.loads(path.read_text())
            expected={(name,a['annotation_key']) for name in models for a in item['annotations']}
            actual={(r['model'],r['annotation_key']) for r in cached['predictions']}
            if cached['fingerprint'] != fingerprint or expected != actual or len(actual)!=len(cached['predictions']):
                raise ValueError(f'Invalid case cache: {path}')
            results.extend(cached['predictions'])
        else:
            pending.append(item)
    print(f'{len(cohort)} cases; {len(entries)} annotations; {len(pending)} cases pending',flush=True)
    kwargs = dict(num_workers=args.workers,batch_size=None,collate_fn=identity)
    if args.workers:
        kwargs.update(multiprocessing_context='spawn',worker_init_fn=worker_init,prefetch_factor=1)
    loader=DataLoader(AnnotationCases(pending,metadata['V4']['roi_shape']),**kwargs)
    started=time.monotonic()
    for i,item in enumerate(loader,1):
        predictions=[]
        for annotation in item['annotations']:
            predictions.extend(infer_annotation(annotation,models,device,args.semantic_input))
        write_json(args.output/'cases'/f"{item['case_id']}.json",
                   dict(fingerprint=fingerprint,case_id=item['case_id'],predictions=predictions))
        results.extend(predictions)
        print(json.dumps(dict(event='case_complete',case_id=item['case_id'],
            completed=i,total=len(pending),annotations=len(item['annotations']),
            elapsed_seconds=round(time.monotonic()-started,1))),flush=True)
    if len(results)!=audit['expected_prediction_rows']:
        raise AssertionError('Prediction count differs from complete annotation cohort')
    for name,parts in models.items():
        after=bank_fingerprint(parts['risk'])
        if after != metadata[name]['bank_fingerprint_before']:
            raise AssertionError(f'{name}: fitted bank changed during test evaluation')
        audit['models'][name]['bank_unchanged']=True
        audit['models'][name]['bank_fingerprint_after']=after
    # Flatten array-valued provenance as JSON inside CSV fields.
    frame=pd.DataFrame(results).sort_values(['model','case_id','annotation_id'])
    for col in frame.columns:
        if frame[col].map(lambda x:isinstance(x,(list,dict))).any():
            frame[col]=frame[col].map(lambda x:json.dumps(x) if isinstance(x,(list,dict)) else x)
    frame.to_csv(args.output/'predictions.csv',index=False)
    audit.update(status='complete',completed_at=datetime.now(timezone.utc).isoformat(),
                 prediction_rows=len(frame),
                 input_status_counts=frame.groupby(['model','input_status']).size().to_dict())
    audit['input_status_counts']={f'{k[0]}/{k[1]}':v for k,v in audit['input_status_counts'].items()}
    write_json(provenance,audit)
    from .report import generate_report
    generate_report(args.output)
    print(f'Analysis complete: {args.output.resolve()}',flush=True)


if __name__=='__main__':
    main()
