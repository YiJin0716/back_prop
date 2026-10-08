"""Reproduce the descriptive V4 semantic and epoch-log audit on CPU."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
RUNS = Path('/usr/project/rudinlab/datasets/LIDC_IDRI/joint_model/model_v4_runs')
NAMES = ['lobulation', 'margin', 'sphericity', 'spiculation', 'subtlety', 'texture']


def json_rows(path):
    result = []
    for line in path.open():
        try:
            result.append(json.loads(line))
        except ValueError:
            continue
    return result


def stats(array):
    return dict(mean=array.mean(0).tolist(), std=array.std(0).tolist(),
                min=array.min(0).tolist(), max=array.max(0).tolist())


def main():
    prediction_path = REPO/'back_prop/evaluate/per_annotation/results/predictions.csv'
    data = pd.read_csv(prediction_path)
    audit = dict(names=NAMES, test_prediction_file=str(prediction_path),
                 test_prediction_sha256=hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
                 test_cohort_scope='Official individual-reader masks; no segmentation error in the input mask.',
                 test={}, sources={})
    feature_rows = []
    for model, group in data.groupby('model'):
        y = group[[f'gt_{n}' for n in NAMES]].to_numpy(float)
        pred = group[[f'pred_{n}' for n in NAMES]].to_numpy(float)
        p = group[[f'{n}_p{i}' for n in NAMES for i in range(1, 6)]].to_numpy(float).reshape(-1,6,5)
        near = np.abs(pred-np.median(pred, axis=0)) < .001
        audit['test'][model] = dict(n=len(group), cases=group.case_id.nunique(),
            physical_nodules=group[['case_id','nodule_id']].drop_duplicates().shape[0],
            predicted=stats(pred), ground_truth=stats(y),
            max_probability_std=float(p.std(0).max()),
            fraction_all_six_within_0001_of_median=float(near.all(1).mean()),
            fraction_each_within_0001_of_median=near.mean(0).tolist())
        for i, n in enumerate(NAMES):
            feature_rows.append(dict(model=model, feature=n, predicted_mean=pred[:,i].mean(),
                predicted_std=pred[:,i].std(), gt_std=y[:,i].std(),
                predicted_min=pred[:,i].min(), predicted_max=pred[:,i].max()))
    pd.DataFrame(feature_rows).to_csv(HERE/'feature_dispersion.csv', index=False)
    native_path = REPO/'back_prop/evaluate/visualize_mask/results/v4_epoch018/LIDC-IDRI-0301__scan301/result.json'
    native = json.loads(native_path.read_text())
    arr = np.array([[p['semantic_features'][n] for n in NAMES] for p in native['predictions']])
    audit['native_predicted_masks'] = dict(source=str(native_path), case_id=native['case_id'],
        n_candidates=len(arr), features=stats(arr), scope='One previously evaluated test CT, all selected candidates.')
    epochs = json_rows(RUNS/'12723354/metrics.jsonl') + json_rows(RUNS/'12748637/metrics.jsonl')
    fields = ['epoch','semantic','teacher_probability','training_samples','valid_risk_cases','diagnostic_scale','temperature']
    epoch_frame = pd.DataFrame([{**{k:r[k] for k in fields}, 'lr_medicalnet':r['lr']['medicalnet']} for r in epochs])
    assert epoch_frame.epoch.tolist() == list(range(1,19))
    epoch_frame.to_csv(HERE/'epoch_metrics.csv', index=False)
    audit['epoch_metrics'] = epoch_frame.to_dict('records')
    step_summaries = []
    for run in ['12723354','12748637']:
        rows = [r for r in json_rows(REPO/f'back_prop/model_v4/logs/train_{run}.out') if r.get('event') == 'step']
        for epoch in sorted({r['epoch'] for r in rows}):
            e = [r for r in rows if r['epoch'] == epoch]
            losses = np.array([r['semantic'] for r in e])
            nonzero = losses > 0
            step_summaries.append(dict(run=run, epoch=epoch, n_rank0_steps=len(e),
                zero_semantic_steps=int((~nonzero).sum()), all_mean=float(losses.mean()),
                nonzero_mean=float(losses[nonzero].mean()),
                teacher_forced=sum(r['teacher_forced'] for r in e),
                fallbacks=sum(r['fallbacks'] for r in e),
                zero_cases=[r['case_id'] for r in e if r['semantic'] == 0],
                skipped_optimizer_steps=sum(r['optimizer_step_skipped'] for r in e)))
    audit['rank0_steps'] = step_summaries
    audit['rank0_scope_note'] = 'Step logs include rank 0 only; do not extrapolate the exact zero count to all four ranks.'
    e17 = next(r for r in step_summaries if r['epoch']==17)
    e18 = next(r for r in step_summaries if r['epoch']==18 and r['run']=='12748637')
    assert np.isclose(e18['all_mean'], e18['nonzero_mean']*(1-e18['zero_semantic_steps']/e18['n_rank0_steps']))
    audit['rank0_decomposition'] = dict(epoch17_mean=e17['all_mean'], epoch18_mean=e18['all_mean'],
        epoch18_mean_on_nonzero=e18['nonzero_mean'],
        zero_denominator_effect=e18['all_mean']-e18['nonzero_mean'],
        note='Not a paired cohort comparison: shuffling and valid-ROI selection differ between epochs.')
    audit['warmup'] = json.loads((RUNS/'12723354/warmup.json').read_text())
    original = json.loads((RUNS/'12723354/config.json').read_text())
    resumed = json.loads((RUNS/'12748637/config.json').read_text())
    audit['resume_config_differences'] = {k:dict(before=original.get(k), after=resumed.get(k))
        for k in sorted(set(original)|set(resumed)) if original.get(k)!=resumed.get(k)}
    for name in ['back_prop/model_v4/model.py','back_prop/model_v4/loss.py','back_prop/model_v4/train.py',
                 'back_prop/model_v4/features.py','back_prop/model_v4/warmup.py','back_prop/common/coarse_model.py']:
        audit['sources'][name] = hashlib.sha256((REPO/name).read_bytes()).hexdigest()
    audit['ddp_recovery'] = json.loads((REPO/'back_prop/model_v4/logs/ddp_recovery_20261002.json').read_text())
    (HERE/'audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')

    fig, axes = plt.subplots(1,2, figsize=(11,4.1), layout='constrained')
    ax=axes[0]
    ax.plot(epoch_frame.epoch, epoch_frame.semantic, 'o-', color='#226b93')
    ax.axvline(17.5, color='#b6432d', linestyle='--', linewidth=1)
    ax.set(xlabel='Training epoch', ylabel='Logged semantic loss (all CT presentations)',
           title='Teacher routing and GT fallback stop at epoch 18', xticks=[1,5,10,15,17,18])
    ax.grid(alpha=.2)
    ax=axes[1]
    group=data[data.model=='V4']
    pred=group[[f'pred_{n}' for n in NAMES]].std(ddof=0).to_numpy()
    gt=group[[f'gt_{n}' for n in NAMES]].std(ddof=0).to_numpy()
    x=np.arange(6)
    ax.bar(x-.18, gt, width=.36, label='Official reader ratings', color='#829397')
    ax.bar(x+.18, pred, width=.36, label='V4 predictions', color='#226b93')
    ax.set(yscale='log', ylabel='Standard deviation across 1,423 annotations',
           xticks=x, xticklabels=NAMES, title='Prediction variation is roughly 1,000× smaller')
    ax.tick_params(axis='x', rotation=30)
    ax.legend(frameon=False)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
    fig.savefig(HERE/'diagnosis.png', dpi=180)
    fig.savefig(HERE/'diagnosis.pdf')
    print(json.dumps(dict(epoch17=epochs[-2]['semantic'], epoch18=epochs[-1]['semantic'],
                         rank0_decomposition=audit['rank0_decomposition'],
                         test=audit['test'], native=audit['native_predicted_masks']), indent=2))


if __name__ == '__main__':
    main()
