"""Fixed paired ROC analysis; bootstrap patients within each held-out fold."""
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

MODELS = ('v3', 'sybil', 'deeplung')
DISPLAY = {'v3': 'V3', 'sybil': 'Sybil', 'deeplung': 'DeepLung'}


def summarize(records, expected_keys, repetitions=2000):
    assert len(records) == len(expected_keys) == len(set(expected_keys))
    assert {r['key'] for r in records} == set(expected_keys)
    included = [r for r in records if r['included']]
    y = np.array([r['label'] for r in included])
    assert set(y) == {0, 1}
    scores = {m: np.array([r['scores'][m] for r in included]) for m in MODELS}
    assert all(np.isfinite(s).all() and ((s >= 0) & (s <= 1)).all() for s in scores.values())
    patient_fold = {}
    for r in records:
        assert patient_fold.setdefault(r['patient_id'], r['fold']) == r['fold'], 'Patient crosses test folds'
    groups = []
    for fold in sorted({r['fold'] for r in included}):
        patients = sorted({r['patient_id'] for r in included if r['fold'] == fold})
        groups.append([np.array([i for i, r in enumerate(included) if r['patient_id'] == p]) for p in patients])
    rng = np.random.default_rng(42); samples = []
    for _ in range(repetitions):
        indices = np.concatenate([g[j] for g in groups for j in rng.integers(0, len(g), len(g))])
        if len(np.unique(y[indices])) == 2:
            samples.append([roc_auc_score(y[indices], scores[m][indices]) for m in MODELS])
    assert len(samples) >= .9 * repetitions
    samples = np.array(samples)
    auc = {m: float(roc_auc_score(y, scores[m])) for m in MODELS}
    report = {'test_scans': len(records), 'evaluated_scans': len(included),
              'evaluated_patients': len({r['patient_id'] for r in included}),
              'positive_scans': int(y.sum()), 'negative_scans': int((y == 0).sum()),
              'excluded_case_keys': [r['key'] for r in records if not r['included']],
              'bootstrap': 'paired patient-cluster sampling within folds; 2000 repeats, seed 42; conditional on trained models',
              'models': {m: {'auc': auc[m], 'auc_ci95': np.percentile(samples[:, i], [2.5, 97.5]).tolist()} for i, m in enumerate(MODELS)},
              'paired_auc_differences': {m: {'v3_minus_baseline': auc['v3'] - auc[m],
                  'ci95': np.percentile(samples[:, 0] - samples[:, i], [2.5, 97.5]).tolist()} for i, m in enumerate(MODELS) if i}}
    return report, {m: roc_curve(y, scores[m], drop_intermediate=False) for m in MODELS}


def save_results(records, expected_keys, output, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    report, curves = summarize(records, expected_keys)
    (output / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    with (output / 'predictions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['fold', 'key', 'patient_id', 'included', 'label', *MODELS])
        writer.writeheader()
        writer.writerows({**{k: r[k] for k in ('fold', 'key', 'patient_id', 'included', 'label')}, **r['scores']} for r in records)
    with (output / 'roc_points.csv').open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['model', 'fpr', 'tpr', 'threshold'])
        for m, values in curves.items():
            writer.writerows((m, *point) for point in zip(*values))
    fig, ax = plt.subplots(figsize=(7.5, 7))
    for m, color in zip(MODELS, ('#2166ac', '#d6604d', '#1b9e77')):
        metric = report['models'][m]; lo, hi = metric['auc_ci95']
        ax.plot(curves[m][0], curves[m][1], color=color, lw=2,
                label=f"{DISPLAY[m]} AUC {metric['auc']:.3f} (95% CI {lo:.3f}–{hi:.3f})")
    ax.plot([0, 1], [0, 1], '--', color='.6', lw=1)
    ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel='False positive rate', ylabel='True positive rate',
           title=f'{title} · {len([r for r in records if r["included"]])} CTs')
    ax.set_aspect('equal'); ax.legend(loc='lower right', fontsize=9); ax.grid(alpha=.15)
    fig.text(.5, .025, 'Reader-derived malignancy; DeepLung public pretraining overlap unresolved.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, 1))
    for ext in ('png', 'pdf'):
        fig.savefig(output / f'test_roc.{ext}', dpi=300)
    plt.close(fig)
    return report
