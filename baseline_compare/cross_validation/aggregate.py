"""Combine exactly one held-out prediction per eligible CT across all five folds."""
import json
from pathlib import Path
import numpy as np
from .build import HERE_CV, PLOTS
from .statistics import MODELS, save_results
from ..cohort import load, sha256
from ..evaluate_roc import write_json


def main():
    plan = json.loads((HERE_CV / 'plan.json').read_text()); records, keys, fold_metrics = [], [], []
    fingerprints = {}
    for item in plan['folds']:
        directory = PLOTS / f'fold_{item["fold"]}'
        complete = json.loads((directory / 'complete.json').read_text())
        assert complete['cohort_sha256'] == item['cohort_sha256'] and complete['bank_unchanged']
        assert complete['models'] == list(MODELS)
        assert sha256(directory / 'test_roc.png') == complete['plot_sha256']
        fingerprints[str(item['fold'])] = complete['fingerprint']
        cohort = load(item['cohort'])
        for row in cohort['splits']['testing']:
            record = json.loads((directory / 'cases' / (row['key'] + '.json')).read_text())
            assert record['key'] == row['key'] and record['fold'] == item['fold']
            assert record['patient_id'] == row['patient_id'] and record['fingerprint'] == complete['fingerprint']
            records.append(record); keys.append(row['key'])
        fold_metrics.append(json.loads((directory / 'metrics.json').read_text()))
    assert len(records) == len(set(keys)) == plan['eligible_scans']
    report = save_results(records, keys, PLOTS, 'Five-fold out-of-fold ROC')
    report.update(folds=fold_metrics, limitations=plan['limitations'], fold_fingerprints=fingerprints,
                  fold_auc_summary={m: {'mean': float(np.mean([f['models'][m]['auc'] for f in fold_metrics])),
                                        'sample_sd': float(np.std([f['models'][m]['auc'] for f in fold_metrics], ddof=1)),
                                        'v3_wins': sum(f['models']['v3']['auc'] > f['models'][m]['auc'] for f in fold_metrics) if m != 'v3' else None}
                                    for m in MODELS})
    write_json(PLOTS / 'metrics.json', report)
    write_json(PLOTS / 'complete.json', {'folds': 5, 'models': list(MODELS), 'eligible_scans': len(records),
               'fold_fingerprints': fingerprints, 'plot_sha256': sha256(PLOTS / 'test_roc.png')})
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
