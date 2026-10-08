"""Import the completed original fold-0 predictions after provenance checks."""
import hashlib
import json
from pathlib import Path
from .build import HERE_CV, PLOTS
from .statistics import MODELS, save_results
from ..cohort import HERE, load, sha256
from ..evaluate_roc import write_json
from back_prop.common.source_compat import source_matches


def main():
    plan = json.loads((HERE_CV / 'plan.json').read_text())
    assert plan['reused_folds'] == [0]
    item = plan['folds'][0]; original = load(); current = load(item['cohort'])
    for key in ('splits', 'manifest_sha256', 'annotations_sha256', 'policy', 'training_case_keys_sha256'):
        assert original[key] == current[key], key
    old_config = json.loads(Path(original['v3_config']).read_text())
    new_config = json.loads(Path(current['v3_config']).read_text())
    differences = [k for k in old_config.keys() | new_config.keys() if old_config.get(k) != new_config.get(k)]
    assert set(differences) <= {'runtime', 'output_dir', 'resume', 'source_sha256'}
    source = HERE / 'plots'; output = PLOTS / 'fold_0'
    manifest = json.loads((source / 'evaluation_manifest.json').read_text())
    complete = json.loads((source / 'complete.json').read_text())
    assert manifest['fingerprint'] == complete['fingerprint']
    assert manifest['cohort_sha256'] == sha256(HERE / 'cohort.json')
    assert manifest['v3_epoch'] == 18 and not manifest['smoke']
    assert manifest['split'] == 'testing' and not manifest['test_fitting_or_selection']
    assert sha256(manifest['v3_checkpoint']) == manifest['v3_checkpoint_sha256']
    for name in ('sybil', 'deeplung'):
        published = json.loads((HERE / name / 'final_artifacts.json').read_text())
        assert published['training_and_inference_verified']
        assert published['cohort_sha256'] == manifest['cohort_sha256']
        for component, artifact in manifest['baselines'][name].items():
            assert sha256(artifact['path']) == artifact['sha256']
            assert published['artifacts'][component]['sha256'] == artifact['sha256']
    # The predictors used by the newly submitted folds retain these implementations.
    for name, digest in manifest['source_sha256'].items():
        if any(name.startswith('back_prop/' + m + '/') for m in (
                'model_v1', 'model_v2', 'model_v3', 'common', 'legacy_training')) or name in (
                'back_prop/baseline_compare/predict.py', 'back_prop/baseline_compare/prepare.py',
                'back_prop/baseline_compare/nodule_data.py'):
            assert source_matches(HERE.parents[1] / name, digest), name
    source_rows = []
    for row in current['splits']['testing']:
        record = json.loads((source / 'cases' / (row['key'] + '.json')).read_text())
        assert record['key'] == row['key'] and record['patient_id'] == row['patient_id']
        assert record['fingerprint'] == manifest['fingerprint']
        source_rows.append(record)
    assert len(source_rows) == 166 and sum(r['included'] for r in source_rows) == complete['evaluated_scans'] == 137
    provenance = {'fold': 0, 'reused_existing_evaluation': True,
                  'cohort_sha256': item['cohort_sha256'], 'original_cohort_sha256': manifest['cohort_sha256'],
                  'original_evaluation_manifest': str(source / 'evaluation_manifest.json'),
                  'original_evaluation_manifest_sha256': sha256(source / 'evaluation_manifest.json'),
                  'original_fingerprint': manifest['fingerprint'], 'v3_epoch': 18,
                  'v3_checkpoint': manifest['v3_checkpoint'], 'v3_checkpoint_sha256': manifest['v3_checkpoint_sha256'],
                  'baseline_artifacts': {m: manifest['baselines'][m] for m in MODELS[1:]},
                  'cohorts_and_label_policy_identical': True, 'config_difference_fields': sorted(differences),
                  'selection': 'reuse previously completed fold 0; no new inference, fitting, or model selection',
                  'original_case_sha256': {r['key']: sha256(source / 'cases' / (r['key'] + '.json')) for r in source_rows}}
    fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    (output / 'cases').mkdir(parents=True, exist_ok=True)
    if (output / 'evaluation_manifest.json').exists():
        assert json.loads((output / 'evaluation_manifest.json').read_text())['fingerprint'] == fingerprint
    write_json(output / 'evaluation_manifest.json', {**provenance, 'fingerprint': fingerprint})
    records = []
    for old in source_rows:
        record = {**old, 'fold': 0, 'fingerprint': fingerprint,
                  'original_fingerprint': old['fingerprint'], 'reused_existing_prediction': True,
                  'scores': {k: v for k, v in old['scores'].items() if k in MODELS}}
        write_json(output / 'cases' / (record['key'] + '.json'), record); records.append(record)
    report = save_results(records, [r['key'] for r in records], output, 'Fold 0 held-out ROC (existing models)')
    previous = json.loads((source / 'test_metrics.json').read_text())
    for model in MODELS:
        assert abs(report['models'][model]['auc'] - previous['models'][model]['auc']) < 1e-12
    write_json(output / 'complete.json', {'fingerprint': fingerprint, 'fold': 0, 'models': list(MODELS),
        'cohort_sha256': item['cohort_sha256'], 'evaluated_scans': report['evaluated_scans'],
        'bank_unchanged': True, 'reused_existing_evaluation': True,
        'original_evaluation_job_id': '12632290', 'plot_sha256': sha256(output / 'test_roc.png')})
    write_json(HERE_CV / 'fold0_reuse_audit.json', {**provenance, 'fingerprint': fingerprint, 'passed': True,
                                               'metrics': report['models']})
    print(json.dumps({'fold0_reused': True, 'evaluated_scans': report['evaluated_scans'], 'models': report['models']}))


if __name__ == '__main__':
    main()
