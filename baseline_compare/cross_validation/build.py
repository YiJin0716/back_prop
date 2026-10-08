"""Freeze all folds before training; no test-driven configuration selection."""
import json
from pathlib import Path
from ..cohort import HERE, V3_CONFIG, build, sha256
from back_prop.common.base_data import WholeCTLIDCDataset, load_cases

ROOT = HERE.parents[1]
HERE_CV = HERE / 'cross_validation'
RUN = Path('/usr/project/rudinlab/datasets/LIDC_IDRI/joint_model/cross_validation_20260921')
PLOTS = HERE / 'plots/cross_validation_20260921'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, allow_nan=False) + '\n'
    if path.exists():
        assert json.loads(path.read_text()) == value, f'Refusing to alter frozen plan: {path}'
    else:
        path.write_text(text)


def main():
    original = json.loads(V3_CONFIG.read_text())
    folds, union, patient_test_fold = [], {}, {}
    for fold in range(5):
        manifest = ROOT / f'vista3D/data/folds/fold_{fold}.json'
        directory = RUN / f'fold_{fold}'
        vista = ROOT / f'vista3D/results/fold_{fold}/checkpoints/best_metric_model.pt'
        assert vista.is_file()
        # Audit the initial archived run and the continuation against the same fold.
        configs = [ROOT / f'vista3D/archive/results_patch160_job12377744_20260820/fold_{fold}/configs.yaml']
        continuation = ROOT / f'vista3D/results/fold_{fold}/configs.yaml'
        if continuation.exists():
            configs.append(continuation)
        for path in configs:
            assert Path(next(line.split(':', 1)[1].strip() for line in path.read_text().splitlines() if line.startswith('data_list_file_path:'))).resolve() == manifest.resolve()
        dataset = WholeCTLIDCDataset(load_cases(manifest, 'training'))
        config = {**original, 'manifest': str(manifest), 'vista_checkpoint': str(vista.resolve()),
                  'output_dir': str(directory / 'v3'), 'resume': None,
                  'malignancy_policy': dataset.policy_summary}
        for key in ('runtime', 'source_sha256'):
            config.pop(key, None)
        write(directory / 'planned_v3_config.json', config)
        cohort = build(directory / 'planned_v3_config.json')
        write(directory / 'cohort.json', cohort)
        train = cohort['splits']['training']; test = cohort['splits']['testing']
        for row in train + test:
            if row['key'] in union:
                assert union[row['key']] == row
            union[row['key']] = row
        for patient in {r['patient_id'] for r in test}:
            assert patient not in patient_test_fold, 'Repeated held-out patient'
            patient_test_fold[patient] = fold
        folds.append({'fold': fold, 'directory': str(directory), 'cohort': str(directory / 'cohort.json'),
                      'cohort_sha256': sha256(directory / 'cohort.json'),
                      'manifest_sha256': sha256(manifest), 'vista_checkpoint': str(vista.resolve()),
                      'vista_checkpoint_sha256': sha256(vista),
                      'vista_configs': {str(p): sha256(p) for p in configs},
                      'train_scans': len(train), 'test_scans': len(test),
                      'train_patients': len({r['patient_id'] for r in train}),
                      'test_patients': len({r['patient_id'] for r in test})})
    for item in folds:
        cohort = json.loads(Path(item['cohort']).read_text())
        train = {r['key'] for r in cohort['splits']['training']}
        test = {r['key'] for r in cohort['splits']['testing']}
        assert not train & test and train | test == set(union)
    assert set(patient_test_fold) == {r['patient_id'] for r in union.values()}
    public = HERE / 'deeplung/upstream/detector/dpnmodel/fd0066.ckpt'
    plan = {'schema': 1, 'run': str(RUN), 'plots': str(PLOTS), 'folds': folds,
            'eligible_scans': len(union), 'eligible_patients': len(patient_test_fold),
            'reused_folds': [0], 'new_training_folds': [1, 2, 3, 4],
            'reuse_reason': 'Original fold 0 already trained and evaluated; user requests training only the four remaining folds.',
            'seed': 42, 'epochs': {'v3': 18, 'sybil': 3, 'detector': 5, 'classifier': 700, 'adapt': 105},
            'selection': 'fixed final epochs; no held-out selection or calibration',
            'deeplung_public_checkpoint': str(public), 'deeplung_public_sha256': sha256(public),
            'limitations': ['Public DeepLung LUNA16 pretraining may include held-out LIDC patients.',
                           'Fold 0 was previously inspected; this is not a prospectively untouched five-fold experiment.',
                           'Endpoint is reader-derived current malignancy, not future cancer incidence.']}
    write(RUN / 'case_union.json', [union[k] for k in sorted(union)])
    write(HERE_CV / 'plan.json', plan)
    print(json.dumps(plan, indent=2))


if __name__ == '__main__':
    main()
