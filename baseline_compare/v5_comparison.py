"""Fine-tune baselines and add paired whole-CT ROC curves to the V5 report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from .cohort import HERE, load, sha256

ROOT = HERE.parents[1]
OUTPUT = ROOT / 'back_prop/evaluate/v5_joint_analysis/results_12877415_epoch18'
PLAN = HERE / 'v5_12877415/plan.json'
RUN = Path('/usr/project/rudinlab/datasets/LIDC_IDRI/joint_model/baseline_compare/v5_12877415_20261015')
COHORT = HERE / 'cohort.json'
CACHE = RUN.parent / 'cache_v1'
INITIAL_CLASSIFIER = RUN.parent / 'deeplung/classifier/12622903/last.pt'
POINT = 'volume_lvl/volume_auc_accuracy'
NOTICE = 'DeepLung public LUNA16 pretraining may overlap the LIDC test cohort.'
MODELS = ('v5', 'deeplung', 'sybil')
DISPLAY = {'v5': 'V5', 'deeplung': 'DeepLung', 'sybil': 'Sybil'}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def prepare():
    """Verify actual V5 training identities, cached labels and all 178 test CTs."""
    import torch
    from back_prop.common.base_data import load_cases, _case_key
    from .cohort import build
    cohort = load(COHORT)
    protocol = json.loads((OUTPUT / '_data/protocol_shard_0.json').read_text())
    checkpoint = Path(protocol['checkpoint'])
    assert sha256(checkpoint) == protocol['checkpoint_sha256']
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    keys = {r['key'] for r in cohort['splits']['training']}
    assert keys == set(saved['model']['rashomon._extra_state']['allowed_cases'])
    rebuilt = build(checkpoint.parent / 'config.json')
    assert rebuilt['splits']['training'] == cohort['splits']['training']
    assert rebuilt['manifest_sha256'] == protocol['cohort']['manifest_sha256']
    assert rebuilt['annotations_sha256'] == protocol['cohort']['identities_csv_sha256']
    assert sha256(protocol['cohort']['annotations_csv']) == protocol['cohort']['annotations_csv_sha256']
    cohort_digest = sha256(COHORT)
    valid = 0
    for row in cohort['splits']['training']:
        directory = CACHE / row['key']
        meta = json.loads((directory / 'metadata.json').read_text())
        assert meta['cohort_sha256'] == cohort_digest and meta['key'] == row['key']
        valid += bool(meta['risk_target_valid'])
        for name in ('sybil.npy', 'sybil_attention.npy', 'hu_1mm.npy'):
            assert (directory / name).is_file(), directory / name
    test = []
    for row in load_cases(cohort['manifest'], 'testing'):
        patient, scan = _case_key(row)
        test.append(dict(case_id=f'{patient}__scan{scan}', patient_id=patient, image=row['image']))
    assert len(test) == len(protocol['expected_cases']) == 178
    assert {r['case_id'] for r in test} == set(protocol['expected_cases'])
    assert not ({r['patient_id'] for r in test} & {r['patient_id'] for r in cohort['splits']['training']})
    artifact = json.loads((HERE / 'deeplung/final_artifacts.json').read_text())
    audit = artifact['training_audits']['deeplung_initial_classifier']
    assert sha256(INITIAL_CLASSIFIER) == audit['checkpoint_sha256']
    initial = torch.load(INITIAL_CLASSIFIER, map_location='cpu', weights_only=False, mmap=True)
    assert initial['cohort_sha256'] == sha256(COHORT) and initial['epoch'] == 700
    assert initial['config']['init'] is None and initial['config']['proposals'] is None
    assert json.loads((INITIAL_CLASSIFIER.parent / 'complete.json').read_text())['epochs'] == 700
    plan = dict(schema=1, output=str(OUTPUT), run=str(RUN), cohort=str(COHORT),
                cohort_sha256=sha256(COHORT), cache=str(CACHE), training_scans=len(keys),
                valid_training_scan_labels=valid, training_keys_sha256=cohort['training_case_keys_sha256'],
                test=sorted(test, key=lambda r: r['case_id']), manifest=cohort['manifest'],
                manifest_sha256=cohort['manifest_sha256'], v5_checkpoint=str(checkpoint),
                v5_checkpoint_sha256=protocol['checkpoint_sha256'],
                v5_predictions_sha256=sha256(OUTPUT / POINT / 'predictions.csv'),
                initial_classifier=str(INITIAL_CLASSIFIER),
                initial_classifier_sha256=audit['checkpoint_sha256'],
                sybil_public=str(HERE / 'sybil/weights/28a7cd44f5bcd3e6cc760b65c7e0d54d.ckpt'),
                deeplung_public=str(HERE / 'deeplung/upstream/detector/dpnmodel/fd0066.ckpt'),
                epochs=dict(sybil=3, detector=5, adapt=105),
                selection='Fixed final epochs; no test or overlapping validation selection',
                endpoint='LIDC reader-derived current malignancy, not pathology or future incidence',
                training_target='Same V5 retained-nodule mean >3; mean=3 ignored; ambiguous scan BCE masked',
                test_target='Same any-positive-reader volume labels and three rules as the existing V5 plot',
                pretraining_notice=NOTICE, public_weights_explicitly_authorized=True)
    plan['public_sha256'] = {name: sha256(plan[name + '_public']) for name in ('sybil', 'deeplung')}
    if PLAN.exists():
        assert json.loads(PLAN.read_text()) == plan, 'Refusing to change an existing run plan'
    else:
        write_json(PLAN, plan)
    print(json.dumps(dict(plan=str(PLAN), train=len(keys), valid_scan_labels=valid, test=len(test))))


def read_plan(path):
    plan = json.loads(Path(path).read_text())
    assert sha256(plan['cohort']) == plan['cohort_sha256']
    assert sha256(plan['manifest']) == plan['manifest_sha256']
    assert sha256(Path(plan['output']) / POINT / 'predictions.csv') == plan['v5_predictions_sha256']
    return plan


def train_command(plan, stage, output=None, smoke=False):
    run = Path(plan['run'])
    command = [sys.executable, '-m']
    common = ['--cohort', plan['cohort'], '--cache', plan['cache'], '--output', str(output or run / stage)]
    if stage == 'sybil':
        module = 'sybil.train'
        extra = ['--checkpoint', plan['sybil_public'], '--epochs', str(plan['epochs'][stage])]
    elif stage == 'detector':
        module = 'deeplung.train_detector'
        extra = ['--checkpoint', plan['deeplung_public'], '--epochs', str(plan['epochs'][stage])]
    elif stage == 'proposals':
        module = 'deeplung.prepare_proposals'
        extra = ['--checkpoint', str(run / 'detector/last.pt')]
    elif stage == 'adapt':
        module = 'deeplung.train_classifier'
        extra = ['--init', plan['initial_classifier'], '--proposals', str(run / 'proposals/proposals.json'),
                 '--epochs', str(plan['epochs'][stage])]
    else:
        raise ValueError(stage)
    return command + ['back_prop.baseline_compare.' + module, *common, *extra] + (['--smoke'] if smoke else [])


def completed(plan, stage):
    directory = Path(plan['run']) / stage
    marker = json.loads((directory / 'complete.json').read_text())
    assert marker['cohort_sha256'] == plan['cohort_sha256']
    assert marker['epochs'] == plan['epochs'][stage]
    assert marker.get('training_scans', marker.get('scans')) == plan['training_scans']
    if stage == 'adapt':
        assert marker['gbm_fitted']
    return directory


def stage_train(plan, stage):
    for model in ('sybil', 'deeplung'):
        assert sha256(plan[model + '_public']) == plan['public_sha256'][model]
    assert sha256(plan['initial_classifier']) == plan['initial_classifier_sha256']
    if stage in ('proposals', 'adapt'):
        completed(plan, 'detector')
    if stage == 'adapt':
        proposal = json.loads((Path(plan['run']) / 'proposals/proposals.json').read_text())
        assert not proposal['smoke_subset'] and len(proposal['cases']) == plan['training_scans']
        assert proposal['detector_sha256'] == sha256(Path(plan['run']) / 'detector/last.pt')
    destination = Path(plan['run']) / stage
    if (destination / 'complete.json').exists():
        completed(plan, stage)
        return
    command = train_command(plan, stage)
    if (destination / 'last.pt').exists():
        # Resume only this run's optimizer state; do not replay completed epochs.
        if stage == 'adapt':
            index = command.index('--init')
            del command[index:index + 2]
        command += ['--resume', str(destination / 'last.pt')]
    print(json.dumps(dict(stage=stage, command=command)), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def infer(plan, model):
    import gc
    import nibabel as nib
    import numpy as np
    from scipy.ndimage import zoom
    import torch
    from .predict import sybil_predict, deeplung_predict
    torch.set_num_threads(2)
    run = Path(plan['run'])
    directory = completed(plan, 'sybil' if model == 'sybil' else 'adapt')
    artifacts = {'checkpoint': directory / 'last.pt'}
    if model == 'deeplung':
        artifacts.update(detector_checkpoint=completed(plan, 'detector') / 'last.pt', gbm=directory / 'gbm.pkl')
    digest = {name: sha256(path) for name, path in artifacts.items()}
    target = run / ('inference_' + model)
    target.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(plan['test']):
        path = target / (row['case_id'] + '.json')
        if path.exists():
            previous = json.loads(path.read_text())
            assert previous['artifacts_sha256'] == digest and previous['image'] == row['image']
            continue
        # Only a CT and fitted model artifacts enter the prediction functions.
        image = nib.as_closest_canonical(nib.load(row['image']))
        raw = np.asarray(image.dataobj, dtype=np.float32)
        assert raw.ndim == 3 and np.isfinite(raw).all()
        spacing = nib.affines.voxel_sizes(image.affine)
        if model == 'sybil':
            result = sybil_predict(raw, spacing, artifacts['checkpoint'], torch.device('cuda'))
        else:
            hu = zoom(raw, spacing, order=1, mode='nearest', prefilter=False)
            result = deeplung_predict(hu, artifacts['checkpoint'], artifacts['detector_checkpoint'],
                                      artifacts['gbm'], torch.device('cuda'), 4, 20)
            del hu
        assert np.isfinite(result['risk']) and 0 <= result['risk'] <= 1
        write_json(path, dict(**row, model=model, artifacts_sha256=digest, prediction=result))
        print(json.dumps(dict(model=model, done=i + 1, total=len(plan['test']), risk=result['risk'])), flush=True)
        del raw, image
        gc.collect()
    write_json(target / 'complete.json', dict(model=model, cases=[r['case_id'] for r in plan['test']],
               artifacts_sha256=digest, cohort_sha256=plan['cohort_sha256']))


def paired_rows(volumes, predictions):
    """Never drop a failed/missing model prediction or change label exclusions."""
    import numpy as np
    from back_prop.evaluate.v5_joint_analysis.metrics import MODES
    expected = set(volumes.case_id)
    assert len(volumes) == len(expected) * len(MODES)
    for mode in MODES:
        rows = volumes[volumes['mode'] == mode]
        assert len(rows) == len(expected) and set(rows.case_id) == expected
    for name in MODELS[1:]:
        assert set(predictions[name]) == expected, f'Incomplete {name} test predictions'
        assert all(np.isfinite(x) and 0 <= x <= 1 for x in predictions[name].values())
    result = volumes[['case_id', 'patient_id', 'mode', 'any_reader_target', 'predicted_risk']].copy()
    result = result.rename(columns={'any_reader_target': 'label', 'predicted_risk': 'v5'})
    assert np.isfinite(result.v5).all() and result.v5.between(0, 1).all()
    assert result.groupby('case_id').v5.nunique().eq(1).all()
    for name in MODELS[1:]:
        result[name] = result.case_id.map(predictions[name])
    return result


def draw_comparison(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    from back_prop.evaluate.v5_joint_analysis.metrics import MODES, binary_metrics
    labels = {'exclude_3': 'Exclude 3', 'ge_3': 'Rating ≥ 3', 'gt_3': 'Rating > 3'}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42, 'savefig.dpi': 170})
    fig, axes = plt.subplots(3, 1, figsize=(8.3, 11.2))
    metrics = []
    for ax, mode in zip(axes, MODES):
        group = rows[rows['mode'] == mode]
        included = group[group.label.notna()]
        for name, color in zip(MODELS, ('#2563a6', '#d55e00', '#009e73')):
            m = binary_metrics([None if x != x else x for x in group.label], group[name].tolist())
            metrics.append(dict(model=name, mode=mode, **m))
            if m['auc'] is not None:
                fpr, tpr, _ = roc_curve(included.label, included[name])
                ax.plot(fpr, tpr, color=color, lw=1.8, label=f"{DISPLAY[name]} (AUC = {m['auc']:.3f})")
        ax.plot([0, 1], [0, 1], ls='--', color='.65', lw=.8)
        ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel='False positive rate', ylabel='True positive rate',
               title=f"{labels[mode]} · n = {len(included)}")
        ax.legend(frameon=False, loc='lower right')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=.15)
    fig.suptitle('Volume-level malignancy: V5 and fine-tuned baselines', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .97))
    return fig, metrics


def refresh_plot(output):
    """Called by the normal report writer too, so it retains baseline curves."""
    import pandas as pd
    import matplotlib.pyplot as plt
    folder = Path(output) / POINT
    path = folder / 'baseline_predictions.csv'
    if not path.exists():
        return False
    rows = pd.read_csv(path)
    volumes = pd.read_csv(folder / 'predictions.csv')
    predictions = {name: rows.drop_duplicates('case_id').set_index('case_id')[name].to_dict() for name in MODELS[1:]}
    expected = paired_rows(volumes, predictions)
    pd.testing.assert_frame_equal(rows, expected, check_dtype=False, rtol=1e-12, atol=1e-12)
    figure, metrics = draw_comparison(rows)
    for extension in ('png', 'pdf'):
        temp = folder / ('malignancy_roc.tmp.' + extension)
        figure.savefig(temp, bbox_inches='tight')
        temp.replace(folder / ('malignancy_roc.' + extension))
    plt.close(figure)
    pd.DataFrame(metrics).to_csv(folder / 'baseline_metrics.csv', index=False)
    return True


def render(plan, *, mark_run_complete=True):
    import pandas as pd
    predictions = {}
    artifacts = {}
    for name in MODELS[1:]:
        directory = Path(plan['run']) / ('inference_' + name)
        done = json.loads((directory / 'complete.json').read_text())
        assert done['cohort_sha256'] == plan['cohort_sha256']
        assert set(done['cases']) == {r['case_id'] for r in plan['test']}
        predictions[name] = {}
        for row in plan['test']:
            item = json.loads((directory / (row['case_id'] + '.json')).read_text())
            assert item['case_id'] == row['case_id'] and item['patient_id'] == row['patient_id']
            assert item['artifacts_sha256'] == done['artifacts_sha256']
            predictions[name][row['case_id']] = item['prediction']['risk']
        artifacts[name] = done['artifacts_sha256']
    folder = Path(plan['output']) / POINT
    rows = paired_rows(pd.read_csv(folder / 'predictions.csv'), predictions)
    temporary = folder / 'baseline_predictions.csv.tmp'
    rows.to_csv(temporary, index=False)
    temporary.replace(folder / 'baseline_predictions.csv')
    assert refresh_plot(plan['output'])
    write_json(folder / 'baseline_protocol.json', dict(**plan, trained_artifacts_sha256=artifacts))
    readme = folder / 'README.md'
    note = ('\n## Fine-tuned baselines\n\nV5, DeepLung and Sybil use identical test CTs and reader-derived labels.'
            '\n\n[Paired predictions](baseline_predictions.csv) · [AUC/accuracy](baseline_metrics.csv)'
            ' · [Training protocol](baseline_protocol.json)\n')
    text = readme.read_text().split('\n## Fine-tuned baselines')[0]
    readme.write_text(text + note)
    if mark_run_complete:
        write_json(Path(plan['run']) / 'complete.json', dict(test_cts=len(plan['test']), updated=str(folder / 'malignancy_roc.png')))


def compare_existing(plan, output):
    """Pair completed baseline predictions with another V5 checkpoint evaluation."""
    import pandas as pd
    import torch
    output = Path(output)
    protocol = json.loads((output / '_data/protocol_shard_0.json').read_text())
    assert protocol['cohort']['manifest_sha256'] == plan['manifest_sha256']
    expected = {row['case_id'] for row in plan['test']}
    assert set(protocol['expected_cases']) == expected
    finished = []
    for shard in range(protocol['shards']):
        done = json.loads((output / '_data' / f'complete_shard_{shard}.json').read_text())
        assert done['fingerprint'] == protocol['fingerprint'] and done['bank_unchanged']
        finished.extend(done['cases'])
    assert len(finished) == len(expected) and set(finished) == expected
    old = pd.read_csv(Path(plan['output']) / POINT / 'predictions.csv')
    new = pd.read_csv(output / POINT / 'predictions.csv')
    columns = ['case_id', 'patient_id', 'mode', 'any_reader_target']
    pd.testing.assert_frame_equal(old[columns].sort_values(columns[:3]).reset_index(drop=True),
                                  new[columns].sort_values(columns[:3]).reset_index(drop=True))
    assert sha256(protocol['checkpoint']) == protocol['checkpoint_sha256']
    checkpoint = torch.load(protocol['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    cohort = load(plan['cohort'])
    assert set(checkpoint['model']['rashomon._extra_state']['allowed_cases']) == {
        row['key'] for row in cohort['splits']['training']}
    target = dict(plan, output=str(output.resolve()), v5_checkpoint=protocol['checkpoint'],
                  v5_checkpoint_sha256=protocol['checkpoint_sha256'],
                  v5_predictions_sha256=sha256(output / POINT / 'predictions.csv'))
    render(target, mark_run_complete=False)


def smoke(plan):
    """Exercise public-weight backward passes and the existing classifier adapter."""
    Path(plan['run']).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='smoke_', dir=plan['run']) as tmp:
        for name in ('sybil', 'detector', 'adapt'):
            command = train_command(plan, name, Path(tmp) / name, smoke=True)
            if name == 'adapt':
                index = command.index('--proposals')
                command[index + 1] = str(RUN.parent / 'deeplung/proposals/12622703/proposals.json')
            subprocess.run(command, cwd=ROOT, check=True)
    print('GPU backward/checkpoint smoke passed; temporary weights removed.', flush=True)


def submit(path):
    plan = read_plan(path)
    run = Path(plan['run'])
    (run / 'logs').mkdir(parents=True, exist_ok=True)
    jobs_path = path.parent / 'jobs.json'
    if jobs_path.exists():
        raise FileExistsError('Jobs already submitted: ' + str(jobs_path))
    jobs = {}
    stages = [('sybil', []), ('detector', []), ('proposals', ['detector']), ('adapt', ['proposals']),
              ('infer_sybil', ['sybil']), ('infer_deeplung', ['adapt']), ('render', ['infer_sybil', 'infer_deeplung'])]
    for name, dependencies in stages:
        cpu = name == 'render'
        command = ['sbatch', '--parsable', '--job-name=v5base-' + name, '--account=rudin',
                   '--partition=' + ('compsci' if cpu else 'rudin'), '--cpus-per-task=4',
                   '--mem=' + ('8G' if cpu else '64G'), '--time=' + ('00:30:00' if cpu else '2-00:00:00'),
                   '--output=' + str(run / 'logs' / (name + '_%j.out')),
                   '--error=' + str(run / 'logs' / (name + '_%j.err')), '--kill-on-invalid-dep=yes']
        if not cpu:
            command += ['--gres=gpu:6000_ada:1']
        if dependencies:
            command += ['--dependency=afterok:' + ':'.join(str(jobs[k]['job_id']) for k in dependencies)]
        command += [str(HERE / 'v5_comparison.sbatch'), name, '--plan', str(path)]
        result = subprocess.check_output(command, text=True).strip()
        job_id = int(result.split(';')[0])
        jobs[name] = dict(job_id=job_id, dependencies=dependencies, command=command)
        write_json(jobs_path, dict(plan=str(path), jobs=jobs, submission_complete=False))
    write_json(jobs_path, dict(plan=str(path), jobs=jobs, submission_complete=True))
    print(json.dumps(jobs, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'smoke', 'submit', 'compare', 'sybil', 'detector', 'proposals',
                                          'adapt', 'infer_sybil', 'infer_deeplung', 'render'])
    parser.add_argument('--plan', type=Path, default=PLAN)
    parser.add_argument('--output', type=Path, help='Completed V5 evaluation directory for compare')
    args = parser.parse_args()
    if args.stage == 'compare' and args.output is None:
        parser.error('compare requires --output')
    if args.stage == 'prepare':
        prepare()
    elif args.stage == 'submit':
        submit(args.plan)
    else:
        plan = read_plan(args.plan)
        if args.stage == 'smoke':
            smoke(plan)
        elif args.stage == 'compare':
            compare_existing(plan, args.output)
        elif args.stage == 'render':
            render(plan)
        elif args.stage.startswith('infer_'):
            infer(plan, args.stage.removeprefix('infer_'))
        else:
            stage_train(plan, args.stage)


if __name__ == '__main__':
    main()
