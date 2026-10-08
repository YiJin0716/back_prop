"""Overlay the corresponding segmentation-only fold-0 VISTA3D on the V3 example."""
import argparse
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

from back_prop.evaluate.legacy_visualization.visualize_test_ct import ROOT, DEFAULT_OUTPUT, digest, load_result, mesh, overlap_metrics
from back_prop.common.source_compat import source_matches

VISTA_CHECKPOINT = Path('/usr/project/rudinlab/datasets/LIDC_IDRI/vista3D_workspace/results/fold_0/checkpoints/best_metric_model.pt')
FOLD = ROOT / 'vista3D/data/folds/fold_0.json'
MANIFEST = ROOT / 'vista3D/data/folds/fold_0_manifest.json'
V3_COHORT = ROOT / 'back_prop/baseline_compare/cohort.json'
INFER_CONFIG = ROOT / 'vista3D/configs/infer_lidc.yaml'
TRAIN_CONFIG = ROOT / 'vista3D/results/fold_0/configs.yaml'


def audit_split(case_key):
    cohort = json.loads(V3_COHORT.read_text())
    config_path = Path(cohort['v3_config'])
    config = json.loads(config_path.read_text())
    assert digest(config_path) == cohort['v3_config_sha256']
    assert Path(config['manifest']).resolve() == FOLD.resolve()
    assert digest(FOLD) == cohort['manifest_sha256']
    assert Path(config['vista_checkpoint']).resolve() == VISTA_CHECKPOINT.resolve()
    fold, manifest = json.loads(FOLD.read_text()), json.loads(MANIFEST.read_text())
    key = lambda r: Path(r['image']).name.removesuffix('.nii.gz')
    v3_train = {key(r) for r in cohort['splits']['training']}
    vista_train = {key(r) for r in fold['training']}
    assert v3_train <= vista_train
    assert case_key in {key(r) for r in cohort['splits']['testing']}
    assert case_key in {key(r) for r in fold['testing']}
    patient = case_key.split('__scan')[0]
    assert patient in manifest['held_out_patients'] and patient not in manifest['train_patients']
    assert patient not in {r['patient_id'] for r in fold['training'] + fold['validation']}
    assert not ({r['patient_id'] for r in cohort['splits']['testing']} &
                {r['patient_id'] for r in fold['training'] + fold['validation']})
    return dict(fold=0, same_source_split=True, exact_training_cases_equal=v3_train == vista_train,
                v3_training_cts=len(v3_train), vista_training_cts=len(vista_train),
                common_training_cts=len(v3_train & vista_train),
                vista_only_training_cases=sorted(vista_train-v3_train),
                current_case_held_out_from_both=True, all_v3_test_patients_held_out=True,
                note='Same fold-0 patient partition; V3 retains 638/702 CTs after malignancy-label filtering. Not an identical-training-cohort ablation.')


def infer(output=DEFAULT_OUTPUT):
    import torch
    from scripts.infer import InferClass
    output = Path(output)
    v3, _, official, affine = load_result(output)
    audit = audit_split(v3['case_key'])
    if not torch.cuda.is_available():
        raise RuntimeError('Allocate a CUDA GPU to generate a new standalone VISTA3D prediction.')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    runtime = output / 'standalone_vista_runtime'
    runtime.mkdir(exist_ok=True)
    inferer = InferClass(config_file=str(INFER_CONFIG), bundle_root=str(runtime), fold=0,
                        **{'infer#ckpt_name': str(VISTA_CHECKPOINT),
                           'infer#output_path': str(runtime),
                           'infer#log_output_file': str(runtime/'inference.log')})
    prediction = inferer.infer(image_file=v3['image'], label_prompt=[23], save_mask=False)
    binary = np.asarray(prediction.detach().cpu()).squeeze() > 0
    source = nib.load(v3['image'])
    assert binary.shape == source.shape
    native_path = output/'vista_seg_only_native.nii.gz'
    nib.save(nib.Nifti1Image(binary.astype(np.uint8), source.affine), native_path)
    # Follow the same canonical orientation + nearest-neighbour 1-mm zoom as V3 GT.
    canonical = nib.as_closest_canonical(nib.Nifti1Image(binary.astype(np.uint8), source.affine))
    aligned = ndimage.zoom(np.asarray(canonical.dataobj), nib.affines.voxel_sizes(canonical.affine),
                           order=0, mode='nearest', prefilter=False) > 0
    assert aligned.shape == official.shape
    path = output/'vista_seg_only.nii.gz'
    nii = nib.Nifti1Image(aligned.astype(np.uint8), affine)
    nii.header.set_xyzt_units('mm')
    nib.save(nii, path)
    sources = [VISTA_CHECKPOINT, FOLD, MANIFEST, V3_COHORT, INFER_CONFIG, TRAIN_CONFIG,
               Path(__file__), ROOT/'vista3D/VISTA/vista3d/scripts/infer.py',
               Path(v3['image'])]
    report = dict(case_key=v3['case_key'], checkpoint=str(VISTA_CHECKPOINT),
                  training_objective='segmentation only (DiceCELoss)', split_audit=audit,
                  source_sha256={str(p): digest(p) for p in sources},
                  artifact_sha256={p.name: digest(p) for p in (path, native_path)},
                  reference_v3_checkpoint_sha256=v3['checkpoint_sha256'],
                  inference='original standalone InferClass; class prompt 23; no annotation/point/box prompt',
                  display_alignment='native mask -> canonical RAS -> V3 nearest-neighbour 1-mm grid',
                  all_nodule_metrics=overlap_metrics(aligned, official),
                  ground_truth_used_for_prediction=False, slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    (output/'vista_seg_only_metadata.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


def load_vista(output=DEFAULT_OUTPUT):
    output = Path(output)
    report = json.loads((output/'vista_seg_only_metadata.json').read_text())
    assert report['split_audit'] == audit_split(report['case_key'])
    for path, sha in report['source_sha256'].items():
        assert source_matches(path, sha), f'Changed standalone VISTA source: {path}'
    for name, sha in report['artifact_sha256'].items():
        assert digest(output/name) == sha, f'Changed standalone VISTA artifact: {name}'
    reference, _, _, affine = load_result(output)
    assert report['case_key'] == reference['case_key']
    assert report['reference_v3_checkpoint_sha256'] == reference['checkpoint_sha256']
    nii = nib.load(output/'vista_seg_only.nii.gz')
    assert list(nii.shape) == reference['shape'] and np.allclose(nii.affine, affine)
    return report, np.asarray(nii.dataobj).astype(bool)


def render_comparison(output=DEFAULT_OUTPUT):
    import plotly.graph_objects as go
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    output = Path(output)
    v3, prediction, official, affine = load_result(output)
    vista, standalone = load_vista(output)
    surfaces = [(mesh(official, affine), 'Official consensus', 'blue', .4),
                (mesh(prediction, affine), 'V3 final segmentation', 'crimson', .5),
                (mesh(standalone, affine), 'VISTA3D segmentation-only (fold 0)', 'forestgreen', .45)]
    traces = []
    for geometry, name, color, opacity in surfaces:
        if geometry is None:
            traces.append(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers',
                                      marker=dict(color=color), name=name+' (empty mask)'))
            continue
        vertices, faces = geometry
        traces.append(go.Mesh3d(x=vertices[:,0], y=vertices[:,1], z=vertices[:,2],
                               i=faces[:,0], j=faces[:,1], k=faces[:,2], color=color,
                               opacity=opacity, name=name, showlegend=True))
    title = f"V3 vs standalone VISTA3D vs Official — {v3['case_key']}"
    note = 'Same fold-0 split; training CTs: V3=638, segmentation-only VISTA3D=702'
    fig = go.Figure(traces)
    fig.update_layout(title=title+'<br><sup>'+note+'</sup>',
                      scene=dict(xaxis_title='R (mm)', yaxis_title='A (mm)', zaxis_title='S (mm)', aspectmode='data'),
                      width=1000, height=750, legend=dict(itemsizing='constant'))
    fig.write_html(output/'v3_vista_vs_official_3d.html', include_plotlyjs=True)
    static = plt.figure(figsize=(11, 8))
    ax = static.add_subplot(projection='3d')
    all_vertices = []
    for geometry, name, color, opacity in surfaces:
        if geometry is not None:
            vertices, faces = geometry
            ax.add_collection3d(Poly3DCollection(vertices[faces], facecolor=color, alpha=opacity, edgecolor='none'))
            all_vertices.append(vertices)
    combined = np.concatenate(all_vertices)
    lower, upper = combined.min(0), combined.max(0)
    for setter, lo, hi in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), lower, upper):
        setter(lo-2, hi+2)
    ax.set_box_aspect(np.maximum(upper-lower, 1))
    ax.set(xlabel='R (mm)', ylabel='A (mm)', zlabel='S (mm)')
    ax.legend(handles=[Patch(color=color, label=name, alpha=opacity)
                       for _, name, color, opacity in surfaces], loc='upper left', fontsize=9)
    static.suptitle(title+'\n'+note, fontsize=12)
    static.tight_layout()
    static.savefig(output/'v3_vista_vs_official_3d.png', dpi=170)
    plt.close(static)
    return fig, vista


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    if args.render_only:
        render_comparison(args.output)
    else:
        infer(args.output)
