"""Audit labels, checkpoint provenance, and cached cohort semantic variation."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[5]
DEST=Path(__file__).resolve().parent
NAMES=('lobulation','margin','sphericity','spiculation','subtlety','texture')


def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def main():
    base=ROOT/'back_prop/model_v3_hard/compare'
    manifest=json.loads((base/'semantic_roc/gt_localized/manifest_0.json').read_text())
    checkpoint=manifest['weights']['v3']
    actual_sha=sha256(Path(checkpoint['path']))
    assert actual_sha==checkpoint['sha256']
    values=[];truth=[];fingerprints=set();case_count=0
    for path in sorted((base/'semantic_roc/gt_localized/cases').glob('*.json')):
        row=json.loads(path.read_text());fingerprints.add(row['fingerprint']);case_count+=1
        values.extend(row['models']['v3']['semantic_means'])
        gt=json.loads((base/'semantic_roc/ground_truth'/path.name).read_text())
        by_id={n['nodule_id']:n for n in gt['nodules']}
        truth.extend([by_id[n]['ratings'] for n in row['nodule_ids']])
    assert fingerprints=={manifest['fingerprint']}
    x=np.asarray(values);deviation=np.max(abs(x-np.median(x,axis=0)),axis=1)
    cohort=dict(cases=case_count,nodules=len(x),mode='GT-box-localized; model-predicted masks',
        within_1e_3_of_componentwise_median=int((deviation<1e-3).sum()),
        within_1e_2_of_componentwise_median=int((deviation<1e-2).sum()),
        features={name:dict(min=float(x[:,j].min()),max=float(x[:,j].max()),
            std=float(x[:,j].std()),median=float(np.median(x[:,j])),
            p1_p99=np.percentile(x[:,j],[1,99]).tolist()) for j,name in enumerate(NAMES)})
    for name in ('texture','subtlety','spiculation'):
        y=np.asarray([n[name] for n in truth]);j=NAMES.index(name)
        cohort['features'][name].update(gt_std=float(y.std()),
            prediction_mae=float(abs(x[:,j]-y).mean()))

    # Training labels must equal official reader ratings, rather than constants
    # accidentally copied by the physical-nodule grouping CSV.
    keys=['patient_id','scan_index','annotation_index']
    official=pd.read_csv(ROOT/'all_ct_annotations.csv',usecols=keys+list(NAMES)+['malignancy']).dropna(subset=keys)
    identity=pd.read_csv(ROOT/'nodule_iden.csv',usecols=keys+list(NAMES)+['malignancy','nodule_id']).dropna(subset=keys+['nodule_id'])
    assert not official.duplicated(keys).any()
    joined=identity.merge(official,on=keys,suffixes=('_training','_official'),validate='many_to_one',how='left',indicator=True)
    label_audit=dict(identity_rows=len(identity),missing_official_rows=int((joined['_merge']!='both').sum()),
        mismatches={name:int((joined[name+'_training']!=joined[name+'_official']).sum()) for name in (*NAMES,'malignancy')})
    assert label_audit['missing_official_rows']==0 and not any(label_audit['mismatches'].values())
    fold=json.loads((ROOT/'vista3D/data/folds/fold_0.json').read_text())
    training={(r['patient_id'],int(r['scan_id'])) for r in fold['training']}
    kept=identity[[ (p,int(s)) in training for p,s in zip(identity.patient_id,identity.scan_index) ]]
    means=kept.groupby(['patient_id','scan_index','nodule_id'])[[*NAMES,'malignancy']].mean()
    means=means[~np.isclose(means.malignancy,3)]
    label_audit['training_physical_nodules_before_empty_mask_filter']=len(means)
    label_audit['training_semantic_mean']=means[list(NAMES)].mean().to_dict()
    label_audit['training_semantic_std']=means[list(NAMES)].std(ddof=0).to_dict()
    metric_path=Path(checkpoint['path']).parent/'metrics.jsonl'
    metrics=[json.loads(line) for line in metric_path.read_text().splitlines()]
    result=dict(checkpoint=checkpoint['path'],checkpoint_sha256=actual_sha,
        cohort_manifest=str(base/'semantic_roc/gt_localized/manifest_0.json'),cohort=cohort,
        labels=label_audit,training_ordinal_loss=[dict(epoch=r['epoch'],ordinal=r['ordinal']) for r in metrics],
        source_sha256=sha256(Path(__file__)))
    (DEST/'cohort_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
