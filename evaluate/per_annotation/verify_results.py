"""Independent full-cohort audit: official labels, rank AUC, checkpoint risk math.

Run after the GPU job: python -m back_prop.evaluate.per_annotation.verify_results
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata
import torch

from back_prop.common.features import SEMANTIC_NAMES,RADIOMICS_NAMES
from .data import build_cohort,sha256


def rank_auc(labels,scores):
    y,s=np.asarray(labels,float),np.asarray(scores,float)
    finite=np.isfinite(y)&np.isfinite(s);y,s=y[finite],s[finite]
    pos=y==1;n1,n0=int(pos.sum()),int((y==0).sum())
    if not n1 or not n0:return np.nan
    return (rankdata(s)[pos].sum()-n1*(n1+1)/2)/(n1*n0)


def verify(directory):
    directory=Path(directory)
    meta=json.loads((directory/'provenance.json').read_text())
    assert meta['status']=='complete' and meta['full_test_set']
    assert meta['protocol']['semantic_threshold_rule']=='ge'
    assert meta['protocol']['malignancy_semantic_input']=='official'
    cohort,info=build_cohort(meta['cohort']['manifest'])
    official=pd.DataFrame([a for c in cohort for a in c['annotations']]).set_index('annotation_key')
    predictions=pd.read_csv(directory/'predictions.csv')
    semantic=pd.read_csv(directory/'semantic_threshold_auc.csv')
    malignancy=pd.read_csv(directory/'malignancy_auc.csv')
    assert len(cohort)==meta['evaluated_cases']
    assert len(predictions)==2*len(official)==meta['prediction_rows']
    assert len(semantic)==2*6*5 and len(malignancy)==2
    results={}
    for name in ('V4','V4oversample'):
        m=meta['models'][name]
        assert m['epoch']==18 and m['bank_unchanged'] and m['test_patient_overlap']==0
        assert sha256(m['checkpoint'])==m['checkpoint_sha256']
        p=predictions[predictions.model==name].set_index('annotation_key')
        assert p.index.is_unique and set(p.index)==set(official.index)
        p=p.loc[official.index]
        for field in ['case_id','nodule_id','annotation_id','mask_path']+[
                'gt_'+f for f in (*SEMANTIC_NAMES,'malignancy')]:
            np.testing.assert_array_equal(p[field].to_numpy(),official[field].to_numpy())
        np.testing.assert_allclose(p.malignancy_target,official.malignancy_target,equal_nan=True)
        assert (p.input_status=='ok').all(), 'Some annotations lack a valid mask; inspect before declaring complete'
        assert (p.mask_voxels>0).all() and (p.native_mask_voxels>0).all()
        for feature in SEMANTIC_NAMES:
            prob=p[[f'{feature}_p{k}' for k in range(1,6)]].to_numpy()
            np.testing.assert_allclose(prob.sum(1),1,atol=1e-6)
            assert np.isfinite(prob).all() and (prob>=0).all()
            np.testing.assert_allclose(prob@np.arange(1,6),p['pred_'+feature],atol=1e-6)
            for t in range(1,6):
                row=semantic[(semantic.model==name)&(semantic.feature==feature)&(semantic.threshold==t)].iloc[0]
                truth=(official['gt_'+feature]>=t).to_numpy()
                expected=rank_auc(truth,prob[:,t-1:].sum(1))
                np.testing.assert_allclose(row.auc,expected,atol=1e-12,equal_nan=True)
                assert row.n_positive==int(truth.sum()) and row.n_negative==int((~truth).sum())
                assert row.n_evaluated==len(p) and row.n_excluded==0
        state=torch.load(m['checkpoint'],map_location='cpu',weights_only=False,mmap=True)['model']
        def array(key):return state['rashomon.'+key].numpy()
        r=p[['radiomics_'+f for f in RADIOMICS_NAMES]].to_numpy(dtype=np.float32)
        s=official[['gt_'+f for f in SEMANTIC_NAMES]].to_numpy(dtype=np.float32)
        baseline=(np.clip((r-array('radiomics_mean'))/array('radiomics_scale'),-10,10)*array('radiomics_varying'))@array('radiomics_weights')+array('radiomics_intercept')
        normalized=np.clip((s-array('mean'))/array('scale'),-10,10)*array('varying')
        k=int(array('count'))
        z=baseline[:,None]+normalized@array('weights')[:k].T+array('intercepts')[:k]
        expected=expit(z).mean(1).clip(1e-6,1-1e-6)
        actual=p.malignancy_probability_official.to_numpy()
        np.testing.assert_allclose(actual,expected,atol=3e-6,rtol=3e-6)
        row=malignancy[malignancy.model==name].iloc[0]
        np.testing.assert_allclose(row.auc,rank_auc(official.malignancy_target,actual),atol=1e-12)
        assert row.n_positive==int((official.gt_malignancy>=4).sum())
        assert row.n_negative==int((official.gt_malignancy<=2).sum())
        assert row.n_indeterminate==int((official.gt_malignancy==3).sum())
        results[name]=dict(annotations=len(p),risk_auc=float(row.auc),
            risk_probability_reconstruction_max_error=float(abs(actual-expected).max()),
            semantic_auc_checks=30,all_official_labels_match=True)
    required=['semantic_threshold_auc.png','semantic_threshold_auc.pdf','semantic_roc.pdf',
              'malignancy_roc.png','malignancy_roc.pdf','report.html','summary.json']
    assert all((directory/p).is_file() and (directory/p).stat().st_size>0 for p in required)
    results=dict(passed=True,cohort=info,models=results,
                 verification='official annotation join, all 60 rank-based AUCs, frozen-checkpoint risk probabilities, label counts, output files')
    (directory/'verification.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    print(json.dumps(results['models'],indent=2))
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',nargs='?',type=Path,default=Path(__file__).parent/'results')
    verify(parser.parse_args().directory)
