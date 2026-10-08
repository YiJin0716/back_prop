"""Generate CSV, PNG/PDF plots and an HTML report from saved annotation rows."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from back_prop.evaluate.eval_package.types import SEMANTIC_NAMES
from .metrics import binary_auc, threshold_auc

MODEL_NAMES=('V4','V4oversample')
COLORS={'V4':'#2563eb','V4oversample':'#dc6b24'}


def save_figure(fig,path):
    for suffix in ('png','pdf'):
        fig.savefig(path.with_suffix('.'+suffix),dpi=180,bbox_inches='tight')
    plt.close(fig)


def generate_report(directory):
    directory=Path(directory)
    frame=pd.read_csv(directory/'predictions.csv')
    audit=json.loads((directory/'provenance.json').read_text())
    if set(frame.model)!=set(MODEL_NAMES):
        raise ValueError('Both V4 and V4oversample must be present')
    expected=set(pd.read_csv(directory/'annotations.csv').annotation_key)
    for model in MODEL_NAMES:
        part=frame[frame.model==model]
        if set(part.annotation_key)!=expected or part.annotation_key.duplicated().any():
            raise ValueError(f'{model}: annotation cohort missing or duplicated')
    rule=audit['protocol']['semantic_threshold_rule']
    operator='≥' if rule=='ge' else '>'
    semantic_rows,curves,roc_records=[],{},[]
    for model in MODEL_NAMES:
        data=frame[frame.model==model]
        for feature in SEMANTIC_NAMES:
            probability=data[[f'{feature}_p{k}' for k in range(1,6)]].to_numpy()
            for threshold in range(1,6):
                stat,curve=threshold_auc(data['gt_'+feature],probability,threshold,rule)
                semantic_rows.append(dict(model=model,feature=feature,threshold=threshold,
                                          positive_rule=f'rating {operator} {threshold}',**stat))
                curves[model,feature,threshold]=curve
                if curve is not None:
                    roc_records.extend(dict(model=model,feature=feature,rating_threshold=threshold,
                        fpr=float(f),tpr=float(t),score_threshold=float(s))
                        for f,t,s in zip(*curve))
    semantic=pd.DataFrame(semantic_rows)
    semantic.to_csv(directory/'semantic_threshold_auc.csv',index=False)
    table=semantic.pivot(index=['model','feature'],columns='threshold',values='auc')
    table.to_csv(directory/'semantic_auc_table.csv')
    pd.DataFrame(roc_records).to_csv(directory/'semantic_roc_points.csv',index=False)

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.5),layout='constrained')
    for ax,model in zip(axes,MODEL_NAMES):
        values=table.loc[model].reindex(SEMANTIC_NAMES).to_numpy()
        cmap=plt.get_cmap('YlGnBu').copy();cmap.set_bad('#eeeeee')
        im=ax.imshow(np.ma.masked_invalid(values),vmin=0,vmax=1,cmap=cmap,aspect='auto')
        ax.set_xticks(range(5),[str(i) for i in range(1,6)])
        ax.set_yticks(range(6),[x.capitalize() for x in SEMANTIC_NAMES])
        ax.set_title(model);ax.set_xlabel(f'Positive label: official rating {operator} threshold')
        for i in range(6):
            for j in range(5):
                value=values[i,j]
                ax.text(j,i,'N/A' if not np.isfinite(value) else f'{value:.3f}',ha='center',va='center',
                        color='white' if np.isfinite(value) and value>.65 else '#222222')
    fig.colorbar(im,ax=axes,label='ROC AUC',shrink=.8)
    fig.suptitle('Semantic extractor · individual official reader masks')
    save_figure(fig,directory/'semantic_threshold_auc')
    # Each feature gets a separate page so all five requested thresholds,
    # including the single-class endpoint, remain visible and legible.
    with PdfPages(directory/'semantic_roc.pdf') as pdf:
        for feature in SEMANTIC_NAMES:
            fig,axes=plt.subplots(2,3,figsize=(11,7),layout='constrained')
            for threshold,ax in zip(range(1,6),axes.flat):
                for model in MODEL_NAMES:
                    curve=curves[model,feature,threshold]
                    score=semantic.loc[(semantic.model==model)&(semantic.feature==feature)&
                                       (semantic.threshold==threshold),'auc'].iloc[0]
                    if curve is not None:
                        ax.plot(curve[0],curve[1],color=COLORS[model],label=f'{model}: {score:.3f}')
                ax.plot([0,1],[0,1],'--',color='.7',lw=1)
                ax.set(xlim=(0,1),ylim=(0,1.02),xlabel='False positive rate',ylabel='True positive rate',
                       title=f'Official rating {operator} {threshold}')
                if ax.get_legend_handles_labels()[0]:ax.legend(loc='lower right',fontsize=9)
                else:ax.text(.5,.5,'AUC undefined\n(single target class)',ha='center',va='center')
            axes.flat[-1].axis('off')
            axes.flat[-1].text(0,.8,'Unit: one reader annotation\nInput: that reader’s official mask\nScore: cumulative class probability',va='top')
            fig.suptitle(feature.capitalize());pdf.savefig(fig);plt.close(fig)

    sources=audit['protocol']['malignancy_semantic_input']
    sources=('official','predicted') if sources=='both' else (sources,)
    risk_rows,risk_points=[],[]
    fig,axes=plt.subplots(1,len(sources),figsize=(6.3*len(sources),5.5),squeeze=False,layout='constrained')
    for ax,source in zip(axes.flat,sources):
        for model in MODEL_NAMES:
            data=frame[frame.model==model]
            stat,curve=binary_auc(data.malignancy_target,data[f'malignancy_probability_{source}'])
            stat.update(model=model,semantic_input=source,n_indeterminate=int((data.gt_malignancy==3).sum()),
                        n_invalid_input=int((data.input_status!='ok').sum()))
            risk_rows.append(stat)
            if curve is not None:
                ax.plot(curve[0],curve[1],color=COLORS[model],lw=2,label=f'{model}: AUC = {stat["auc"]:.4f}')
                risk_points.extend(dict(model=model,semantic_input=source,fpr=float(f),tpr=float(t),
                    score_threshold=float(s)) for f,t,s in zip(*curve))
        ax.plot([0,1],[0,1],'--',color='.6',lw=1,label='Chance')
        ax.set(xlim=(0,1),ylim=(0,1.02),xlabel='False positive rate',ylabel='True positive rate',
               title=f'Official masks + {source} semantic features')
        ax.legend(loc='lower right');ax.set_aspect('equal')
    fig.suptitle('Per-annotation nodule malignancy ROC\nReader ratings 1–2 vs 4–5; rating 3 excluded')
    save_figure(fig,directory/'malignancy_roc')
    risk=pd.DataFrame(risk_rows)
    risk.to_csv(directory/'malignancy_auc.csv',index=False)
    pd.DataFrame(risk_points).to_csv(directory/'malignancy_roc_points.csv',index=False)
    summary=dict(full_test_set=audit['full_test_set'],cases=audit['evaluated_cases'],
                 annotations=audit['expected_annotations'],prediction_rows=len(frame),
                 semantic_auc=semantic.where(pd.notna(semantic),None).astype(object).to_dict('records'),
                 malignancy_auc=risk.astype(object).where(pd.notna(risk),None).to_dict('records'))
    # Convert NaN through explicit record iteration; JSON never encodes a fake AUC.
    for row in summary['semantic_auc']:
        if row['auc'] is not None and not np.isfinite(row['auc']):row['auc']=None
    (directory/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    scope='FULL TEST SET' if audit['full_test_set'] else 'SMOKE SUBSET — NOT FULL TEST SET'
    protocol=audit['protocol']
    overview=[f'<li><b>{html.escape(k)}</b>: {html.escape(str(v))}</li>' for k,v in protocol.items()
              if k not in ('source_sha256','semantic_names','radiomics_names')]
    checkpoints=''.join(f'<li>{html.escape(name)}: {html.escape(meta["checkpoint"])} '
                        f'(epoch {meta["epoch"]})</li>' for name,meta in audit['models'].items())
    content=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>V4 per-annotation test analysis</title>
<style>body{{font:16px/1.6 Georgia,serif;max-width:1150px;margin:40px auto;padding:0 24px;color:#222}}
table{{border-collapse:collapse;width:100%;font:14px/1.5 system-ui,sans-serif;margin:20px 0}}
th,td{{border:0;border-bottom:1px solid #ddd;padding:8px;text-align:right}}thead{{border-top:2px solid #333;border-bottom:2px solid #333}}
img{{max-width:100%}}li{{overflow-wrap:anywhere}}h1,h2{{font-weight:600}}.scope{{color:#444}}</style>
<h1>V4 and V4oversample: per-annotation test analysis</h1>
<p class="scope">{scope} · {audit['evaluated_cases']} CT scans · {audit['expected_annotations']} reader annotations</p>
<p>Each annotation uses its own official mask and six ratings. Readers are not averaged and masks are not merged.
All trained weights and risk banks are frozen. This is an evaluation of the diagnostic components with official inputs.</p>
<h2>Semantic feature extractor</h2><p>For every feature and threshold t = 1, 2, 3, 4, 5, the positive target is
official rating {operator} t. ROC scores are the corresponding sums of predicted class probabilities.
Single-class targets have undefined AUC (N/A), not an assigned chance score.</p>
<img src="semantic_threshold_auc.png" alt="Semantic threshold AUC heatmaps">
{table.round(4).to_html(na_rep='N/A',border=0)}
<p><a href="semantic_threshold_auc.csv">AUC counts and values (CSV)</a> · <a href="semantic_roc.pdf">All semantic ROC curves (PDF)</a></p>
<h2>Nodule malignancy</h2><p>Radiomics is computed from original HU and each exact binary reader mask.
The saved radiomics baseline is combined with the saved semantic correction models; probabilities are averaged
across the models selected by the checkpoint's evaluation mode. The target is that same reader's malignancy rating:
1–2 negative, 4–5 positive. Rating 3 retains predictions but is excluded from this binary ROC.</p>
<img src="malignancy_roc.png" alt="V4 and V4oversample malignancy ROC">
{risk.round(4).to_html(index=False,na_rep='N/A',border=0)}
<p>Annotations of the same nodule/patient are correlated. These are annotation-weighted point estimates;
no independent-annotation confidence intervals or significance claims are made. LIDC reader-derived malignancy
is not a pathology endpoint. Official semantic inputs evaluate the risk component under supplied reader features.</p>
<h2>Reproducibility</h2><ul>{checkpoints}</ul><ul>{''.join(overview)}</ul>
<p><a href="predictions.csv">All annotation predictions</a> · <a href="annotations.csv">Test annotation cohort</a> ·
<a href="provenance.json">Checkpoint hashes and protocol</a> · <a href="malignancy_roc.pdf">Malignancy ROC (PDF)</a></p></html>'''
    (directory/'report.html').write_text(content)
    print(risk[['model','semantic_input','auc','n_evaluated','n_positive','n_negative']].to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    generate_report(parser.parse_args().directory)
