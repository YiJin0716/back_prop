"""Aggregate the complete test cohort and render the requested evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import roc_curve

from back_prop.evaluate.eval_package.types import SEMANTIC_NAMES
from .metrics import (MODES, binary_metrics, semantic_metrics, malignancy_label,
                      volume_label, froc, match_predictions, scan_probability)
from .run import DEFAULT_OUTPUT, write_json, storage_root

MODE_LABELS = {'exclude_3':'Exclude 3', 'ge_3':'Rating ≥ 3', 'gt_3':'Rating > 3'}
COLORS = ['#2563a6','#d55e00','#009e73']


def operating_indices(case, cutoff=0.):
    return [i for i,p in enumerate(case['predictions'])
            if p['object_probability'] >= case['object_threshold'] and p['diameter_mm'] > cutoff]


def matched_indices(case, cutoff=0.):
    predictions=case['predictions']
    scores=[p['object_probability'] if p['diameter_mm'] > cutoff else -1. for p in predictions]
    iou=np.asarray(case['best_reader_iou']).reshape(len(scores),len(case['nodules']))
    return match_predictions(iou,scores,case['object_threshold'],case['minimum_iou'])[0]


def collect(cases):
    """Preserve missing predictions as misses; semantic scores stay conditional."""
    sem={k:dict(ratings=[],probabilities=[]) for k in ('annotation','nodule_closest','nodule_consensus')}
    risk={k:[] for k in ('annotation_official','annotation_predicted','nodule_closest',
                         'nodule_all_readers','nodule_consensus_matched','nodule_consensus_all')}
    nodules=[];annotations=[];predictions=[];volumes=[];exclusions=[]
    for case in cases:
        common=dict(case_id=case['case_id'],patient_id=case['patient_id'])
        matched={r['nodule_index']:r['prediction_index'] for r in case['matches']}
        for row in case['annotations']:
            annotations.append(row)
            if row['input_status']!='ok':continue
            ratings=[row['gt_'+f] for f in SEMANTIC_NAMES]
            probability=[[row[f'{f}_p{k}'] for k in range(1,6)] for f in SEMANTIC_NAMES]
            sem['annotation']['ratings'].append(ratings);sem['annotation']['probabilities'].append(probability)
            for source in ('official','predicted'):
                risk['annotation_'+source].append((row['gt_malignancy'],row['malignancy_probability_'+source]))
        for j,gt in enumerate(case['nodules']):
            i=matched.get(j)
            p=None if i is None else case['predictions'][i]
            risk['nodule_consensus_all'].append((gt['malignancy_mean'],p['malignancy_probability'] if p else 0.))
            n=dict(common,nodule_id=gt['nodule_id'],matched=p is not None,diameter_mm=gt['diameter_mm'],
                   malignancy_mean=gt['malignancy_mean'],query_id=p['query_id'] if p else None,
                   predicted_risk=p['malignancy_probability'] if p else None,
                   intersection_empty=gt['intersection_voxels']==0)
            if p is not None:
                detail=case['pair_details'][j][i]
                for reference,ratings in (('closest',detail['closest_ratings']),('consensus',gt['consensus_ratings'])):
                    sem['nodule_'+reference]['ratings'].append(ratings)
                    sem['nodule_'+reference]['probabilities'].append(p['semantic_probabilities'])
                risk['nodule_closest'].append((detail['closest_malignancy'],p['malignancy_probability']))
                risk['nodule_consensus_matched'].append((gt['malignancy_mean'],p['malignancy_probability']))
                risk['nodule_all_readers'].extend((r['malignancy'],p['malignancy_probability']) for r in gt['readers'])
                n['closest_annotation_id']=detail['closest_annotation_id']
                for reference in ('closest','union','intersection','consensus'):
                    for metric in ('dice','iou'):
                        n[reference+'_'+metric]=detail[reference][metric]
            for reference in ('closest','union','intersection','consensus'):
                for metric in ('dice','iou'):
                    n.setdefault(reference+'_'+metric,0.)
            nodules.append(n)
        selected=operating_indices(case,cutoff=-1.)
        matched_prediction={r['prediction_index'] for r in case['matches']}
        max_risk=max((case['predictions'][i]['malignancy_probability'] for i in selected),default=-1.)
        for i in selected:
            p=case['predictions'][i]
            predictions.append(dict(common,query_id=p['query_id'],matched=i in matched_prediction,
                object_probability=p['object_probability'],predicted_risk=p['malignancy_probability'],
                diameter_mm=p['diameter_mm'],mask_voxels=p['mask_voxels']))
        for row in case['exclusions']:
            p=case['predictions'][row['prediction_index']]
            exclusions.append(dict(common,**row,predicted_risk=p['malignancy_probability'],
                diameter_mm=p['diameter_mm'],tied_maximum_risk=p['malignancy_probability'] >= max_risk-1e-8))
        for mode in MODES:
            label=volume_label([malignancy_label(n['malignancy_mean'],mode) for n in case['nodules']])
            any_reader_label=volume_label([malignancy_label(r['malignancy'],mode) for n in case['nodules'] for r in n['readers']])
            volumes.append(dict(common,mode=mode,target=label,any_reader_target=any_reader_label,
                predicted_risk=case['scan_probability'],matched_only_risk=case['matched_only_scan_probability']))
    return dict(sem=sem,risk=risk,nodules=nodules,annotations=annotations,
                predictions=predictions,volumes=volumes,exclusions=exclusions)


def size_analysis(cases):
    filtering=[];cumulative=[]
    for mode in MODES:
        for cutoff in range(16):
            labels=[];scores=[];scan_labels=[];scan_scores=[]
            removed=removed_malignant_predictions=lost_malignant_detections=0
            for c in cases:
                selected=operating_indices(c,cutoff)
                removed_indices=set(operating_indices(c,-1.))-set(selected)
                removed+=len(removed_indices)
                removed_malignant_predictions+=sum(c['predictions'][i]['malignancy_probability']>=.5 for i in removed_indices)
                mapping={j:i for i,j in matched_indices(c,cutoff)}
                original={r['nodule_index'] for r in c['matches']}
                gt_labels=[malignancy_label(n['malignancy_mean'],mode) for n in c['nodules']]
                for j,label in enumerate(gt_labels):
                    labels.append(label)
                    scores.append(c['predictions'][mapping[j]]['malignancy_probability'] if j in mapping else 0.)
                    lost_malignant_detections+=int(label==1 and j in original and j not in mapping)
                scan_labels.append(volume_label(gt_labels))
                scan_scores.append(scan_probability([c['predictions'][i] for i in selected]))
            row=dict(mode=mode,cutoff_mm=cutoff,removed_predictions=removed,
                     removed_predicted_malignant=removed_malignant_predictions,
                     additional_malignant_nodules_lost=lost_malignant_detections,
                     **binary_metrics(labels,scores))
            row.update({'volume_'+k:v for k,v in binary_metrics(scan_labels,scan_scores).items()})
            filtering.append(row)
        for upper in (3,4,5,6,8,10,15,20,30,50):
            labels=[];scores=[]
            for c in cases:
                mapping={r['nodule_index']:r['prediction_index'] for r in c['matches']}
                for j,n in enumerate(c['nodules']):
                    if n['diameter_mm']<=upper:
                        labels.append(malignancy_label(n['malignancy_mean'],mode))
                        scores.append(c['predictions'][mapping[j]]['malignancy_probability'] if j in mapping else 0.)
            cumulative.append(dict(mode=mode,upper_diameter_mm=upper,**binary_metrics(labels,scores)))
    return filtering,cumulative


def aggregate(cases):
    data=collect(cases)
    sem_overall=[];sem_threshold=[]
    for unit,values in data['sem'].items():
        rows=semantic_metrics(values['ratings'],values['probabilities'])
        for feature,row in zip(SEMANTIC_NAMES,rows):
            sem_overall.append(dict(unit=unit,feature=feature,**{k:v for k,v in row.items() if k!='thresholds'}))
            for t in row['thresholds']:sem_threshold.append(dict(unit=unit,feature=feature,**t))
    risk_rows=[]
    for unit,rows in data['risk'].items():
        for mode in MODES:
            risk_rows.append(dict(unit=unit,mode=mode,**binary_metrics(
                [malignancy_label(y,mode) for y,p in rows],[p for y,p in rows])))
    for unit,target,score in (('volume','target','predicted_risk'),
                              ('volume_any_reader','any_reader_target','predicted_risk'),
                              ('volume_matched_only','target','matched_only_risk')):
        for mode in MODES:
            rows=[r for r in data['volumes'] if r['mode']==mode]
            risk_rows.append(dict(unit=unit,mode=mode,**binary_metrics([r[target] for r in rows],[r[score] for r in rows])))
    segmentation=[]
    for cohort in ('matched','all_nodules'):
        nodes=[n for n in data['nodules'] if n['matched'] or cohort=='all_nodules']
        for reference in ('closest','union','intersection','consensus'):
            for metric in ('dice','iou'):
                values=[n[reference+'_'+metric] for n in nodes]
                segmentation.append(dict(cohort=cohort,reference=reference,metric=metric,n=len(values),
                    mean=float(np.mean(values)) if values else None,
                    median=float(np.median(values)) if values else None))
    exclusions=data['exclusions']
    excluded=dict(total=len(exclusions),cases=len({r['case_id'] for r in exclusions}),
        predicted_malignant=sum(r['predicted_malignant'] for r in exclusions),
        maximum_risk_nodules=sum(r['tied_maximum_risk'] for r in exclusions),
        maximum_risk_cases=len({r['case_id'] for r in exclusions if r['tied_maximum_risk']}),
        reasons={reason:sum(r['reason']==reason for r in exclusions) for reason in sorted({r['reason'] for r in exclusions})})
    filtering,cumulative=size_analysis(cases)
    summary=dict(scans=len(cases),patients=len({c['patient_id'] for c in cases}),
        nodules=len(data['nodules']),annotations=len(data['annotations']),
        annotation_inputs_empty=sum(r['input_status']!='ok' for r in data['annotations']),
        matched_nodules=sum(n['matched'] for n in data['nodules']),
        selected_predictions=len(data['predictions']),exclusions=excluded,
        segmentation=segmentation,semantics=sem_overall,semantic_thresholds=sem_threshold,
        malignancy=risk_rows,size_filtering=filtering,size_cumulative=cumulative,
        froc=froc(cases),empty_intersection_nodules=sum(n['intersection_empty'] for n in data['nodules']))
    return data,summary


def clean(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(alpha=.16);ax.set_axisbelow(True)


def fmt(value):
    return 'N/A' if value is None else f'{value:.3f}'


def table(ax,rows,columns,widths=None,size=9):
    ax.axis('off')
    t=ax.table(cellText=rows,colLabels=columns,colWidths=widths,cellLoc='center',loc='center')
    t.auto_set_font_size(False);t.set_fontsize(size);t.scale(1,1.55)
    for (row,col),cell in t.get_celld().items():
        cell.visible_edges='TB' if row==0 else 'B' if row==len(rows) else ''
        cell.set_linewidth(.6)
        if row==0:
            cell.set_text_props(weight='bold')
            if any('\n' in c for c in columns):cell.set_height(cell.get_height()*1.8)
    return t


def overview(summary,protocol):
    s=summary;fig=plt.figure(figsize=(8.3,11.7))
    fig.text(.07,.95,'V5 joint model: independent test evaluation',fontsize=15,weight='bold')
    run_id=Path(protocol['checkpoint']).parent.name
    fig.text(.07,.915,f"Run {run_id} · epoch {protocol['epoch']} · frozen checkpoint",fontsize=10)
    fig.text(.07,.875,f"{s['scans']} CT scans / {s['patients']} patients · {s['nodules']} nodules · {s['annotations']} reader annotations\n"
        f"No training-patient overlap. Ground truth is the official LIDC reader rating.\n"
        f"At objectness ≥ 0.5: {s['matched_nodules']}/{s['nodules']} nodules matched "
        f"({s['matched_nodules']/s['nodules']:.1%}); {s['exclusions']['total']} predictions unmatched.",fontsize=10,va='top',linespacing=1.6)
    fig.text(.07,.76,'Malignancy classification',fontsize=12,weight='bold')
    labels={'annotation_official':'Reader: official features','nodule_consensus_matched':'Nodule: matched',
            'nodule_consensus_all':'Nodule: all, misses = 0',
            'volume_any_reader':'CT: any positive reader','volume':'CT: nodule consensus'}
    rows=[]
    for unit,label in labels.items():
        for mode in MODES:
            r=next(r for r in s['malignancy'] if r['unit']==unit and r['mode']==mode)
            rows.append([label,MODE_LABELS[mode],r['n'],fmt(r['auc']),fmt(r['accuracy'])])
    table(fig.add_axes([.06,.365,.88,.38]),rows,['Evaluation','Label rule','N','AUC','Accuracy'],[.39,.23,.10,.14,.14])
    ex=s['exclusions']
    fig.text(.07,.32,'Unmatched prediction exclusions',fontsize=12,weight='bold')
    text=(f"{ex['total']} predictions excluded from paired nodule metrics; "
          f"{ex['predicted_malignant']} have predicted malignancy ≥ 0.5.\n"
          f"{ex['maximum_risk_cases']} scans have an excluded prediction tied for highest nodule risk.\n"
          'These predictions remain in the deployed whole-CT risk and count as FROC false positives.\n\n')
    row=next(r for r in s['size_filtering'] if r['mode']=='exclude_3' and r['cutoff_mm']==5)
    inclusive=next(r for r in s['size_filtering'] if r['mode']=='ge_3' and r['cutoff_mm']==5)
    small=next(r for r in s['size_cumulative'] if r['mode']=='exclude_3' and r['upper_diameter_mm']==10)
    text+=(f"Removing predicted nodules ≤ 5 mm removes {row['removed_predictions']} predictions,\n"
           f"including {row['removed_predicted_malignant']} classified malignant at probability ≥ 0.5. Additional malignant\n"
           f"detections lost: {row['additional_malignant_nodules_lost']} under exclude-3 / >3; "
           f"{inclusive['additional_malignant_nodules_lost']} under ≥3.\n\n"
           f"Among GT nodules ≤ 10 mm (exclude-3), {small['tp']}/{small['positives']} malignant nodules\n"
           'are classified positive, including detection misses in the denominator.\n\n'
           'Binary predictions use a fixed probability threshold of 0.5.\n'
           'The three label rules apply to reader scores or mean nodule scores, as appropriate.\n'
           'All results are test-set estimates; no thresholds or weights were fitted on this set.')
    fig.text(.07,.29,text,fontsize=10,va='top',linespacing=1.5)
    return fig


def semantic_figure(summary, units=None, title='Semantic features: threshold AUC'):
    """Annotated heatmaps on a common AUC scale, one panel per reference."""
    if units is None:
        units=[('annotation','Official reader mask'),('nodule_closest','Predicted mask / closest reader'),
               ('nodule_consensus','Predicted mask / reader mean')]
    fig,axes=plt.subplots(len(units),1,figsize=(7.4,4.5*len(units)),constrained_layout=True)
    cmap=plt.get_cmap('YlGnBu').copy()
    cmap.set_bad('#eeeeee')
    for ax,(unit,label) in zip(np.atleast_1d(axes),units):
        rows={(r['feature'],r['threshold']):r['auc']
              for r in summary['semantic_thresholds'] if r['unit']==unit}
        values=np.asarray([[np.nan if rows[name,t] is None else rows[name,t]
                            for t in range(1,6)] for name in SEMANTIC_NAMES])
        im=ax.imshow(np.ma.masked_invalid(values),vmin=0,vmax=1,cmap=cmap,aspect='auto')
        ax.set_xticks(range(5),[str(t) for t in range(1,6)])
        ax.set_yticks(range(len(SEMANTIC_NAMES)),[name.capitalize() for name in SEMANTIC_NAMES])
        ax.set(title=label,xlabel='Positive label: official rating ≥ threshold')
        for i,j in np.ndindex(values.shape):
            value=values[i,j]
            ax.text(j,i,'N/A' if not np.isfinite(value) else f'{value:.3f}',ha='center',va='center',
                    color='white' if np.isfinite(value) and value>.65 else '#222222')
        ax.spines[['top','right','bottom','left']].set_visible(False)
        fig.colorbar(im,ax=ax,label='ROC AUC',shrink=.8)
    if title:
        fig.suptitle(title,fontsize=13)
    return fig


def risk_figure(data,summary,units=None,title='Malignancy ROC curves',modes=MODES):
    fig,axes=plt.subplots(len(modes),1,figsize=(8.3,3.6*len(modes)+.2),constrained_layout=True)
    axes=np.atleast_1d(axes)
    if units is None:
        units=[('annotation_official','Reader mask + official features'),
               ('nodule_consensus_matched','Predicted mask + features (matched)'),
               ('volume_any_reader','Whole CT: any positive reader')]
    volume_fields={'volume':('target','predicted_risk'),
                   'volume_any_reader':('any_reader_target','predicted_risk'),
                   'volume_matched_only':('target','matched_only_risk')}
    for mode,ax in zip(modes,axes):
        series=[]
        for unit,label in units:
            if unit in volume_fields:
                target,score=volume_fields[unit]
                rows=[r for r in data['volumes'] if r['mode']==mode]
                series.append((label,[r[target] for r in rows],[r[score] for r in rows]))
            else:
                rows=data['risk'][unit]
                series.append((label,[malignancy_label(y,mode) for y,p in rows],[p for y,p in rows]))
        for color,(label,y,p) in zip(COLORS,series):
            keep=[i for i,v in enumerate(y) if v is not None]
            y=np.asarray([y[i] for i in keep]);p=np.asarray([p[i] for i in keep])
            metric=binary_metrics(y,p)
            if metric['auc'] is not None:
                fpr,tpr,_=roc_curve(y,p)
                ax.plot(fpr,tpr,color=color,label=f"{label}: AUC {metric['auc']:.3f}")
        clean(ax);ax.plot([0,1],[0,1],':',color='gray',lw=.8)
        ax.set(title=MODE_LABELS[mode],xlabel='False positive rate',ylabel='Sensitivity',xlim=(0,1),ylim=(0,1))
        ax.legend(frameon=False,fontsize=8,loc='lower right')
    fig.suptitle(title,fontsize=14)
    return fig


def semantic_accuracy_figure(summary):
    fig=plt.figure(figsize=(8.3,7))
    fig.text(.07,.93,'Semantic feature accuracy and rating error',fontsize=14,weight='bold')
    rows=[]
    for feature in SEMANTIC_NAMES:
        entries=[next(r for r in summary['semantics'] if r['feature']==feature and r['unit']==unit)
                 for unit in ('annotation','nodule_closest','nodule_consensus')]
        rows.append([feature.capitalize()]+[fmt(r['accuracy']) for r in entries]+[fmt(r['mae']) for r in entries])
    table(fig.add_axes([.06,.40,.88,.40]),rows,
          ['Feature','Reader\naccuracy','Closest\naccuracy','Consensus\naccuracy',
           'Reader\nMAE','Closest\nMAE','Consensus\nMAE'],[.19,.135,.135,.135,.135,.135,.135],size=8)
    counts={unit:next(r['n'] for r in summary['semantics'] if r['unit']==unit)
            for unit in ('annotation','nodule_closest','nodule_consensus')}
    fig.text(.07,.32,
        f"Reader masks: {counts['annotation']} annotations; predicted masks: {counts['nodule_closest']} matched nodules.\n\n"
        'Reader: individual official mask → that reader’s score.\n'
        'Closest: predicted mask → highest-IoU reader’s score.\n'
        'Consensus: predicted mask → mean of all reader scores.\n\n'
        'Accuracy compares the most probable class with the target class (1–5).\n'
        'Consensus targets are rounded half-up for accuracy; MAE uses unrounded scores.\n'
        'Missed nodules have no semantic prediction and are counted separately in detection.',
        fontsize=9,va='top',linespacing=1.5)
    return fig


def detection_figure(data,summary):
    fig,axes=plt.subplots(3,1,figsize=(8.3,11),gridspec_kw={'height_ratios':[1.2,1,.6]},constrained_layout=True)
    curve=summary['froc'];points=curve['points']
    axes[0].step([r['false_positives_per_scan'] for r in points],[r['sensitivity'] for r in points],where='post',color=COLORS[0])
    axes[0].scatter([r['achieved_fp_per_scan'] for r in curve['operating_points']],
                    [r['sensitivity'] for r in curve['operating_points']],color=COLORS[1],s=20)
    axes[0].set(title='FROC: match to any reader annotation',xlabel='False positives per CT',ylabel='Nodule sensitivity',
                ylim=(0,1),xlim=(0,max(8,max(r['false_positives_per_scan'] for r in points))))
    refs=('closest','union','intersection','consensus')
    values=[[n[ref+'_dice'] for n in data['nodules'] if n['matched']] for ref in refs]
    axes[1].boxplot(values,labels=['Closest reader','Reader union','Reader intersection','2-vote consensus'],showfliers=False)
    axes[1].set(title='Segmentation Dice of matched nodules',ylabel='Dice',ylim=(0,1))
    for ax in axes[:2]:clean(ax)
    cells=[]
    for ref in refs:
        values=[]
        for cohort in ('matched','all_nodules'):
            for metric in ('dice','iou'):
                row=next(r for r in summary['segmentation'] if r['cohort']==cohort and r['reference']==ref and r['metric']==metric)
                values.append(fmt(row['mean']))
        cells.append([ref.capitalize()]+values)
    table(axes[2],cells,['Reference','Matched Dice','Matched IoU','All-GT Dice','All-GT IoU'],[.24,.19,.19,.19,.19],size=8)
    axes[2].set_title('Mean overlap; missed nodules contribute zero in all-GT metrics',fontsize=10)
    fig.suptitle(f"Detection and segmentation · {summary['matched_nodules']}/{summary['nodules']} nodules matched at objectness ≥ 0.5",fontsize=12)
    return fig


def size_figure(summary):
    fig,axes=plt.subplots(3,2,figsize=(8.3,10),constrained_layout=True)
    for i,mode in enumerate(MODES):
        for j,(field,xkey,label) in enumerate([('size_filtering','cutoff_mm','Remove predictions ≤ d (mm)'),
                                               ('size_cumulative','upper_diameter_mm','Evaluate GT nodules ≤ d (mm)')]):
            rows=[r for r in summary[field] if r['mode']==mode]
            ax=axes[i,j]
            for key,color in zip(('sensitivity','specificity'),COLORS):
                ax.plot([r[xkey] for r in rows],[np.nan if r[key] is None else r[key] for r in rows],color=color,label=key.capitalize())
            clean(ax);ax.axvline(5,color='gray',ls=':',lw=1)
            ax.set(title=MODE_LABELS[mode],xlabel=label,ylabel='Nodule classification',ylim=(0,1))
            if i==0 and j==0:ax.legend(frameon=False,fontsize=8)
    fig.suptitle('Effect of nodule size\nFixed malignancy threshold 0.5; missed nodules receive risk 0',fontsize=13)
    return fig


def definitions_figure(summary):
    fig=plt.figure(figsize=(8.3,11.7))
    fig.text(.07,.95,'Evaluation definitions',fontsize=15,weight='bold')
    sections=[
      ('Cohort and labels',
       'Every testing CT and physical nodule is included, including mean malignancy = 3.\n'
       'The risk bank and neural weights are frozen; no test patient fitted the bank.\n'
       'Training excluded nodules with mean malignancy = 3; the two inclusion rules\n'
       'also evaluate these nodules in the test cohort.\n'
       'Malignancy is derived from official reader ratings, as requested; no biopsy labels\n'
       'are available in the supplied table. These are not pathology-confirmed outcomes.'),
      ('Per-annotation component evaluation',
       'Each reader mask is independently applied to CT. The semantic network receives\n'
       'masked CT and mask. The malignancy stage receives that same reader’s six official\n'
       'features plus the radiomics baseline computed from the reader’s mask.\n'
       'An additional table reports risk from predicted semantic features on reader masks.'),
      ('Per-nodule evaluation',
       'Whole-CT inference receives no GT boxes, masks or ratings. Predictions match\n'
       'physical nodules one-to-one at mask IoU ≥ 0.1 against any reader. Matching first\n'
       'maximizes valid pair count, then total IoU. The closest reader has the largest IoU;\n'
       'ties use annotation ID. Union, intersection and 2-vote masks use all readers.\n'
       'Unmatched predictions are excluded from paired semantic/malignancy metrics.\n'
       'Missed GT remains in detection and all-nodule metrics; its operational risk is 0.\n'
       'Consensus semantic labels use reader means, rounded half-up for exact accuracy.\n'
       'Threshold labels use the unrounded mean; t = 1 AUC is undefined.\n'
       'Sparse endpoints: closest-reader sphericity at t = 2 has one negative;\n'
       'consensus lobulation at t = 5 has one positive. Counts are in the threshold CSV.'),
      ('Per-volume evaluation and FROC',
       'Default whole-CT risk retains unmatched predictions and uses the model’s\n'
       'object-weighted noisy-OR within each bank member, followed by ensemble averaging.\n'
       'CT metrics include both any positive reader and any positive mean nodule score.\n'
       'When excluding 3, a scan with an indeterminate label and no positive is unknown.\n'
       'ROC plots use any positive reader; the tables also include oracle matched-only risk.\n'
       'FROC sweeps all post-NMS candidates from object threshold 0; duplicates count as\n'
       f"false positives. All {summary['nodules']} physical nodules remain in the sensitivity denominator."),
      ('Size and interpretation',
       'Size is the maximum physical 3D Feret diameter of occupied voxel cells. GT size\n'
       'averages reader diameters. The left size plots remove predicted masks ≤ d and\n'
       'rematch; the right plots evaluate GT nodules whose size is ≤ d without filtering.\n'
       'All size analyses keep missed malignant nodules in the denominator.')]
    y=.905
    for title,body in sections:
        fig.text(.07,y,title,fontsize=11,weight='bold',va='top');y-=.026
        fig.text(.07,y,body,fontsize=9,va='top',linespacing=1.4)
        y-=.018*len(body.splitlines())+.027
    return fig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args();out=args.output
    storage=storage_root(out)
    protocols=[json.loads(p.read_text()) for p in sorted(storage.glob('protocol_shard_*.json'))]
    if not protocols:raise RuntimeError('No inference protocol')
    protocol=protocols[0]
    assert len(protocols)==protocol['shards']
    assert len({p['fingerprint'] for p in protocols})==1
    expected=set(protocol['expected_cases'])
    completed=set()
    for p in protocols:
        marker=json.loads((storage/f"complete_shard_{p['shard']}.json").read_text())
        assert marker['bank_unchanged'] and marker['fingerprint']==protocol['fingerprint']
        assert not completed & set(marker['cases'])
        completed.update(marker['cases'])
    assert completed==expected, 'Full held-out cohort has not finished'
    files=sorted((storage/'cases').glob('*.json'))
    cases=[json.loads(p.read_text()) for p in files]
    assert {c['case_id'] for c in cases}==expected and len(cases)==len(expected)
    assert all(c['fingerprint']==protocol['fingerprint'] for c in cases)
    assert sum(len(c['nodules']) for c in cases)==protocol['cohort']['physical_nodules']
    assert sum(len(c['annotations']) for c in cases)==protocol['cohort']['annotations']
    data,summary=aggregate(cases)
    write_json(storage/'summary.json',summary)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'savefig.dpi':170})
    with PdfPages(out/'v5_joint_evaluation.pdf') as pdf:
        figure=overview(summary,protocol);pdf.savefig(figure);plt.close(figure)
        for name,figure in [('semantic_threshold_auc',semantic_figure(summary)),
                            ('semantic_accuracy',semantic_accuracy_figure(summary)),
                            ('malignancy_roc',risk_figure(data,summary)),
                            ('detection_segmentation',detection_figure(data,summary)),
                            ('nodule_size',size_figure(summary))]:
            pdf.savefig(figure);plt.close(figure)
        figure=definitions_figure(summary);pdf.savefig(figure);plt.close(figure)
    from .organize import write_points
    write_points(out,cases,data,summary,protocol)
    print(json.dumps(dict(report=str(out/'v5_joint_evaluation.pdf'),scans=summary['scans'],
        nodules=summary['nodules'],matched=summary['matched_nodules'],exclusions=summary['exclusions']),indent=2))


if __name__=='__main__':main()
