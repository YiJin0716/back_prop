"""Export one directory per evaluation point from existing inference results."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from back_prop.evaluate.eval_package.types import SEMANTIC_NAMES
from .report import collect, clean, table, fmt, semantic_figure, risk_figure, COLORS, MODE_LABELS
from .metrics import MODES
from .run import DEFAULT_OUTPUT, storage_root


POINTS = {
    'annotation_lvl/annotator_segmentation_to_annotator_lidc_feature': (
        'Annotator Segmentation → Annotator LIDC Feature',
        'Each individual reader mask is used to predict that same reader’s six feature ratings.'),
    'annotation_lvl/annotator_lidc_feature_to_annotator_lidc_malignancy': (
        'Annotator LIDC Feature → Annotator LIDC Malignancy',
        'Input: the same reader’s six official ratings and radiomics from their mask.'),
    '_data/supplementary/official_segmentation_to_malignancy': (
        'Official Segmentation → Predicted LIDC Feature → Malignancy',
        'Input: predicted features and radiomics from each official reader mask. This is the additional predicted-feature comparison.'),
    'nodule_lvl/predicted_segmentation_to_overlap': (
        'Predicted Segmentation → Overlap (DICE / IoU)',
        'Compare predicted masks with the closest annotation, reader union and reader intersection. Additional two-vote consensus metrics remain in the CSV. All-GT metrics assign zero overlap to misses.'),
    'nodule_lvl/predicted_segmentation_to_lidc_feature': (
        'Predicted Segmentation → LIDC Feature',
        'Matched physical nodules only. Compare predicted features with the closest reader’s ratings and the mean reader ratings.'),
    'nodule_lvl/predicted_segmentation_to_lidc_feature_to_annotator_malignancy': (
        'Predicted Segmentation → LIDC Feature → LIDC Annotator Malignancy',
        'Use predicted features and radiomics from predicted masks. Matched comparisons use closest-reader, all-reader and mean-reader labels; the separate all-nodule curve assigns risk zero to misses.'),
    'volume_lvl/exclusion_metrics': (
        'Capture Metrics about Exclusions',
        'Total exclusions, exclusions predicted malignant and exclusions tied for the highest-risk nodule. The malignancy_vs_nodule_size subdirectory contains sensitivity/specificity curves and the 5-mm cutoff analysis.'),
    'volume_lvl/exclude_unmatched_segmentations': (
        'Exclude Segmentations without an Annotation Match',
        'Exclude unmatched predicted segmentations from paired per-nodule evaluations. The retained matches and excluded predictions are listed here. Excluded predictions remain in whole-CT risk and count as FROC false positives.'),
    'volume_lvl/volume_auc_accuracy': (
        'Volume-level AUC / Accuracy',
        'The primary target is any positive reader diagnosis, using the previously agreed reader-score labels. Additional mean-nodule and matched-only comparisons are retained separately.'),
    'volume_lvl/froc': (
        'Whole CT → Nodule Detection (FROC)',
        'Sweep objectness over all post-NMS candidates. Every official physical nodule remains in the sensitivity denominator.'),
    'volume_lvl/exclusion_metrics/malignancy_vs_nodule_size/filter_small_predictions': (
        'Filter Small Predicted Nodules',
        'Remove predictions with diameter ≤ d, then rematch while retaining all official nodules. Includes the 5-mm cutoff and whole-CT metrics.'),
    'volume_lvl/exclusion_metrics/malignancy_vs_nodule_size/small_nodule_performance': (
        'Performance on Small Official Nodules',
        'Evaluate official nodules with diameter ≤ d without filtering predictions. Missed nodules receive malignancy risk zero.'),
}

OVERLAP_POINT = 'nodule_lvl/predicted_segmentation_to_overlap'
SEMANTIC_POINT = 'nodule_lvl/predicted_segmentation_to_lidc_feature'
RISK_POINT = 'nodule_lvl/predicted_segmentation_to_lidc_feature_to_annotator_malignancy'
for reference in ('closest', 'union', 'intersection'):
    POINTS[OVERLAP_POINT+'/vs_'+reference] = (
        'Predicted Segmentation vs. '+reference.capitalize(),
        f'DICE and IoU against the {reference} annotation reference, with matched-only and all-GT summaries.')
for reference, description in (
    ('closest_annotation', 'Use the highest-IoU annotation to determine LIDC feature targets.'),
    ('consensus_annotation', 'Use mean reader scores as consensus LIDC feature targets.')):
    POINTS[SEMANTIC_POINT+'/'+reference] = ('LIDC Features: '+reference.replace('_', ' ').title(), description)
for mode in MODES:
    POINTS[RISK_POINT+'/'+mode] = ('Malignancy: '+MODE_LABELS[mode],
        'Matched-nodule AUC and accuracy for this label rule; compare with closest-reader, all-reader and reader-mean targets.')


def save_figure(figure, folder, name):
    for extension in ('png', 'pdf'):
        figure.savefig(folder/f'{name}.{extension}', bbox_inches='tight')
    plt.close(figure)


def csv(folder, name, rows):
    pd.DataFrame(rows).to_csv(folder/f'{name}.csv', index=False)


def subset(summary, key, units):
    return [row for row in summary[key] if row['unit'] in units]


def annotation_threshold_heatmap(summary):
    return semantic_figure(summary, [('annotation', 'V5 · individual official reader masks')], title=None)


def accuracy_figure(summary, units, title):
    fig, axes = plt.subplots(2, 1, figsize=(8.3, 6.8), constrained_layout=True)
    x = np.arange(len(SEMANTIC_NAMES))
    width = .7 / len(units)
    for ax, metric, ylabel in zip(axes, ('accuracy', 'mae'), ('Exact-class accuracy', 'Mean absolute rating error')):
        for index, (unit, label) in enumerate(units):
            values = [next(row[metric] for row in summary['semantics']
                           if row['unit'] == unit and row['feature'] == feature) for feature in SEMANTIC_NAMES]
            ax.bar(x + (index-(len(units)-1)/2)*width, values, width, label=label, color=COLORS[index])
        ax.set(xticks=x, xticklabels=[name.capitalize() for name in SEMANTIC_NAMES], ylabel=ylabel)
        ax.tick_params(axis='x', labelsize=9)
        if metric == 'accuracy':
            ax.set_ylim(0, 1)
        clean(ax)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=13)
    return fig


def segmentation_figure(data, summary, references=('closest', 'union', 'intersection')):
    fig, axes = plt.subplots(2, 1, figsize=(8.3, 6), gridspec_kw={'height_ratios': [2, 1]}, constrained_layout=True)
    values = [[n[ref+'_dice'] for n in data['nodules'] if n['matched']] for ref in references]
    axes[0].boxplot(values, showfliers=False)
    axes[0].set_xticks(range(1, len(references)+1), [ref.capitalize() for ref in references])
    axes[0].set(ylabel='Dice', ylim=(0, 1), title='Matched predicted segmentations')
    clean(axes[0])
    rows = []
    for ref in references:
        values = [next(r['mean'] for r in summary['segmentation']
                       if r['reference'] == ref and r['cohort'] == cohort and r['metric'] == metric)
                  for cohort in ('matched', 'all_nodules') for metric in ('dice', 'iou')]
        rows.append([ref.capitalize(), *map(fmt, values)])
    table(axes[1], rows, ['Reference', 'Matched Dice', 'Matched IoU', 'All-GT Dice', 'All-GT IoU'], size=8)
    fig.suptitle('Per nodule: Predicted Segmentation → Overlap', fontsize=13)
    return fig


def froc_figure(summary):
    fig, ax = plt.subplots(figsize=(7.4, 5), constrained_layout=True)
    curve = summary['froc']
    ax.step([r['false_positives_per_scan'] for r in curve['points']],
            [r['sensitivity'] for r in curve['points']], where='post', color=COLORS[0])
    ax.scatter([r['achieved_fp_per_scan'] for r in curve['operating_points']],
               [r['sensitivity'] for r in curve['operating_points']], color=COLORS[1])
    ax.set(xlabel='False positives per CT', ylabel='Nodule sensitivity', ylim=(0, 1),
           title=f"Whole CT: nodule detection ({summary['nodules']} official nodules)")
    clean(ax)
    return fig


def size_figure(summary, field, *, volume=False):
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 9), constrained_layout=True)
    filtering = field == 'size_filtering'
    xkey = 'cutoff_mm' if filtering else 'upper_diameter_mm'
    for ax, mode in zip(axes, MODES):
        rows = [r for r in summary[field] if r['mode'] == mode]
        for key, color in zip(('sensitivity', 'specificity'), COLORS):
            source = ('volume_' if volume else '') + key
            ax.plot([r[xkey] for r in rows], [np.nan if r[source] is None else r[source] for r in rows],
                    marker='o', markersize=3, color=color, label=key.capitalize())
        ax.axvline(5, color='gray', ls=':', lw=1)
        ax.set(title=MODE_LABELS[mode], ylim=(0, 1), ylabel='Whole-CT classification' if volume else 'Nodule classification',
               xlabel='Remove predicted nodules ≤ d (mm)' if filtering else 'Evaluate official nodules ≤ d (mm)')
        clean(ax)
    axes[0].legend(frameon=False)
    fig.suptitle('Effect of size filtering' if filtering else 'Performance on small official nodules', fontsize=13)
    return fig


def nodule_semantic_rows(cases):
    rows = []
    for case in cases:
        for match in case['matches']:
            i, j = match['prediction_index'], match['nodule_index']
            predicted, target = case['predictions'][i], case['nodules'][j]
            detail = case['pair_details'][j][i]
            row = dict(case_id=case['case_id'], nodule_id=target['nodule_id'], query_id=predicted['query_id'],
                       closest_annotation_id=detail['closest_annotation_id'])
            for k, name in enumerate(SEMANTIC_NAMES):
                row.update({f'pred_{name}': predicted['semantic_features'][k],
                            f'closest_{name}': detail['closest_ratings'][k],
                            f'mean_{name}': target['consensus_ratings'][k]})
                row.update({f'{name}_p{t}': p for t, p in enumerate(predicted['semantic_probabilities'][k], 1)})
            rows.append(row)
    return rows


def write_points(output, cases, data, summary, protocol):
    output = Path(output)
    folders = {point: output/point for point in POINTS}
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42, 'savefig.dpi': 170})

    semantic_points = [
        ('annotation_lvl/annotator_segmentation_to_annotator_lidc_feature', [('annotation', 'Individual reader')], 'Per annotation: official mask → LIDC features'),
        ('nodule_lvl/predicted_segmentation_to_lidc_feature', [('nodule_closest', 'Closest reader'), ('nodule_consensus', 'Reader mean')], 'Per nodule: predicted mask → LIDC features'),
    ]
    for point, units, title in semantic_points:
        folder = folders[point]
        names = [unit for unit, _ in units]
        figure = (annotation_threshold_heatmap(summary) if names == ['annotation']
                  else semantic_figure(summary, units, title))
        save_figure(figure, folder, 'threshold_auc')
        save_figure(accuracy_figure(summary, units, title), folder, 'accuracy_and_mae')
        csv(folder, 'semantic_metrics', subset(summary, 'semantics', names))
        csv(folder, 'threshold_metrics', subset(summary, 'semantic_thresholds', names))
    csv(folders[semantic_points[0][0]], 'annotations', data['annotations'])
    csv(folders[semantic_points[1][0]], 'nodule_predictions', nodule_semantic_rows(cases))
    for suffix, unit, label in [('closest_annotation', 'nodule_closest', 'Closest annotation'),
                                ('consensus_annotation', 'nodule_consensus', 'Consensus annotation')]:
        folder = folders[SEMANTIC_POINT+'/'+suffix]
        units = [(unit, label)]
        save_figure(semantic_figure(summary, units, 'Per nodule: predicted mask → LIDC features'), folder, 'threshold_auc')
        save_figure(accuracy_figure(summary, units, label), folder, 'accuracy_and_mae')
        csv(folder, 'semantic_metrics', subset(summary, 'semantics', [unit]))
        csv(folder, 'threshold_metrics', subset(summary, 'semantic_thresholds', [unit]))

    risk_points = [
        ('annotation_lvl/annotator_lidc_feature_to_annotator_lidc_malignancy', [('annotation_official', 'Official reader features')]),
        ('_data/supplementary/official_segmentation_to_malignancy', [('annotation_predicted', 'Predicted features on reader mask')]),
        ('nodule_lvl/predicted_segmentation_to_lidc_feature_to_annotator_malignancy', [('nodule_closest', 'Closest reader'), ('nodule_all_readers', 'All readers'), ('nodule_consensus_matched', 'Reader mean')]),
        ('volume_lvl/volume_auc_accuracy', [('volume_any_reader', 'Any positive reader')]),
    ]
    for point, units in risk_points:
        folder = folders[point]
        title = 'Predicted Segmentation → LIDC Feature\n→ LIDC Annotator Malignancy' if point == RISK_POINT else POINTS[point][0]
        save_figure(risk_figure(data, summary, units, title), folder, 'malignancy_roc')
        names = [unit for unit, _ in units]
        if point.startswith('nodule_lvl'):
            names.append('nodule_consensus_all')
            save_figure(risk_figure(data, summary, [('nodule_consensus_all', 'All nodules; missed risk = 0')],
                                   'Per nodule: malignancy including misses'), folder, 'malignancy_roc_all_nodules')
        csv(folder, 'malignancy_metrics', subset(summary, 'malignancy', names))
        source = data['annotations'] if units[0][0].startswith('annotation_') else data['nodules'] if point.startswith('nodule_lvl') else data['volumes']
        csv(folder, 'predictions', source)
        if point == RISK_POINT:
            for mode in MODES:
                child = folders[point+'/'+mode]
                save_figure(risk_figure(data, summary, units, 'Per nodule: malignancy', modes=(mode,)), child, 'malignancy_roc')
                csv(child, 'malignancy_metrics', [r for r in summary['malignancy'] if r['unit'] in [u for u, _ in units] and r['mode'] == mode])
    csv(folders['volume_lvl/volume_auc_accuracy'], 'additional_comparisons',
        subset(summary, 'malignancy', ['volume', 'volume_matched_only']))

    folder = folders['nodule_lvl/predicted_segmentation_to_overlap']
    save_figure(segmentation_figure(data, summary), folder, 'segmentation_overlap')
    csv(folder, 'segmentation_metrics', summary['segmentation'])
    csv(folder, 'nodules', data['nodules'])
    for reference in ('closest', 'union', 'intersection'):
        child = folders[OVERLAP_POINT+'/vs_'+reference]
        save_figure(segmentation_figure(data, summary, (reference,)), child, 'segmentation_overlap')
        csv(child, 'segmentation_metrics', [r for r in summary['segmentation'] if r['reference'] == reference])
    folder = folders['volume_lvl/exclude_unmatched_segmentations']
    csv(folder, 'retained_nodule_matches', [r for r in data['nodules'] if r['matched']])
    csv(folder, 'excluded_predictions', data['exclusions'])
    folder = folders['volume_lvl/exclusion_metrics']
    csv(folder, 'unmatched_predictions', data['exclusions'])
    csv(folder, 'counts', [{k: v for k, v in summary['exclusions'].items() if k != 'reasons'}])
    csv(folder, 'reasons', [dict(reason=k, count=v) for k, v in summary['exclusions']['reasons'].items()])
    folder = folders['volume_lvl/froc']
    save_figure(froc_figure(summary), folder, 'froc')
    csv(folder, 'froc', summary['froc']['points'])
    csv(folder, 'operating_points', summary['froc']['operating_points'])
    csv(folder, 'nodules', data['nodules'])
    csv(folder, 'predictions', data['predictions'])

    folder = folders['volume_lvl/exclusion_metrics/malignancy_vs_nodule_size/filter_small_predictions']
    save_figure(size_figure(summary, 'size_filtering'), folder, 'nodule_classification')
    save_figure(size_figure(summary, 'size_filtering', volume=True), folder, 'volume_classification')
    csv(folder, 'size_filtering', summary['size_filtering'])
    csv(folder, 'cutoff_5mm', [r for r in summary['size_filtering'] if r['cutoff_mm'] == 5])
    folder = folders['volume_lvl/exclusion_metrics/malignancy_vs_nodule_size/small_nodule_performance']
    save_figure(size_figure(summary, 'size_cumulative'), folder, 'nodule_classification')
    csv(folder, 'size_cumulative', summary['size_cumulative'])

    from back_prop.baseline_compare.v5_comparison import refresh_plot
    has_baselines = refresh_plot(output)
    for point, (title, description) in POINTS.items():
        folder = folders[point]
        files = sorted(p.name for p in folder.iterdir() if p.name != 'README.md')
        details = ''
        if point.endswith('lidc_feature') or point in (SEMANTIC_POINT+'/closest_annotation', SEMANTIC_POINT+'/consensus_annotation'):
            thresholds = 'The heatmap shows t = 1–5; AUC at t = 1 is undefined (N/A). '
            details = ('\nThreshold AUC uses the target rating ≥ t and predicted probability P(rating ≥ t). '
                       + thresholds +
                       'Exact-class accuracy uses the most probable class; reader means are rounded half-up only for accuracy.\n')
        if 'malignancy' in point or point == 'volume_lvl/volume_auc_accuracy':
            details += '\nLabel rules: exclude rating 3; rating ≥ 3 positive; rating > 3 positive. Labels come from LIDC readers.\n'
        if point == 'volume_lvl/volume_auc_accuracy' and has_baselines:
            details += '\nV5, DeepLung and Sybil use identical test CTs and label exclusions.\n'
        (folder/'README.md').write_text(f'# {title}\n\n{description}\n{details}\n'
            + '\n'.join(f'- [{name}]({name})' for name in files)
            + '\n\n[All evaluation points]('+os.path.relpath(output/'README.md', folder)+')\n')
    run_id = Path(protocol['checkpoint']).parent.name
    lines = ['# V5 evaluation results', '',
             f"Run **{run_id}**, epoch **{protocol['epoch']}**. Checkpoint frozen during evaluation.", '',
             '[Complete report](v5_joint_evaluation.pdf)', '',
             'Choose the evaluation level, then the input → output point. PNG figures, PDF figures and CSV tables are stored together.', '']
    for level in ('annotation_lvl', 'volume_lvl', 'nodule_lvl'):
        lines += [f'## {level}', '']
        primary = [point for point in POINTS if point.startswith(level+'/') and point.count('/') == 1]
        if level == 'volume_lvl':
            primary = [level+'/'+name for name in ('exclude_unmatched_segmentations', 'froc',
                                                   'exclusion_metrics', 'volume_auc_accuracy')]
        lines += [f'- [{POINTS[point][0]}]({point}/): {POINTS[point][1]}' for point in primary]
        lines.append('')
    lines += ['## Supporting data', '',
              '`_data/` retains case JSON predictions, summary metrics and inference protocols needed to redraw these results.',
              'Additional annotation-mask → predicted-feature → malignancy results are in `_data/supplementary/official_segmentation_to_malignancy/`.',
              'The root PDF is the combined report. Point-specific figures contain only their corresponding evaluation level.', '']
    (output/'README.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    storage = storage_root(args.output)
    summary = json.loads((storage/'summary.json').read_text())
    cases = [json.loads(p.read_text()) for p in sorted((storage/'cases').glob('*.json'))]
    protocol = json.loads((storage/'protocol_shard_0.json').read_text())
    assert {c['case_id'] for c in cases} == set(protocol['expected_cases'])
    assert len(cases) == summary['scans']
    assert all(c['fingerprint'] == protocol['fingerprint'] for c in cases)
    write_points(args.output, cases, collect(cases), summary, protocol)
    print(json.dumps(dict(output=str(args.output), levels=3,
                         points=sum(point.count('/') == 1 for point in POINTS))))


if __name__ == '__main__':
    main()
