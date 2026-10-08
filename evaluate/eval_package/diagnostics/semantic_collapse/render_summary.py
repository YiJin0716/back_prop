from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
base=Path(__file__).resolve().parent
p=json.loads((base/'probe_results.json').read_text())
c=json.loads((base/'cohort_audit.json').read_text())
by_name={r['label']:r for r in p['probes']}
fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
for key,label,color in [('native_masked_128_to_64','Checkpoint BN statistics','#b2182b'),('native_with_batch_statistics','Current-batch BN statistics (diagnostic only)','#2166ac')]:
 stages=by_name[key]['stages']
 names=list(stages)
 axes[0].plot(names,[stages[n]['pair_difference_rms'] for n in names],marker='o',label=label,color=color)
axes[0].set_yscale('log');axes[0].set_ylabel('RMS activation difference: query 7 vs 23')
axes[0].set_title('Where image dependence is suppressed')
axes[0].tick_params(axis='x',rotation=45);axes[0].legend(fontsize=8);axes[0].grid(axis='y',alpha=.2)
names=['spiculation','subtlety','texture'];idx=np.arange(len(names));f=c['cohort']['features']
axes[1].bar(idx-.18,[f[n]['std'] for n in names],width=.36,label='V3 predicted score',color='#b2182b')
axes[1].bar(idx+.18,[f[n]['gt_std'] for n in names],width=.36,label='Official reader mean',color='#2166ac')
axes[1].set_xticks(idx,names);axes[1].set_ylabel('Across-nodule standard deviation')
axes[1].set_title('527 GT-localized nodules / 166 test CTs');axes[1].legend();axes[1].grid(axis='y',alpha=.2)
fig.savefig(base/'diagnostic_summary.png',dpi=160)
fig.savefig(base/'diagnostic_summary.pdf')
plt.close(fig)
