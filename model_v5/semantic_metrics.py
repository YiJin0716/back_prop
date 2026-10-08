"""Fixed-model semantic accuracy and spread checks, separate from online logs."""
import torch
from back_prop.common.features import SEMANTIC_NAMES


def semantic_metrics(probability, histogram):
    p, h = probability.float().cpu(), histogram.float().cpu()
    classes = torch.arange(1,6)
    x, y = (p*classes).sum(-1), (h*classes).sum(-1)
    xc, yc = x-x.mean(0), y-y.mean(0)
    ps, ts = x.std(0, unbiased=False), y.std(0, unbiased=False)
    correlation = (xc*yc).mean(0)/(ps*ts).clamp_min(1e-8)
    return dict(samples=len(x), predicted_std=ps.tolist(), target_std=ts.tolist(),
        std_ratio=(ps/ts.clamp_min(1e-8)).tolist(), correlation=correlation.tolist(),
        mae=(x-y).abs().mean(0).tolist(),
        nll=float(-(h*p.clamp_min(1e-7).log()).sum(-1).mean()),
        prior_nll=float(-(h*h.mean(0).clamp_min(1e-7).log()).sum(-1).mean()))


def semantic_failures(metrics, *, min_ratio=.15, min_correlation=.25,
                      nll_margin=.01, reference=None):
    failures=[]
    for domain,m in metrics.items():
        if m['samples'] < 32:
            failures.append(f'{domain}: fewer than 32 supervised probe nodules')
            continue
        variable=[i for i,t in enumerate(m['target_std']) if t>.25]
        if m['nll'] > m['prior_nll']-nll_margin:
            failures.append(f'{domain}: NLL does not improve on the constant distribution')
        if variable and sum(m['correlation'][i] for i in variable)/len(variable)<min_correlation:
            failures.append(f'{domain}: mean rating correlation is too low')
        for i in variable:
            if m['std_ratio'][i] < min_ratio:
                failures.append(f'{domain}/{SEMANTIC_NAMES[i]}: output spread too small')
        if reference:
            baseline=reference[domain]
            if m['nll']>baseline['nll']+.1:
                failures.append(f'{domain}: NLL worsened by more than 0.1 since warmup')
            for i in variable:
                if m['predicted_std'][i]<.5*baseline['predicted_std'][i]:
                    failures.append(f'{domain}/{SEMANTIC_NAMES[i]}: lost over half of warmup spread')
                if m['correlation'][i]<baseline['correlation'][i]-.2:
                    failures.append(f'{domain}/{SEMANTIC_NAMES[i]}: correlation fell by over 0.2')
    return failures
