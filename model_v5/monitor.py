"""Epoch-only metrics with explicit effective sample counts and missing values."""
from back_prop.model_v4.monitor import Monitor as BaseMonitor
from .loss import TERM_NAMES


class Monitor(BaseMonitor):
    def __init__(self,args,config,saved=None):
        super().__init__(args,config,saved)
        if self.run is not None:
            self.run.tags = ('v5','staged-preparation','semantic-retention','valid-supervision')
            for group in ('geometry','semantic_warmup'):
                self.run.define_metric(group+'/*',step_metric='warmup_epoch')
            self.run.define_metric('03_semantic/*',step_metric='epoch')


def epoch_metrics(record, weights):
    result = dict(epoch=record['epoch'], global_step=record['global_step'])
    result.update({f'01_components/{k}': record[k] for k in TERM_NAMES})
    result.update({f'00_training/{k}': record[k] for k in
                   ('total', 'active_total', 'teacher_probability', 'supervised_nodules',
                    'semantic_supervised_nodules', 'valid_risk_cases', 'teacher_forced', 'fallbacks')})
    result.update({f'00_training/lr_{k}': v for k, v in record['lr'].items()})
    result.update({f'02_objectives/count_{k}': v for k, v in record['loss_counts'].items()})
    result.update({f'02_objectives/semantic_std_{i}': v for i, v in
                   enumerate(record['semantic_audit']['predicted_std'])})
    result.update({f'02_objectives/predicted_roi_semantic_std_{i}': v for i, v in
                   enumerate(record['predicted_semantic_audit']['predicted_std'])})
    for domain,metrics in record['semantic_audit']['domains'].items():
        result[f'03_semantic/{domain}_nll'] = metrics['nll']
        result[f'03_semantic/{domain}_prior_nll'] = metrics['prior_nll']
        for kind in ('std_ratio','correlation'):
            result.update({f'03_semantic/{domain}_{kind}_{i}':v for i,v in enumerate(metrics[kind])})
    result['03_semantic/consecutive_failed_audits'] = record['semantic_guard_failures']
    return result
