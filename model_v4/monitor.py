"""Epoch-only W&B metrics and an explicit chart order shared by both variants."""
import json
import uuid
from back_prop.common.rng import preserve_rng

COMPONENTS = ('box_l1', 'box_giou', 'coarse_dice', 'coarse_focal', 'fine_dice', 'fine_focal')
GROUPS = ('missing_nodule', 'semantic', 'nodule', 'risk', 'detr_loss', 'finetuner_loss')
LOSS_NAMES = ('total',) + COMPONENTS + GROUPS


def epoch_metrics(record, weights):
    result = {'epoch': record['epoch'], 'global_step': record['global_step']}
    result['00_training/total'] = record['total']
    for name, value in record['lr'].items():
        result[f'00_training/lr_{name}'] = value
    result['00_training/temperature'] = record['temperature']
    for index, name in enumerate(COMPONENTS, 1):
        result[f'01_components/{index:02d}_{name}'] = record[name]
    for index, name in enumerate(GROUPS, 1):
        result[f'02_objectives/{index:02d}_{name}'] = record[name]
    # Group plots use effective weighted geometry losses. Others are raw;
    # separate weighted values make the exact total auditable.
    weighted = {k: record[k]*v*(record['diagnostic_scale'] if k in ('nodule','risk') else 1.)
                for k,v in weights.items()}
    result.update({f'00_training/weighted_{k}': v for k,v in weighted.items()})
    result['00_training/weighted_sum'] = sum(weighted.values())
    if abs(result['00_training/weighted_sum']-record['total']) > 1e-4*max(1,abs(record['total'])):
        raise AssertionError('Epoch loss contributions do not sum to total')
    return result


class Monitor:
    def __init__(self, args, config, saved=None):
        self.output_dir = args.output_dir
        self.mode = args.wandb_mode
        self.identity = saved or dict(id=uuid.uuid4().hex[:12], entity=args.wandb_entity,
                                     project=args.wandb_project)
        self.run = None
        if self.mode != 'disabled':
            with preserve_rng():
                import wandb
                self.run = wandb.init(**self.identity, name=args.wandb_name, mode=self.mode,
                    resume='must' if saved and self.mode == 'online' else None,
                    job_type='smoke' if args.max_cases else 'train',
                    tags=[config['variant'], 'groupnorm', 'risk-from-epoch1', 'gt-warmup', 'epoch-loss'],
                    config=json.loads(json.dumps(config, default=str)), dir=str(args.output_dir),
                    settings=wandb.Settings(console='off', save_code=False, disable_git=True))
                self.run.define_metric('epoch')
                self.run.define_metric('warmup_epoch')
                self.run.define_metric('warmup/*', step_metric='warmup_epoch')
                for prefix in ('00_training', '01_components', '02_objectives', 'sampling'):
                    self.run.define_metric(prefix+'/*', step_metric='epoch')
        state = {**self.identity, 'mode': self.mode,
                 'url': self.run.url if self.run is not None and self.mode == 'online' else None}
        (self.output_dir/'wandb_run.json').write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps(dict(event='wandb', **state)), flush=True)

    def log(self, metrics):
        if not ('epoch' in metrics or 'warmup_epoch' in metrics):
            raise ValueError('Only epoch-aggregated metrics may be logged')
        with (self.output_dir/'monitor.jsonl').open('a') as stream:
            stream.write(json.dumps(metrics, allow_nan=False)+'\n')
        if self.run is not None:
            with preserve_rng():
                self.run.log(metrics)

    def state_dict(self):
        return self.identity if self.mode != 'disabled' else None

    def finish(self):
        if self.run is not None:
            with preserve_rng():
                self.run.finish()
