"""Resume all three warmup components to a measured optimization plateau.

Each invocation trains one component, restoring its weights and Adam moments.
The existing joint run and its preparation directory are read-only inputs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from back_prop.common.checkpoint_compat import adapt_cuda_rng_state
from back_prop.common.coarse_data import load_cases
from back_prop.common.refiner import FineMaskRefiner3D
from back_prop.common.rng import preserve_rng
from back_prop.common.train_utils import atomic_save, distributed_setup, finish_scaled_step
from .convergence import PlateauStop
from .detection_metrics import detection_counts, summarize_detection_counts
from .detector import select_mask_queries
from .features import CNNSemantics
from .loss import WholeCTCriterionV5, LossAccumulator
from .monitor import Monitor
from .native_proposals import NativeProposals
from .parallel_warmup import coarse_output, refiner_forward, RefinerROIs, write_json
from .preparation import SCHEMA
from .repair_detector import detector_from_config
from .warmup import SemanticROIs, audit_semantics

CONTINUATION_SCHEMA = 'v5_warmup_convergence_v1'
DETR_TERMS = ('object', 'box_l1', 'box_giou', 'coarse_dice', 'coarse_focal', 'missing_nodule')


def one_item(rows):
    return rows[0]


class ProjectedCases(Dataset):
    def __init__(self, projected, native, config):
        self.projected, self.native = Path(projected), Path(native)
        marker = json.loads((self.projected/'complete.json').read_text())
        native_marker = json.loads((self.native/'complete.json').read_text())
        provenance = native_marker['provenance']
        from . import native_proposals
        if (marker['cases'] != native_marker['cases'] or native_marker['split'] != 'training'
                or provenance != config['native_cache_provenance']
                or provenance['module_sha256'] != hashlib.sha256(Path(native_proposals.__file__).read_bytes()).hexdigest()
                or Path(marker['source_checkpoint']).resolve() != Path(provenance['source_checkpoint']).resolve()
                or provenance['manifest_sha256'] != hashlib.sha256(Path(config['manifest']).read_bytes()).hexdigest()):
            raise ValueError('Frozen detector inputs do not match the continued checkpoint')
        self.case_ids = native_marker['case_ids']
        training_ids = {f"{r['patient_id']}__scan{int(r['scan_id'])}" for r in load_cases(config['manifest'])}
        if len(set(self.case_ids)) != len(self.case_ids) or not set(self.case_ids) <= training_ids:
            raise ValueError('Detector cache is not a unique subset of the training cohort')
        for directory in (self.projected, self.native):
            if not all((directory/f'{i:06d}.pt').is_file() for i in range(len(self))):
                raise ValueError(f'Incomplete detector cache: {directory}')

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, index):
        row = torch.load(self.projected/f'{index:06d}.pt', map_location='cpu', weights_only=False, mmap=True)
        native = torch.load(self.native/f'{index:06d}.pt', map_location='cpu', weights_only=False, mmap=True)
        if row['case_id'] != self.case_ids[index] or native['case_id'] != row['case_id']:
            raise ValueError('Detector cache case alignment mismatch')
        if tuple(native['isotropic_shape']) != tuple(row['isotropic_shape']):
            raise ValueError('Detector cache geometry mismatch')
        row['native'] = native['proposals']
        return row


def detection_forward(module, row, device):
    p = row['native']
    native = NativeProposals(p.boxes.to(device), p.object_logits.to(device),
        p.coarse_logits.to(device), p.origins, [])
    objects, boxes, masks, _ = module(row['screening'].to(device), native)
    return objects[0], boxes[0], masks[0]


def optimizer_for(branch, module, config):
    if branch == 'detr':
        heads = ('query_projection.', 'local_projection.', 'prediction_head.')
        named = list(module.named_parameters())
        groups = [dict(params=[p for k,p in named if not k.startswith((*heads, 'native_'))]),
                  dict(params=[p for k,p in named if k.startswith(heads)]),
                  dict(params=[p for k,p in named if k.startswith('native_')])]
    else:
        groups = module.parameters()
    return torch.optim.AdamW(groups, lr=config['lr_main'], weight_decay=config['weight_decay'])


def module_for(branch, config):
    if branch == 'detr':
        return detector_from_config(config)
    if branch == 'refiner':
        return FineMaskRefiner3D(input_channels=4+config['context_channels'],
            query_dim=config['detr_hidden_dim'], base_channels=config['refiner_base'])
    return CNNSemantics(config['semantic_roi_size'], config['semantic_width'])


def restore_rng(saved, rank):
    # Preserve available streams across a GPU topology change; record that
    # topology in the new checkpoint instead of claiming bitwise replay.
    rng = saved['rng_states'][rank % len(saved['rng_states'])]
    torch.set_rng_state(rng['torch'])
    torch.cuda.set_rng_state(adapt_cuda_rng_state(rng['cuda'], torch.cuda.get_rng_state().numel()))
    np.random.set_state(rng['numpy'])
    random.setstate(rng['python'])


def payload(module, optimizer, scaler, branch, epoch, audit, control, config, source, rank, world, monitor):
    rng = dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(),
               numpy=np.random.get_state(), python=random.getstate())
    states = [None]*world
    if world > 1:
        dist.all_gather_object(states, rng)
    else:
        states = [rng]
    if rank:
        return None
    return dict(schema=SCHEMA, continuation_schema=CONTINUATION_SCHEMA, branch=branch,
        epoch=epoch, model=module.state_dict(), optimizer=optimizer.state_dict(),
        scaler=scaler.state_dict(), rng_states=states, world_size=world, audit=audit,
        ready=False, consecutive_passes=control.passing, convergence=control.state_dict(),
        config=config, source_checkpoint=str(source.resolve()),
        source='continued_official_training_annotations', wandb=monitor.state_dict())


@torch.no_grad()
def probe(branch, module, data, config, device, rank, world, official):
    """Fixed random corruptions and fixed training-cohort samples each epoch."""
    module.eval()
    with preserve_rng():
        torch.manual_seed(config['seed']+9833)
        if branch == 'semantic':
            args = SimpleNamespace(**{**config, 'semantic_probe_nodules':512})
            audit = audit_semantics(SimpleNamespace(semantics=module, device=device), args, official)
            loss = sum(m['nll'] for m in audit['domains'].values())/len(audit['domains'])
            return loss, audit
        size = min(len(data), 128 if branch == 'detr' else 256)
        order = torch.randperm(len(data), generator=torch.Generator().manual_seed(config['seed']+9731))[:size]
        if branch == 'refiner':
            totals = []
            for i in order:
                with torch.autocast('cuda', dtype=getattr(torch, config['amp_dtype'])):
                    loss, metrics = refiner_forward(module, data[int(i)], device)
                totals.append([float(loss), float(metrics['dice']), float(metrics['prior_dice'])])
            values = np.asarray(totals).mean(0)
            audit = dict(nodules=size, dice=float(values[1]), prior_dice=float(values[2]))
            audit['passed'] = audit['dice'] >= config['refiner_min_dice'] and audit['dice'] >= audit['prior_dice']
            return float(values[0]), audit
        names = ('nodules','scans','selected','coverage','coarse_dice','box_iou',
                 'center_error_mm','true_positives','false_positives')
        totals = torch.zeros(len(names), device=device, dtype=torch.float64)
        accumulator = LossAccumulator(device)
        criterion = criterion_for(config)
        for i in order[rank::world]:
            row = data[int(i)]
            with torch.autocast('cuda', dtype=getattr(torch, config['amp_dtype'])):
                prediction = detection_forward(module, row, device)
                out = coarse_output(prediction, row)
                losses = criterion(out, row, diagnostic_scale=0., semantic_scale=0., synchronize=False)
            accumulator.update(losses)
            objects, boxes, masks = prediction
            selected = select_mask_queries(objects, masks, config['inference_object_threshold'], config['detector_nms_iou'])
            values = detection_counts(objects, boxes, masks, row, selected,
                                      tuple(config['roi_shape']), config['minimum_roi_coverage'])
            totals += totals.new_tensor([values[k] for k in names])
        if world > 1:
            dist.all_reduce(totals)
        audit = summarize_detection_counts(dict(zip(names, totals.cpu().tolist())))
        audit['passed'] = (audit['coverage'] >= config['geometry_min_coverage']
            and audit['coarse_dice'] >= config['geometry_min_coarse_dice']
            and audit['detection_recall'] >= .65 and audit['false_positives_per_scan'] <= 4.)
        audit['losses'] = accumulator.compute(detector_weights(criterion), synchronize=world>1)
        return audit['losses']['active_total'], audit


def criterion_for(config):
    return WholeCTCriterionV5(missing_nodule_weight=config['missing_nodule_weight'],
                             object_background_weight=config['object_background_weight'])


def detector_weights(criterion):
    return {k:(v if k in DETR_TERMS else 0.) for k,v in criterion.loss_weights.items()}


def train_branch(args):
    rank, local_rank, world = distributed_setup()
    if args.branch != 'detr' and world != 1:
        raise ValueError('ROI branches use one independent GPU each')
    device = torch.device('cuda', local_rank)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    config = json.loads((args.source_prepared/'config.json').read_text())
    manifest = json.loads(Path(config['manifest']).read_text())
    roi_root = args.roi_cache_root or args.source_prepared
    official_marker = json.loads((roi_root/'official_cache_complete.json').read_text())
    if official_marker['manifest_sha256'] != hashlib.sha256(Path(config['manifest']).read_bytes()).hexdigest():
        raise ValueError('Official ROI cache manifest changed')
    training = {r['patient_id'] for r in manifest['training']}
    if training & {r['patient_id'] for r in manifest['testing']}:
        raise ValueError('Training/test patients overlap')
    source = (args.detector_checkpoint if args.branch == 'detr'
              else args.source_prepared/f'{args.branch}_warmup.pt')
    root = args.output_dir/args.branch
    root.mkdir(parents=True, exist_ok=True)
    if (root/'converged.pt').exists():
        raise ValueError('This branch is already complete; refusing to overwrite it')
    latest = root/'latest.pt'
    restore = latest if latest.exists() else (root/'best.pt' if (root/'best.pt').exists() else source)
    saved = torch.load(restore, map_location='cpu', weights_only=False)
    resuming = saved.get('continuation_schema') == CONTINUATION_SCHEMA
    if resuming and (saved['branch'] != args.branch or Path(saved['source_checkpoint']).resolve() != source.resolve()):
        raise ValueError('Continuation source checkpoint changed')
    module = module_for(args.branch, config).to(device)
    module.load_state_dict(saved['model'], strict=True)
    optimizer = optimizer_for(args.branch, module, config)
    optimizer.load_state_dict(saved['optimizer'])
    scaler = torch.amp.GradScaler('cuda', enabled=config['amp_dtype']=='float16', init_scale=config['amp_init_scale'])
    if 'scaler' in saved:
        scaler.load_state_dict(saved['scaler'])
    control = PlateauStop(**saved['convergence']) if resuming else PlateauStop(minimum_epochs=args.minimum_additional_epochs)
    config.update(convergence_schema=CONTINUATION_SCHEMA, convergence_options=dict(
        minimum_additional_epochs=control.minimum_epochs, maximum_additional_epochs=args.maximum_additional_epochs,
        fixed_probe_relative_delta=control.relative_delta, training_relative_delta=control.training_delta,
        patience=control.patience, lr_patience=control.lr_patience,
        minimum_lr_reductions=2, maximum_lr_reductions=3),
        convergence_probe_split='training; previously used validation patients are not held out',
        original_checkpoint=str(source.resolve()), continued_world_size=world)
    project = Path(__file__).resolve().parents[2]
    config['original_source_sha256'] = config.pop('source_sha256', {})
    config['source_sha256'] = {str(p.relative_to(project)):hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ('common','model_v4','model_v5')
        for p in sorted((project/'back_prop'/folder).glob('*.py'))}
    config['original_runtime'] = config.pop('runtime', {})
    config['runtime'] = dict(torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(device), world_size=world,
        nccl_environment={k:v for k,v in os.environ.items() if k.startswith('NCCL_')})
    official = roi_root/'official_warmup'
    if args.branch == 'detr':
        data = ProjectedCases(args.projected_cache, args.native_cache, config)
        batch_size, collate = 1, one_item
    elif args.branch == 'refiner':
        data = RefinerROIs(sorted((roi_root/'refiner_official').glob('*.pt')), config['teacher_jitter'])
        batch_size, collate = None, None
    else:
        data = SemanticROIs(sorted(official.glob('*.pt')), config['teacher_jitter'])
        batch_size, collate = config['warmup_batch_size'], None
    if args.branch != 'detr' and len(data) != official_marker['nodules']:
        raise ValueError('Incomplete official ROI cache')
    sampler = DistributedSampler(data, num_replicas=world, rank=rank, seed=config['seed']+171)
    loader = DataLoader(data, batch_size=batch_size, sampler=sampler, collate_fn=collate,
                        num_workers=0)
    monitor = None
    if rank == 0:
        write_json(config, root/'config.json')
        monitor = Monitor(SimpleNamespace(output_dir=root, wandb_mode=args.wandb_mode,
            wandb_entity=config['wandb_entity'], wandb_project=config['wandb_project'], max_cases=None,
            wandb_name=f'v5-continue-{args.branch}-{args.output_dir.name}'), config,
            saved.get('wandb') if resuming else None)
        if monitor.run is not None:
            monitor.run.tags = ('v5', 'continued-warmup', 'convergence', args.branch)
    restore_rng(saved, rank)
    start = int(saved['epoch'])
    initial_epoch = start-control.completed
    wrapped = DDP(module, broadcast_buffers=False, find_unused_parameters=True) if world>1 else module
    criterion = criterion_for(config)
    weights = detector_weights(criterion)
    if not resuming:
        initial_loss, audit = probe(args.branch, module, data, config, device, rank, world, official)
        control.baseline(initial_loss)
        state = payload(module, optimizer, scaler, args.branch, start, audit, control, config, source, rank, world, monitor)
        if rank == 0:
            state['probe_loss'] = initial_loss
            atomic_save(state, latest)
            if audit['passed']:
                atomic_save(state, root/'best.pt')
            write_json(dict(epoch=start, loss=initial_loss, audit=audit), root/'initial_probe.json')
            print(json.dumps(dict(event='continued_warmup_start', branch=args.branch, epoch=start,
                loss=initial_loss, samples=len(data), optimizer_restored=True, audit=audit)), flush=True)
    best_loss = None
    if (root/'best.pt').exists():
        best_loss = torch.load(root/'best.pt', map_location='cpu', weights_only=False)['probe_loss']
    if world>1:
        dist.barrier()
    for epoch in range(start, initial_epoch+args.maximum_additional_epochs):
        started = time.monotonic()
        sampler.set_epoch(epoch)
        module.train()
        sums = {}
        count = 0
        accumulator = LossAccumulator(device)
        for step, row in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=getattr(torch, config['amp_dtype'])):
                if args.branch == 'detr':
                    out = coarse_output(detection_forward(wrapped, row, device), row)
                    losses = criterion(out, row, diagnostic_scale=0., semantic_scale=0., synchronize=world>1)
                    loss = losses.total
                    accumulator.update(losses)
                    metrics, n = {}, 1
                elif args.branch == 'refiner':
                    loss, metrics = refiner_forward(module, row, device)
                    n = 1
                else:
                    roi, hist = row
                    p, _ = module(roi.to(device), prepared=True)
                    loss = -(hist.to(device)*p.float().clamp_min(1e-7).log()).sum(-1).mean()
                    metrics, n = {}, len(roi)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite continued warmup loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm, _, _ = finish_scaled_step(optimizer, scaler, list(module.parameters()), max_norm=5.)
            count += n
            for key, value in dict(loss=loss, **metrics).items():
                sums[key] = sums.get(key, 0.)+float(value.detach())*n
            if rank == 0 and (step == 1 or step%200 == 0):
                print(json.dumps(dict(event='continued_warmup_step', branch=args.branch,
                    epoch=epoch+1, step=step, loss=float(loss.detach()), gradient_norm=norm)), flush=True)
        if args.branch == 'detr':
            losses = accumulator.compute(weights, synchronize=world>1)
            training_loss = losses['active_total']
            metrics = {'loss_'+k:losses[k] for k in DETR_TERMS}
        else:
            metrics = {k:v/count for k,v in sums.items()}
            training_loss = metrics['loss']
        probe_loss, audit = probe(args.branch, module, data, config, device, rank, world, official)
        decision = control.update(probe_loss, training_loss, audit['passed'])
        if decision['reduce_lr']:
            for group in optimizer.param_groups:
                group['lr'] *= .5
        state = payload(module, optimizer, scaler, args.branch, epoch+1, audit, control, config, source, rank, world, monitor)
        if rank == 0:
            state['probe_loss'] = probe_loss
            atomic_save(state, latest)
            if audit['passed'] and (best_loss is None or probe_loss < best_loss):
                best_loss = probe_loss
                atomic_save(state, root/'best.pt')
            scalars = dict(epoch=epoch+1, training_loss=training_loss, fixed_probe_loss=probe_loss,
                best_probe_loss=best_loss, seconds=time.monotonic()-started, **metrics, **decision)
            scalars.update({k:v for k,v in audit.items() if isinstance(v,(int,float,bool))})
            scalars.update({f'lr_{i}':g['lr'] for i,g in enumerate(optimizer.param_groups)})
            if args.branch == 'semantic':
                from back_prop.common.features import SEMANTIC_NAMES
                for domain, values in audit['domains'].items():
                    scalars[domain+'_nll'] = values['nll']
                    for kind in ('std_ratio','correlation','nll_by_attribute'):
                        scalars.update({f'{domain}_{kind}_{name}':v for name,v in zip(SEMANTIC_NAMES,values[kind])})
            monitor.log(dict(warmup_epoch=epoch+1, **{args.branch+'_warmup/'+k:v for k,v in scalars.items()}))
            with (root/'history.jsonl').open('a') as f:
                f.write(json.dumps(dict(probe_loss=probe_loss, audit=audit, **scalars))+'\n')
            print(json.dumps(dict(event='continued_warmup_epoch', branch=args.branch, **scalars)), flush=True)
            if decision['converged']:
                if best_loss is None:
                    raise RuntimeError('No qualifying checkpoint to assemble')
                best = torch.load(root/'best.pt', map_location='cpu', weights_only=False)
                best.update(ready=True, convergence_complete=True, convergence_decision=decision,
                            stopped_epoch=epoch+1, consecutive_passes=control.passing)
                atomic_save(best, root/'converged.pt')
                write_json(dict(selected_epoch=best['epoch'], stopped_epoch=epoch+1,
                    initial_epoch=initial_epoch, probe_loss=best['probe_loss'], audit=best['audit'],
                    convergence=decision), root/'converged.json')
        if world>1:
            dist.barrier()
        if decision['converged']:
            break
    else:
        raise RuntimeError('Epoch cap reached without convergence; latest state retained; joint training is blocked')
    if rank == 0:
        monitor.finish()
    if world>1:
        dist.destroy_process_group()


def export_preparation(args):
    """The only route to joint training requires all three convergence results."""
    states = {}
    for branch in ('detr', 'refiner', 'semantic'):
        state = torch.load(args.output_dir/branch/'converged.pt', map_location='cpu', weights_only=False)
        if (state.get('continuation_schema') != CONTINUATION_SCHEMA or not state.get('convergence_complete')
                or not state['convergence_decision']['converged'] or not state['audit']['passed']):
            raise ValueError(f'{branch} has not converged with passing quality checks')
        module_for(branch, state['config']).load_state_dict(state['model'], strict=True)
        states[branch] = state
    root = args.output_dir/'prepared'
    root.mkdir(exist_ok=False)
    old = torch.load(args.source_prepared/'detr_warmup.pt', map_location='cpu', weights_only=False)
    combined = {k:v for k,v in old['model'].items() if k.startswith('segmenter.')}
    combined.update({f'detr.{k}':v for k,v in states['detr']['model'].items()})
    # Completed preparation imports weights only. The source branch checkpoints
    # retain resumable optimizers, which have different parameter groups here.
    states['detr']['model'] = combined
    for branch, state in states.items():
        atomic_save(state, root/f'{branch}_warmup.pt')
        write_json(dict(epoch=state['epoch'], audit=state['audit'], converged=True), root/f'{branch}_ready.json')
    for name in ('official_warmup', 'refiner_official'):
        (root/name).symlink_to((args.source_prepared/name).resolve(), target_is_directory=True)
    shutil.copy2(args.source_prepared/'official_cache_complete.json', root/'official_cache_complete.json')
    pointer = torch.load(args.source_prepared/'preparation.pt', map_location='cpu', weights_only=False)
    pointer.update(cache_root=str(root.resolve()), convergence_schema=CONTINUATION_SCHEMA,
                   convergence_source=str(args.output_dir.resolve()))
    pointer['config'].update(output_dir=str(root.resolve()), prepare_resume=str((root/'preparation.pt').resolve()))
    atomic_save(pointer, root/'preparation.pt')
    config = json.loads((args.source_prepared/'config.json').read_text())
    config.update(output_dir=str(root.resolve()), prepare_resume=str((root/'preparation.pt').resolve()),
                  convergence_source=str(args.output_dir.resolve()))
    write_json(config, root/'config.json')
    print(json.dumps(dict(event='converged_preparation_exported', path=str(root/'preparation.pt'),
        epochs={k:v['epoch'] for k,v in states.items()})), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--branch', choices=('detr','refiner','semantic','export'), required=True)
    parser.add_argument('--source-prepared', type=Path, required=True)
    parser.add_argument('--detector-checkpoint', type=Path)
    parser.add_argument('--projected-cache', type=Path)
    parser.add_argument('--native-cache', type=Path)
    parser.add_argument('--roi-cache-root', type=Path,
                        help='Optional local copy of the two official ROI caches and their provenance marker')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--minimum-additional-epochs', type=int, default=20)
    parser.add_argument('--maximum-additional-epochs', type=int, default=200)
    parser.add_argument('--wandb-mode', choices=('online','offline','disabled'), default='online')
    args = parser.parse_args()
    if args.output_dir.resolve() == args.source_prepared.resolve():
        parser.error('Continuation must use a new output directory')
    if not 1 <= args.minimum_additional_epochs <= args.maximum_additional_epochs:
        parser.error('Invalid continuation epoch bounds')
    if args.branch == 'export':
        export_preparation(args)
    else:
        train_branch(args)


if __name__ == '__main__':
    main()
