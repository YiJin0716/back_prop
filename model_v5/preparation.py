"""Geometry preparation and resumable checkpoints before joint training."""
import json
import hashlib
import random
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from back_prop.common.semantic_data import whole_ct_collate
from back_prop.common.train_utils import atomic_save, finish_scaled_step
from back_prop.common.checkpoint_compat import adapt_cuda_rng_state
from .loss import WholeCTCriterionV5, LossAccumulator

SCHEMA = 'v5_semantic_repair_v1'


def preparation_save(model, optimizer, args, rank, world, *, phase, epoch, **metadata):
    rng = dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(model.device),
               numpy=np.random.get_state(), python=random.getstate())
    states = [None] * world if world > 1 else [rng]
    if world > 1:
        dist.all_gather_object(states, rng)
    if rank == 0:
        atomic_save(dict(preparation_schema=SCHEMA, phase=phase, epoch=epoch,
            model=model.state_dict(), optimizer=optimizer.state_dict() if optimizer else None,
            config={k:str(v) for k,v in vars(args).items()}, world_size=world,
            manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            rng_states=states, **metadata), args.output_dir/'preparation.pt')
    if world > 1:
        dist.barrier()


def preparation_restore(model, args, rank, world):
    if args.prepare_resume is None:
        return None
    saved = torch.load(args.prepare_resume, map_location='cpu', weights_only=False)
    if saved.get('preparation_schema') != SCHEMA or saved['world_size'] != world:
        raise ValueError('Preparation checkpoint schema/world size mismatch')
    if saved['manifest_sha256'] != hashlib.sha256(args.manifest.read_bytes()).hexdigest():
        raise ValueError('Preparation training manifest changed')
    mutable = {'output_dir','prepare_resume','resume','init_v2','wandb_name',
               'geometry_max_epochs','warmup_epochs','save_every','stop_after_epoch'}
    if args.preparation_smoke and saved['config'].get('preparation_smoke') == 'True':
        mutable.update(('epochs','smoke_curriculum_epoch'))
    for key,value in vars(args).items():
        if key not in mutable and str(value) != saved['config'].get(key):
            raise ValueError(f'Preparation resume argument differs: {key}')
    model.load_state_dict(saved['model'], strict=True)
    rng = saved['rng_states'][rank]
    torch.set_rng_state(rng['torch'])
    torch.cuda.set_rng_state(adapt_cuda_rng_state(rng['cuda'],
        torch.cuda.get_rng_state(model.device).numel()), model.device)
    np.random.set_state(rng['numpy']); random.setstate(rng['python'])
    # Cached crops belong to the geometry checkpoint that produced them.
    saved['cache_root'] = saved.get('cache_root', str(Path(args.prepare_resume).parent))
    return saved


def geometry_teacher(epoch, minimum_epochs):
    return max(0., 1. - epoch / max(minimum_epochs - 1, 1))


@torch.no_grad()
def geometry_audit(model, dataset, args, rank, world):
    model.eval()
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed+731))
    order = order[:args.geometry_probe_cases]
    totals = torch.zeros(4, device=model.device, dtype=torch.float64)
    for index in order[rank::world]:
        batch = dataset[int(index)]
        out = model(batch['image'], batch=batch, geometry_only=True, update_bank=False)
        assert out.teacher_forced_count == out.fallback_count == 0
        rows = ((out.refined_target_indices >= 0) & out.fine_supervision_valid).nonzero().flatten()
        # Uncovered GT has zero quality: no survivor-only Dice average.
        totals += totals.new_tensor([len(batch['nodule_ids']), len(rows),
            float(out.fine_soft_dice[rows].sum()), int(out.semantic_supervision_valid.sum())])
    if world > 1:
        dist.all_reduce(totals)
    n = int(totals[0]); denominator = max(n, 1)
    result = dict(nodules=n, coverage=float(totals[1]/denominator),
                  dice=float(totals[2]/denominator), usable_fraction=float(totals[3]/denominator))
    result['passed'] = (n > 0 and result['coverage'] >= args.geometry_min_coverage
        and result['dice'] >= args.geometry_min_dice
        and result['usable_fraction'] >= args.geometry_min_usable)
    return result


def run_geometry_warmup(model, dataset, args, rank, world, monitor, saved=None):
    if saved and saved['phase'] != 'geometry':
        return saved['geometry']
    # Freeze the semantic CNN and pretrained VISTA backbone; the pyramid
    # projections, DETR, context and refiner remain trainable.
    model.requires_grad_(True); model.semantics.requires_grad_(False)
    backbone = getattr(model.segmenter, 'model', None)
    if backbone is not None:
        backbone.requires_grad_(False)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr_main, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda',enabled=args.amp_dtype=='float16',init_scale=args.amp_init_scale)
    start = 0
    if saved:
        optimizer.load_state_dict(saved['optimizer']); start = saved['epoch'] + 1
        scaler.load_state_dict(saved['scaler'])
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, seed=args.seed+91)
    loader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0,
                        collate_fn=whole_ct_collate)
    wrapped = DistributedDataParallel(model, find_unused_parameters=True,
        broadcast_buffers=False) if world > 1 else model
    criterion = WholeCTCriterionV5(missing_nodule_weight=args.missing_nodule_weight,
                                   semantic_anchor_weight=args.semantic_anchor_weight)
    weights = {k: (0. if k in ('semantic','semantic_anchor','nodule','risk') else v)
               for k,v in criterion.loss_weights.items()}
    result = None
    for epoch in range(start, args.geometry_max_epochs):
        sampler.set_epoch(epoch); wrapped.train()
        if backbone is not None:
            backbone.eval()
        teacher = geometry_teacher(epoch, args.geometry_min_epochs)
        accumulator = LossAccumulator(model.device)
        for step,batch in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=getattr(torch,args.amp_dtype)):
                out = wrapped(batch['image'], batch=batch, teacher_probability=teacher,
                              geometry_only=True, update_bank=False)
                loss = criterion(out, batch, semantic_scale=0., diagnostic_scale=0.)
            if not torch.isfinite(loss.total):
                raise FloatingPointError('Nonfinite geometry warmup loss')
            scaler.scale(loss.total).backward(); scaler.unscale_(optimizer)
            finish_scaled_step(optimizer,scaler,parameters,max_norm=5.)
            accumulator.update(loss)
            if rank == 0:
                print(json.dumps(dict(event='geometry_step', epoch=epoch+1, step=step,
                    teacher=teacher, loss=float(loss.total.detach()))), flush=True)
            if args.preparation_smoke and step >= args.benchmark_steps:
                break
        metrics = accumulator.compute(weights)
        result = geometry_audit(model, dataset, args, rank, world)
        result.update(epochs=epoch+1, teacher_probability=teacher, losses=metrics)
        ready = epoch+1 >= args.geometry_min_epochs and result['passed']
        if args.preparation_smoke:
            ready = True  # Pipeline validation only; never accepted by normal resume.
        if rank == 0:
            print(json.dumps(dict(event='geometry_epoch', **result)), flush=True)
            monitor.log({'warmup_epoch':epoch+1,
                         **{f'geometry/{k}':v for k,v in result.items() if k!='losses'},
                         **{f'geometry/loss_{k}':metrics[k] for k in weights}})
        preparation_save(model, optimizer, args, rank, world,
            phase='geometry_complete' if ready else 'geometry', epoch=epoch,
            geometry=result, scaler=scaler.state_dict(),cache_root=str(args.output_dir.resolve()))
        if ready:
            break
    del wrapped, optimizer
    if result is None or (not args.preparation_smoke and not ready):
        raise RuntimeError('Geometry readiness not reached; preparation.pt preserves progress. '
                           'Inspect coverage/Dice before extending geometry_max_epochs.')
    return result
