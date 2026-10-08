#!/usr/bin/env python3
"""DDP training entry point for the coarse-to-fine model v5."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import Tensor, nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from back_prop.lidc_policy import POLICY_NAME
from back_prop.common.semantic_data import DEFAULT_MANIFEST, WholeCTLIDCDataset, load_cases, whole_ct_collate
from .loss import WholeCTCriterionV5, LossAccumulator
from .model import ARCHITECTURE, WholeCTJointModelV5
from back_prop.common.schedule import LearningRateSchedule, make_lr_scheduler, assert_learning_rates
from .features import NORMALIZATION, CONVOLUTION
from .warmup import run_warmup, AnchorStream
from .preparation import SCHEMA
from .monitor import Monitor, epoch_metrics
from .warmup import audit_semantics
from .diagnostics import SemanticAccumulator
from back_prop.common.features import FEATURE_NAMES, SEMANTIC_NAMES, RADIOMICS_NAMES
from back_prop.common.checkpoint_compat import adapt_cuda_rng_state
from back_prop.common.base_model import DEFAULT_VISTA_CHECKPOINT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--vista-checkpoint", type=Path, default=DEFAULT_VISTA_CHECKPOINT)
    parser.add_argument("--semantic-roi-size", type=int, default=64)
    parser.add_argument("--semantic-width", type=int, default=16)
    parser.add_argument("--inference-object-threshold", type=float, default=.5)
    parser.add_argument("--init-v2", type=Path)
    parser.add_argument("--amp-dtype", choices=['bfloat16', 'float16'], default='bfloat16')
    parser.add_argument("--sparsity", type=int, default=5)
    parser.add_argument("--coefficient-bound", type=float, default=5.0)
    parser.add_argument("--beam", type=int, default=10)
    parser.add_argument("--pool-size", type=int, default=30)
    parser.add_argument("--gap", type=float, default=0.05)
    parser.add_argument("--sample-fraction", type=float, default=0.3)
    parser.add_argument("--bank-min-samples", type=int, default=32)
    parser.add_argument("--bank-refresh-every", type=int, default=1)
    parser.add_argument("--bank-mode", choices=['sample', 'single', 'all', 'best'], default='sample')
    parser.add_argument("--missing-nodule-weight", type=float, default=2.0)
    parser.add_argument("--baseline-l2", type=float, default=1e-3)
    parser.add_argument("--lr-min-ratio", type=float, default=0.1)
    parser.add_argument("--lr-schedule-epochs", type=int, default=18)
    parser.add_argument("--stop-after-epoch", type=int, help="Save then stop; only for a limited smoke cohort")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--window-size", type=int, nargs=3, default=(144, 160, 160))
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--screen-shape", type=int, nargs=3, default=(64, 64, 64))
    parser.add_argument("--roi-shape", type=int, nargs=3, default=(128, 128, 128))
    parser.add_argument("--num-queries", type=int, default=24)
    parser.add_argument("--detr-hidden-dim", type=int, default=128)
    parser.add_argument("--detr-coarse-shape", type=int, nargs=3, default=(12, 12, 12))
    parser.add_argument("--detr-nheads", type=int, default=4)
    parser.add_argument("--detr-encoder-layers", type=int, default=2)
    parser.add_argument("--detr-decoder-layers", type=int, default=2)
    parser.add_argument("--context-channels", type=int, default=8)
    parser.add_argument("--refiner-base", type=int, default=8)
    parser.add_argument("--hard-negatives", type=int, default=2)
    parser.add_argument("--teacher-full-epochs", type=int, default=1)
    parser.add_argument("--teacher-zero-epoch", type=int, default=11)
    parser.add_argument("--teacher-jitter", type=int, default=8)
    parser.add_argument("--minimum-roi-coverage", type=float, default=0.8)
    parser.add_argument("--lr-main", type=float, default=1e-4)
    parser.add_argument("--lr-vista", type=float, default=1e-6)
    parser.add_argument("--lr-semantic", type=float, default=3e-5)
    parser.add_argument('--lr-semantic-warmup', type=float, default=3e-4)
    parser.add_argument('--geometry-min-epochs', type=int, default=5)
    parser.add_argument('--geometry-max-epochs', type=int, default=10)
    parser.add_argument('--geometry-probe-cases', type=int, default=32)
    parser.add_argument('--geometry-min-coverage', type=float, default=.75)
    parser.add_argument('--geometry-min-dice', type=float, default=.25)
    parser.add_argument('--geometry-min-usable', type=float, default=.5)
    parser.add_argument('--semantic-min-dice', type=float, default=.3)
    parser.add_argument('--semantic-min-epochs', type=int, default=8)
    parser.add_argument('--semantic-probe-nodules', type=int, default=128)
    parser.add_argument('--semantic-min-std-ratio', type=float, default=.15)
    parser.add_argument('--semantic-min-correlation', type=float, default=.25)
    parser.add_argument('--semantic-nll-margin', type=float, default=.01)
    parser.add_argument('--semantic-anchor-weight', type=float, default=.5)
    parser.add_argument('--semantic-anchor-batch-size', type=int, default=4)
    parser.add_argument('--semantic-guard-patience', type=int, default=2)
    parser.add_argument('--prepare-resume', type=Path)
    parser.add_argument('--preparation-smoke', action='store_true',
                        help='Limited pipeline test only; bypass preparation readiness gates')
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--amp-init-scale",
        type=float,
        default=4.0,
        help=(
            "Initial GradScaler value for optional FP16 runs. BF16 runs do not "
            "use loss scaling."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument(
        "--smoke-curriculum-epoch",
        type=int,
        help="Test one later curriculum phase; allowed only with --epochs 1 and --max-cases",
    )
    parser.add_argument("--save-every", type=int, default=3)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--skip-save", action="store_true")
    parser.add_argument("--benchmark-steps", type=int,
                        help="Stop after this many synchronized steps; requires --skip-save")
    parser.add_argument('--warmup-epochs', type=int, default=20,
                        help='Maximum augmented semantic preparation epochs')
    parser.add_argument('--warmup-batch-size', type=int, default=8)
    parser.add_argument('--wandb-mode', choices=['online','offline','disabled'], default='online')
    parser.add_argument('--wandb-entity', default='jin20020716-duke-university')
    parser.add_argument('--wandb-project', default='imaging-feature')
    parser.add_argument('--wandb-name')
    return parser.parse_args()


from back_prop.common.train_utils import (
    distributed_setup, reduce, gradient_norm, finish_scaled_step, atomic_save,
)


def set_training_phase(model: WholeCTJointModelV5, epoch: int) -> str:
    vista_base = getattr(model.segmenter, "model", None)
    if vista_base is not None:
        for parameter in vista_base.parameters():
            parameter.requires_grad_(False)
        if epoch >= 15:
            image_encoder = getattr(vista_base, "image_encoder", None)
            if image_encoder is None or not len(image_encoder.up_layers_auto):
                raise RuntimeError("Cannot locate VISTA3D final automatic decoder block")
            for parameter in image_encoder.up_layers_auto[-1].parameters():
                parameter.requires_grad_(True)
            for parameter in vista_base.class_head.parameters():
                parameter.requires_grad_(True)
    if epoch < 15:
        return "joint_diagnostics"
    return "end_to_end_unfrozen"


def diagnostic_scale(epoch: int) -> float:
    # Official-mask warmup precedes this loop. Both risk objectives participate
    # at their full configured weights from the first joint-training update.
    return 1.0


def main() -> None:
    args = parse_args()
    variant = 'v5'
    if args.preparation_smoke and not (args.max_cases and args.benchmark_steps and args.skip_save):
        raise ValueError('--preparation-smoke requires --max-cases, --benchmark-steps and --skip-save')
    if not (1 <= args.geometry_min_epochs <= args.geometry_max_epochs
            and 1 <= args.semantic_min_epochs <= args.warmup_epochs):
        raise ValueError('Preparation epoch bounds are invalid')
    if min(args.semantic_anchor_batch_size,args.geometry_probe_cases,args.semantic_probe_nodules,
           args.semantic_guard_patience) < 1:
        raise ValueError('Preparation/probe/anchor sizes and guard patience must be positive')
    for key in ('geometry_min_coverage','geometry_min_dice','geometry_min_usable',
                'semantic_min_dice','semantic_min_std_ratio','semantic_min_correlation'):
        if not 0 <= getattr(args,key) <= 1:
            raise ValueError(f'{key} must be in [0, 1]')
    if sum(p is not None for p in (args.resume,args.prepare_resume,args.init_v2)) > 1:
        raise ValueError('Choose joint resume, preparation resume, or geometry initialization')
    if not 0 <= args.inference_object_threshold <= 1:
        raise ValueError('Object threshold must be in [0, 1]')
    if args.smoke_curriculum_epoch is None and args.epochs - args.teacher_zero_epoch + 1 < 3:
        raise ValueError('V5 requires at least three epochs after teacher reaches zero')
    if args.benchmark_steps is not None and (args.benchmark_steps < 1 or not args.skip_save):
        raise ValueError('--benchmark-steps requires a positive value and --skip-save')
    if args.smoke_curriculum_epoch is not None and (
        args.epochs != 1 or args.max_cases is None or args.smoke_curriculum_epoch < 0
    ):
        raise ValueError(
            "--smoke-curriculum-epoch requires --epochs 1, --max-cases, and a nonnegative value"
        )
    if args.stop_after_epoch is not None and (args.max_cases is None or args.stop_after_epoch < 1):
        raise ValueError('--stop-after-epoch requires a limited smoke cohort and positive epoch')
    lr_schedule = LearningRateSchedule(min_ratio=args.lr_min_ratio, epochs=args.lr_schedule_epochs)
    base_lrs = dict(main=args.lr_main, vista=args.lr_vista, semantic=args.lr_semantic)
    rank, local_rank, world = distributed_setup()
    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    manifest_cases = load_cases(args.manifest, "training")
    dataset = WholeCTLIDCDataset(
        manifest_cases,
        target_mask_shape=tuple(args.screen_shape),
        max_cases=args.max_cases,
    )
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=args.seed)
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=0,
        pin_memory=False,
        collate_fn=whole_ct_collate,
    )
    # Only the manifest training cohort is allowed to update the model bank.
    test_cases = load_cases(args.manifest, "testing")
    training_patients = {c['patient_id'] for c in dataset.cases}
    if training_patients & {c['patient_id'] for c in test_cases}:
        raise ValueError('Training/testing patient overlap')
    device = torch.device("cuda", local_rank)
    amp_dtype = getattr(torch, args.amp_dtype)
    if amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError('This GPU does not support BF16; explicitly select another AMP dtype')
    if args.resume and args.init_v2:
        raise ValueError('Use either --resume or --init-v2')
    bare_model = WholeCTJointModelV5(
        window_size=tuple(args.window_size),
        overlap=args.overlap,
        screen_shape=tuple(args.screen_shape),
        roi_shape=tuple(args.roi_shape),
        num_queries=args.num_queries,
        detr_hidden_dim=args.detr_hidden_dim,
        detr_coarse_shape=tuple(args.detr_coarse_shape),
        detr_nheads=args.detr_nheads,
        detr_encoder_layers=args.detr_encoder_layers,
        detr_decoder_layers=args.detr_decoder_layers,
        context_channels=args.context_channels,
        refiner_base_channels=args.refiner_base,
        hard_negatives=args.hard_negatives,
        teacher_full_epochs=args.teacher_full_epochs,
        teacher_zero_epoch=args.teacher_zero_epoch,
        teacher_jitter=args.teacher_jitter,
        minimum_roi_coverage=args.minimum_roi_coverage,
        vista_checkpoint=args.vista_checkpoint,
        semantic_roi_size=args.semantic_roi_size,
        semantic_width=args.semantic_width,
        semantic_min_dice=args.semantic_min_dice,
        inference_object_threshold=args.inference_object_threshold,
        amp_dtype=amp_dtype,
        baseline_l2=args.baseline_l2,
        bank_config=dict(sparsity=args.sparsity, bound=args.coefficient_bound,
                         beam=args.beam, pool_size=args.pool_size, gap=args.gap,
                         fraction=args.sample_fraction, min_samples=args.bank_min_samples,
                         refresh_every=args.bank_refresh_every, mode=args.bank_mode),
    ).to(device)
    if args.init_v2:
        initial = torch.load(args.init_v2, map_location='cpu', weights_only=False)
        shared = {k: v for k, v in initial['model'].items()
                  if not k.startswith(('ordinal_heads.', 'radiomics.'))}
        incompatible = bare_model.load_state_dict(shared, strict=False)
        if incompatible.unexpected_keys or any(not k.startswith(('semantics.', 'rashomon.', 'mask_temperature'))
                                               for k in incompatible.missing_keys):
            raise RuntimeError(f'Geometry warm-start mismatch: {incompatible}')
        del initial, shared
    bare_model.rashomon.allowed_cases = {f"{c['patient_id']}__scan{int(c['scan_id'])}" for c in dataset.cases}
    criterion = WholeCTCriterionV5(
        missing_nodule_weight=args.missing_nodule_weight,
        semantic_anchor_weight=args.semantic_anchor_weight,
    ).to(device)

    vista_base_module = getattr(bare_model.segmenter, "model", None)
    vista_parameters = list(vista_base_module.parameters()) if vista_base_module else []
    semantic_parameters = list(bare_model.semantics.parameters())
    special_ids = {id(parameter) for parameter in vista_parameters + semantic_parameters}
    main_parameters = [
        parameter for parameter in bare_model.parameters() if id(parameter) not in special_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": main_parameters, "lr": args.lr_main, "name": "main"},
            {"params": vista_parameters, "lr": args.lr_vista, "name": "vista"},
            {"params": semantic_parameters, "lr": args.lr_semantic, "name": "semantic"},
        ],
        weight_decay=args.weight_decay,
    )
    if args.amp_init_scale <= 0:
        raise ValueError("--amp-init-scale must be positive")
    scaler = torch.cuda.amp.GradScaler(
        enabled=amp_dtype == torch.float16,
        init_scale=args.amp_init_scale,
        growth_interval=1000,
    )
    start_epoch = 0
    global_step = 0
    saved_lr_scheduler = None
    resume_summary = None
    saved_wandb = None
    warmup_state = None
    semantic_guard_failures = 0
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if saved.get("architecture") != ARCHITECTURE:
            raise ValueError(f"Resume architecture mismatch: {saved.get('architecture')}")
        if saved['config'].get('variant') != variant:
            raise ValueError('Cannot resume a different training variant')
        saved_wandb = saved.get('wandb')
        warmup_state = saved.get('warmup')
        if not warmup_state or not warmup_state['complete'] or warmup_state.get('schema') != SCHEMA:
            raise ValueError('Joint resume requires repaired staged preparation; old V5 remains inference-compatible')
        if warmup_state.get('smoke_only') and not args.preparation_smoke:
            raise ValueError('Smoke preparation is not a production checkpoint')
        semantic_guard_failures = saved.get('semantic_guard_failures',0)
        saved_policy = saved.get("config", {}).get("malignancy_policy", {}).get("name")
        if saved_policy != POLICY_NAME:
            raise ValueError(f"Resume malignancy policy mismatch: {saved_policy}")
        if saved['lr_schedule'] != lr_schedule.state_dict():
            raise ValueError('Resume LR schedule mismatch')
        if saved['config']['loss_weights'] != criterion.loss_weights:
            raise ValueError('Resume loss weights mismatch')
        mutable = {'output_dir', 'resume', 'prepare_resume', 'init_v2', 'epochs', 'stop_after_epoch', 'save_every', 'wandb_name'}
        for name, value in vars(args).items():
            if name not in mutable and str(value) != str(saved['config'][name]):
                raise ValueError(f'Resume argument mismatch: {name}')
        if saved['config']['world_size'] != world:
            raise ValueError('Resume requires the same number of ranks')
        if saved['config']['manifest_sha256'] != hashlib.sha256(args.manifest.read_bytes()).hexdigest():
            raise ValueError('Resume manifest contents changed')
        saved_lr_scheduler = saved['lr_scheduler']
        global_step = saved['global_step']
        bare_model.load_state_dict(saved["model"], strict=True)
        if float(bare_model.mask_temperature) != 1.0:
            raise ValueError('V5 requires fixed mask temperature 1.0')
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        start_epoch = int(saved["epoch"]) + 1
        allowed = {f"{c['patient_id']}__scan{int(c['scan_id'])}" for c in dataset.cases}
        if bare_model.rashomon.allowed_cases != allowed:
            raise ValueError('Resume training cohort differs from cached model bank')
        if saved['config']['amp_dtype'] != args.amp_dtype:
            raise ValueError('Resume AMP dtype differs from checkpoint')
        rng_states = saved.get('rng_states', [])
        cuda_rng_format = None
        if len(rng_states) == world:
            rng = rng_states[rank]
            torch.set_rng_state(rng['torch'])
            cuda_rng = adapt_cuda_rng_state(rng['cuda'], torch.cuda.get_rng_state(device).numel())
            torch.cuda.set_rng_state(cuda_rng, device)
            if not torch.equal(cuda_rng, torch.cuda.get_rng_state(device)):
                raise RuntimeError('CUDA RNG state changed while restoring checkpoint')
            cuda_rng_format = dict(saved_bytes=rng['cuda'].numel(), restored_bytes=cuda_rng.numel())
            np.random.set_state(rng['numpy'])
            random.setstate(rng['python'])
        resume_summary = dict(checkpoint=str(args.resume), completed_epochs=start_epoch,
                              cache_size=len(bare_model.rashomon.memory),
                              bank_size=int(bare_model.rashomon.count),
                              bank_fits=int(bare_model.rashomon.fit_count),
                              optimizer_state_entries=len(optimizer.state),
                              cuda_rng_format=cuda_rng_format,
                              rng_restored=len(rng_states) == world)
        del saved

    next_lr_epoch = args.smoke_curriculum_epoch if args.smoke_curriculum_epoch is not None else start_epoch
    lr_scheduler = make_lr_scheduler(optimizer, lr_schedule, base_lrs,
                                     next_epoch=next_lr_epoch, saved_state=saved_lr_scheduler)
    run_config = {
        **vars(args),
        "architecture": ARCHITECTURE,
        "variant": variant,
        "preparation_schema": SCHEMA,
        "semantic_normalization": NORMALIZATION,
        "semantic_convolution": CONVOLUTION,
        "ddp_output_handling": "registered WholeCTOutputV5 pytree; supports unused loss branches",
        "diagnostic_schedule": {"kind": "constant", "scale": 1.0, "start_epoch": 1},
        "world_size": world,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "lr_schedule": lr_schedule.state_dict(),
        "mask_temperature": 1.0,
        "temperature_schedule": "constant",
        "risk_stages": "radiomics L2 logistic; frozen-offset continuous FasterRisk on six semantics",
        "residual_definition": "y - sigmoid(radiomics_logit); modeled by logistic offset correction",
        "semantic_loss_baseline_detached": True,
        "runtime": dict(python=sys.version, executable=sys.executable,
                        torch=torch.__version__, cuda=torch.version.cuda,
                        cudnn=torch.backends.cudnn.version(),
                        nccl=torch.cuda.nccl.version(),
                        nccl_environment={key: value for key, value in os.environ.items()
                                          if key.startswith('NCCL_')},
                        gpu=torch.cuda.get_device_name(device)),
        "malignancy_policy": dataset.policy_summary,
        "feature_names": FEATURE_NAMES,
        "semantic_names": SEMANTIC_NAMES,
        "radiomics_names": RADIOMICS_NAMES,
        "fasterrisk_version": "0.1.10",
        "bank_fit_source": "latest detached features of training physical nodules; no test data",
        "loss_weights": criterion.loss_weights,
        "radiomics_definition": "18 fixed differentiable 3-D soft-mask descriptors on 1-mm CT; no STE",
        "endpoint": "scan-level reader-derived malignancy surrogate",
        "source_sha256": {
            str(path.relative_to(Path(__file__).parents[2])): hashlib.sha256(path.read_bytes()).hexdigest()
            for directory in ('common', 'model_v4', 'model_v5')
            for path in sorted((Path(__file__).parents[1] / directory).glob('*.py'))
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        (args.output_dir / "config.json").write_text(
            json.dumps(run_config, default=str, indent=2) + "\n"
        )
        print(
            json.dumps(
                {
                    "event": "start",
                    "architecture": ARCHITECTURE,
                    "manifest_cases": len(manifest_cases),
                    "training_cases": len(dataset),
                    "world_size": world,
                    "runtime": run_config['runtime'],
                    "resume": resume_summary,
                    "malignancy_policy": dataset.policy_summary,
                }
            ),
            flush=True,
        )

    monitor = Monitor(args, run_config, saved_wandb) if rank == 0 else None
    if warmup_state is None:
        warmup_state = run_warmup(bare_model, dataset, args, rank, world, monitor)
    anchor_stream = AnchorStream(warmup_state['cache_dir'],args,rank,world)
    if world > 1:
        dist.barrier()
    model: nn.Module = bare_model
    if world > 1:
        model = DistributedDataParallel(bare_model, device_ids=None, output_device=None,
            find_unused_parameters=True, broadcast_buffers=False)

    consecutive_overflows = 0
    for epoch in range(start_epoch, args.epochs):
        curriculum_epoch = args.smoke_curriculum_epoch if args.smoke_curriculum_epoch is not None else epoch
        sampler.set_epoch(epoch)
        anchor_stream.set_epoch(epoch)
        model.train()
        phase = set_training_phase(bare_model, curriculum_epoch)
        if phase != 'end_to_end_unfrozen' and vista_base_module is not None:
            vista_base_module.eval()
        diag_scale = diagnostic_scale(curriculum_epoch)
        teacher_p = bare_model.teacher_probability(curriculum_epoch)
        if lr_scheduler.last_epoch != curriculum_epoch:
            raise ValueError('LR scheduler is not at the current curriculum epoch')
        assert_learning_rates(optimizer, base_lrs, lr_schedule(curriculum_epoch))
        epoch_lrs = {g['name']:g['lr'] for g in optimizer.param_groups}
        accumulator = LossAccumulator(device)
        semantic_accumulator = SemanticAccumulator(device)
        sums = torch.zeros(7, device=device, dtype=torch.float64)
        epoch_start = time.time()
        torch.cuda.reset_peak_memory_stats(device)
        for step,batch in enumerate(loader,start=1):
            started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            anchor_roi,anchor_histogram = anchor_stream.next(device)
            with torch.autocast('cuda',dtype=amp_dtype):
                output = model(batch['image'],batch=batch,epoch=curriculum_epoch,
                    teacher_probability=teacher_p,compute_diagnostics=diag_scale>0,
                    semantic_anchor_roi=anchor_roi)
                losses = criterion(output,batch,diagnostic_scale=diag_scale,
                                   anchor=(output.anchor_probabilities,anchor_histogram))
            if not torch.isfinite(losses.total):
                raise FloatingPointError(f"Nonfinite loss for {batch['case_id']}")
            scaler.scale(losses.total).backward()
            scaler.unscale_(optimizer)
            grads = dict(vista=gradient_norm(vista_parameters),detr=gradient_norm(list(bare_model.detr.parameters())),
                refiner=gradient_norm(list(bare_model.refiner.parameters())),semantic=gradient_norm(semantic_parameters))
            total_grad,before_scale,after_scale = finish_scaled_step(optimizer,scaler,bare_model.parameters(),max_norm=5.)
            skipped = after_scale < before_scale
            consecutive_overflows = consecutive_overflows+1 if skipped else 0
            if consecutive_overflows >= 8:
                raise FloatingPointError('Eight consecutive AMP overflows')
            accumulator.update(losses)
            semantic_accumulator.update(output, batch)
            sums += torch.tensor([1, losses.positive_candidates, losses.refined_candidates,
                losses.supervised_candidates, int(losses.risk_valid), output.teacher_forced_count,
                output.fallback_count], device=device, dtype=torch.float64)
            global_step += 1
            if rank == 0:
                # Batch details remain in the local SLURM log, never W&B.
                print(json.dumps(dict(event='step',epoch=epoch+1,step=step,global_step=global_step,
                    case_id=batch['case_id'],phase=phase,total=float(losses.total),
                    semantic=losses.mean('semantic'),missing_nodule=losses.mean('missing_nodule'),
                    nodule=losses.mean('nodule'),risk=losses.mean('risk'),grad_norm=total_grad,
                    gradients=grads,optimizer_step_skipped=skipped,bank_ready=output.bank_ready,
                    bank_size=int(bare_model.rashomon.count),bank_last_fit=bare_model.rashomon.last_fit,
                    teacher_forced=output.teacher_forced_count,fallbacks=output.fallback_count,
                    seconds=time.monotonic()-started)),flush=True)
            if args.smoke_curriculum_epoch is not None and output.bank_ready and losses.supervised_candidates:
                if not np.isfinite(grads['semantic']) or grads['semantic'] <= 0:
                    raise AssertionError('Smoke requires a nonzero finite CNN gradient')
            del output,losses,batch,anchor_roi,anchor_histogram
            if args.benchmark_steps is not None and step >= args.benchmark_steps:
                break
        if diag_scale > 0 and not int(bare_model.rashomon.count):
            raise RuntimeError('No fitted semantic residual model')
        sums = reduce(sums,world)
        loss_record = accumulator.compute(criterion.loss_weights, diagnostic_scale=diag_scale)
        semantic_audit = audit_semantics(bare_model,args,warmup_state['cache_dir'],
                                        reference=warmup_state['semantic_reference'])
        semantic_guard_failures = 0 if semantic_audit['passed'] else semantic_guard_failures+1
        # Online moments mix model states; fixed-checkpoint probes decide the
        # regression stop, after the epoch checkpoint has been preserved.
        predicted_semantic_audit = semantic_accumulator.compute(enforce=False)
        peak = torch.tensor(torch.cuda.max_memory_allocated(device),device=device,dtype=torch.float64)
        if world > 1:
            dist.all_reduce(peak,op=dist.ReduceOp.MAX)
        count = float(sums[0])
        lr_scheduler.step()
        assert_learning_rates(optimizer,base_lrs,lr_schedule(curriculum_epoch+1))
        rng = dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device),
                   numpy=np.random.get_state(),python=random.getstate())
        rng_states = [None]*world if world>1 else [rng]
        if world>1:
            dist.all_gather_object(rng_states,rng)
        if rank==0:
            record = dict(event='epoch',epoch=epoch+1,phase=phase,global_step=global_step,
                **loss_record, semantic_audit=semantic_audit, predicted_semantic_audit=predicted_semantic_audit,
                semantic_guard_failures=semantic_guard_failures,
                lr=epoch_lrs,next_lr={g['name']:g['lr'] for g in optimizer.param_groups},
                diagnostic_scale=diag_scale,temperature=float(bare_model.mask_temperature),
                curriculum_epoch=curriculum_epoch,teacher_probability=teacher_p,
                mean_positive_candidates=float(sums[1]/count),mean_refined_candidates=float(sums[2]/count),
                supervised_nodules=int(sums[3]), valid_risk_cases=int(sums[4]),
                semantic_supervised_nodules=loss_record['loss_counts']['semantic'],
                teacher_forced=int(sums[5]), fallbacks=int(sums[6]),training_samples=int(count),seconds=time.time()-epoch_start,
                bank_fits=int(bare_model.rashomon.fit_count),bank_last_fit=bare_model.rashomon.last_fit,
                max_gpu_memory_gib=float(peak)/2**30)
            print(json.dumps(record),flush=True)
            with (args.output_dir/'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(record)+'\n')
            metrics = epoch_metrics(record,criterion.loss_weights)
            monitor.log(metrics)
            if not args.skip_save:
                payload = dict(epoch=epoch,global_step=global_step,lr_schedule=lr_schedule.state_dict(),
                    lr_scheduler=lr_scheduler.state_dict(),model=bare_model.state_dict(),optimizer=optimizer.state_dict(),
                    scaler=scaler.state_dict(),config=run_config,architecture=ARCHITECTURE,rng_states=rng_states,
                    warmup=warmup_state,wandb=monitor.state_dict())
                payload['semantic_guard_failures'] = semantic_guard_failures
                atomic_save(payload,args.output_dir/'latest.pt')
                if (epoch+1)%args.save_every==0 or epoch+1==args.epochs:
                    atomic_save(payload,args.output_dir/f'epoch_{epoch+1:03d}.pt')
        if world>1:
            dist.barrier()
        if not args.preparation_smoke and semantic_guard_failures >= args.semantic_guard_patience:
            raise RuntimeError('Semantic regression persisted; checkpoint saved before stopping: '
                               + '; '.join(semantic_audit['failures']))
        if args.benchmark_steps is not None or (args.stop_after_epoch and epoch+1>=args.stop_after_epoch):
            break
    if rank==0:
        monitor.finish()
    if world>1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
