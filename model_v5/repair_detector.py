"""Fine-tune the mask-guided detector while retaining completed V5 branches.

Inputs are fixed, prediction-independent CT features from the source run's
frozen VISTA/projection weights. Only the detector is optimized in this phase.
"""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler

from back_prop.common.rng import preserve_rng
from back_prop.common.train_utils import atomic_save,distributed_setup
from .detector import MaskGuidedDETR,NativeSeededDETR,select_mask_queries
from .detection_metrics import detection_counts,summarize_detection_counts
from .loss import WholeCTCriterionV5,LossAccumulator
from .monitor import Monitor
from .parallel_warmup import coarse_output,write_json
from .preparation import WarmupReadiness

REPAIR_SCHEMA='v5_mask_guided_detector_repair_v1'


def detector_from_config(config):
    factory=NativeSeededDETR if config.get('detector_kind')=='native_seeded' else MaskGuidedDETR
    extra=({'confidence_residual_bound':config.get('native_confidence_residual_bound')}
           if factory is NativeSeededDETR else {})
    return factory(num_queries=config['num_queries'],hidden_dim=config['detr_hidden_dim'],
        coarse_shape=tuple(config['detr_coarse_shape']),nheads=config['detr_nheads'],
        encoder_layers=config['detr_encoder_layers'],decoder_layers=config['detr_decoder_layers'],
        input_channels=config.get('vista_feature_dim',48)+2,**extra)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--detector-init',type=Path,required=True)
    parser.add_argument('--projected-cache',type=Path,required=True)
    parser.add_argument('--native-cache',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--minimum-epochs',type=int,default=15)
    parser.add_argument('--maximum-epochs',type=int,default=30)
    parser.add_argument('--ready-epochs',type=int,default=3)
    parser.add_argument('--object-background-weight',type=float,default=.1)
    parser.add_argument('--lr-body',type=float,default=1e-5)
    parser.add_argument('--lr-head',type=float,default=1e-4)
    parser.add_argument('--lr-native-head',type=float,default=1e-3)
    parser.add_argument('--native-confidence-residual-bound',type=float)
    parser.add_argument('--wandb-mode',choices=('online','offline','disabled'),default='online')
    args=parser.parse_args()
    if not 1<=args.minimum_epochs<=args.maximum_epochs or args.ready_epochs<1:
        raise ValueError('Invalid repair epoch requirements')
    if args.native_confidence_residual_bound is not None and (
            args.native_confidence_residual_bound<=0 or not args.native_cache):
        raise ValueError('A positive confidence bound requires native priors')
    rank,local_rank,world=distributed_setup()
    device=torch.device('cuda',local_rank)
    config=json.loads((args.source_run/'config.json').read_text())
    config.update(detector_kind='native_seeded' if args.native_cache else 'mask_guided',detector_nms_iou=.5,
                  object_background_weight=args.object_background_weight,
                  native_confidence_residual_bound=args.native_confidence_residual_bound)
    seed=config['seed']+rank
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=True
    model=detector_from_config(config).to(device)
    initial=torch.load(args.detector_init,map_location='cpu',weights_only=False)
    incompatible=model.load_state_dict(initial['model'],strict=False)
    if incompatible.unexpected_keys or any(not (args.native_cache and k.startswith('native_'))
                                           for k in incompatible.missing_keys):
        raise ValueError(f'Incompatible detector initialization: {incompatible}')
    marker=json.loads((args.projected_cache/'complete.json').read_text())
    source=args.source_run/'detr_warmup.pt'
    if Path(marker['source_checkpoint']).resolve()!=source.resolve() or marker['epoch']!=30:
        raise ValueError('Projected cache does not match the repair source')
    if args.native_cache:
        import back_prop.model_v5.native_proposals as native_module
        native_marker=json.loads((args.native_cache/'training'/'complete.json').read_text())
        provenance=native_marker['provenance']
        expected_hash=hashlib.sha256(Path(native_module.__file__).read_bytes()).hexdigest()
        if (native_marker['cases']!=marker['cases'] or provenance['module_sha256']!=expected_hash
            or Path(provenance['source_checkpoint']).resolve()!=source.resolve()
            or provenance['source_mtime_ns']!=source.stat().st_mtime_ns
            or provenance['manifest_sha256']!=hashlib.sha256(Path(config['manifest']).read_bytes()).hexdigest()
            or provenance['maximum']!=config['num_queries'] or provenance['screen_shape']!=config['screen_shape']):
            raise ValueError('Native priors do not match the fixed projection source')
        config['native_proposal_threshold']=provenance['threshold']
    rows=[]
    for i in range(marker['cases']):
        row=torch.load(args.projected_cache/f'{i:06d}.pt',map_location='cpu',weights_only=False,mmap=True)
        row['screening']=row['screening'].to(device)
        if args.native_cache:
            from .native_proposals import NativeProposals
            native=torch.load(args.native_cache/'training'/f'{i:06d}.pt',map_location='cpu',weights_only=False)
            if native['case_id']!=row['case_id'] or tuple(native['isotropic_shape'])!=tuple(row['isotropic_shape']):
                raise ValueError('Native prior case alignment mismatch')
            p=native['proposals']
            row['native']=NativeProposals(p.boxes.to(device),p.object_logits.to(device),
                p.coarse_logits.to(device),p.origins,[])  # Fine crops are unnecessary for detector-only training.
        rows.append(row)
    wrapped=DDP(model,broadcast_buffers=False,find_unused_parameters=bool(args.native_cache)) if world>1 else model
    heads=[p for name,p in model.named_parameters() if name.startswith(('query_projection.','local_projection.','prediction_head.'))]
    body=[p for name,p in model.named_parameters() if not name.startswith(('query_projection.','local_projection.','prediction_head.','native_'))]
    groups=[dict(params=body,lr=args.lr_body),dict(params=heads,lr=args.lr_head)]
    base_lrs=[args.lr_body,args.lr_head]
    if args.native_cache:
        groups.append(dict(params=[p for name,p in model.named_parameters() if name.startswith('native_')],lr=args.lr_native_head))
        base_lrs.append(args.lr_native_head)
    optimizer=torch.optim.AdamW(groups,weight_decay=config['weight_decay'])
    criterion=WholeCTCriterionV5(missing_nodule_weight=config['missing_nodule_weight'],
                                object_background_weight=args.object_background_weight)
    weights={k:(v if k in ('object','box_l1','box_giou','coarse_dice','coarse_focal','missing_nodule') else 0.)
             for k,v in criterion.loss_weights.items()}
    sampler=DistributedSampler(rows,num_replicas=world,rank=rank,seed=config['seed']+171)
    order=torch.randperm(len(rows),generator=torch.Generator().manual_seed(config['seed']+731))[:config['geometry_probe_cases']]
    names=('nodules','scans','selected','coverage','coarse_dice','box_iou','center_error_mm','true_positives','false_positives')
    readiness=WarmupReadiness(args.minimum_epochs,args.ready_epochs)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    monitor=None
    if rank==0:
        run_config=dict(config,repair_schema=REPAIR_SCHEMA,repair={k:str(v) for k,v in vars(args).items()},
                        repair_initial_step=initial.get('step'),repair_world_size=world,
                        repair_policy=config['detector_kind']+' DETR; frozen VISTA/projections; completed refiner/semantic retained')
        project=Path(__file__).resolve().parents[2]
        run_config['source_sha256']={str(path.relative_to(project)):hashlib.sha256(path.read_bytes()).hexdigest()
            for folder in ('common','model_v4','model_v5')
            for path in sorted((project/'back_prop'/folder).glob('*.py'))}
        if args.native_cache:run_config['native_cache_provenance']=provenance
        write_json(run_config,args.output_dir/'config.json')
        monitor_args=SimpleNamespace(output_dir=args.output_dir,wandb_mode=args.wandb_mode,
            wandb_entity=config['wandb_entity'],wandb_project=config['wandb_project'],
            wandb_name='v5-detector-repair-'+args.output_dir.name,max_cases=None)
        monitor=Monitor(monitor_args,run_config)
    ready=False
    for epoch in range(args.maximum_epochs):
        sampler.set_epoch(epoch);model.train()
        factor=.3+.7*(1+math.cos(math.pi*epoch/args.maximum_epochs))/2
        for group,base in zip(optimizer.param_groups,base_lrs):group['lr']=base*factor
        accumulator=LossAccumulator(device)
        for step,index in enumerate(sampler,1):
            row=rows[index];optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                obj,box,mask,_=(wrapped(row['screening'],row['native']) if args.native_cache else wrapped(row['screening']))
                out=coarse_output((obj[0],box[0],mask[0]),row)
                loss=criterion(out,row,diagnostic_scale=0.,semantic_scale=0.,synchronize=world>1)
            if not torch.isfinite(loss.total):raise FloatingPointError('Nonfinite detector repair loss')
            loss.total.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            optimizer.step();accumulator.update(loss)
            if rank==0 and (step==1 or step%100==0):
                print(json.dumps(dict(event='detector_repair_step',epoch=epoch+1,step=step,
                    loss=float(loss.total.detach()),gradient_norm=float(norm))),flush=True)
        metrics=accumulator.compute(weights,synchronize=world>1)
        with preserve_rng(),torch.no_grad():
            model.eval();totals=torch.zeros(len(names),device=device,dtype=torch.float64)
            for i in order[rank::world]:
                row=rows[int(i)]
                obj,box,mask,_=(model(row['screening'],row['native']) if args.native_cache else model(row['screening']))
                obj,box,mask=obj[0],box[0],mask[0]
                selected=select_mask_queries(obj,mask,config['inference_object_threshold'],config['detector_nms_iou'])
                values=detection_counts(obj,box,mask,row,selected,tuple(config['roi_shape']),config['minimum_roi_coverage'])
                totals+=totals.new_tensor([values[k] for k in names])
            if world>1:dist.all_reduce(totals)
            audit=summarize_detection_counts(dict(zip(names,totals.cpu().tolist())))
        # Preserve the original coverage/Dice gates, and additionally reject a
        # flood of misplaced candidates that happens to cover large ROI crops.
        audit['passed']=(audit['coverage']>=config['geometry_min_coverage']
            and audit['coarse_dice']>=config['geometry_min_coarse_dice']
            and audit['detection_recall']>=.65 and audit['false_positives_per_scan']<=4.)
        ready=readiness.update(epoch+1,audit['passed'])
        rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),numpy=np.random.get_state(),python=random.getstate())
        states=[None]*world
        if world>1:dist.all_gather_object(states,rng)
        else:states=[rng]
        if rank==0:
            payload=dict(repair_schema=REPAIR_SCHEMA,model=model.state_dict(),optimizer=optimizer.state_dict(),
                epoch=epoch+1,ready=ready,consecutive_passes=readiness.consecutive_passes,rng_states=states,
                config=run_config,audit=audit,metrics=metrics,wandb=monitor.state_dict())
            atomic_save(payload,args.output_dir/'latest.pt')
            record=dict(warmup_epoch=epoch+1,**{'detr_warmup/'+k:v for k,v in dict(
                epoch=epoch+1,ready=ready,consecutive_passes=readiness.consecutive_passes,**audit).items()})
            record.update({'detr_warmup/loss_'+k:metrics[k] for k,v in weights.items() if v})
            monitor.log(record)
            with (args.output_dir/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print(json.dumps(dict(event='detector_repair_epoch',epoch=epoch+1,ready=ready,audit=audit)),flush=True)
            if ready:write_json(dict(epoch=epoch+1,audit=audit),args.output_dir/'ready.json')
        if world>1:dist.barrier()
        if ready:break
    if rank==0:monitor.finish()
    if world>1:dist.destroy_process_group()
    if not ready:raise RuntimeError('Detector repair exhausted its epoch limit without passing; latest checkpoint retained')


if __name__=='__main__':main()
