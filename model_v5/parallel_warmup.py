"""Concurrent, independent DETR / refiner / semantic training from official labels.

Ranks 0..W-3 train DETR with DDP; rank W-2 trains the refiner and rank W-1
trains semantics. No warmup branch consumes another branch's outputs.
"""
import hashlib
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from back_prop.common.checkpoint_compat import adapt_cuda_rng_state
from back_prop.common.coarse_loss import masked_dice_loss, sigmoid_focal_loss
from back_prop.common.matcher import match_coarse, pairwise_box_iou
from back_prop.common.roi import integer_crop, paste_compact_mask
from back_prop.common.rng import preserve_rng
from back_prop.common.semantic_data import whole_ct_collate
from back_prop.common.train_utils import atomic_save, finish_scaled_step
from .loss import WholeCTCriterionV5, LossAccumulator, coarse_loss_support
from .preparation import SCHEMA, WarmupReadiness
from .warmup import official_row, SemanticROIs, audit_semantics, build_prediction_cache, fit_prediction_bank

BRANCHES = ('detr', 'refiner', 'semantic')


def write_json(value, path):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, allow_nan=False, default=str)+'\n')
    temporary.replace(path)


class DetectorBranch(nn.Module):
    """Only the CT encoder/projections and DETR belong to this optimizer."""
    def __init__(self, model):
        super().__init__()
        self.segmenter, self.detr = model.segmenter, model.detr
        object.__setattr__(self, '_source', model)

    def forward(self, image=None, *, frozen_vista=None):
        if frozen_vista is None:
            screening, _, _, _, native = self._source.detection_inputs(image)
        else:
            if self._source.detector_kind=='native_seeded':
                raise ValueError('Native-seeded warmup requires native priors as well as frozen features')
            from .feature_cache import cached_screening
            screening = cached_screening(self.segmenter, frozen_vista)
            native=None
        prediction=self.detr(screening,native) if native is not None else self.detr(screening)
        objects, boxes, masks, _ = prediction
        return objects[0], boxes[0], masks[0]


def detector_forward(module, batch):
    if 'frozen_vista' in batch:
        return module(frozen_vista=batch['frozen_vista'])
    return module(batch['image'])


def coarse_output(prediction, batch):
    objects, boxes, masks = prediction
    match = match_coarse(objects, boxes, masks, batch['target_boxes'], batch['target_masks'])
    empty = torch.empty(0, device=objects.device, dtype=torch.long)
    empty_mask = torch.empty((0,1,1,1), device=objects.device, dtype=torch.bool)
    return SimpleNamespace(object_logits=objects, boxes=boxes, coarse_mask_logits=masks,
        matched_query_indices=match.query_indices, matched_target_indices=match.target_indices,
        refined_query_indices=empty, refined_target_indices=empty,
        roi_loss_valid=empty_mask, fine_supervision_valid=empty.bool(),
        semantic_supervision_valid=empty.bool(), bank_ready=False)


@torch.no_grad()
def make_refiner_row(model, batch, index, jitter):
    center = model._box_center_voxel(batch['target_boxes'][index], batch['image'].shape[-3:])
    extent = batch['target_mask_crops'][index].shape
    crop_shape = tuple(max(base, int(n)+4) for base,n in zip(model.roi_shape, extent))
    shape = tuple(n+2*jitter for n in crop_shape)
    ct, origin, valid = integer_crop(batch['image'], center, shape)
    mask = paste_compact_mask(batch['target_mask_crops'][index],
                             batch['target_mask_origins'][index], origin, shape).bool()[None]
    ignored = torch.zeros_like(mask)
    for m,o in zip(batch.get('ignored_target_mask_crops',[]), batch.get('ignored_target_mask_origins',[])):
        ignored |= paste_compact_mask(m,o,origin,shape).bool()[None]
    support = valid & (~ignored | mask)
    if mask.sum() != batch['target_mask_crops'][index].sum():
        raise ValueError('Official refiner crop truncated its target')
    return dict(ct=ct.half(),mask=mask,valid=valid,support=support,crop_shape=crop_shape,
                source='official',key=(batch['case_id'],int(batch['nodule_ids'][index])))


@torch.no_grad()
def build_official_caches(model, dataset, args, rank, world, root):
    official, refiner = root/'official_warmup', root/'refiner_official'
    official.mkdir(exist_ok=True); refiner.mkdir(exist_ok=True)
    marker = root/'official_cache_complete.json'
    if marker.exists():
        record = json.loads(marker.read_text())
        if record['schema'] != SCHEMA or record['manifest_sha256'] != hashlib.sha256(args.manifest.read_bytes()).hexdigest():
            raise ValueError('Official cache provenance mismatch')
        return official,refiner
    # Only annotation processing: no encoder, detector or refiner forward call.
    for index in range(rank,len(dataset),world):
        batch=dataset[index]
        for j in range(len(batch['nodule_ids'])):
            name=f"{batch['case_id']}__nodule{int(batch['nodule_ids'][j])}.pt"
            atomic_save(official_row(model,batch,j),official/name)
            atomic_save(make_refiner_row(model,batch,j,args.teacher_jitter),refiner/name)
        if index//world % 20 == 0:
            print(json.dumps(dict(event='official_cache',rank=rank,case=index,total=len(dataset))),flush=True)
    dist.barrier()
    if rank==0:
        write_json(dict(schema=SCHEMA,manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                        nodules=len(list(official.glob('*.pt')))),marker)
    dist.barrier()
    return official,refiner


class RefinerROIs(Dataset):
    def __init__(self, paths, jitter):
        self.paths,self.jitter=paths,jitter

    def __len__(self):
        return len(self.paths)

    def __getitem__(self,index):
        row=torch.load(self.paths[index],map_location='cpu',weights_only=False)
        center=torch.tensor(row['ct'].shape[-3:])//2
        shift=torch.randint(-self.jitter,self.jitter+1,(3,))
        if float(torch.rand(()))<.2:shift.zero_()
        fields={k:integer_crop(row[k],center+shift,row['crop_shape'])[0]
                for k in ('ct','mask','valid','support')}
        if fields['mask'].sum()<.9*row['mask'].sum():
            fields={k:integer_crop(row[k],center,row['crop_shape'])[0] for k in fields}
        return fields


def synthetic_prior(mask):
    """GT-derived imperfect prior: translation, dilation/erosion and low resolution.

    This is an input corruption, never the target. A floor probability avoids
    an identity shortcut from near-binary logits; some examples have no prior.
    """
    size=mask.shape[-3:]
    offset=torch.randint(-4,5,(3,))
    prior=integer_crop(mask[0].cpu(),torch.tensor(size)//2+offset,size)[0][None].to(mask)
    radius=int(torch.randint(0,4,()))
    if radius:
        k=2*radius+1
        if float(torch.rand(()))<.7:
            prior=F.max_pool3d(prior,k,1,radius)
        else:
            prior=1-F.max_pool3d(1-prior,k,1,radius)
    factor=int(torch.randint(2,6,()))
    small=tuple(max(2,n//factor) for n in size)
    prior=F.interpolate(F.interpolate(prior,size=small,mode='trilinear',align_corners=False),
                        size=size,mode='trilinear',align_corners=False)
    if float(torch.rand(()))<.15:prior.zero_()
    return (.02+.88*prior).clamp(.02,.9)


def refiner_forward(refiner,row,device):
    ct=row['ct'][None].to(device).float()
    mask=row['mask'][None].to(device).float()
    valid=row['valid'][None].to(device).float()
    support=row['support'][None].to(device).bool()
    prior=synthetic_prior(mask)*valid
    logits=torch.logit(prior.clamp(.001,.999))
    # Auxiliary VISTA/context/query slots have no other model's predictions.
    # They stay neutral here and gain their usual conditioning in joint training.
    channels=refiner.input_channels-4
    context=torch.zeros((1,channels,*ct.shape[-3:]),device=device)
    inputs=torch.cat((ct,torch.zeros_like(prior),prior,valid,context),dim=1)
    query=torch.zeros((1,refiner.query_dim),device=device)
    refined=logits+refiner(inputs,query).float()
    dice=masked_dice_loss(refined,mask,valid=support)
    focal=sigmoid_focal_loss(refined,mask,valid=support)
    with torch.no_grad():
        prior_dice=1-masked_dice_loss(logits,mask,valid=support)
    return 2*dice+focal, dict(dice_loss=dice,focal_loss=focal,dice=1-dice,prior_dice=prior_dice)


@torch.no_grad()
def detr_audit(module,model,dataset,args,rank,world,group):
    module.eval()
    if getattr(model,'detector_kind','legacy') in ('mask_guided','native_seeded'):
        from .detection_metrics import detection_counts,summarize_detection_counts
        order=torch.randperm(len(dataset),generator=torch.Generator().manual_seed(args.seed+731))[:args.geometry_probe_cases]
        names=('nodules','scans','selected','coverage','coarse_dice','box_iou','center_error_mm','true_positives','false_positives')
        totals=torch.zeros(len(names),device=model.device,dtype=torch.float64)
        for index in order[rank::world]:
            batch=dataset[int(index)]
            objects,boxes,masks=detector_forward(module,batch)
            selected=model.select_detection_queries(objects,masks)
            values=detection_counts(objects,boxes,masks,batch,selected,model.roi_shape,args.minimum_roi_coverage)
            totals+=totals.new_tensor([values[k] for k in names])
        if world>1:dist.all_reduce(totals,group=group)
        result=summarize_detection_counts(dict(zip(names,totals.cpu().tolist())))
        result['passed']=result['nodules']>0 and result['coverage']>=args.geometry_min_coverage and result['coarse_dice']>=args.geometry_min_coarse_dice
        return result
    order=torch.randperm(len(dataset),generator=torch.Generator().manual_seed(args.seed+731))[:args.geometry_probe_cases]
    totals=torch.zeros(6,device=model.device,dtype=torch.float64)
    for index in order[rank::world]:
        batch=dataset[int(index)]
        out=coarse_output(detector_forward(module,batch),batch)
        q,t=out.matched_query_indices,out.matched_target_indices
        p=out.coarse_mask_logits[q].float().sigmoid()
        g=batch['target_masks'].to(p)[t]
        support=coarse_loss_support(batch,g)
        if support is not None:p,g=p*support,g*support
        p,g=p.flatten(1),g.flatten(1)
        dice=2*(p*g).sum(1)/(p.sum(1)+g.sum(1)).clamp_min(1e-8)
        selected=out.object_logits.sigmoid()>=model.inference_object_threshold
        iou=pairwise_box_iou(out.boxes[q],batch['target_boxes'].to(out.boxes)[t]).diag()
        # Coverage uses DETR crop centers, without running the refiner.
        covered=0
        for qi,ti in zip(q.tolist(),t.tolist()):
            center=model._box_center_voxel(out.boxes[qi],batch['isotropic_shape'])
            origin=torch.floor(center.cpu()+.5).long()-torch.tensor(model.roi_shape)//2
            gt=batch['target_mask_crops'][ti]
            visible=paste_compact_mask(gt,batch['target_mask_origins'][ti],origin,model.roi_shape).sum()
            covered+=int(selected[qi] and visible>=args.minimum_roi_coverage*gt.sum())
        totals+=totals.new_tensor([len(t),covered,float(dice.sum()),float(iou.sum()),int(selected.sum()),1])
    if world>1:dist.all_reduce(totals,group=group)
    n=max(float(totals[0]),1)
    result=dict(nodules=int(totals[0]),coverage=float(totals[1]/n),coarse_dice=float(totals[2]/n),
                box_iou=float(totals[3]/n),selected_queries_per_scan=float(totals[4]/totals[5].clamp_min(1)))
    result['passed']=bool(totals[0]>0 and result['coverage']>=args.geometry_min_coverage
                          and result['coarse_dice']>=args.geometry_min_coarse_dice)
    return result


@torch.no_grad()
def refiner_audit(module,data,args,device):
    module.eval()
    order=torch.randperm(len(data),generator=torch.Generator().manual_seed(args.seed+832))[:args.refiner_probe_nodules]
    totals=[]
    with preserve_rng():
        torch.manual_seed(args.seed+833)
        for i in order:
            with torch.autocast(device.type,enabled=device.type=='cuda',dtype=getattr(torch,args.amp_dtype)):
                _,metrics=refiner_forward(module,data[int(i)],device)
            totals.append([float(metrics[k]) for k in ('dice','prior_dice')])
    average=torch.tensor(totals).mean(0)
    result=dict(nodules=len(totals),dice=float(average[0]),prior_dice=float(average[1]))
    result['passed']=result['dice']>=args.refiner_min_dice and result['dice']>=result['prior_dice']
    return result


def save_branch(module,optimizer,scaler,args,branch,epoch,ready,streak,audit,rank,world,group,root):
    rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),numpy=np.random.get_state(),python=random.getstate())
    states=[None]*world
    if world>1:dist.all_gather_object(states,rng,group=group)
    else:states=[rng]
    if rank==0:
        atomic_save(dict(schema=SCHEMA,branch=branch,epoch=epoch,ready=ready,consecutive_passes=streak,
            model=module.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
            rng_states=states,audit=audit,source='official_annotations_only'),root/f'{branch}_warmup.pt')
        if ready:write_json(dict(epoch=epoch,audit=audit),root/f'{branch}_ready.json')


class Journal:
    """Only global rank zero writes W&B; other branches publish local JSONL."""
    def __init__(self,root,monitor):
        self.root,self.monitor,self.counts=root,monitor,{b:0 for b in BRANCHES}

    def drain(self):
        if self.monitor is None:return
        for branch in BRANCHES:
            path=self.root/f'{branch}_history.jsonl'
            if not path.exists():continue
            for line in path.read_text().splitlines()[self.counts[branch]:]:
                try:record=json.loads(line)
                except json.JSONDecodeError:break
                self.monitor.log(record)
                self.counts[branch]+=1


def run_branch(branch,model,dataset,official,refiner_dir,args,rank,world,group,root,journal,resume):
    local_rank=rank if branch=='detr' else 0
    replicas=world-2 if branch=='detr' else 1
    model.requires_grad_(False)
    if branch=='detr':
        module=DetectorBranch(model)
        module.requires_grad_(True)
        backbone=getattr(module.segmenter,'model',None)
        if backbone is not None:backbone.requires_grad_(False)
        data=dataset;batch_size=1;collate=whole_ct_collate
        minimum,maximum=args.geometry_min_epochs,args.geometry_max_epochs
        lr=args.lr_main
    elif branch=='refiner':
        module=model.refiner;module.requires_grad_(True);module.use_checkpoint=True
        data=RefinerROIs(sorted(refiner_dir.glob('*.pt')),args.teacher_jitter)
        batch_size=None;collate=None
        minimum,maximum=args.refiner_min_epochs,args.refiner_max_epochs;lr=args.lr_main
    else:
        module=model.semantics;module.requires_grad_(True)
        data=SemanticROIs(sorted(official.glob('*.pt')),args.teacher_jitter)
        batch_size=args.warmup_batch_size;collate=None
        minimum,maximum=args.semantic_min_epochs,args.warmup_epochs;lr=args.lr_semantic_warmup
    parameters=[p for p in module.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=lr,weight_decay=args.weight_decay)
    scaler=torch.amp.GradScaler('cuda',enabled=args.amp_dtype=='float16',init_scale=args.amp_init_scale)
    readiness=WarmupReadiness(minimum,args.warmup_ready_epochs)
    start=0
    if resume and (root/f'{branch}_warmup.pt').exists():
        saved=torch.load(root/f'{branch}_warmup.pt',map_location='cpu',weights_only=False)
        if saved['schema']!=SCHEMA:raise ValueError('Incompatible branch checkpoint')
        module.load_state_dict(saved['model']);start=saved['epoch']
        if saved['ready']:
            # Completed branches contribute weights only. In particular, a
            # detector-only repair may use different optimizer parameter groups.
            module.eval().requires_grad_(False)
            if local_rank==0:write_json(dict(epoch=start,audit=saved['audit']),root/f'{branch}_ready.json')
            return
        optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        readiness.consecutive_passes=saved['consecutive_passes']
        rng=saved['rng_states'][local_rank];torch.set_rng_state(rng['torch'])
        torch.cuda.set_rng_state(adapt_cuda_rng_state(rng['cuda'],torch.cuda.get_rng_state().numel()))
        np.random.set_state(rng['numpy']);random.setstate(rng['python'])
    if branch=='detr' and getattr(args,'detr_feature_cache',None) is not None:
        from .feature_cache import CachedDetectorDataset
        # Validate against the restored backbone, not fresh initialization.
        data=CachedDetectorDataset(args.detr_feature_cache,model,dataset,args.manifest)
    sampler=DistributedSampler(data,num_replicas=replicas,rank=local_rank,seed=args.seed+171)
    loader=DataLoader(data,batch_size=batch_size,sampler=sampler,num_workers=0,collate_fn=collate)
    wrapped=DDP(module,process_group=group,broadcast_buffers=False,find_unused_parameters=True) if replicas>1 else module
    criterion=WholeCTCriterionV5(missing_nodule_weight=args.missing_nodule_weight,
        object_background_weight=getattr(args,'object_background_weight',1.))
    weights={k:(v if k in ('object','box_l1','box_giou','coarse_dice','coarse_focal','missing_nodule') else 0.)
             for k,v in criterion.loss_weights.items()}
    print(json.dumps(dict(event='independent_warmup_start',branch=branch,rank=rank,gpu=torch.cuda.current_device(),
                         samples=len(data),source='official_annotations_only',start_epoch=start)),flush=True)
    ready=False
    for epoch in range(start,maximum):
        sampler.set_epoch(epoch);module.train();model.training=True
        if branch=='detr' and backbone is not None:backbone.eval()
        accumulator=LossAccumulator(model.device)
        totals={};count=0
        for step,item in enumerate(loader,1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=getattr(torch,args.amp_dtype)):
                if branch=='detr':
                    output=coarse_output(detector_forward(wrapped,item),item)
                    losses=criterion(output,item,diagnostic_scale=0.,semantic_scale=0.,
                                     synchronize=replicas>1,process_group=group)
                    loss=losses.total;accumulator.update(losses);metrics={};n=1
                elif branch=='refiner':
                    loss,metrics=refiner_forward(module,item,model.device);n=1
                else:
                    roi,hist=item
                    probability,_=module(roi.to(model.device),prepared=True)
                    loss=-(hist.to(model.device)*probability.float().clamp_min(1e-7).log()).sum(-1).mean()
                    metrics=dict(nll=loss);n=len(roi)
            if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite {branch} warmup loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            norm,_,_=finish_scaled_step(optimizer,scaler,parameters,max_norm=5.)
            count+=n
            for k,v in dict(loss=loss,**metrics).items():totals[k]=totals.get(k,0.)+float(v.detach())*n
            if local_rank==0 and (step==1 or step%20==0):
                print(json.dumps(dict(event='independent_warmup_step',branch=branch,epoch=epoch+1,step=step,
                                      loss=float(loss.detach()),gradient_norm=norm)),flush=True)
            journal.drain()
            if args.preparation_smoke and step>=args.benchmark_steps:break
        with preserve_rng():
            if branch=='detr':
                metrics=accumulator.compute(weights,synchronize=replicas>1,process_group=group)
                audit=detr_audit(module,model,data,args,local_rank,replicas,group)
                metrics={f'loss_{k}':metrics[k] for k,v in weights.items() if v}
            elif branch=='refiner':
                metrics={k:v/count for k,v in totals.items()}
                audit=refiner_audit(module,data,args,model.device)
            else:
                metrics={k:v/count for k,v in totals.items()}
                audit=audit_semantics(model,args,official)
        ready=readiness.update(epoch+1,audit['passed'])
        if args.preparation_smoke:ready=True
        save_branch(module,optimizer,scaler,args,branch,epoch+1,ready,readiness.consecutive_passes,audit,
                    local_rank,replicas,group,root)
        if local_rank==0:
            prefix=branch+'_warmup/'
            record=dict(warmup_epoch=epoch+1)
            scalars=dict(epoch=epoch+1,ready=ready,consecutive_passes=readiness.consecutive_passes,**metrics)
            scalars.update({k:v for k,v in audit.items() if isinstance(v,(int,float,bool))})
            if branch=='semantic':
                from back_prop.common.features import SEMANTIC_NAMES
                for domain,values in audit['domains'].items():
                    scalars[domain+'_nll']=values['nll']
                    for kind in ('std_ratio','correlation','nll_by_attribute'):
                        scalars.update({f'{domain}_{kind}_{name}':v for name,v in zip(SEMANTIC_NAMES,values[kind])})
            record.update({prefix+k:v for k,v in scalars.items()})
            with (root/f'{branch}_history.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
            print(json.dumps(dict(event='independent_warmup_epoch',branch=branch,epoch=epoch+1,ready=ready,audit=audit)),flush=True)
        journal.drain()
        # A fast non-writing rank must not raise at the epoch limit while rank
        # zero is still publishing its checkpoint/history: torchrun would kill
        # the writer before atomic_save can rename the completed temporary file.
        if replicas>1:dist.barrier(group=group)
        if ready:break
    if not ready:raise RuntimeError(f'{branch} exhausted {maximum} epochs without readiness; checkpoint saved')
    module.requires_grad_(False)
    del wrapped,optimizer


def validate_parallel_resume(args,saved,world):
    if saved.get('preparation_schema')!=SCHEMA or saved['world_size']!=world:
        raise ValueError('Only matching parallel official-mask preparation can resume')
    if saved['manifest_sha256']!=hashlib.sha256(args.manifest.read_bytes()).hexdigest():
        raise ValueError('Preparation manifest changed')
    mutable={'output_dir','prepare_resume','wandb_name','geometry_max_epochs','refiner_max_epochs','warmup_epochs',
             'detr_feature_cache'}
    legacy_defaults={'detector_kind':'legacy','detector_nms_iou':'0.5','object_background_weight':'1.0',
                     'native_proposal_threshold':'0.1','native_confidence_residual_bound':'None'}
    for k,v in vars(args).items():
        if k not in mutable and str(v)!=saved['config'].get(k,legacy_defaults.get(k)):
            raise ValueError(f'Preparation resume differs: {k}')


def run_parallel_warmup(model,dataset,args,rank,world,monitor):
    if world<3:raise ValueError('Independent simultaneous warmup needs >=3 GPU processes (launcher uses 4)')
    root=args.output_dir
    resume=args.prepare_resume is not None
    if resume:
        saved=torch.load(args.prepare_resume,map_location='cpu',weights_only=False)
        validate_parallel_resume(args,saved,world)
        root=Path(saved['cache_root'])
    detr_group=dist.new_group(ranks=list(range(world-2)))
    if rank==0:
        atomic_save(dict(preparation_schema=SCHEMA,world_size=world,phase='parallel',
            manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            config={k:str(v) for k,v in vars(args).items()},cache_root=str(root.resolve())),args.output_dir/'preparation.pt')
    model.eval()
    official,refiner=build_official_caches(model,dataset,args,rank,world,root)
    branch='detr' if rank<world-2 else ('refiner' if rank==world-2 else 'semantic')
    journal=Journal(root,monitor)
    # A short initial collective only; no global collective until all three
    # independent loops finish. Fast branches never block DETR's DDP group.
    dist.barrier()
    run_branch(branch,model,dataset,official,refiner,args,rank,world,detr_group,root,journal,resume)
    while not all((root/f'{b}_ready.json').exists() for b in BRANCHES):
        journal.drain();time.sleep(5)
    dist.barrier();journal.drain()
    audits={}
    for name,module in [('detr',DetectorBranch(model)),('refiner',model.refiner),('semantic',model.semantics)]:
        state=torch.load(root/f'{name}_warmup.pt',map_location='cpu',weights_only=False)
        module.load_state_dict(state['model']);audits[name]=dict(epochs=state['epoch'],audit=state['audit'])
    # All ranks start joint training with identical previously unused context.
    for p in model.context_compressor.parameters():dist.broadcast(p.data,src=0)
    model.refiner.use_checkpoint=False  # Joint forward checkpoints the complete refiner.
    model.eval();model.requires_grad_(False)
    # Prediction-only risk initialization happens AFTER all three warmups.
    # Its crops are never input to any warmup optimizer.
    official,predicted=build_prediction_cache(model,dataset,args,rank,world,root)
    paths=sorted(predicted.glob('*.pt'))
    if args.preparation_smoke:
        # Random fresh DETR may select nothing in a two-step plumbing test.
        # Do not fabricate prediction quality or call this production readiness.
        bank_paths=paths if len(paths)>=args.bank_min_samples else sorted(official.glob('*.pt'))
    else:
        bank_paths=paths
        if len(paths)<args.bank_min_samples:raise RuntimeError('Too few usable predictions for joint risk initialization; independent warmups retained')
    fit_prediction_bank(model,bank_paths,rank,world)
    audit=audit_semantics(model,args,official)
    result=dict(schema=SCHEMA,complete=True,smoke_only=args.preparation_smoke,
        epochs=max(v['epochs'] for v in audits.values()),branches=audits,
        official_nodules=len(list(official.glob('*.pt'))),predicted_nodules=len(paths),
        cache_dir=str(official.resolve()),predicted_cache_dir=str(predicted.resolve()),
        semantic_reference=audit['domains'],baseline_samples=len(model.rashomon.radiomics_memory),
        semantic_samples=len(model.rashomon.memory))
    if rank==0:
        write_json(result,args.output_dir/'warmup.json')
        atomic_save(dict(model=model.state_dict(),audit=result),args.output_dir/'warmup.pt')
    model.requires_grad_(True)
    return result
