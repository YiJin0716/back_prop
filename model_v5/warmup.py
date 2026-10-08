"""Geometry preparation, augmented semantic warmup, and fixed-model probes."""
import json
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from back_prop.common.roi import integer_crop, paste_compact_mask
from back_prop.common.train_utils import atomic_save
from .augmentation import official_view
from .semantic_metrics import semantic_metrics, semantic_failures
from .preparation import SCHEMA, preparation_restore, preparation_save, run_geometry_warmup

def official_row(model, batch, index):
    """Use exact binary official mask on the same 1-mm grid as joint training."""
    device = model.device
    center = model._box_center_voxel(batch['target_boxes'][index], batch['image'].shape[-3:])
    extent = batch['target_mask_crops'][index].shape
    shape = tuple(max(base, int(n)+4) for base,n in zip(model.roi_shape, extent))
    ct, origin, valid = integer_crop(batch['image'], center, shape)
    hu, _, _ = integer_crop(batch['image_hu'], center, shape)
    mask = paste_compact_mask(batch['target_mask_crops'][index],
                             batch['target_mask_origins'][index], origin, shape)
    if int(mask.sum()) != int(batch['target_mask_crops'][index].sum()):
        raise ValueError('Warmup crop does not contain the complete official nodule')
    mask = mask[None,None].to(device).float()
    logits = torch.where(mask > 0, torch.inf, -torch.inf)
    valid = valid[None].to(device).float()
    with torch.no_grad():
        roi = model.semantics.prepare_roi(ct[None].to(device), logits, valid)[0].cpu()
        radio = model.radiomics(hu[None].to(device), logits, valid)[0].cpu()
    mask_cpu=mask[0,0].cpu()
    coordinates=mask_cpu.nonzero();lower=coordinates.amin(0);upper=coordinates.amax(0)+1
    slices=tuple(slice(int(a),int(b)) for a,b in zip(lower,upper))
    compact_mask=mask_cpu[slices][None]
    compact=torch.cat((ct[(slice(None),*slices)]*compact_mask,compact_mask),dim=0)
    return dict(roi=roi, compact=compact,compact_origin=lower,crop_shape=shape,
                spacing=torch.tensor(shape)/model.semantics.roi_size, radiomics=radio, histogram=batch['semantic_histograms'][index].cpu(),
                semantics=batch['semantic_targets'][index].cpu(),
                label=batch['malignancy_targets'][index].cpu(),
                key=(batch['case_id'], int(batch['nodule_ids'][index])))


class SemanticROIs(Dataset):
    def __init__(self, paths, jitter=0):
        self.paths, self.jitter = paths, jitter

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        row = torch.load(self.paths[index], weights_only=False, map_location='cpu')
        roi = row['roi']
        # Predicted crops already have the actual localization/segmentation
        # errors. Augment official crops; reject translations that truncate GT.
        if self.jitter and row.get('source', 'official') == 'official':
            shift=torch.randint(-int(self.jitter),int(self.jitter)+1,(3,))
            if float(torch.rand(())) >= .2:
                roi = official_view(row,shift)
        return roi, row['histogram']


class AnchorStream:
    def __init__(self, directory, args, rank, world):
        data = SemanticROIs(sorted(Path(directory).glob('*.pt')), args.teacher_jitter)
        self.sampler = DistributedSampler(data, num_replicas=world, rank=rank,
                                          seed=args.seed+553)
        self.loader = DataLoader(data, sampler=self.sampler,
                                batch_size=args.semantic_anchor_batch_size, num_workers=0)
        self.cycle = 0
        self.iterator = None

    def set_epoch(self, epoch):
        self.cycle = epoch * 10000
        self.sampler.set_epoch(self.cycle)
        self.iterator = iter(self.loader)

    def next(self, device):
        if self.iterator is None:
            self.set_epoch(0)
        try:
            roi, hist = next(self.iterator)
        except StopIteration:
            self.cycle += 1
            self.sampler.set_epoch(self.cycle)
            self.iterator = iter(self.loader)
            roi, hist = next(self.iterator)
        return roi.to(device), hist.to(device)


@torch.no_grad()
def audit_semantics(model, args, cache_dir=None, *, reference=None):
    root = Path(cache_dir).parent if cache_dir else args.output_dir
    metrics = {}
    was_training = model.semantics.training
    model.semantics.eval()
    try:
        for domain, directory in [('centered','official_warmup'), ('predicted','predicted_warmup')]:
            paths = sorted((root/directory).glob('*.pt'))
            order = torch.randperm(len(paths), generator=torch.Generator().manual_seed(args.seed+415))
            rows = [torch.load(paths[int(i)], map_location='cpu', weights_only=False)
                    for i in order[:args.semantic_probe_nodules]]
            if not rows:
                raise RuntimeError(f'No {domain} semantic probe ROIs')
            x = torch.stack([r['roi'] for r in rows])
            hist = torch.stack([r['histogram'] for r in rows])
            def evaluate(source):
                p = torch.cat([model.semantics(c.to(model.device), prepared=True)[0].cpu()
                               for c in source.split(args.warmup_batch_size)])
                return semantic_metrics(p, hist)
            metrics[domain] = evaluate(x)
            if domain == 'centered':
                shifts = torch.randint(-args.teacher_jitter,args.teacher_jitter+1,(len(x),3),
                                       generator=torch.Generator().manual_seed(args.seed+881))
                metrics['shifted'] = evaluate(torch.stack([official_view(r,s) for r,s in zip(rows,shifts)]))
        failures = semantic_failures(metrics, min_ratio=args.semantic_min_std_ratio,
            min_correlation=args.semantic_min_correlation, nll_margin=args.semantic_nll_margin,
            reference=reference)
        return dict(**metrics['centered'], domains=metrics, passed=not failures, failures=failures)
    finally:
        model.semantics.train(was_training)


@torch.no_grad()
def build_caches(model, dataset, args, rank, world, root):
    official, predicted = root/'official_warmup', root/'predicted_warmup'
    official.mkdir(parents=True,exist_ok=True); predicted.mkdir(parents=True,exist_ok=True)
    marker = root/'semantic_cache_complete.json'
    if marker.exists():
        record = json.loads(marker.read_text())
        if record.get('schema') != SCHEMA:
            raise ValueError('Cannot reuse a pre-repair semantic cache')
        return official,predicted
    model.eval()
    for index in range(rank,len(dataset),world):
        batch = dataset[index]
        for j in range(len(batch['nodule_ids'])):
            row = official_row(model,batch,j)
            atomic_save(row,official/f"{row['key'][0]}__nodule{row['key'][1]}.pt")
        # GT supplies matching and loss flags only; eval routing uses predictions.
        out = model(batch['image'],batch=batch,compute_diagnostics=False,update_bank=False)
        assert out.teacher_forced_count == out.fallback_count == 0
        valid_rows = out.semantic_supervision_valid.nonzero().flatten()
        for i in valid_rows.tolist():
            target = int(out.refined_target_indices[i])
            center = out.crop_origins[i].cpu() + torch.tensor(model.roi_shape)//2
            ct,_,valid = integer_crop(batch['image'],center,model.roi_shape)
            roi = model.semantics.prepare_roi(ct[None].to(model.device),
                out.fine_mask_logits[i][None,None], valid[None].to(model.device))[0].cpu()
            key = (batch['case_id'],int(batch['nodule_ids'][target]))
            atomic_save(dict(roi=roi, spacing=torch.tensor(model.roi_shape)/model.semantics.roi_size,
                source='predicted', key=key, radiomics=out.radiomics_features[i].cpu(),
                histogram=batch['semantic_histograms'][target],semantics=batch['semantic_targets'][target],
                label=batch['malignancy_targets'][target],soft_dice=float(out.fine_soft_dice[i])),
                predicted/f'{key[0]}__nodule{key[1]}.pt')
        print(json.dumps(dict(event='semantic_cache',rank=rank,case_id=batch['case_id'],
                              official=len(batch['nodule_ids']),predicted=len(valid_rows))),flush=True)
    if world>1:
        dist.barrier()
    if rank==0:
        marker.write_text(json.dumps(dict(schema=SCHEMA,
            official=len(list(official.glob('*.pt'))),predicted=len(list(predicted.glob('*.pt'))))))
    if world>1:
        dist.barrier()
    return official,predicted


@torch.no_grad()
def fit_prediction_bank(model, paths, rank, world):
    radio,semantic,labels,keys=[],[],[],[]
    model.semantics.eval();model.rashomon.train()
    # Refit from current CNN predictions on usable prediction-only crops, never
    # initialize a classifier on reader scores and silently switch its inputs.
    model.rashomon.memory.clear();model.rashomon.radiomics_memory.clear()
    model.rashomon.baseline_count.zero_();model.rashomon.count.zero_()
    for path in paths[rank::world]:
        row=torch.load(path,map_location='cpu',weights_only=False)
        value=model.semantics(row['roi'][None].to(model.device),prepared=True)[1][0]
        radio.append(row['radiomics']);semantic.append(value);labels.append(row['label']);keys.append(row['key'])
    radio=torch.stack(radio).to(model.device) if radio else torch.empty(0,18,device=model.device)
    semantic=torch.stack(semantic) if semantic else torch.empty(0,6,device=model.device)
    labels=torch.stack(labels).to(model.device) if labels else torch.empty(0,device=model.device)
    model.rashomon.observe_radiomics(radio,labels,keys,refit=True)
    model.rashomon.observe_semantics(semantic,labels,keys,refit=True)
    if not int(model.rashomon.count) or not int(model.rashomon.baseline_count):
        raise RuntimeError('Usable predicted crops must contain enough nodules from both malignancy classes')


def run_warmup(model,dataset,args,rank,world,monitor):
    saved=preparation_restore(model,args,rank,world)
    geometry=run_geometry_warmup(model,dataset,args,rank,world,monitor,saved)
    root=Path(saved['cache_root']) if saved and saved['phase']!='geometry' else args.output_dir
    if saved and saved['phase']=='ready':
        model.requires_grad_(True)
        return saved['warmup']
    model.requires_grad_(False);model.semantics.requires_grad_(True)
    official,predicted=build_caches(model,dataset,args,rank,world,root)
    predicted_paths=sorted(predicted.glob('*.pt'))
    if len(predicted_paths)<args.bank_min_samples:
        raise RuntimeError('Insufficient usable predicted ROIs for semantic/risk preparation')
    samples=SemanticROIs(sorted(official.glob('*.pt'))+predicted_paths,args.teacher_jitter)
    sampler=DistributedSampler(samples,num_replicas=world,rank=rank,seed=args.seed+171)
    loader=DataLoader(samples,batch_size=args.warmup_batch_size,sampler=sampler,num_workers=0)
    sem=model.semantics
    wrapped=DistributedDataParallel(sem,broadcast_buffers=False) if world>1 else sem
    optimizer=torch.optim.AdamW(sem.parameters(),lr=args.lr_semantic_warmup,weight_decay=args.weight_decay)
    start=0;history=[];ready=False
    if saved and saved['phase']=='semantic':
        optimizer.load_state_dict(saved['optimizer']);start=saved['epoch']+1
        history=saved.get('history',[])
    for epoch in range(start,args.warmup_epochs):
        sampler.set_epoch(epoch);wrapped.train();sums=torch.zeros(2,device=model.device,dtype=torch.float64)
        for step,(roi,histogram) in enumerate(loader,1):
            optimizer.zero_grad(set_to_none=True)
            probability,_=wrapped(roi.to(model.device),prepared=True)
            loss=-(histogram.to(model.device)*probability.clamp_min(1e-7).log()).sum(-1).mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite semantic warmup loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(sem.parameters(),5.,error_if_nonfinite=True)
            optimizer.step();sums+=sums.new_tensor([float(loss.detach())*len(roi),len(roi)])
            if args.preparation_smoke and step>=args.benchmark_steps:break
        if world>1:dist.all_reduce(sums)
        audit=audit_semantics(model,args,official)
        record=dict(epoch=epoch+1,nll=float(sums[0]/sums[1]),audit=audit)
        history.append(record)
        ready=epoch+1>=args.semantic_min_epochs and audit['passed']
        if args.preparation_smoke:ready=True
        if rank==0:
            print(json.dumps(dict(event='semantic_warmup_epoch',**record)),flush=True)
            monitor.log({'warmup_epoch':geometry['epochs']+epoch+1,
                'semantic_warmup/epoch':epoch+1,'semantic_warmup/nll':record['nll'],
                **{f'semantic_warmup/{k}_nll':v['nll'] for k,v in audit['domains'].items()}})
        preparation_save(model,optimizer,args,rank,world,phase='semantic',epoch=epoch,
            geometry=geometry,history=history,cache_root=str(root.resolve()))
        if ready:break
    del wrapped,optimizer
    if not ready:
        raise RuntimeError('Semantic readiness not reached; inspect warmup metrics. preparation.pt preserves progress.')
    fit_prediction_bank(model,predicted_paths,rank,world)
    result=dict(schema=SCHEMA,complete=True,smoke_only=args.preparation_smoke,epochs=epoch+1,
        official_nodules=len(list(official.glob('*.pt'))),predicted_nodules=len(predicted_paths),
        cache_dir=str(official.resolve()),predicted_cache_dir=str(predicted.resolve()),geometry=geometry,
        semantic_reference=audit['domains'],history=history,
        baseline_samples=len(model.rashomon.radiomics_memory),semantic_samples=len(model.rashomon.memory))
    preparation_save(model,None,args,rank,world,phase='ready',epoch=epoch,geometry=geometry,
                     warmup=result,cache_root=str(root.resolve()))
    if rank==0:
        (args.output_dir/'warmup.json').write_text(json.dumps(result,indent=2)+'\n')
        atomic_save(dict(semantics=sem.state_dict(),risk=model.rashomon.state_dict(),audit=result),
                    args.output_dir/'warmup.pt')
    model.requires_grad_(True)
    return result
