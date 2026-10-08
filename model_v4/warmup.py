"""Training-only official-mask MedicalNet warmup and official-feature risk fits."""
import json
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from back_prop.common.roi import integer_crop, paste_compact_mask
from back_prop.common.train_utils import atomic_save
from .features import NORMALIZATION, CONVOLUTION, normalization_probe, check_semantic_spread


class OfficialROIs(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        row = torch.load(self.paths[index], weights_only=False, map_location='cpu')
        return row['roi'], row['histogram']


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
    return dict(roi=roi, radiomics=radio, histogram=batch['semantic_histograms'][index].cpu(),
                semantics=batch['semantic_targets'][index].cpu(),
                label=batch['malignancy_targets'][index].cpu(),
                key=(batch['case_id'], int(batch['nodule_ids'][index])))


def run_warmup(model, dataset, args, rank, world, monitor):
    if args.warmup_epochs < 1:
        raise ValueError('V4 requires official-annotation warmup before joint training')
    directory = args.output_dir/'official_warmup'
    directory.mkdir(parents=True, exist_ok=True)
    local, paths = [], []
    model.train()
    for index in range(rank, len(dataset), world):
        batch = dataset[index]
        for j in range(len(batch['nodule_ids'])):
            row = official_row(model, batch, j)
            path = directory/f"{row['key'][0]}__nodule{row['key'][1]}.pt"
            atomic_save(row, path)
            paths.append(str(path))
            local.append({k:v for k,v in row.items() if k not in ('roi','histogram')})
        print(json.dumps(dict(event='warmup_cache', rank=rank, case_id=batch['case_id'],
                              nodules=len(batch['nodule_ids']))), flush=True)
    gathered = [None]*world if world > 1 else [paths]
    if world > 1:
        dist.all_gather_object(gathered, paths)
    all_paths = sorted(p for group in gathered for p in group)
    radio = torch.stack([r['radiomics'] for r in local]).to(model.device) if local else torch.empty(0,18,device=model.device)
    semantic = torch.stack([r['semantics'] for r in local]).to(model.device) if local else torch.empty(0,6,device=model.device)
    labels = torch.stack([r['label'] for r in local]).to(model.device) if local else torch.empty(0,device=model.device)
    keys = [r['key'] for r in local]
    # Each physical nodule contributes once, independent of later CT repeats.
    model.rashomon.observe_radiomics(radio, labels, keys, refit=True)
    model.rashomon.observe_semantics(semantic, labels, keys, refit=True)
    if not int(model.rashomon.count) or not int(model.rashomon.baseline_count):
        raise RuntimeError('Official training GT must initialize both malignancy stages')
    del local, radio, semantic, labels
    samples = OfficialROIs(all_paths)
    sampler = DistributedSampler(samples, num_replicas=world, rank=rank, seed=args.seed)
    loader = DataLoader(samples, batch_size=args.warmup_batch_size, sampler=sampler, num_workers=0)
    semantics = model.semantics
    wrapped = DistributedDataParallel(semantics, broadcast_buffers=False) if world > 1 else semantics
    optimizer = torch.optim.AdamW(semantics.parameters(), lr=args.lr_medicalnet, weight_decay=args.weight_decay)
    initial = next(semantics.encoder.parameters()).detach().clone()
    bn = [m for m in semantics.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    gn = [m for m in semantics.modules() if isinstance(m, torch.nn.GroupNorm)]
    if bn or not gn:
        raise AssertionError('MedicalNet warmup requires GroupNorm and no BatchNorm')
    history = []
    for epoch in range(args.warmup_epochs):
        sampler.set_epoch(epoch)
        wrapped.train()
        sums = torch.zeros(2, device=model.device, dtype=torch.float64)
        for roi, histogram in loader:
            optimizer.zero_grad(set_to_none=True)
            probability, _ = wrapped(roi.to(model.device), prepared=True)
            loss = -(histogram.to(model.device)*probability.clamp_min(1e-7).log()).sum(-1).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite official-mask semantic warmup loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(semantics.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            sums += torch.tensor([float(loss)*len(roi), len(roi)], device=model.device)
        if world > 1:
            dist.all_reduce(sums)
        record = dict(warmup_epoch=epoch+1, **{'warmup/semantic':float(sums[0]/sums[1]),
            'warmup/nodules':len(samples), 'warmup/baseline_loss':model.rashomon.last_fit['baseline_loss'],
            'warmup/residual_risk_loss':model.rashomon.last_fit['best_loss']})
        history.append(record)
        if rank == 0:
            monitor.log(record)
            print(json.dumps(dict(event='warmup_epoch', **record)), flush=True)
    changed = float((next(semantics.encoder.parameters()).detach()-initial).abs().max())
    if changed <= 0:
        raise AssertionError('MedicalNet encoder did not update during warmup')
    # Check the actual pretrained encoder after warmup on the same fixed ROIs
    # individually, in groups, and in eval mode. GN has no EMA to calibrate.
    probe_indices = torch.randperm(len(samples), generator=torch.Generator().manual_seed(args.seed))[:64]
    probe_rows = [samples[int(i)] for i in probe_indices]
    probe = torch.stack([r[0] for r in probe_rows]).to(model.device)
    normalization_audit = normalization_probe(semantics, probe)
    target_means = (torch.stack([r[1] for r in probe_rows]) * torch.arange(1,6)).sum(-1)
    spread_audit = check_semantic_spread(normalization_audit['semantic_std'], target_means)
    result = dict(complete=True, epochs=args.warmup_epochs, official_nodules=len(samples),
        source='training official consensus masks and reader semantic histograms/means',
        encoder_max_change=changed, bn_layers=len(bn), groupnorm_layers=len(gn),
        normalization=NORMALIZATION, convolution=CONVOLUTION, normalization_probe=normalization_audit,
        semantic_spread=spread_audit,
        semantic_probe_std=normalization_audit['semantic_std'], history=history,
        baseline_samples=len(model.rashomon.radiomics_memory), semantic_samples=len(model.rashomon.memory))
    if rank == 0:
        (args.output_dir/'warmup.json').write_text(json.dumps(result,indent=2)+'\n')
        atomic_save(dict(semantics=semantics.state_dict(), risk=model.rashomon.state_dict(),
                         audit=result), args.output_dir/'warmup.pt')
    del wrapped, optimizer
    return result
