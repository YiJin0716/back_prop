"""Node-local frozen VISTA features for DETR warmup.

Only the raw backbone maps are cached. The four affine 1x1 projections remain
trainable: they commute with interpolation and normalized window blending in
real arithmetic. Applying them after blending in FP32 introduces small AMP
rounding differences from the original per-window BF16 projections.
This cache must never be used when the backbone is being trained.
"""
import argparse
import hashlib
import itertools
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset

from back_prop.common.base_model import sliding_starts
from back_prop.common.coarse_model import _gaussian_weight
from back_prop.common.coarse_data import WholeCTLIDCDataset, load_cases
from back_prop.common.features import SEMANTIC_INDICES_V2
from back_prop.common.train_utils import atomic_save

CACHE_SCHEMA = 'v5_frozen_raw_vista_fp32_v1'


def frozen_digest(segmenter):
    digest = hashlib.sha256()
    for name, tensor in sorted(segmenter.model.state_dict().items()):
        digest.update(f'{name}:{tensor.dtype}:{tuple(tensor.shape)}'.encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def provenance(model, dataset, manifest):
    common = Path(__file__).resolve().parents[1] / 'common'
    sources = [Path(__file__), *(common / name for name in (
        'base_model.py', 'semantic_model.py', 'coarse_model.py',
        'base_data.py', 'coarse_data.py', 'semantic_data.py'))]
    return dict(schema=CACHE_SCHEMA, backbone_sha256=frozen_digest(model.segmenter),
        manifest_sha256=hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
        annotations_sha256=hashlib.sha256(json.dumps(sorted(dataset.annotations.items()),
            sort_keys=True,default=lambda value:value.tolist()).encode()).hexdigest(),
        target_spacing=dataset.target_spacing,min_votes=dataset.min_votes,
        official_mask_dir=str(dataset.official_mask_dir.resolve()),
        sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        cases=[f"{c['patient_id']}__scan{int(c['scan_id'])}" for c in dataset.cases],
        window_size=list(model.window_size), screen_shape=list(model.screen_shape),
        overlap=model.overlap, amp_dtype=str(model.amp_dtype),
        prompt_class=model.segmenter.prompt_class,
        policy=dataset.policy_summary)


def cached_screening(segmenter, cached):
    if any(p.requires_grad for p in segmenter.model.parameters()) or segmenter.model.training:
        raise ValueError('Feature cache requires a frozen backbone in eval mode')
    projections = segmenter.pyramid_projections
    if len(projections) != len(cached['raw_pyramid']):
        raise ValueError('Feature pyramid length mismatch')
    device = next(projections.parameters()).device
    # FP32 preserves the affine/interpolation identity and projection gradients;
    # the DETR that consumes these maps still runs under the caller's AMP mode.
    with torch.autocast(device.type, enabled=False):
        maps = [layer(raw.to(device=device, dtype=torch.float32))
                for layer, raw in zip(projections, cached['raw_pyramid'])]
        return torch.cat((cached['ct'].to(device), cached['probability'].to(device), *maps), dim=1)


@torch.no_grad()
def extract_raw_features(model, image, *, reference=False):
    """Same windows, interpolation coordinates and weights as discover()."""
    if model.segmenter.model.training or any(p.requires_grad for p in model.segmenter.model.parameters()):
        raise ValueError('Freeze the VISTA backbone before extracting features')
    if image.ndim != 4 or image.shape[0] != 1 or image.device.type != 'cpu':
        raise ValueError('Expected a CPU CT tensor [1,X,Y,Z]')
    projections = model.segmenter.pyramid_projections
    for layer in projections:
        if not isinstance(layer, torch.nn.Conv3d) or layer.kernel_size != (1,1,1) or layer.stride != (1,1,1) or layer.padding != (0,0,0):
            raise ValueError('Caching requires pointwise affine pyramid projections')
    raw = {}
    hooks = [layer.register_forward_pre_hook(lambda _m, inputs, i=i: raw.__setitem__(i, inputs[0]))
             for i, layer in enumerate(projections)]
    shape = tuple(image.shape[-3:])
    sums = [torch.zeros((1, layer.in_channels, *model.screen_shape), device=model.device)
            for layer in projections]
    probability_sum = torch.zeros((1,1,*model.screen_shape), device=model.device)
    weight_sum = torch.zeros_like(probability_sum)
    reference_sum = torch.zeros((1,model.vista_feature_dim,*model.screen_shape), device=model.device) if reference else None
    axes = [sliding_starts(n, w, model.overlap) for n,w in zip(shape,model.window_size)]
    count = 0
    try:
        for starts in itertools.product(*axes):
            valid_shape = tuple(min(w,n-s) for s,w,n in zip(starts,model.window_size,shape))
            lower = tuple(int(np.floor(s*t/n)) for s,t,n in zip(starts,model.screen_shape,shape))
            upper = tuple(max(lo+1,int(np.ceil((s+v)*t/n)))
                          for s,v,t,n,lo in zip(starts,valid_shape,model.screen_shape,shape,lower))
            block_shape = tuple(hi-lo for lo,hi in zip(lower,upper))
            destination = (..., *(slice(lo,hi) for lo,hi in zip(lower,upper)))
            patch = model._window(image,starts).unsqueeze(0).to(model.device)
            raw.clear()
            with torch.autocast(model.device.type, enabled=model.device.type=='cuda', dtype=model.amp_dtype):
                logits, projected = model.segmenter.forward_with_features(patch)
            if len(raw) != len(projections):
                raise RuntimeError('Backbone did not produce every pyramid level')
            probability = F.interpolate(logits[(..., *(slice(0,n) for n in valid_shape))].float().sigmoid(),
                                        size=block_shape,mode='trilinear',align_corners=False)
            weight = _gaussian_weight(block_shape,model.device,probability.dtype)
            probability_sum[destination] += probability*weight
            weight_sum[destination] += weight
            originals = []
            for i in range(len(projections)):
                valid = tuple(slice(0,max(1,int(np.ceil(v*f/p))))
                              for v,f,p in zip(valid_shape,raw[i].shape[-3:],patch.shape[-3:]))
                summarized = F.interpolate(raw[i][(...,*valid)].float(),size=block_shape,
                                           mode='trilinear',align_corners=False)
                sums[i][destination] += summarized*weight
                if reference:
                    originals.append(F.interpolate(projected[i][(...,*valid)].float(),size=block_shape,
                                                   mode='trilinear',align_corners=False))
            if reference:
                reference_sum[destination] += torch.cat(originals,dim=1)*weight
            count += 1
    finally:
        for hook in hooks: hook.remove()
    denominator = weight_sum.clamp_min(1e-6)
    ct = F.interpolate(image[None].to(model.device),size=model.screen_shape,mode='trilinear',align_corners=False)
    cached = dict(ct=ct.cpu(),probability=(probability_sum/denominator).cpu(),
                  raw_pyramid=tuple((value/denominator).cpu() for value in sums), windows=count)
    original = torch.cat((ct,probability_sum/denominator,reference_sum/denominator),dim=1) if reference else None
    return cached, original


class CachedDetectorDataset(Dataset):
    def __init__(self, root, model, dataset, manifest):
        self.root = Path(root)
        marker = json.loads((self.root/'complete.json').read_text())
        if marker['provenance'] != provenance(model,dataset,manifest):
            raise ValueError('Frozen feature cache provenance mismatch')
        self.cases = dataset.cases
        self.paths = [self.root/f'{i:06d}.pt' for i in range(len(dataset))]
        if not all(p.is_file() for p in self.paths):
            raise ValueError('Incomplete frozen feature cache')

    def __len__(self): return len(self.paths)

    def __getitem__(self, index):
        return torch.load(self.paths[index],map_location='cpu',weights_only=False,mmap=True)


def load_builder(config_path, checkpoint, device):
    from .model import WholeCTJointModelV5
    from .parallel_warmup import DetectorBranch
    from .preparation import SCHEMA
    config = json.loads(Path(config_path).read_text())
    keys = ('window_size','overlap','screen_shape','roi_shape','num_queries','detr_hidden_dim',
            'detr_coarse_shape','detr_nheads','detr_encoder_layers','detr_decoder_layers',
            'context_channels','hard_negatives','teacher_full_epochs','teacher_zero_epoch',
            'teacher_jitter','minimum_roi_coverage','vista_checkpoint','semantic_roi_size',
            'semantic_width','semantic_min_dice','inference_object_threshold')
    kwargs = {k:config[k] for k in keys}
    for k in ('window_size','screen_shape','roi_shape','detr_coarse_shape'): kwargs[k]=tuple(kwargs[k])
    model = WholeCTJointModelV5(**kwargs,refiner_base_channels=config['refiner_base'],
                               amp_dtype=getattr(torch,config['amp_dtype'])).to(device)
    saved = torch.load(checkpoint,map_location='cpu',weights_only=False)
    if saved.get('schema') != SCHEMA or saved.get('branch') != 'detr':
        raise ValueError('Expected an independent DETR warmup checkpoint')
    DetectorBranch(model).load_state_dict(saved['model'])
    model.eval().requires_grad_(False)
    dataset = WholeCTLIDCDataset(load_cases(Path(config['manifest']),'training'),
                                target_mask_shape=tuple(config['screen_shape']),max_cases=config['max_cases'])
    return model,dataset,config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--detr-checkpoint',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args = parser.parse_args()
    rank,world = int(os.environ.get('RANK',0)),int(os.environ.get('WORLD_SIZE',1))
    device = torch.device('cuda',int(os.environ.get('LOCAL_RANK',0)))
    torch.cuda.set_device(device)
    if world>1: dist.init_process_group('nccl')
    torch.backends.cudnn.benchmark = True
    model,dataset,config = load_builder(args.config,args.detr_checkpoint,device)
    args.output_dir.mkdir(parents=True,exist_ok=True,mode=0o700)
    record = provenance(model,dataset,config['manifest'])
    identity = args.output_dir/'provenance.json'
    if identity.exists() and json.loads(identity.read_text()) != record:
        raise ValueError('Refusing to mix different frozen feature caches')
    if rank==0: identity.write_text(json.dumps(record)+'\n')
    if world>1: dist.barrier()
    started = time.monotonic()
    for ordinal,index in enumerate(range(rank,len(dataset),world),1):
        destination = args.output_dir/f'{index:06d}.pt'
        if destination.exists(): continue  # atomic writes allow an interrupted build to resume
        batch = dataset[index]
        features,_ = extract_raw_features(model,batch['image'])
        # Coarse loader avoids the extra, unused full-volume HU read. Retain
        # the same six-attribute target contract as the joint dataset.
        for key in ('semantic_histograms','semantic_targets','semantic_reader_counts'):
            batch[key] = batch[key][:,list(SEMANTIC_INDICES_V2)]
        batch.pop('image')
        batch['frozen_vista'] = features
        atomic_save(batch,destination)
        print(json.dumps(dict(event='frozen_feature_cache',rank=rank,completed=ordinal,
            case=index,total=len(dataset),elapsed_seconds=time.monotonic()-started,
            windows=features['windows'])),flush=True)
        del batch,features
    if world>1: dist.barrier()
    if rank==0:
        paths = [args.output_dir/f'{i:06d}.pt' for i in range(len(dataset))]
        if not all(p.is_file() for p in paths): raise RuntimeError('Missing cache entries')
        marker = dict(provenance=record,cases=len(paths),bytes=sum(p.stat().st_size for p in paths))
        temporary = args.output_dir/'complete.tmp'
        temporary.write_text(json.dumps(marker)+'\n')
        temporary.replace(args.output_dir/'complete.json')
        print(json.dumps(dict(event='frozen_feature_cache_complete',cases=len(paths),bytes=marker['bytes'])),flush=True)
    if world>1: dist.destroy_process_group()


if __name__=='__main__': main()
