"""High-resolution VISTA proposals alongside the learned DETR queries.

Proposal selection is detached and uses predictions only. The original stitched
features retain their gradients; DETR keeps its own full set of loss targets.
"""
from dataclasses import dataclass
import itertools

import numpy as np
from scipy import ndimage
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from back_prop.common.base_model import sliding_starts
from back_prop.common.coarse_model import _gaussian_weight


@dataclass
class NativeProposals:
    boxes: torch.Tensor
    object_logits: torch.Tensor
    coarse_logits: torch.Tensor
    origins: list[tuple[int, int, int]]
    probabilities: list[torch.Tensor]


def discover_native(model, image):
    """Preserve native probabilities without changing DETR's existing inputs.

    VISTA runs once per window. Native probability accumulation is detached to
    avoid retaining full-CT graphs solely for discrete proposal generation.
    """
    if image.ndim != 4 or image.shape[0] != 1 or image.device.type != 'cpu':
        raise ValueError('Expected a CPU CT tensor [1,X,Y,Z]')
    shape=tuple(image.shape[-3:]);device=model.device
    native_sum=torch.zeros(shape,device=device)
    native_weight=torch.zeros_like(native_sum)
    probability_sum=torch.zeros((1,1,*model.screen_shape),device=device)
    feature_sum=torch.zeros((1,model.vista_feature_dim,*model.screen_shape),device=device)
    weight_sum=torch.zeros_like(probability_sum)
    axes=[sliding_starts(n,w,model.overlap) for n,w in zip(shape,model.window_size)]
    trainable=any(p.requires_grad for p in model.segmenter.parameters())
    windows=0
    for start in itertools.product(*axes):
        valid=tuple(min(n-s,w) for n,s,w in zip(shape,start,model.window_size))
        lower=tuple(int(np.floor(s*t/n)) for s,t,n in zip(start,model.screen_shape,shape))
        upper=tuple(max(lo+1,int(np.ceil((s+v)*t/n)))
            for s,v,t,n,lo in zip(start,valid,model.screen_shape,shape,lower))
        block=tuple(hi-lo for hi,lo in zip(upper,lower))
        patch=model._window(image,start)[None].to(device)

        def summarize(value,valid=valid,block=block):
            if hasattr(model.segmenter,'forward_with_features'):
                logits,pyramid=model.segmenter.forward_with_features(value)
            else:
                logits=model.segmenter(value);pyramid=(logits,)
            native=logits[(...,*(slice(0,n) for n in valid))].float().sigmoid()
            maps=[]
            for feature in pyramid:
                extent=tuple(max(1,int(np.ceil(v*f/p))) for v,f,p in
                             zip(valid,feature.shape[-3:],value.shape[-3:]))
                maps.append(F.interpolate(feature[(...,*(slice(0,n) for n in extent))].float(),
                    size=block,mode='trilinear',align_corners=False))
            return native,torch.cat(maps,dim=1)

        with torch.autocast(device.type,enabled=device.type=='cuda',dtype=model.amp_dtype):
            if model.training and model.use_checkpoint and trainable:
                native,features=checkpoint(summarize,patch,use_reentrant=False)
            else:native,features=summarize(patch)
        if features.shape[1]!=model.vista_feature_dim:
            raise ValueError('Unexpected VISTA feature width')
        probability=F.interpolate(native,size=block,mode='trilinear',align_corners=False)
        weight=_gaussian_weight(block,device,probability.dtype)
        dest=(...,*(slice(lo,hi) for lo,hi in zip(lower,upper)))
        probability_sum[dest]+=probability*weight
        feature_sum[dest]+=features*weight
        weight_sum[dest]+=weight
        native_dest=tuple(slice(s,s+n) for s,n in zip(start,valid))
        native_importance=_gaussian_weight(valid,device,torch.float32).reshape(valid)
        native_sum[native_dest]+=native[0,0].detach()*native_importance
        native_weight[native_dest]+=native_importance
        windows+=1
    probability=probability_sum/weight_sum.clamp_min(1e-6)
    features=feature_sum/weight_sum.clamp_min(1e-6)
    ct=F.interpolate(image[None].to(device),size=model.screen_shape,mode='trilinear',align_corners=False)
    screening=torch.cat((ct,probability,features),dim=1)
    native=native_sum/native_weight.clamp_min(1e-6)
    return screening,probability,native,windows


@torch.no_grad()
def native_proposals(probability,screen_shape,*,threshold=.5,minimum_voxels=8,maximum=24,margin=8):
    """Connected predictions with native centers and compact soft-mask priors."""
    if probability.ndim!=3 or not 0<threshold<1 or minimum_voxels<1 or maximum<1:
        raise ValueError('Invalid native proposal inputs')
    device=probability.device;p=probability.detach().float().cpu().numpy()
    labels,count=ndimage.label(p>=threshold)
    mass=np.bincount(labels.ravel())
    ids=np.flatnonzero(mass>=minimum_voxels);ids=ids[ids!=0]
    scores=np.array(ndimage.maximum(p,labels,ids)) if len(ids) else np.empty(0)
    order=np.argsort(-scores,kind='stable')[:maximum]
    bounds=ndimage.find_objects(labels,max_label=count)
    spatial=np.array(p.shape);coarse_shape=np.array(screen_shape)
    boxes=[];coarse=[];origins=[];crops=[];selected_scores=[]
    for index in order:
        identity=ids[index];bound=bounds[identity-1]
        lo=np.array([s.start for s in bound]);hi=np.array([s.stop for s in bound])
        local=labels[bound]==identity
        positions=np.argwhere(local)+lo
        values=p[tuple(positions.T)]
        center=((positions+.5)*values[:,None]).sum(0)/values.sum()
        boxes.append(np.concatenate((center/spatial,(hi-lo)/spatial)))
        target=np.floor(positions*coarse_shape/spatial).astype(np.int64)
        target=np.minimum(target,coarse_shape-1)
        pooled=np.zeros(tuple(screen_shape),dtype=np.float32)
        np.maximum.at(pooled,tuple(target.T),values)
        coarse.append(torch.from_numpy(pooled).to(device))
        crop_lo=np.maximum(lo-margin,0);crop_hi=np.minimum(hi+margin,spatial)
        extent=tuple(slice(a,b) for a,b in zip(crop_lo,crop_hi))
        values=p[extent].copy();local_labels=labels[extent]
        # Remove other instances, preserving soft boundary probabilities.
        values[(local_labels!=0)&(local_labels!=identity)]=0
        origins.append(tuple(int(n) for n in crop_lo))
        crops.append(torch.from_numpy(values).to(device))
        selected_scores.append(scores[index])
    boxes=torch.as_tensor(np.array(boxes).reshape(-1,6),device=device,dtype=torch.float32)
    scores=torch.as_tensor(selected_scores,device=device,dtype=torch.float32)
    masks=torch.stack(coarse) if coarse else probability.new_empty((0,*screen_shape))
    return NativeProposals(boxes,torch.logit(scores.clamp(1e-6,1-1e-6)),
        torch.logit(masks.clamp(1e-6,1-1e-6)),origins,crops)


def native_roi(proposals,index,origin,shape):
    """Place a compact predicted probability crop in a fine-refinement ROI."""
    crop=proposals.probabilities[index]
    offset=torch.tensor(proposals.origins[index])-torch.as_tensor(origin,device='cpu')
    lower=torch.maximum(offset,torch.zeros(3,dtype=torch.long))
    upper=torch.minimum(offset+torch.tensor(crop.shape),torch.tensor(shape))
    result=crop.new_zeros((1,1,*shape))
    if bool((upper>lower).all()):
        dst=tuple(slice(int(a),int(b)) for a,b in zip(lower,upper))
        src=tuple(slice(int(a-s),int(b-s)) for a,b,s in zip(lower,upper,offset))
        result[(0,0,*dst)]=crop[src]
    return torch.logit(result.clamp(1e-6,1-1e-6))
