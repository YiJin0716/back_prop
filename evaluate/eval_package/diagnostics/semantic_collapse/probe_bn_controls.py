"""Isolate first-BN suppression, input scaling, and foreground crop controls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import ndimage
import torch
from torch import nn

from probe_semantics import ROOT, Encoder, MedicalNetSemantics, crop, resize, probe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    case = ROOT/'back_prop/evaluate/eval_package/try_results/v3_epoch018_seed4/LIDC-IDRI-0301__scan301'
    report = json.loads((case/'result.json').read_text())
    saved = torch.load(report['metadata']['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    state = {k.removeprefix('semantics.'): v for k,v in saved['model'].items() if k.startswith('semantics.')}
    model = MedicalNetSemantics(encoder_factory=Encoder, use_checkpoint=False).eval()
    model.load_state_dict(state, strict=True)
    rois, hu_rois, mask_rois, small_rois, small_masks = [], [], [], [], []
    crop_records = []
    with np.load(case/'arrays.npz', allow_pickle=False) as arrays:
        hu = arrays['image_hu']
        image = (np.clip(hu, -1024, 1024) + 1024) / 2048
        for query in (7,23):
            i = next(i for i,p in enumerate(report['predictions']) if p['query_id'] == query)
            p = report['predictions'][i]
            probability = arrays[f'probability_{i}']
            ct = crop(image, p['origin'], probability.shape)
            rois.append(torch.from_numpy(ct)[None])
            hu_rois.append(torch.from_numpy(crop(hu,p['origin'],probability.shape))[None])
            mask_rois.append(torch.from_numpy(probability)[None])
            # A disconnected mask's overall bounding-box centre can be empty.
            # This control instead uses the largest 26-connected component.
            components, count = ndimage.label(probability >= .5, structure=np.ones((3,3,3)))
            sizes = np.bincount(components.ravel()); sizes[0] = 0
            positions = np.argwhere(components == int(sizes.argmax()))
            center = np.round((positions.min(0) + positions.max(0))/2).astype(int)
            small_rois.append(torch.from_numpy(crop(ct,center-16,np.array([32]*3)))[None])
            small_mask = crop(probability,center-16,np.array([32]*3))
            small_masks.append(torch.from_numpy(small_mask)[None])
            crop_records.append(dict(query_id=query, component_count=count,
                largest_component_voxels=int(sizes.max()), retained_hard_voxels=int((small_mask>=.5).sum())))
    ct, hu, mask = map(torch.stack, (rois, hu_rois, mask_rois))
    native = resize(ct * mask)
    records = []
    # Directly quantify the scale applied by the checkpoint's first BN.
    bn = model.encoder.network.bn1
    with torch.inference_mode():
        convolution = model.encoder.network.conv1(native)
        actual_variance = convolution.var((0,2,3,4), unbiased=False)
        scale = bn.weight / torch.sqrt(bn.running_var+bn.eps)
    bn_record = dict(running_std_quantiles=torch.quantile(torch.sqrt(bn.running_var),torch.tensor([0.,.5,1.])).tolist(),
        actual_std_quantiles=torch.quantile(actual_variance.sqrt(),torch.tensor([0.,.5,1.])).tolist(),
        normalization_gain_quantiles=torch.quantile(scale.abs(),torch.tensor([0.,.5,1.])).tolist())
    bn.train()
    records.append(probe(model,native,'only_first_bn_uses_batch_statistics'))
    model.load_state_dict(state,strict=True);model.eval()
    records.append(probe(model,resize(hu*mask),'raw_hu_soft_mask_same_roi'))
    records.append(probe(model,native*2048,'native_input_x2048_same_roi'))
    small_ct,small_mask=map(torch.stack,(small_rois,small_masks))
    records.append(probe(model,resize(small_ct*small_mask),'largest_component_32_masked'))
    unmasked=resize(small_ct)
    records.append(probe(model,unmasked,'largest_component_32_unmasked'))
    normalized=(unmasked-unmasked.mean((2,3,4),keepdim=True))/unmasked.std((2,3,4),keepdim=True).clamp_min(1e-3)
    records.append(probe(model,normalized,'largest_component_32_unmasked_zscore'))
    for module in model.encoder.modules():
        if isinstance(module,nn.modules.batchnorm._BatchNorm): module.train()
    records.append(probe(model,normalized,'largest_component_32_zscore_batch_bn'))
    result = dict(case_id=report['case_id'],checkpoint=report['metadata']['checkpoint'],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        first_bn=bn_record,crops=crop_records,probes=records,
        warning='These interventions measure sensitivity only, not held-out accuracy. No checkpoint is written.')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(bn_record),flush=True)


if __name__=='__main__': main()
