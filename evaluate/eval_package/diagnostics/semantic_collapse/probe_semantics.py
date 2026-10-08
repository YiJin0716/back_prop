"""Read-only controlled probes of the V3 semantic branch.

Uses saved evaluation CT/masks and checkpoint tensors; never changes a trained
checkpoint, runs an optimizer, or updates a production BatchNorm/Rashomon bank.
Alternative inputs/BN/geometry are diagnostic interventions, not predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from monai.networks.nets import resnet50
from back_prop.common.features import MedicalNetSemantics, SEMANTIC_NAMES


class Encoder(nn.Module):
    output_dim = 2048

    def __init__(self):
        super().__init__()
        self.network = resnet50(spatial_dims=3, n_input_channels=1,
                                feed_forward=False, bias_downsample=False)

    def forward(self, value):
        return self.network(value)


def crop(image, origin, shape):
    result = np.zeros(shape, dtype=np.float32)
    origin = np.asarray(origin)
    lower = np.maximum(origin, 0)
    upper = np.minimum(origin + shape, image.shape)
    if np.all(upper > lower):
        source = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))
        target = tuple(slice(int(a), int(b)) for a, b in zip(lower-origin, upper-origin))
        result[target] = image[source]
    return result


def resize(value):
    return F.interpolate(value, (64, 64, 64), mode='trilinear', align_corners=False)


def head(model, embedding):
    score = model.score(embedding)
    thresholds = torch.cat((model.threshold_base[:, None],
        model.threshold_base[:, None] + F.softplus(model.threshold_steps).cumsum(1)), 1)
    cumulative = torch.sigmoid(thresholds[None] - score[..., None])
    probabilities = torch.cat((cumulative[..., :1],
        cumulative[..., 1:] - cumulative[..., :-1], 1-cumulative[..., -1:]), -1).clamp_min(1e-7)
    probabilities = probabilities / probabilities.sum(-1, keepdim=True)
    return (probabilities * model.classes).sum(-1), probabilities, score


def probe(model, value, label):
    records = {}
    def statistics(name):
        def hook(module, args, output):
            x = output.detach().float()
            records[name] = dict(shape=list(x.shape), mean=float(x.mean()),
                std=float(x.std()), zero_fraction=float((x == 0).float().mean()),
                pair_difference_rms=float((x[0]-x[1]).square().mean().sqrt()),
                pair_difference_max=float((x[0]-x[1]).abs().max()))
        return hook
    network = model.encoder.network
    names = ('conv1', 'bn1', 'act', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool')
    handles = [getattr(network, name).register_forward_hook(statistics(name)) for name in names]
    started = time.monotonic()
    with torch.inference_mode():
        embedding = model.encoder(value)
        means, probabilities, scores = head(model, embedding)
    for handle in handles:
        handle.remove()
    record = dict(label=label, seconds=time.monotonic()-started,
        input_std=[float(x.std()) for x in value],
        input_nonzero_fraction=[float((x != 0).float().mean()) for x in value],
        input_fraction_above_1e_3=[float((x.abs() > 1e-3).float().mean()) for x in value],
        stages=records, semantic_means=means.cpu().tolist(),
        semantic_max_pair_difference=float((means[0]-means[1]).abs().max()),
        semantic_probabilities=probabilities.cpu().tolist(), linear_scores=scores.cpu().tolist())
    print(json.dumps({key: record[key] for key in ('label', 'seconds', 'input_std',
                     'semantic_max_pair_difference', 'semantic_means')}), flush=True)
    return record


def original_geometry(network):
    """Match Tencent/MedicalNet ResNet50 backbone strides and dilation."""
    network.conv1.stride = (2, 2, 2)
    for name, dilation in (('layer3', 2), ('layer4', 4)):
        layer = getattr(network, name)
        for block in layer:
            block.conv2.stride = (1, 1, 1)
            block.conv2.dilation = (dilation,) * 3
            block.conv2.padding = (dilation,) * 3
        layer[0].downsample[0].stride = (1, 1, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    case = ROOT/'back_prop/evaluate/eval_package/try_results/v3_epoch018_seed4/LIDC-IDRI-0301__scan301'
    report = json.loads((case/'result.json').read_text())
    checkpoint = torch.load(report['metadata']['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    state = {k.removeprefix('semantics.'): v for k, v in checkpoint['model'].items() if k.startswith('semantics.')}
    model = MedicalNetSemantics(encoder_factory=Encoder, use_checkpoint=False).to(args.device).eval()
    model.load_state_dict(state, strict=True)
    indices = [next(i for i, p in enumerate(report['predictions']) if p['query_id'] == q) for q in (7, 23)]
    masks, cts, tight_cts, tight_masked, gt_cts, gt_masks = [], [], [], [], [], []
    with np.load(case/'arrays.npz', allow_pickle=False) as arrays:
        image = (np.clip(arrays['image_hu'], -1024, 1024) + 1024) / 2048
        for i in indices:
            pred = report['predictions'][i]
            probability = arrays[f'probability_{i}']
            ct = crop(image, pred['origin'], probability.shape)
            cts.append(torch.from_numpy(ct)[None])
            masks.append(torch.from_numpy(probability)[None])
            positions = np.argwhere(probability >= 0.5)
            center = np.round((positions.min(0)+positions.max(0))/2).astype(int)
            ct_small = crop(ct, center-16, np.array([32]*3))
            mask_small = crop(probability, center-16, np.array([32]*3))
            tight_cts.append(torch.from_numpy(ct_small)[None])
            tight_masked.append(torch.from_numpy(ct_small*mask_small)[None])
        for i, gt in enumerate(report['ground_truth']):
            mask = arrays[f'gt_{i}'].astype(np.float32)
            origin = np.asarray(gt['origin'])
            center = np.round(origin + (np.asarray(mask.shape)-1)/2).astype(int)
            gt_cts.append(torch.from_numpy(crop(image, center-64, np.array([128]*3)))[None])
            gt_masks.append(torch.from_numpy(crop(mask, center-64-origin, np.array([128]*3)))[None])
    ct = torch.stack(cts).to(args.device)
    probability = torch.stack(masks).to(args.device)
    native = resize(ct * probability)
    variants = {
        'native_masked_128_to_64': native,
        'zero_vs_native_query23': torch.cat((torch.zeros_like(native[:1]), native[1:]), 0),
        'unmasked_128_to_64': resize(ct),
        'masked_input_x100': native * 100,
        'tight_masked_32_to_64': resize(torch.stack(tight_masked).to(args.device)),
        'tight_unmasked_32_to_64': resize(torch.stack(tight_cts).to(args.device)),
        'gt_masked_128_to_64': resize(torch.stack(gt_cts).to(args.device) * torch.stack(gt_masks).to(args.device)),
    }
    unmasked = variants['tight_unmasked_32_to_64']
    variants['tight_unmasked_zscore'] = (unmasked-unmasked.mean((2,3,4),keepdim=True))/unmasked.std((2,3,4),keepdim=True).clamp_min(1e-3)
    results = [probe(model, value, name) for name, value in variants.items()]
    expected = torch.tensor([[report['predictions'][i]['semantic_features'][name] for name in SEMANTIC_NAMES] for i in indices])
    reconstruction_error = float((torch.tensor(results[0]['semantic_means'])-expected).abs().max())
    assert reconstruction_error < 1e-4, reconstruction_error

    # Temporary BN intervention: use this input batch's statistics. No buffers are saved.
    for module in model.encoder.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.train()
    results.append(probe(model, native, 'native_with_batch_statistics'))
    results.append(probe(model, variants['tight_unmasked_zscore'], 'tight_zscore_with_batch_statistics'))
    model.load_state_dict(state, strict=True)
    model.eval()
    original_geometry(model.encoder.network)
    results.append(probe(model, native, 'original_medicalnet_geometry_native_input'))
    results.append(probe(model, variants['tight_unmasked_zscore'], 'original_medicalnet_geometry_tight_zscore'))

    # Check when loss of image dependence first appears on the same fixed ROIs.
    history = []
    history_model = MedicalNetSemantics(encoder_factory=Encoder, use_checkpoint=False).to(args.device).eval()
    for epoch in (3, 6, 9, 12, 15):
        path = Path(report['metadata']['checkpoint']).with_name(f'epoch_{epoch:03d}.pt')
        old = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        old_state = {k.removeprefix('semantics.'): v for k,v in old['model'].items() if k.startswith('semantics.')}
        history_model.load_state_dict(old_state, strict=True)
        history.append(probe(history_model, native, f'epoch_{epoch:03d}_fixed_current_rois'))
        del old, old_state
    # Pretrained encoder, unchanged epoch-18 head: isolates the encoder weights.
    pretrained_path = Path(checkpoint['config']['medicalnet_checkpoint'])
    pretrained = torch.load(pretrained_path, map_location='cpu', weights_only=False)
    pretrained = pretrained.get('state_dict', pretrained)
    pretrained = {k.removeprefix('module.'):v for k,v in pretrained.items()}
    history_model.load_state_dict(state, strict=True)
    history_model.encoder.network.load_state_dict(pretrained, strict=True)
    results.append(probe(history_model, native, 'pretrained_encoder_epoch18_head_native'))
    results.append(probe(history_model, variants['tight_unmasked_zscore'], 'pretrained_encoder_epoch18_head_tight_zscore'))
    bn = {k: dict(max_abs_change=float((v-pretrained[k.removeprefix('encoder.network.')]).abs().max()))
          for k,v in state.items() if k.endswith(('running_mean','running_var'))}
    output = dict(case_id=report['case_id'], query_ids=[7,23], checkpoint=report['metadata']['checkpoint'],
        device=args.device, torch=torch.__version__, semantic_names=SEMANTIC_NAMES,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        native_reconstruction_max_error=reconstruction_error, probes=results, history=history,
        bn_running_stats_vs_pretrained=bn,
        notes=['GT input is an explicitly labelled diagnostic control, never inference.',
               'Input/BN/geometry controls measure sensitivity, not accuracy or calibration.',
               'Checkpoint and optimizer files are read only.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    print('Saved', args.output, flush=True)


if __name__ == '__main__':
    main()
