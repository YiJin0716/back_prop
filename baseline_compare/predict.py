"""Annotation-free whole-NIfTI CT -> scan-risk inference for the three baselines.

Only the image and trained model artifacts are accepted. No annotation CSV,
cohort metadata, manual mask, or ground-truth box is read during prediction.
"""
from __future__ import annotations
import argparse
import itertools
import json
import math
from pathlib import Path
import pickle
import sys
import time

import cv2
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
import torch

from .nodule_data import crop_cube, box_masked_patch


def report(**kwargs):
    print(json.dumps(kwargs), file=sys.stderr, flush=True)


def load_checkpoint(path):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if 'model' not in saved:
        raise ValueError(f'Expected a trained baseline checkpoint: {path}')
    return saved


def cube_nms(candidates, overlap=.1, keep=20):
    """Candidates are score, center-x/y/z, diameter (1-mm array coordinates)."""
    if not len(candidates):
        return np.empty((0, 5), dtype=np.float32)
    candidates = np.asarray(candidates, dtype=np.float32)
    valid = np.isfinite(candidates).all(1) & (candidates[:, 4] > 0)
    candidates = candidates[valid]
    order = np.argsort(-candidates[:, 0], kind='stable')[:10000]
    selected = []
    while len(order) and len(selected) < keep:
        i = order[0]; selected.append(i)
        remaining = order[1:]
        lo = np.maximum(candidates[i, 1:4] - candidates[i, 4] / 2,
                        candidates[remaining, 1:4] - candidates[remaining, 4, None] / 2)
        hi = np.minimum(candidates[i, 1:4] + candidates[i, 4] / 2,
                        candidates[remaining, 1:4] + candidates[remaining, 4, None] / 2)
        intersection = np.maximum(hi - lo, 0).prod(1)
        union = candidates[i, 4] ** 3 + candidates[remaining, 4] ** 3 - intersection
        order = remaining[intersection / np.maximum(union, 1e-12) <= overlap]
    return candidates[selected]


def sybil_predict(raw, spacing, checkpoint, device):
    from .prepare import sybil_transform
    from .sybil.model import SybilNet, HERE
    saved = load_checkpoint(checkpoint)
    model = SybilNet.from_pretrained(HERE / 'weights/28a7cd44f5bcd3e6cc760b65c7e0d54d.ckpt')
    model.load_state_dict(saved['model'], strict=True)
    model = model.to(device).eval()
    image, _ = sybil_transform(raw, spacing, np.zeros(raw.shape, dtype=np.uint8))
    image = torch.from_numpy(image.astype(np.float32))[None, None].expand(-1, 3, -1, -1, -1).to(device)
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        logit = model(image)['logit'][0, 0].float()
    return {'risk': float(logit.sigmoid()), 'logit': float(logit), 'candidates': [],
            'aggregation': 'adapted first Sybil logit', 'checkpoint_epoch': saved['epoch']}


def deeplung_detect(hu, checkpoint, device, batch_size=4, max_candidates=20):
    from .deeplung.ported.detector import DPN92_3D
    from .deeplung.detection_data import lps_volume, coordinate_grid, normalize_hu
    saved = load_checkpoint(checkpoint)
    model = DPN92_3D(); model.load_state_dict(saved['model'], strict=True)
    model = model.to(device).eval()
    volume = lps_volume(hu)
    # Each 96³ tile owns its middle 64³ region, with 16-voxel context on all sides.
    origins = list(itertools.product(*(range(-16, math.ceil(int(s) / 64) * 64 - 16, 64)
                                       for s in volume.shape)))
    grid = np.stack(np.meshgrid(*([np.arange(4, 20) * 4 + 1.5] * 3), indexing='ij'), -1)
    candidates = []
    anchors = np.asarray([5., 10., 20.], dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(origins), batch_size):
            group = origins[start:start + batch_size]
            images, coordinates = [], []
            for origin in group:
                patch, actual = crop_cube(volume, np.asarray(origin) + 48, 96, fill=0)
                assert tuple(actual) == tuple(origin)
                images.append(normalize_hu(patch)[None])
                coordinates.append(coordinate_grid(origin, 96, volume.shape))
            output = model(torch.from_numpy(np.stack(images)).to(device),
                           torch.from_numpy(np.stack(coordinates)).to(device))
            output = output[:, 4:20, 4:20, 4:20].float().cpu().numpy()
            for origin, prediction in zip(group, output):
                select = prediction[..., 0] > -2
                if not select.any():
                    continue
                z, y, x, a = np.where(select)
                values = prediction[z, y, x, a]
                centers = grid[z, y, x] + values[:, 1:4] * anchors[a, None] + np.asarray(origin)
                diameter = np.exp(np.clip(values[:, 4], -10, 10)) * anchors[a]
                valid = ((centers >= 0) & (centers < np.asarray(volume.shape))).all(1)
                centers, diameter, values = centers[valid], diameter[valid], values[valid]
                # Convert ZYX DICOM-like centers to the classifier's canonical XYZ.
                xyz = np.stack([hu.shape[0] - centers[:, 2], hu.shape[1] - centers[:, 1], centers[:, 0]], 1)
                scores = 1 / (1 + np.exp(-np.clip(values[:, 0], -80, 80)))
                candidates.extend(np.column_stack([scores, xyz, diameter]).tolist())
            if start % (batch_size * 20) == 0:
                report(stage='deeplung_detection', tiles_done=min(start + batch_size, len(origins)), tiles=len(origins))
    return cube_nms(candidates, keep=max_candidates), saved


def deeplung_predict(hu, checkpoint, detector_checkpoint, gbm_path, device, batch_size, max_candidates):
    from .deeplung.ported.classifier import DPN92_3D
    candidates, detector_saved = deeplung_detect(hu, detector_checkpoint, device, batch_size, max_candidates)
    saved = load_checkpoint(checkpoint)
    with Path(gbm_path).open('rb') as stream:
        gbm = pickle.load(stream)
    if not (saved['cohort_sha256'] == detector_saved['cohort_sha256'] == gbm['cohort_sha256']):
        raise ValueError('Detector, classifier and GBM cohorts differ')
    model = DPN92_3D(); model.load_state_dict(saved['model'], strict=True)
    model = model.to(device).eval()
    results = []
    with torch.inference_mode():
        for candidate in candidates:
            score, x, y, z, diameter = candidate.tolist()
            raw, _ = crop_cube(hu, [x, y, z], 32)
            raw = (np.clip((raw + 1200) / 1800, 0, 1) * 255).astype(np.uint8).astype(np.float32)
            patch = (raw - gbm['pixel_mean']) / gbm['pixel_std']
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                logits, features = model(torch.from_numpy(patch)[None, None].to(device))
            feature = np.concatenate([features.float().cpu().numpy()[0], raw[8:25, 8:25, 8:25].ravel() / 255,
                                      [diameter / gbm['max_diameter']]])
            probability = float(gbm['model'].predict_proba(feature[None])[0, list(gbm['model'].classes_).index(1)])
            results.append({'center_xyz_1mm': [x, y, z], 'diameter_mm': diameter, 'detection_probability': score,
                            'malignancy_probability': probability, 'cnn_probability': float(logits.float().softmax(-1)[0, 1])})
    return {'risk': max((n['malignancy_probability'] for n in results), default=0.), 'candidates': results,
            'aggregation': 'maximum GBM malignancy over detected nodules; zero if none',
            'checkpoint_epoch': saved['epoch'], 'detector_epoch': detector_saved['epoch']}


def edicnet_predict(raw, spacing, hu, checkpoint, detector_checkpoint, device, batch_size, max_candidates):
    from .edicnet.detector import RetinaNet
    from .edicnet.hscnn import HSCNN
    detector_saved = load_checkpoint(detector_checkpoint)
    saved = load_checkpoint(checkpoint)
    if saved['cohort_sha256'] != detector_saved['cohort_sha256']:
        raise ValueError('Detector and classifier cohorts differ')
    detector = RetinaNet(pretrained=None); detector.load_state_dict(detector_saved['model'], strict=True)
    detector = detector.to(device).eval()
    candidates = []
    # ndimage.zoom matches the training cache's end-point coordinate convention.
    # Centers refer to voxel-boundary coordinates, so use the output/input shape ratio.
    ratio = np.asarray(hu.shape) / np.asarray(raw.shape)
    with torch.inference_mode():
        for start in range(0, raw.shape[2], batch_size):
            images = []
            slices = list(range(start, min(start + batch_size, raw.shape[2])))
            for z in slices:
                image = np.clip((raw[:, :, z] + 1000) / 1500, 0, 1)
                image = cv2.resize(image, (608, 608), interpolation=cv2.INTER_LINEAR)
                padded = np.zeros((3, 640, 640), dtype=np.float32)
                padded[:, :608, :608] = image[None]; images.append(padded)
            output = detector(torch.from_numpy(np.stack(images)).to(device))
            for z, result in zip(slices, output):
                scores = result['scores'].float().cpu().numpy()
                boxes = result['boxes'].float().cpu().numpy().clip(0, 608)
                order = np.argsort(-scores)[:100]
                for score, (left, top, right, bottom) in zip(scores[order], boxes[order]):
                    if right <= left or bottom <= top:
                        continue
                    lower = np.asarray([top / 608 * hu.shape[0], left / 608 * hu.shape[1]])
                    upper = np.asarray([bottom / 608 * hu.shape[0], right / 608 * hu.shape[1]])
                    center = (lower + upper) / 2
                    diameter = float(max(upper - lower))
                    candidates.append([float(score), float(center[0]), float(center[1]), (z + .5) * ratio[2],
                                       diameter, float(upper[0] - lower[0]), float(upper[1] - lower[1])])
            if start % (batch_size * 20) == 0:
                report(stage='edicnet_detection', slices_done=slices[-1] + 1, slices=raw.shape[2])
    del detector
    candidates = cube_nms(candidates, keep=max_candidates)
    classifier = HSCNN(); classifier.load_state_dict(saved['model'], strict=True)
    classifier = classifier.to(device).eval()
    results = []
    with torch.inference_mode():
        for candidate in candidates:
            score, x, y, z, diameter, width_x, width_y = candidate.tolist()
            center = np.asarray([x, y, z])
            extent = np.asarray([width_x, width_y, diameter])
            box = np.concatenate([center - extent / 2, center + extent / 2])
            patch = box_masked_patch(hu, center, box)
            malignant, semantic = classifier(torch.from_numpy(patch)[None, None].to(device))
            results.append({'center_xyz_1mm': center.tolist(), 'diameter_mm': diameter, 'detection_probability': score,
                            'malignancy_probability': float(malignant.softmax(-1)[0, 1]),
                            'semantic_probabilities': [pred.softmax(-1)[0].tolist() for pred in semantic]})
    return {'risk': max((n['malignancy_probability'] for n in results), default=0.), 'candidates': results,
            'aggregation': 'highest-detection-score ROI' if max_candidates == 1 else 'maximum malignancy over detected ROIs',
            'no_detection_risk': 0., 'checkpoint_epoch': saved['epoch'], 'detector_epoch': detector_saved['epoch']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['sybil', 'deeplung', 'edicnet'])
    parser.add_argument('--image', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--detector-checkpoint', type=Path)
    parser.add_argument('--gbm', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-candidates', type=int)
    args = parser.parse_args()
    if args.model != 'sybil' and args.detector_checkpoint is None:
        parser.error('--detector-checkpoint is required for detection-based models')
    if args.model == 'deeplung' and args.gbm is None:
        parser.error('--gbm is required for DeepLung')
    if args.batch_size < 1 or (args.max_candidates is not None and args.max_candidates < 1):
        parser.error('batch size and candidate limit must be positive')
    started = time.time()
    torch.set_num_threads(2); cv2.setNumThreads(1)
    device = torch.device(args.device)
    image = nib.as_closest_canonical(nib.load(args.image))
    raw = np.asarray(image.dataobj, dtype=np.float32)
    if raw.ndim != 3 or not np.isfinite(raw).all():
        raise ValueError('Expected a finite 3D HU volume')
    spacing = nib.affines.voxel_sizes(image.affine)
    report(stage='loaded_ct', model=args.model, shape=list(raw.shape))
    if args.model == 'sybil':
        result = sybil_predict(raw, spacing, args.checkpoint, device)
    else:
        hu = zoom(raw, spacing, order=1, mode='nearest', prefilter=False)
        if args.model == 'deeplung':
            result = deeplung_predict(hu, args.checkpoint, args.detector_checkpoint, args.gbm,
                                      device, args.batch_size, args.max_candidates or 20)
        else:
            result = edicnet_predict(raw, spacing, hu, args.checkpoint, args.detector_checkpoint,
                                     device, args.batch_size, args.max_candidates or 1)
    assert math.isfinite(result['risk']) and 0 <= result['risk'] <= 1
    result.update(model=args.model, input_image=str(args.image), checkpoint=str(args.checkpoint),
                  detector_checkpoint=str(args.detector_checkpoint) if args.detector_checkpoint else None,
                  gbm=str(args.gbm) if args.gbm else None, annotation_inputs=False,
                  endpoint='LIDC reader-derived current malignancy', seconds=time.time() - started)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'model': args.model, 'risk': result['risk'], 'candidates': len(result['candidates']),
                      'seconds': result['seconds'], 'output': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
