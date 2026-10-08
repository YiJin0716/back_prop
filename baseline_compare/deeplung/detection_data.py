"""96^3 DPN crops and original anchor assignments on the frozen V3 cohort."""
import json
from pathlib import Path
import random

import numpy as np
from scipy.ndimage import zoom
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..nodule_data import crop_cube
from .ported.detector import config
from .ported.labels import LabelMapping


def lps_volume(volume):
    return volume[::-1, ::-1, :].transpose(2, 1, 0)


def lps_nodule(nodule, xyz_shape):
    x, y, z = nodule['center_xyz']
    return [z, xyz_shape[1] - y, xyz_shape[0] - x, nodule['diameter_mm']]


def coordinate_grid(origin, extent, volume_shape):
    lower = np.asarray(origin) / np.asarray(volume_shape) - .5
    upper = (np.asarray(origin) + extent) / np.asarray(volume_shape) - .5
    return np.stack(np.meshgrid(*(np.linspace(a, b, 24) for a, b in zip(lower, upper)),
                               indexing='ij')).astype(np.float32)


def normalize_hu(patch):
    return (np.clip((patch + 1200) / 1800 * 255, 0, 255).astype(np.uint8).astype(np.float32) - 128) / 128


class DetectionDataset(Dataset):
    def __init__(self, rows, cache, digest):
        self.cases, self.samples = [], []
        self.mapping = LabelMapping(config, 'train')
        for row in rows:
            directory = Path(cache) / row['key']
            meta = json.loads((directory / 'metadata.json').read_text())
            assert meta['cohort_sha256'] == digest
            shape = meta['array_shapes']['hu_1mm.npy']
            boxes = np.asarray([lps_nodule(n, shape) for n in meta['nodules']], dtype=np.float32)
            case_index = len(self.cases)
            self.cases.append((directory / 'hu_1mm.npy', boxes, row['key']))
            for n, nodule in enumerate(meta['nodules']):
                if not nodule['ignored']:
                    self.samples.append((case_index, n))
        # Original random-background crop fraction. Positive list still includes
        # every retained physical nodule and therefore every V3 training CT.
        self.positives = len(self.samples)
        self.random_count = int(self.positives * .3 / .7)

    def __len__(self):
        return self.positives + self.random_count

    def __getitem__(self, index):
        background = index >= self.positives
        case_index, nodule_index = (random.randrange(len(self.cases)), 0) if background else self.samples[index]
        path, all_boxes, key = self.cases[case_index]
        volume = lps_volume(np.load(path, mmap_mode='r'))
        target = all_boxes[nodule_index].copy()
        size = 96 if background else int(round(96 / np.random.uniform(.75, 1.25)))
        center = np.asarray([np.random.uniform(0, s) for s in volume.shape]) if background else target[:3] + np.random.uniform(-12, 12, 3)
        # Original pad_value=170 in windowed uint8 space corresponds to HU 0.
        crop, origin = crop_cube(volume, center, size, fill=0)
        scale = 96 / size
        if size != 96:
            crop = zoom(crop, scale, order=1, prefilter=False)
        coord = coordinate_grid(origin, size, volume.shape)
        boxes = all_boxes.copy(); boxes[:, :3] -= origin; boxes *= scale
        target[:3] -= origin; target *= scale
        if background:
            target[:] = np.nan
        for axis in (1, 2):
            if np.random.random() < .5:
                crop = np.flip(crop, axis)
                coord = np.flip(coord, axis + 1)
                target[axis] = 96 - target[axis]
                boxes[:, axis] = 96 - boxes[:, axis]
        label = self.mapping((96, 96, 96), target, boxes, key)
        # A large ignored nodule can have IoU < .02 with every small anchor.
        # Protect anchor centers inside any annotated nodule in that case too.
        centers = np.stack(np.meshgrid(*([np.arange(24) * 4 + 1.5] * 3), indexing='ij'), -1)
        for box in boxes:
            inside = (np.abs(centers - box[:3]) <= box[3] / 2).all(-1)
            neutral = inside[..., None] & (label[..., 0] < -.5)
            label[..., 0][neutral] = 0
        return (torch.from_numpy(normalize_hu(crop).copy())[None], torch.from_numpy(coord.copy()),
                torch.from_numpy(label), index, key)


def detection_loss(output, labels):
    # Official BCE + four SmoothL1 terms with two hard negatives per crop.
    batch_size = len(output)
    output, labels = output.float().reshape(-1, 5), labels.reshape(-1, 5)
    positive, negative = labels[:, 0] > .5, labels[:, 0] < -.5
    negatives = output[negative, 0]
    if len(negatives):
        negatives = negatives.topk(min(2 * batch_size, len(negatives))).values
        negative_loss = F.binary_cross_entropy_with_logits(negatives, torch.zeros_like(negatives)) * .5
    else:
        negative_loss = output.sum() * 0
    if positive.any():
        positive_loss = F.binary_cross_entropy_with_logits(output[positive, 0], labels[positive, 0]) * .5
        regression = sum(F.smooth_l1_loss(output[positive, i], labels[positive, i]) for i in range(1, 5))
    else:
        positive_loss = regression = output.sum() * 0
    return negative_loss + positive_loss + regression
