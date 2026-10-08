"""Nodule training crops from the shared full-CT cache (training supervision only)."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def crop_cube(volume, center, size, fill=-1000):
    origin = np.floor(np.asarray(center) - size / 2).astype(int)
    end = origin + size
    lower = np.maximum(origin, 0)
    upper = np.minimum(end, volume.shape)
    result = np.full((size,) * 3, fill, dtype=np.float32)
    if np.all(upper > lower):
        result[tuple(slice(int(a - o), int(b - o)) for a, b, o in zip(lower, upper, origin))] = volume[
            tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))]
    return result, origin


def box_masked_patch(volume, center, box, size=52):
    patch, origin = crop_cube(volume, center, size)
    patch = np.clip((patch + 1000) / 1500, 0, 1)
    # Annotation-free inference uses the detector's box in this same function.
    lower = np.maximum(np.floor(np.asarray(box[:3]) - origin).astype(int), 0)
    upper = np.minimum(np.ceil(np.asarray(box[3:]) - origin).astype(int), size)
    mask = np.zeros_like(patch)
    if np.all(upper > lower):
        mask[tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))] = 1
    return patch * mask


class NoduleDataset(Dataset):
    def __init__(self, rows, cache, method, augment=True):
        self.items = []
        self.method, self.augment = method, augment
        self.patches = []
        for row in rows:
            directory = Path(cache) / row['key']
            meta = json.loads((directory / 'metadata.json').read_text())
            volume = np.load(directory / 'hu_1mm.npy', mmap_mode='r')
            for nodule in meta['nodules']:
                if nodule['ignored']:
                    continue
                if method == 'edicnet':
                    patch = box_masked_patch(volume, nodule['center_xyz'], nodule['box_xyzxyz'])
                else:
                    patch, _ = crop_cube(volume, nodule['center_xyz'], 32)
                    patch = (np.clip((patch + 1200) / 1800, 0, 1) * 255).astype(np.uint8).astype(np.float32)
                self.patches.append(patch)
                self.items.append({'case_key': row['key'], **nodule})
        self.patches = np.stack(self.patches)
        if method == 'deeplung':
            # Upstream mean/std calculation restricted to the training cohort.
            self.mean = float(self.patches.mean(dtype=np.float64))
            self.std = float(self.patches.std(dtype=np.float64))
            assert self.std > 0

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        patch = self.patches[index].copy()
        item = self.items[index]
        if self.augment:
            if self.method == 'deeplung':
                padded = np.pad(patch, 4)
                shifts = np.random.randint(0, 9, size=3)
                patch = padded[tuple(slice(int(s), int(s) + 32) for s in shifts)]
            for axis in range(3):
                if np.random.random() < .5:
                    patch = np.flip(patch, axis)
            patch = patch.copy()
            if self.method == 'deeplung':
                starts = np.random.randint(0, 29, size=3)
                patch[tuple(slice(int(s), int(s) + 4) for s in starts)] = 0
        if self.method == 'deeplung':
            patch = (patch - self.mean) / self.std
        return (torch.from_numpy(patch.copy())[None], item['target'],
                torch.tensor(item['edicnet_semantic_targets']), index)
