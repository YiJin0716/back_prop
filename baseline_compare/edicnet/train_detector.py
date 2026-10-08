"""Train the cited ResNet34 RetinaNet on native CT slices from V3 patients."""
import argparse
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from ..cohort import HERE, load, sha256
from ..prepare import CACHE
from .detector import RetinaNet


class SliceDataset(Dataset):
    def __init__(self, rows, cache, digest, augment=True):
        self.samples = []
        self.augment = augment
        for row in rows:
            directory = Path(cache) / row['key']
            meta = json.loads((directory / 'edicnet_slices.json').read_text())
            assert meta['cohort_sha256'] == digest
            for index, item in enumerate(meta['slices']):
                self.samples.append((directory / 'edicnet_slices.npy', index, item, row['key']))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        cv2.setNumThreads(1)
        path, position, target, key = self.samples[index]
        image = np.load(path, mmap_mode='r')[position].astype(np.float32)
        boxes = np.asarray(target['boxes'], dtype=np.float32).copy()
        height, width = image.shape
        boxes[:, [0, 2]] *= 608 / width; boxes[:, [1, 3]] *= 608 / height
        image = cv2.resize(image, (608, 608), interpolation=cv2.INTER_LINEAR)
        if self.augment and np.random.random() < .5:
            image = image[:, ::-1].copy()
            boxes[:, [0, 2]] = 608 - boxes[:, [2, 0]]
        padded = np.zeros((3, 640, 640), dtype=np.float32)
        padded[:, :608, :608] = image[None]
        return torch.from_numpy(padded), torch.from_numpy(boxes), index


def collate(batch):
    images, boxes, indices = zip(*batch)
    padded = torch.full((len(batch), max(len(b) for b in boxes), 5), -1.)
    for i, target in enumerate(boxes):
        padded[i, :len(target)] = target
    return torch.stack(images), padded, indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cache', default=CACHE, type=Path)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    cohort = load(); digest = sha256(HERE / 'cohort.json')
    rows = cohort['splits']['training'][:1] if args.smoke else cohort['splits']['training']
    dataset = SliceDataset(rows, args.cache, digest)
    if args.smoke:
        dataset = Subset(dataset, list(range(min(2, len(dataset)))))
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2,
                        pin_memory=True, collate_fn=collate)
    args.output.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(cohort_sha256=digest, training_case_keys_sha256=cohort['training_case_keys_sha256'],
                  training_scans=len(rows), training_slices=len(dataset),
                  architecture='RetinaNet ResNet34, unchanged upstream FPN and anchors',
                  initialization='official ImageNet ResNet34', optimizer='Adam lr=1e-5', batch_size=2,
                  indeterminate_policy='ignore overlapping anchors; not negative labels',
                  selection='fixed final epoch; train-loss LR scheduler, no validation/test tuning', seed=42)
    (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    model = RetinaNet().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    start = 0
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        assert saved['cohort_sha256'] == digest
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler']); start = saved['epoch']
    reference = model.classificationModel.output.weight.detach().clone()
    for epoch in range(start, args.epochs):
        model.train()
        started, losses, seen = time.time(), [], []
        for step, (images, boxes, indices) in enumerate(loader):
            images, boxes = images.cuda(), boxes.cuda()
            optimizer.zero_grad(set_to_none=True)
            classification, regression = model(images, boxes)
            loss = classification + regression
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite RetinaNet loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .1, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach())); seen.extend(indices)
            if step % 100 == 0:
                print(json.dumps({'epoch': epoch + 1, 'step': step + 1, 'steps': len(loader),
                                  'loss': losses[-1], 'classification': float(classification.detach()),
                                  'regression': float(regression.detach()), 'grad_norm': float(norm),
                                  'elapsed_seconds': time.time() - started}), flush=True)
        assert sorted(seen) == list(range(len(dataset)))
        assert not torch.equal(reference, model.classificationModel.output.weight.detach())
        mean_loss = float(np.mean(losses)); scheduler.step(mean_loss)
        payload = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                   'scheduler': scheduler.state_dict(), 'epoch': epoch + 1,
                   'cohort_sha256': digest, 'config': config}
        temporary = args.output / 'last.pt.tmp'
        torch.save(payload, temporary); temporary.replace(args.output / 'last.pt')
        with (args.output / 'epochs.jsonl').open('a') as stream:
            stream.write(json.dumps({'epoch': epoch + 1, 'loss': mean_loss, 'slices_seen': len(seen),
                                     'seconds': time.time() - started}) + '\n')
        if args.smoke:
            model.eval()
            with torch.no_grad():
                predictions = model(images[:1])
            assert torch.isfinite(predictions[0]['scores']).all()
            print(json.dumps({'smoke_passed': True, 'parameter_updated': True,
                              'inference_boxes': len(predictions[0]['scores'])}), flush=True)
            return
    (args.output / 'complete.json').write_text(json.dumps({'epochs': args.epochs, 'cohort_sha256': digest,
                                                          'scans': len(rows), 'slices': len(dataset)}) + '\n')


if __name__ == '__main__':
    main()
