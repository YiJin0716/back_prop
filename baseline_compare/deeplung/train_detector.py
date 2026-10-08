"""Fine-tune the official DeepLung DPN26 detector on the V3 cohort."""
import argparse
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ..cohort import HERE, load, sha256
from ..prepare import CACHE
from .ported.detector import DPN92_3D
from .detection_data import DetectionDataset, detection_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=CACHE)
    parser.add_argument('--checkpoint', type=Path, default=HERE / 'deeplung/upstream/detector/dpnmodel/fd0066.ckpt')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--cohort', type=Path, default=HERE / 'cohort.json')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    torch.backends.cudnn.benchmark = True
    cohort = load(args.cohort); digest = sha256(args.cohort)
    rows = cohort['splits']['training'][:2] if args.smoke else cohort['splits']['training']
    dataset = DetectionDataset(rows, args.cache, digest)
    if args.smoke:
        dataset = Subset(dataset, list(range(min(2, len(dataset)))))
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
    args.output.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(cohort_sha256=digest, training_case_keys_sha256=cohort['training_case_keys_sha256'],
                  training_scans=len(rows), samples_per_epoch=len(dataset),
                  pretrained_sha256=sha256(args.checkpoint), architecture='official DPN26 detector',
                  optimizer='SGD lr=.001 momentum=.9 weight_decay=.0001',
                  pretraining_test_overlap='LUNA16 overlaps LIDC; public fold-0 checkpoint exposure unresolved',
                  preprocessing_adaptation='full 1mm CT, HU[-1200,600], without external LUNA lung-mask cropping',
                  selection='fixed final epoch, no held-out tuning', seed=42)
    (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    model = DPN92_3D()
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(saved['state_dict'], strict=True)
    model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=.001, momentum=.9, weight_decay=1e-4)
    start = 0
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        assert saved['cohort_sha256'] == digest
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        start = saved['epoch']
    reference = model.output[-1].weight.detach().clone()
    for epoch in range(start, args.epochs):
        model.train()
        started, losses, seen, cases = time.time(), [], [], set()
        for step, (images, coordinates, labels, indices, keys) in enumerate(loader):
            images, coordinates, labels = images.cuda(), coordinates.cuda(), labels.cuda()
            optimizer.zero_grad(set_to_none=True)
            output = model(images, coordinates)
            loss = detection_loss(output, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite DeepLung loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach())); seen.extend(indices.tolist()); cases.update(keys)
            if step % 50 == 0:
                print(json.dumps({'epoch': epoch + 1, 'step': step + 1, 'steps': len(loader), 'loss': losses[-1],
                                  'grad_norm': float(norm), 'elapsed_seconds': time.time() - started}), flush=True)
        assert sorted(seen) == list(range(len(dataset)))
        if not args.smoke:
            assert cases == {r['key'] for r in rows}
        assert not torch.equal(reference, model.output[-1].weight.detach())
        payload = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch + 1,
                   'cohort_sha256': digest, 'seen_cases': sorted(cases), 'config': config}
        temporary = args.output / 'last.pt.tmp'
        torch.save(payload, temporary); temporary.replace(args.output / 'last.pt')
        with (args.output / 'epochs.jsonl').open('a') as stream:
            stream.write(json.dumps({'epoch': epoch + 1, 'loss': float(np.mean(losses)),
                                     'samples_seen': len(seen), 'scans_seen': len(cases),
                                     'seconds': time.time() - started}) + '\n')
        if args.smoke:
            model.eval()
            with torch.no_grad():
                prediction = model(images[:1], coordinates[:1])
            assert torch.isfinite(prediction).all()
            print(json.dumps({'smoke_passed': True, 'parameter_updated': True,
                              'output_shape': list(prediction.shape)}), flush=True)
            return
    (args.output / 'complete.json').write_text(json.dumps({'epochs': args.epochs, 'cohort_sha256': digest,
                                                          'scans': len(rows)}) + '\n')


if __name__ == '__main__':
    main()
