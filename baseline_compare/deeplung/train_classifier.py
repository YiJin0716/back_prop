"""Official DPN92 nodule classifier schedule, followed by training-only GBM fit."""
import argparse
import json
from pathlib import Path
import pickle
import random
import time

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from ..cohort import HERE, load, sha256
from ..nodule_data import NoduleDataset
from ..prepare import CACHE
from .ported.classifier import DPN92_3D


def fit_gbm(model, dataset, output, depth=1, cohort_path=HERE / 'cohort.json'):
    """Original 2560 DPN + 17³ pixels + diameter features; train only."""
    dataset.augment = False
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)
    model.eval()
    features, labels = [], []
    max_diameter = max(item['diameter_mm'] for item in dataset.items)
    with torch.no_grad():
        for image, target, _, indices in loader:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                _, hidden = model(image.cuda())
            raw = dataset.patches[indices.numpy()]
            pixels = raw[:, 8:25, 8:25, 8:25].reshape(len(raw), -1) / 255
            diameter = np.asarray([dataset.items[int(i)]['diameter_mm'] / max_diameter for i in indices])[:, None]
            features.append(np.concatenate([hidden.float().cpu().numpy(), pixels, diameter], axis=1))
            labels.extend(target.tolist())
    features = np.concatenate(features)
    assert features.shape == (len(dataset), 2560 + 17 ** 3 + 1)
    classifier = GradientBoostingClassifier(max_depth=depth, random_state=0)
    classifier.fit(features, labels)
    with (output / 'gbm.pkl').open('wb') as stream:
        pickle.dump({'model': classifier, 'max_diameter': max_diameter, 'pixel_mean': dataset.mean,
                     'pixel_std': dataset.std, 'cohort_sha256': sha256(cohort_path),
                     'fitting_samples': len(dataset)}, stream)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cache', default=CACHE, type=Path)
    parser.add_argument('--epochs', default=700, type=int)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--init', type=Path)
    parser.add_argument('--proposals', type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--cohort', type=Path, default=HERE / 'cohort.json')
    args = parser.parse_args()
    if args.resume and args.init:
        parser.error('Use --resume for continuation or --init for a new adaptation stage')
    if args.proposals and not (args.init or args.resume):
        parser.error('Proposal adaptation requires a trained classifier via --init or --resume')
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    torch.set_num_threads(2); torch.backends.cudnn.benchmark = True
    cohort = load(args.cohort); digest = sha256(args.cohort)
    rows = cohort['splits']['training'][:2] if args.smoke else cohort['splits']['training']
    for row in rows:
        meta = json.loads((args.cache / row['key'] / 'metadata.json').read_text())
        assert meta['cohort_sha256'] == digest
    full_dataset = NoduleDataset(rows, args.cache, 'deeplung')
    proposal_summary = None
    if args.proposals:
        from .proposal_data import add_proposals
        proposal_summary = add_proposals(full_dataset, rows, args.cache, args.proposals, smoke=args.smoke, cohort_path=args.cohort)
    if args.smoke and args.proposals:
        smoke_indices = [next(i for i, item in enumerate(full_dataset.items) if item['target'] == label)
                         for label in (0, 1)]
        dataset = Subset(full_dataset, smoke_indices)
    else:
        dataset = Subset(full_dataset, list(range(min(2, len(full_dataset))))) if args.smoke else full_dataset
    if not args.smoke and not args.proposals:
        assert len(dataset) == cohort['policy']['retained_physical_nodules']
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    args.output.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(cohort_sha256=digest, training_case_keys_sha256=cohort['training_case_keys_sha256'],
                  training_scans=len(rows), physical_nodules=len(dataset),
                  architecture='official DPN92_3D', initialization='random; no public classifier weights found',
                  optimizer='SGD lr=.1/.01/.001 at epochs 0/300/500, momentum=.9, weight_decay=.0005',
                  batch_size=16, amp='bfloat16', seed=42, pixel_mean=full_dataset.mean, pixel_std=full_dataset.std,
                  source='upstream/nodcls/main_nodcls.py (neptime=2)',
                  gbm='max_depth=1, n_estimators=100, random_state=0, fit once on training features at final epoch',
                  selection='fixed final epoch, no validation/test tuning')
    if args.proposals:
        config.update(training_stage='detected_proposal_adaptation', proposal_summary=proposal_summary,
                      physical_nodules=proposal_summary['gt_samples_retained'], training_samples=len(full_dataset),
                      initialization=str(args.init or args.resume),
                      source='upstream/nodcls/det2cls.py (neptime=.3), plus retained GT samples to preserve cohort',
                      optimizer='SGD lr=.1/.01/.001 at epochs 0/45/90, momentum=.9, weight_decay=.0005',
                      gbm='max_depth=2, n_estimators=100, training features at final epoch')
    (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    model = DPN92_3D().cuda()
    if args.init:
        saved = torch.load(args.init, map_location='cpu', weights_only=False)
        assert saved['cohort_sha256'] == digest
        model.load_state_dict(saved['model'], strict=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9, weight_decay=5e-4)
    start = 0
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        assert saved['cohort_sha256'] == digest
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        start = saved['epoch']
    reference = model.linear.weight.detach().clone()
    for epoch in range(start, args.epochs):
        model.train()
        for group in optimizer.param_groups:
            first, second = (45, 90) if args.proposals else (300, 500)
            group['lr'] = .1 if epoch < first else .01 if epoch < second else .001
        started, losses, seen = time.time(), [], []
        for step, (image, target, _, indices) in enumerate(loader):
            image, target = image.cuda(), target.cuda()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits, _ = model(image)
            loss = F.cross_entropy(logits.float(), target)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite DPN92 loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach())); seen.extend(indices.tolist())
            if step % 50 == 0:
                print(json.dumps({'epoch': epoch + 1, 'step': step + 1, 'steps': len(loader), 'loss': losses[-1],
                                  'grad_norm': float(norm), 'elapsed_seconds': time.time() - started}), flush=True)
        expected_indices = dataset.indices if isinstance(dataset, Subset) else list(range(len(dataset)))
        assert sorted(seen) == sorted(expected_indices)
        assert not torch.equal(reference, model.linear.weight.detach())
        temporary = args.output / 'last.pt.tmp'
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch + 1,
                    'cohort_sha256': digest, 'config': config}, temporary)
        temporary.replace(args.output / 'last.pt')
        with (args.output / 'epochs.jsonl').open('a') as stream:
            stream.write(json.dumps({'epoch': epoch + 1, 'loss': float(np.mean(losses)), 'nodules_seen': len(seen),
                                     'seconds': time.time() - started}) + '\n')
        if args.smoke:
            model.eval()
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                logits, hidden = model(image[:1])
            assert hidden.shape == (1, 2560) and torch.isfinite(logits).all()
            print(json.dumps({'smoke_passed': True, 'parameter_updated': True, 'hidden_dim': 2560}), flush=True)
            return
    fit_gbm(model, full_dataset, args.output, depth=2 if args.proposals else 1, cohort_path=args.cohort)
    (args.output / 'complete.json').write_text(json.dumps({'epochs': args.epochs, 'cohort_sha256': digest,
                                                          'scans': len(rows), 'nodules': len(dataset),
                                                          'gbm_fitted': True}) + '\n')


if __name__ == '__main__':
    main()
