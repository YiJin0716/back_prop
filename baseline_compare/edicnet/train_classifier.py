"""Train the EDICNet HSCNN adaptation on all V3 retained physical nodules."""
import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from ..cohort import HERE, load, sha256
from ..nodule_data import NoduleDataset
from ..prepare import CACHE
from .hscnn import HSCNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cache', default=CACHE, type=Path)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    torch.set_num_threads(2)
    cohort = load()
    digest = sha256(HERE / 'cohort.json')
    rows = cohort['splits']['training'][:2] if args.smoke else cohort['splits']['training']
    for row in rows:
        metadata = json.loads((args.cache / row['key'] / 'metadata.json').read_text())
        assert metadata['cohort_sha256'] == digest
    dataset = NoduleDataset(rows, args.cache, 'edicnet')
    if args.smoke:
        dataset = Subset(dataset, list(range(min(6, len(dataset)))))
    else:
        assert len(dataset) == cohort['policy']['retained_physical_nodules']
    assert len(dataset) >= 2
    loader = DataLoader(dataset, batch_size=6, shuffle=True, num_workers=2, pin_memory=True)
    args.output.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(cohort_sha256=digest, training_case_keys_sha256=cohort['training_case_keys_sha256'],
                  training_scans=len(rows), physical_nodules=len(dataset),
                  architecture='HSCNN: four Conv3d, three 256->64 semantic branches, hierarchical malignancy head',
                  optimizer='Adam lr=0.001', batch_size=6, dropout=.8,
                  source='hscnn_upstream/model/cnn_model_module.py:build_no_direct_connection',
                  initialization='Xavier random; no published EDICNet checkpoint',
                  input_adaptation='box-masked 52^3 patch; training annotation box, inference detector box',
                  semantic_adaptation='LIDC proxies; diameter bbox mean xy, solid texture>=4, smooth margin>=4/lobulation<=2/spiculation<=2',
                  semantic_classes={'diameter': ['<=10', '(10,20]', '>20'], 'consistency': ['non-solid', 'solid'],
                                    'margin': ['smooth', 'other']},
                  seed=42, selection='fixed final epoch; no validation/test selection')
    (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    model = HSCNN().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    start = 0
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        assert saved['cohort_sha256'] == digest
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        start = saved['epoch']
    reference = model.features[0].weight.detach().clone()
    for epoch in range(start, args.epochs):
        model.train()
        started, losses, seen = time.time(), [], []
        for step, (image, target, semantic, indices) in enumerate(loader):
            image, target, semantic = image.cuda(), target.cuda(), semantic.cuda()
            optimizer.zero_grad(set_to_none=True)
            malignant, predictions = model(image)
            risk_loss = F.cross_entropy(malignant, target)
            semantic_loss = sum(weight * F.cross_entropy(pred, semantic[:, i])
                                for i, (weight, pred) in enumerate(zip((.33, .34, .33), predictions)))
            # The cited Keras runner applies l2(.048) to convolution and task-base
            # kernels; task-module and final output layers have no regularizer.
            regularization = .048 * sum(p.square().sum() for name, p in model.named_parameters()
                                        if p.ndim > 1 and (name.startswith('features.') or name.startswith('task_bases.')))
            loss = risk_loss + semantic_loss + regularization
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite HSCNN loss')
            loss.backward()
            # Check finiteness without altering the original optimizer update.
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
            optimizer.step()
            seen.extend(indices.tolist()); losses.append(float(loss.detach()))
            if step % 100 == 0:
                print(json.dumps({'epoch': epoch + 1, 'step': step + 1, 'steps': len(loader),
                                  'loss': losses[-1], 'risk_loss': float(risk_loss.detach()),
                                  'semantic_loss': float(semantic_loss.detach()),
                                  'regularization': float(regularization.detach()),
                                  'grad_norm': float(norm), 'elapsed_seconds': time.time() - started}), flush=True)
        assert len(seen) == len(dataset) and len(set(seen)) == len(dataset)
        assert not torch.equal(reference, model.features[0].weight.detach())
        payload = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch + 1,
                   'cohort_sha256': digest, 'config': config}
        temporary = args.output / 'last.pt.tmp'
        torch.save(payload, temporary); temporary.replace(args.output / 'last.pt')
        with (args.output / 'epochs.jsonl').open('a') as stream:
            stream.write(json.dumps({'epoch': epoch + 1, 'loss': float(np.mean(losses)),
                                     'nodules_seen': len(seen), 'seconds': time.time() - started}) + '\n')
        if args.smoke:
            model.eval()
            with torch.no_grad():
                prob = model(image[:1])[0].softmax(-1)[:, 1]
            assert torch.isfinite(prob).all()
            print(json.dumps({'smoke_passed': True, 'risk': prob.tolist(), 'parameter_updated': True}), flush=True)
            return
    (args.output / 'complete.json').write_text(json.dumps({'epochs': args.epochs, 'cohort_sha256': digest,
                                                          'scans': len(rows), 'nodules': len(dataset)}) + '\n')


if __name__ == '__main__':
    main()
