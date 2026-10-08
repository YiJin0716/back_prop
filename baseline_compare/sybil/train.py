"""Fine-tune official Sybil_1 on the frozen V3 training cohort."""
import argparse
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..cohort import HERE as BASE, load, sha256
from ..prepare import CACHE
from .model import SybilNet, attention_loss


class CTDataset(Dataset):
    def __init__(self, cohort, split, cache):
        self.rows = cohort['splits'][split]
        self.cache = Path(cache)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        dest = self.cache / row['key']
        metadata = json.loads((dest / 'metadata.json').read_text())
        assert metadata['key'] == row['key']
        image = torch.from_numpy(np.load(dest / 'sybil.npy').astype(np.float32))[None]
        mask = torch.from_numpy(np.load(dest / 'sybil_attention.npy'))[None]
        return image, mask, float(metadata['risk_target']), metadata['risk_target_valid'], row['key']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, default=CACHE)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--checkpoint', type=Path, default=BASE / 'sybil/weights/28a7cd44f5bcd3e6cc760b65c7e0d54d.ckpt')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--cohort', type=Path, default=BASE / 'cohort.json')
    args = parser.parse_args()
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    cohort = load(args.cohort)
    digest = sha256(args.cohort)
    data = CTDataset(cohort, 'training', args.cache)
    if args.smoke:
        data.rows = data.rows[:1]
    for row in data.rows:
        meta = json.loads((args.cache / row['key'] / 'metadata.json').read_text())
        assert meta['cohort_sha256'] == digest
    args.output.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(cohort_sha256=digest, training_case_keys_sha256=cohort['training_case_keys_sha256'],
                  training_scans=len(data), pretrained_sha256=sha256(args.checkpoint),
                  endpoint='LIDC scan contains a retained nodule with reader mean malignancy > 3',
                  scan_labels_masked=True, attention_supervision='retained physical nodule consensus masks',
                  selected_epoch='fixed final epoch; no validation or test selection', seed=42,
                  torch=torch.__version__, gpu=torch.cuda.get_device_name())
    (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    model = SybilNet.from_pretrained(args.checkpoint)
    model.configure_finetuning()
    model = model.cuda()
    optimizer = torch.optim.AdamW([
        {'params': [p for p in model.image_encoder.parameters() if p.requires_grad], 'lr': 1e-5},
        {'params': [p for name, p in model.named_parameters()
                    if not name.startswith('image_encoder.') and p.requires_grad], 'lr': 1e-4},
    ], weight_decay=1e-5)
    start = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        assert checkpoint['cohort_sha256'] == digest
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start = checkpoint['epoch']
    loader = DataLoader(data, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
    reference = next(model.pool.hidden_fc.parameters()).detach().clone()
    for epoch in range(start, args.epochs):
        started = time.time()
        model.train()
        seen, values = [], []
        for step, (image, mask, target, valid, keys) in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            image = image.cuda(non_blocking=True).expand(-1, 3, -1, -1, -1)
            mask = mask.cuda(non_blocking=True)
            target = target.float().cuda()
            valid = valid.float().cuda()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                result = model(image)
            binary = F.binary_cross_entropy_with_logits(result['logit'][:, 0].float(), target,
                                                        reduction='none')
            risk_loss = (binary * valid).sum() / valid.sum().clamp_min(1)
            localization = attention_loss(result, mask)
            loss = risk_loss + localization
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss: {keys}')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            seen.extend(keys)
            values.append(float(loss.detach()))
            if step % 20 == 0 or args.smoke:
                print(json.dumps({'epoch': epoch + 1, 'step': step + 1, 'cases': len(loader),
                                  'loss': values[-1], 'risk_loss': float(risk_loss.detach()),
                                  'attention_loss': float(localization.detach()), 'grad_norm': float(norm),
                                  'elapsed_seconds': time.time() - started}), flush=True)
        assert sorted(seen) == sorted(r['key'] for r in data.rows)
        assert not torch.equal(reference, next(model.pool.hidden_fc.parameters()).detach())
        payload = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch + 1,
                   'cohort_sha256': digest, 'seen_case_keys': seen, 'config': config}
        temporary = args.output / 'last.pt.tmp'
        torch.save(payload, temporary)
        temporary.replace(args.output / 'last.pt')
        with (args.output / 'epochs.jsonl').open('a') as stream:
            stream.write(json.dumps({'epoch': epoch + 1, 'loss': np.mean(values),
                                     'scans_seen': len(seen), 'seconds': time.time() - started}) + '\n')
        if args.smoke:
            # Verify saved weights and label-free inference independently of loss.
            model.eval()
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                prediction = model(image)['logit'][:, 0].float().sigmoid()
            assert torch.isfinite(prediction).all()
            print(json.dumps({'smoke_passed': True, 'scan_risk': prediction.tolist(),
                              'parameter_updated': True, 'checkpoint_saved': str(args.output / 'last.pt')}), flush=True)
            return
    (args.output / 'complete.json').write_text(json.dumps({'epochs': args.epochs, 'cohort_sha256': digest,
                                                          'training_scans': len(data)}) + '\n')


if __name__ == '__main__':
    main()
