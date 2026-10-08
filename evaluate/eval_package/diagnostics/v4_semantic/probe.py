"""Read-only V4 semantic and routing audit on saved training checkpoints.

Run from the repository root with the pro6000 environment on a GPU.
No optimizer, backward pass, bank update, or checkpoint write is performed.
"""
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from back_prop.evaluate.eval_package.inference import load_joint_model, _uninitialized_encoder
from back_prop.common.features import MedicalNetSemantics as V3Semantics, SEMANTIC_NAMES
from back_prop.model_v4.features import MedicalNetSemantics
from back_prop.model_v4.warmup import OfficialROIs

HERE = Path(__file__).resolve().parent
RUNS = Path('/usr/project/rudinlab/datasets/LIDC_IDRI/joint_model/model_v4_runs')
OLD = RUNS / '12723354'
FINAL = RUNS / '12748637/epoch_018.pt'


def save(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def say(**value):
    print(json.dumps(value), flush=True)


def official_probe():
    paths = sorted((OLD / 'official_warmup').glob('*.pt'))
    assert len(paths) == 1599
    subset = np.random.default_rng(42).choice(len(paths), 128, replace=False)
    subset.sort()
    cases = [p.stem.rsplit('__nodule', 1)[0] for p in paths]
    unique_cases = sorted(set(cases))
    case_groups = [np.flatnonzero(np.array(cases) == c) for c in unique_cases]
    targets = torch.stack([torch.load(p, map_location='cpu', weights_only=False)['histogram'] for p in paths]).numpy()
    priors = targets.mean(0)
    case_priors = np.stack([targets[g].mean(0) for g in case_groups]).mean(0)
    base = MedicalNetSemantics(V3Semantics(encoder_factory=_uninitialized_encoder,
                                         use_checkpoint=False), weight_standardization=False).cuda().eval()
    summary = dict(names=SEMANTIC_NAMES, n_rois=len(paths), n_cases=len(case_groups),
                   input='cached official consensus-mask ROIs from the training cohort',
                   subset_indices=subset.tolist(), constant_nodule_prior=priors.tolist(),
                   constant_nodule_prior_nll=float(-(targets * np.log(priors.clip(1e-7))).sum(-1).mean()),
                   constant_case_prior_nll=float(np.mean([
                       -(targets[g] * np.log(case_priors.clip(1e-7))).sum(-1).mean() for g in case_groups])),
                   checkpoints=[])
    jobs = [('warmup', OLD/'warmup.pt', True)] + [
        (f'epoch_{e:03}', OLD/f'epoch_{e:03}.pt', False) for e in (3, 6, 9, 12, 15)] + [
        ('epoch_017', OLD/'latest.pt', True), ('epoch_018', FINAL, True)]
    for label, path, full in jobs:
        started = time.time()
        saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        state = saved['semantics'] if label == 'warmup' else {
            k.removeprefix('semantics.'): v for k, v in saved['model'].items() if k.startswith('semantics.')}
        base.load_state_dict(state, strict=True)
        del state, saved
        selected = np.arange(len(paths)) if full else subset
        loader = DataLoader(OfficialROIs([paths[i] for i in selected]), batch_size=8,
                            num_workers=2, pin_memory=True)
        probabilities, features = [], []
        with torch.inference_mode():
            for roi, _ in loader:
                p, f = base(roi.cuda(non_blocking=True), prepared=True)
                probabilities.append(p.cpu().numpy())
                features.append(f.cpu().numpy())
        probabilities, features = np.concatenate(probabilities), np.concatenate(features)
        losses = -(targets[selected] * np.log(probabilities.clip(1e-7))).sum(-1).mean(-1)
        np.savez_compressed(HERE/f'{label}_official.npz', indices=selected,
                            probabilities=probabilities, features=features, nll=losses,
                            targets=targets[selected])
        item = dict(label=label, checkpoint=str(path), n=len(selected),
                    nll=float(losses.mean()), feature_mean=features.mean(0).tolist(),
                    feature_std=features.std(0).tolist(),
                    probability_max_std=float(probabilities.std(0).max()),
                    fixed_128_nll=float(losses[subset].mean()) if full else float(losses.mean()),
                    fixed_128_feature_std=(features[subset] if full else features).std(0).tolist(),
                    seconds=time.time()-started)
        if full:
            item['case_mean_nll'] = float(np.mean([losses[g].mean() for g in case_groups]))
        summary['checkpoints'].append(item)
        save('official_probe.json', summary)
        say(event='official_checkpoint', **item)
    del base
    gc.collect()
    torch.cuda.empty_cache()


def routing_probe():
    from back_prop.model_v4.data import WholeCTLIDCDataset, load_cases
    from back_prop.model_v4.loss import WholeCTCriterionV4
    cfg = json.loads((OLD/'config.json').read_text())
    model = load_joint_model(OLD/'latest.pt', device='cuda')
    # Only the parent training flag controls the GT fallback. Keep all child
    # modules in eval mode to remove dropout and use exactly the same weights.
    model.eval()
    model.training = True
    assert not any(m.training for m in list(model.modules())[1:])
    criterion = WholeCTCriterionV4()
    bank_before = {k:v.detach().clone() for k,v in model.rashomon.named_buffers()}
    bank_updates, bank_memory = model.rashomon.updates, len(model.rashomon.memory)
    keys = ['LIDC-IDRI-0431__scan435', 'LIDC-IDRI-0556__scan562', 'LIDC-IDRI-0634__scan639']
    cases = [c for c in load_cases(cfg['manifest']) if f"{c['patient_id']}__scan{int(c['scan_id'])}" in keys]
    dataset = WholeCTLIDCDataset(cases, target_mask_shape=tuple(cfg['screen_shape']))
    result = dict(checkpoint=str(OLD/'latest.pt'), checkpoint_epoch=17,
                  protocol='Fixed weights, cached discovery, deterministic child modules in eval mode; '
                           'parent training flag enables routing fallback. No bank updates.',
                  cases=[])
    original_discover = model.discover
    for index in range(len(dataset)):
        started = time.time()
        batch = dataset[index]
        say(event='routing_case_start', case_id=batch['case_id'])
        with torch.inference_mode(), torch.autocast('cuda', dtype=model.amp_dtype):
            cached = original_discover(batch['image'])
            model.discover = lambda image: cached
            rows = []
            for label, p in [('fallback_disabled', 0.0), ('fallback_enabled', 1e-12)]:
                torch.manual_seed(42)
                torch.cuda.manual_seed_all(42)
                out = model(batch['image'], batch=batch, teacher_probability=p, update_bank=False)
                loss = criterion(out, batch)
                indices = out.refined_target_indices
                valid = (indices >= 0) & out.fine_supervision_valid
                matched = []
                for j, target in enumerate(indices.tolist()):
                    if target >= 0:
                        coverage = float(out.fine_target_masks[j].sum()) / float(batch['target_mask_crops'][target].sum())
                        matched.append(dict(target_index=target, query_index=int(out.refined_query_indices[j]),
                                            coverage=coverage, valid=bool(out.fine_supervision_valid[j])))
                row = dict(mode=label, teacher_probability=p, semantic_loss=float(loss.semantic),
                           valid_semantic_rois=int(valid.sum()), matched_rois=matched,
                           teacher_forced=out.teacher_forced_count, fallbacks=out.fallback_count,
                           scan_path_valid=out.scan_path_valid, risk_valid=loss.risk_valid,
                           semantic_mean=out.semantic_features.float().mean(0).cpu().tolist())
                if p > 0:
                    assert out.teacher_forced_count == out.fallback_count, 'Unexpected random teacher selection'
                rows.append(row)
                del out, loss
            assert rows[0]['fallbacks'] == 0
            assert rows[1]['valid_semantic_rois'] >= rows[0]['valid_semantic_rois']
        model.discover = original_discover
        result['cases'].append(dict(case_id=batch['case_id'], comparisons=rows, seconds=time.time()-started))
        save('routing_probe.json', result)
        say(event='routing_case', **result['cases'][-1])
        del cached, batch
        gc.collect()
        torch.cuda.empty_cache()
        # Two reproduced zero-vs-positive cases suffice for a causal audit.
        reproduced = sum(c['comparisons'][0]['semantic_loss'] == 0 and
                         c['comparisons'][1]['semantic_loss'] > 0 for c in result['cases'])
        if reproduced >= 2:
            break
    assert model.rashomon.updates == bank_updates and len(model.rashomon.memory) == bank_memory
    assert all(torch.equal(v, bank_before[k]) for k,v in model.rashomon.named_buffers())
    result['bank_unchanged'] = True
    result['reproduced_zero_loss_cases'] = reproduced
    save('routing_probe.json', result)


if __name__ == '__main__':
    torch.set_num_threads(4)
    say(event='start', gpu=torch.cuda.get_device_name())
    official_probe()
    routing_probe()
    say(event='complete')
