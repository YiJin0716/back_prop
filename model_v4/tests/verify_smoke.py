"""Audit real-data checkpoints, resume state, and GT-free CUDA inference."""
import gc
import json
from pathlib import Path
import sys
import torch

from back_prop.model_v4.model import ARCHITECTURE, WholeCTJointModelV4
from back_prop.model_v4.data import WholeCTLIDCDataset, load_cases
from back_prop.model_v4.schedule import LearningRateSchedule
from back_prop.model_v4.features import normalization_probe


def main():
    root = Path(sys.argv[1])
    schedule = LearningRateSchedule()
    audit = {}
    for name, completed in [('first', 1), ('resumed', 2), ('oversample', 2), ('oversample_resumed', 3), ('diagnostic', 1)]:
        saved = torch.load(root/name/'latest.pt', map_location='cpu', weights_only=False)
        config = saved['config']
        assert saved['architecture'] == ARCHITECTURE and saved['epoch']+1 == completed
        assert float(saved['model']['mask_temperature']) == 1.
        epoch = config['smoke_curriculum_epoch'] if config['smoke_curriculum_epoch'] is not None else completed-1
        assert saved['lr_scheduler']['last_epoch'] == epoch+1
        assert len(saved['rng_states']) == 4
        for group in saved['optimizer']['param_groups']:
            assert abs(group['lr']-config['lr_'+group['name']]*schedule(epoch+1)) < 1e-15
        records = [json.loads(line) for line in (root/name/'metrics.jsonl').read_text().splitlines()]
        latest = records[-1]
        assert latest['missing_nodule'] > 0 and latest['temperature'] == 1.
        assert all(row['diagnostic_scale'] == 1. and row['nodule'] > 0 and row['risk'] > 0
                   for row in records)
        assert latest['global_step'] == saved['global_step']
        audit[name] = dict(epoch=completed,global_step=saved['global_step'],lr=latest['lr'],
                           missing_nodule=latest['missing_nodule'],semantic=latest['semantic'],
                           nodule=latest['nodule'],risk=latest['risk'],diagnostic_scale=latest['diagnostic_scale'],
                           bank_count=int(saved['model']['rashomon.count']))
        assert saved['warmup']['complete'] and saved['warmup']['bn_layers'] == 0
        assert saved['warmup']['groupnorm_layers'] == 53
        assert saved['warmup']['normalization_probe']['passed']
        assert config['medicalnet_normalization']['kind'] == 'GroupNorm'
        assert config['diagnostic_schedule'] == dict(kind='constant',scale=1.,start_epoch=1)
        assert not any(k.startswith('semantics.') and
                       k.endswith(('running_mean','running_var','num_batches_tracked'))
                       for k in saved['model'])
        audit[name]['normalization_probe'] = saved['warmup']['normalization_probe']
        assert max(saved['warmup']['semantic_probe_std']) > 1e-5
        assert not ({'object','radiomics','ordinal'} & set(config['loss_weights']))
        assert not ({'object','radiomics','ordinal'} & set(latest))
        if name.startswith('oversample'):
            assert saved['sampler']['last_mining_epoch'] == completed
            assert latest['sampling']['extra_cases'] > 0
            assert latest['mining']['gt_routing'] is False
            assert set(saved['sampler']['history']) == set(saved['model']['rashomon._extra_state']['allowed_cases'])
            audit[name]['extra_cases'] = latest['sampling']['extra_cases']
        monitoring = [json.loads(line) for line in (root/name/'monitor.jsonl').read_text().splitlines()]
        epochs = [row for row in monitoring if 'epoch' in row]
        assert len(epochs) == len(records) and all('global_step' in row for row in epochs)
        for row in epochs:
            assert abs(row['00_training/total']-row['00_training/weighted_sum']) < 1e-3
        if name != 'diagnostic':
            del saved; gc.collect()
    assert audit['resumed']['global_step'] == 2*audit['first']['global_step']
    assert audit['resumed']['lr']['main'] < audit['first']['lr']['main']
    assert int(saved['model']['rashomon.baseline_count']) > 0
    assert int(saved['model']['rashomon.count']) > 0
    assert audit['diagnostic']['semantic'] > 0
    assert audit['oversample_resumed']['global_step'] > audit['oversample']['global_step']
    assert saved['model']['rashomon.weights'].shape[1] == 6
    assert saved['model']['rashomon.radiomics_weights'].shape == (18,)
    model = WholeCTJointModelV4(
        window_size=tuple(config['window_size']), screen_shape=tuple(config['screen_shape']),
        roi_shape=tuple(config['roi_shape']), num_queries=config['num_queries'],
        vista_checkpoint=config['vista_checkpoint'],medicalnet_checkpoint=config['medicalnet_checkpoint'],
        medicalnet_roi_size=config['medicalnet_roi_size'],baseline_l2=config['baseline_l2'],
        bank_config=dict(sparsity=config['sparsity'], bound=config['coefficient_bound'],
            beam=config['beam'], pool_size=config['pool_size'], gap=config['gap'],
            fraction=config['sample_fraction'], min_samples=config['bank_min_samples'],
            refresh_every=config['bank_refresh_every'], mode=config['bank_mode']))
    model.load_state_dict(saved['model'], strict=True)
    del saved; gc.collect()
    model.cuda().eval()
    # A held-out case never enters either training-only feature cache.
    dataset=WholeCTLIDCDataset(load_cases(config['manifest'],'testing'),max_cases=1,
                               target_mask_shape=tuple(config['screen_shape']))
    batch=dataset[0]
    before=(len(model.rashomon.memory),int(model.rashomon.fit_count),int(model.rashomon.baseline_count))
    prepared_rois=[]
    handle=model.semantics.register_forward_pre_hook(
        lambda module,inputs: prepared_rois.append(module.prepare_roi(*inputs).detach()))
    with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16):
        output=model(batch['image'],image_hu=batch['image_hu'])
    handle.remove()
    assert output.teacher_forced_count == 0 and output.fallback_count == 0
    assert torch.isfinite(output.nodule_logits).all() and torch.isfinite(output.risk_logit).all()
    torch.testing.assert_close(output.ensemble_logits,
                               output.radiomics_logits[:,None]+output.semantic_correction_logits)
    torch.testing.assert_close(output.predicted_residuals,
                               output.ensemble_logits.sigmoid()-output.radiomics_logits.sigmoid()[:,None])
    assert before == (len(model.rashomon.memory),int(model.rashomon.fit_count),int(model.rashomon.baseline_count))
    audit['inference']=dict(case_id=batch['case_id'],predictions=len(output.nodule_logits),
                            models=len(output.model_indices),risk=float(output.risk_logit.sigmoid()),
                            cache_unchanged=True,gt_routing=False)
    assert prepared_rois, 'Smoke inference must exercise MedicalNet on actual predicted ROIs'
    audit['predicted_roi_normalization'] = normalization_probe(model.semantics,torch.cat(prepared_rois[:16]))
    (root/'verification.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(dict(event='v4_smoke_verified',**audit)),flush=True)


if __name__=='__main__': main()
