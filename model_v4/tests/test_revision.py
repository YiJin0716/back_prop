import copy
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from torch import nn

from back_prop.common.features import MedicalNetSemantics as OriginalSemantics
from back_prop.model_v4.features import (MedicalNetSemantics, normalization_probe, replace_batchnorm,
                                        standardize_convolutions, WeightStandardizedConv3d,
                                        check_semantic_spread)
from back_prop.model_v4.monitor import epoch_metrics, COMPONENTS, GROUPS
from back_prop.model_v4_oversample.sampling import MissingNoduleSampler
from back_prop.model_v4_oversample.mining import missed_nodules


class BNEncoder(nn.Module):
    output_dim=4
    def __init__(self):
        super().__init__()
        self.conv=nn.Conv3d(1,4,3,padding=1)
        self.bn=nn.BatchNorm3d(4)
    def forward(self,x):
        return self.bn(self.conv(x)).relu().mean((2,3,4))


class RevisionTests(unittest.TestCase):
    def test_warmup_rejects_constant_predictions_for_varying_targets(self):
        targets=torch.linspace(1,5,64)[:,None].expand(-1,6)
        with self.assertRaisesRegex(RuntimeError,'joint training was not started'):
            check_semantic_spread([.001]*6,targets)
        self.assertTrue(check_semantic_spread([.2]*6,targets)['checked'])
        self.assertFalse(check_semantic_spread([0.]*6,targets[:1])['checked'])
        check_semantic_spread([0.]*6,torch.ones(64,6))

    def test_weight_standardization_preserves_parameters_and_removes_filter_offsets(self):
        torch.manual_seed(12)
        original=nn.Sequential(nn.Conv3d(2,4,3,padding=1),nn.GroupNorm(1,4))
        parameters=list(original.parameters())
        state=copy.deepcopy(original.state_dict())
        standardize_convolutions(original)
        self.assertIsInstance(original[0],WeightStandardizedConv3d)
        self.assertTrue(all(a is b for a,b in zip(parameters,original.parameters())))
        original.load_state_dict(state,strict=True)
        x=torch.randn(3,2,8,8,8,requires_grad=True)
        expected=original(x)
        # A DC offset of a pretrained filter must not erase input differences.
        with torch.no_grad():
            original[0].weight.add_(torch.arange(4.)[:,None,None,None,None])
        actual=original(x)
        torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-5)
        (actual*torch.randn_like(actual)).sum().backward()
        self.assertGreater(float(x.grad.abs().sum()),0.)
        self.assertGreater(float(original[0].weight.grad.abs().sum()),0.)
        singleton=nn.Sequential(nn.Conv3d(1,2,1))
        standardize_convolutions(singleton)
        self.assertNotIsInstance(singleton[0],WeightStandardizedConv3d)

    def test_legacy_semantics_retains_unstandardized_convolutions(self):
        torch.manual_seed(5)
        source=OriginalSemantics(roi_size=8,encoder_factory=BNEncoder,use_checkpoint=False)
        legacy=MedicalNetSemantics(copy.deepcopy(source),weight_standardization=False)
        fixed=MedicalNetSemantics(copy.deepcopy(source))
        self.assertNotIsInstance(legacy.encoder.conv,WeightStandardizedConv3d)
        self.assertIsInstance(fixed.encoder.conv,WeightStandardizedConv3d)
        # Both use the same raw pretrained tensors, but different forward paths.
        fixed.load_state_dict(legacy.state_dict(),strict=True)
        ct=torch.randn(4,1,8,8,8)
        self.assertFalse(torch.allclose(legacy(ct,prepared=True)[1],fixed(ct,prepared=True)[1],atol=1e-7))

    def test_groupnorm_batch_mode_invariance_and_checkpoint_gradients(self):
        torch.set_num_threads(2)
        source=OriginalSemantics(roi_size=8,encoder_factory=BNEncoder,use_checkpoint=True)
        with torch.no_grad():
            source.encoder.bn.weight.copy_(torch.tensor([.5,1.,1.5,2.]))
            source.encoder.bn.bias.copy_(torch.tensor([-.2,.1,.3,-.1]))
        affine=source.encoder.bn.weight.detach().clone()
        convolution=source.encoder.conv.weight.detach().clone()
        model=MedicalNetSemantics(source).train()
        direct=copy.deepcopy(model); direct.use_checkpoint=False
        self.assertIsInstance(model.encoder.bn,nn.GroupNorm)
        self.assertEqual(model.encoder.bn.num_groups,1)
        self.assertFalse(any(isinstance(m,nn.modules.batchnorm._BatchNorm) for m in model.modules()))
        torch.testing.assert_close(model.encoder.bn.weight,affine,rtol=0,atol=0)
        torch.testing.assert_close(model.encoder.conv.weight,convolution,rtol=0,atol=0)
        ct=torch.rand(5,1,8,8,8)*torch.tensor([.1,1.,2.,10.,.3])[:,None,None,None,None]
        result=model(ct,prepared=True)[0]; expected=direct(ct,prepared=True)[0]
        result.square().sum().backward(); expected.square().sum().backward()
        torch.testing.assert_close(result,expected)
        torch.testing.assert_close(model.encoder.conv.weight.grad,direct.encoder.conv.weight.grad)
        self.assertGreater(float(model.encoder.conv.weight.grad.abs().sum()),0.)
        before=copy.deepcopy(model.state_dict())
        audit=normalization_probe(model,ct,batch_size=4)
        self.assertTrue(audit['passed'])
        self.assertTrue(model.training and model.use_checkpoint)
        self.assertGreater(max(audit['semantic_std']),1e-5)
        for k,v in model.state_dict().items():
            torch.testing.assert_close(v,before[k],rtol=0,atol=0)
        restored=MedicalNetSemantics(OriginalSemantics(roi_size=8,encoder_factory=BNEncoder))
        restored.load_state_dict(model.state_dict(),strict=True)
        restored.eval(); model.eval()
        torch.testing.assert_close(restored(ct,prepared=True)[0],model(ct,prepared=True)[0],rtol=0,atol=0)

    def test_groupnorm_group_sizes_do_not_recreate_instance_normalization(self):
        layers=nn.Sequential(nn.BatchNorm3d(64),nn.Sequential(nn.BatchNorm3d(128)),nn.BatchNorm3d(2048))
        replace_batchnorm(layers)
        norms=[m for m in layers.modules() if isinstance(m,nn.GroupNorm)]
        self.assertEqual([m.num_groups for m in norms],[8,16,32])
        self.assertTrue(all(m.num_channels//m.num_groups>=8 for m in norms))

    def test_both_risk_losses_use_full_weight_from_first_joint_epoch(self):
        from back_prop.model_v4.train import diagnostic_scale,set_training_phase
        from back_prop.model_v4.loss import WholeCTCriterionV4
        from back_prop.model_v4.tests.test_v4 import V4Tests
        model,batch=V4Tests().tiny()
        self.assertEqual(set_training_phase(model,0),'joint_diagnostics')
        self.assertTrue(all(diagnostic_scale(e)==1. for e in range(18)))
        out=model(batch['image'],batch=batch,epoch=0,update_bank=False,
                  compute_diagnostics=diagnostic_scale(0)>0)
        criterion=WholeCTCriterionV4()
        loss=criterion(out,batch,diagnostic_scale=diagnostic_scale(0))
        self.assertGreater(float(loss.nodule),0.)
        self.assertGreater(float(loss.risk),0.)
        expected=sum(getattr(loss,k)*v for k,v in criterion.loss_weights.items())
        torch.testing.assert_close(loss.total,expected)
        self.assertEqual(criterion.loss_weights['nodule'],.5)
        self.assertEqual(criterion.loss_weights['risk'],.5)

    def test_sampler_ddp_coverage_repeats_caps_resume_and_recovery(self):
        ids=[f'train{i}' for i in range(12)]
        records={c:{'1':i<2,'2':False} for i,c in enumerate(ids)}
        schedules=[]
        for rank in range(4):
            s=MissingNoduleSampler(ids,rank=rank,world_size=4,fraction=.5,max_extra_per_case=2)
            s.update(records,2); s.set_epoch(2)
            schedules.append(list(s))
            self.assertEqual(s.summary['extra_cases'],4)
            self.assertTrue(all(v<=2 for v in s.summary['repeated_case_ids'].values()))
        flat=[x for group in schedules for x in group]
        self.assertEqual(set(flat),set(range(12)))
        self.assertTrue(all(len(x)==4 for x in schedules))
        restored=MissingNoduleSampler(ids,rank=3,world_size=4,fraction=.5,max_extra_per_case=2)
        restored.load_state_dict(copy.deepcopy(s.state_dict())); restored.set_epoch(2)
        self.assertEqual(list(restored),list(s))
        with self.assertRaises(ValueError):
            restored.update({'test':{'1':True}},4)
        restored.update({c:{'1':False,'2':False} for c in ids},4)
        restored.set_epoch(4); self.assertEqual(restored.summary['extra_cases'],0)

    def test_before_mining_sampler_matches_ordinary_v4(self):
        from torch.utils.data import DistributedSampler
        ids=[f'train{i}' for i in range(10)]
        for epoch in (0,1):
            for rank in range(4):
                ordinary=DistributedSampler(ids,num_replicas=4,rank=rank,seed=42)
                hard=MissingNoduleSampler(ids,rank=rank,world_size=4,seed=42)
                ordinary.set_epoch(epoch); hard.set_epoch(epoch)
                self.assertEqual(list(ordinary),list(hard))

    def test_mining_requires_overlap_and_one_to_one_matching(self):
        out=SimpleNamespace(fine_mask_logits=torch.full((1,2,2,2),10.),
            roi_valid=torch.ones(1,2,2,2,dtype=torch.bool),crop_origins=torch.zeros(1,3,dtype=torch.long))
        batch=dict(nodule_ids=torch.tensor([10,11]),target_mask_crops=[torch.ones(2,2,2)]*2,
                   target_mask_origins=torch.zeros(2,3,dtype=torch.long))
        result=missed_nodules(out,batch)
        self.assertEqual(sum(result.values()),1)
        out.fine_mask_logits.fill_(-10)
        self.assertTrue(all(missed_nodules(out,batch).values()))

    def test_epoch_monitor_reconstructs_weighted_total_and_chart_order(self):
        from back_prop.model_v4.loss import WholeCTCriterionV4
        w=WholeCTCriterionV4().loss_weights
        record={k:float(i+1) for i,k in enumerate(w)}
        record.update(epoch=1,global_step=20,lr={'main':1e-4},temperature=1.,diagnostic_scale=.3)
        record['detr_loss']=sum(record[k]*w[k] for k in COMPONENTS[:4])
        record['finetuner_loss']=sum(record[k]*w[k] for k in COMPONENTS[4:])
        record['total']=sum(record[k]*v*(.3 if k in ('nodule','risk') else 1) for k,v in w.items())
        metrics=epoch_metrics(record,w)
        self.assertAlmostEqual(metrics['00_training/weighted_sum'],record['total'])
        ordered=[k.split('_',2)[-1] for k in metrics if k.startswith('02_objectives/')]
        self.assertEqual([k for k in metrics if k.startswith('02_objectives/')],
                         [f'02_objectives/{i:02d}_{k}' for i,k in enumerate(GROUPS,1)])
        self.assertFalse(any('ordinal' in k or '/object' in k or 'radiomics' in k for k in metrics))


if __name__=='__main__': unittest.main()
