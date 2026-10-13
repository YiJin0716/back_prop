"""Branch isolation and official-mask input contracts for parallel warmup."""
import unittest
from unittest.mock import patch
import torch
from back_prop.model_v5.tests.test_v5 import tiny
from back_prop.model_v5.parallel_warmup import (
    DetectorBranch, coarse_output, make_refiner_row, refiner_forward, RefinerROIs,
)
from back_prop.model_v5.loss import WholeCTCriterionV5


class ParallelWarmupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_detr_does_not_call_or_update_refiner_semantics_or_context(self):
        model,batch=tiny()
        module=DetectorBranch(model)
        with patch.object(model.refiner,'forward',side_effect=AssertionError('refiner called')), \
             patch.object(model.semantics,'forward',side_effect=AssertionError('CNN called')), \
             patch.object(model.context_compressor,'forward',side_effect=AssertionError('context called')):
            out=coarse_output(module(batch['image']),batch)
            loss=WholeCTCriterionV5()(out,batch,semantic_scale=0.,diagnostic_scale=0.,synchronize=False)
            loss.total.backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.detr.parameters() if p.grad is not None),0)
        for child in (model.refiner,model.semantics,model.context_compressor):
            self.assertTrue(all(p.grad is None for p in child.parameters()))
        self.assertEqual(loss.counts['box_l1'],len(batch['nodule_ids']))
        self.assertEqual(loss.counts['fine_dice'],0)

    def test_refiner_uses_only_corrupted_gt_and_ct(self):
        model,batch=tiny()
        row=make_refiner_row(model,batch,0,0)
        row.pop('source');row.pop('key');row.pop('crop_shape')
        inputs=[]
        hook=model.refiner.register_forward_pre_hook(lambda m,a:inputs.append(a))
        with patch.object(model.detr,'forward',side_effect=AssertionError('DETR called')), \
             patch.object(model.segmenter,'forward',side_effect=AssertionError('VISTA called')), \
             patch.object(model.semantics,'forward',side_effect=AssertionError('CNN called')):
            loss,metrics=refiner_forward(model.refiner,row,model.device)
            loss.backward()
        hook.remove()
        value,query=inputs[0]
        self.assertEqual(float(query.abs().sum()),0.)
        self.assertEqual(float(value[:,1].abs().sum()),0.)
        self.assertEqual(float(value[:,4:].abs().sum()),0.)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(model.refiner.residual_head.weight.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None for p in model.detr.parameters()))
        self.assertTrue(all(p.grad is None for p in model.semantics.parameters()))
        self.assertEqual(int(row['mask'].sum()),int(batch['target_mask_crops'][0].sum()))

    def test_resume_rejects_old_policy_and_changed_inputs_but_allows_more_epochs(self):
        import hashlib
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from copy import deepcopy
        from back_prop.model_v5.parallel_warmup import validate_parallel_resume
        from back_prop.model_v5.preparation import SCHEMA
        with tempfile.TemporaryDirectory() as directory:
            manifest=Path(directory)/'manifest.json';manifest.write_text('{}')
            args=SimpleNamespace(manifest=manifest,geometry_max_epochs=40,teacher_jitter=8,
                                 preparation_smoke=False,lr_main=.0001)
            config={k:str(v) for k,v in vars(args).items()};config['geometry_max_epochs']='30'
            saved=dict(preparation_schema=SCHEMA,world_size=4,config=config,
                       manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
            validate_parallel_resume(args,saved,4)
            for key,value in [('teacher_jitter',0),('preparation_smoke',True),('lr_main',.1)]:
                changed=deepcopy(args);setattr(changed,key,value)
                with self.assertRaisesRegex(ValueError,key):validate_parallel_resume(changed,saved,4)
            with self.assertRaises(ValueError):validate_parallel_resume(args,saved,3)
            saved['preparation_schema']='v5_localization_repair_v2'
            with self.assertRaises(ValueError):validate_parallel_resume(args,saved,4)

    def test_semantic_does_not_call_or_update_other_models(self):
        from back_prop.model_v5.warmup import official_row
        model,batch=tiny()
        with patch.object(model.detr,'forward',side_effect=AssertionError('DETR called')), \
             patch.object(model.refiner,'forward',side_effect=AssertionError('refiner called')), \
             patch.object(model.segmenter,'forward',side_effect=AssertionError('VISTA called')):
            row=official_row(model,batch,0)
            p,_=model.semantics(row['roi'][None],prepared=True)
            loss=-(row['histogram'][None]*p.log()).sum(-1).mean();loss.backward()
        self.assertTrue(all(p.grad is None for p in model.detr.parameters()))
        self.assertTrue(all(p.grad is None for p in model.refiner.parameters()))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.semantics.parameters() if p.grad is not None),0)

    def test_completed_branch_restores_weights_without_optimizer_or_rng(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from back_prop.model_v5.parallel_warmup import run_branch
        from back_prop.model_v5.preparation import SCHEMA
        model,batch=tiny()
        expected={k:v.detach().clone() for k,v in DetectorBranch(model).state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            torch.save(dict(schema=SCHEMA,model=expected,epoch=15,ready=True,
                audit={'passed':True}),root/'detr_warmup.pt')
            with torch.no_grad():
                for p in model.detr.parameters():p.zero_()
            args=SimpleNamespace(geometry_min_epochs=15,geometry_max_epochs=30,
                lr_main=1e-4,weight_decay=1e-5,amp_dtype='bfloat16',amp_init_scale=4.,warmup_ready_epochs=3)
            run_branch('detr',model,[batch],root,root,args,0,4,None,root,None,True)
            for key,value in DetectorBranch(model).state_dict().items():
                torch.testing.assert_close(value,expected[key])
            self.assertFalse(model.detr.training)
            self.assertTrue(all(not p.requires_grad for p in model.detr.parameters()))
            self.assertTrue((root/'detr_ready.json').exists())

    def test_training_inherits_detector_options_from_resume_and_keeps_legacy_defaults(self):
        import tempfile
        from pathlib import Path
        from back_prop.model_v5.train import parse_args
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            checkpoint=root/'preparation.pt'
            options=['--output-dir',str(root/'output')]
            fresh=parse_args(options)
            self.assertEqual((fresh.detector_kind,fresh.object_background_weight),('legacy',1.))
            torch.save({'config':{}},checkpoint)
            old=parse_args(options+['--prepare-resume',str(checkpoint)])
            self.assertEqual((old.detector_kind,old.object_background_weight),('legacy',1.))
            # Preparation pointers store strings; joint checkpoints store values.
            for stringify in (True,False):
                cfg=dict(detector_kind='native_seeded',object_background_weight=.1,
                         native_proposal_threshold=.1,detector_nms_iou=.5,
                         native_confidence_residual_bound=2.)
                torch.save({'config':{k:str(v) if stringify else v for k,v in cfg.items()}},checkpoint)
                for flag in ('--prepare-resume','--resume','--restart-from'):
                    restored=parse_args(options+[flag,str(checkpoint)])
                    for key,value in cfg.items():self.assertEqual(getattr(restored,key),value)
                explicit=parse_args(options+['--prepare-resume',str(checkpoint),
                    '--detector-kind','mask_guided','--object-background-weight','.2'])
                self.assertEqual((explicit.detector_kind,explicit.object_background_weight),('mask_guided',.2))


if __name__=='__main__':unittest.main()
