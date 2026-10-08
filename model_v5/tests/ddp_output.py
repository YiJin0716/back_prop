"""Regression for rank-dependent loss gating and V5's dataclass DDP output.

Run with torchrun --standalone --nproc_per_node=4 -m
back_prop.model_v5.tests.ddp_output. Uses the actual V5 forward and criterion
with small encoders; compares reduced gradients to independent serial forwards.
"""
import copy
from datetime import timedelta
import json

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from back_prop.model_v5.loss import WholeCTCriterionV5
from back_prop.model_v5.tests.test_v5 import tiny


def main():
    torch.set_num_threads(1)
    dist.init_process_group('gloo', timeout=timedelta(seconds=60))
    rank, world = dist.get_rank(), dist.get_world_size()
    bare, batch = tiny()
    bare.use_checkpoint = True
    bare.semantics.use_checkpoint = True
    reference = copy.deepcopy(bare)
    batch['image'] = batch['image'] + .01 * rank
    drop_semantics = False

    def gate(_module, _inputs, output):
        # A predicted ROI can miss every GT, disabling semantic/nodule/scan
        # losses while coarse geometry and missing-nodule losses remain active.
        if drop_semantics:
            output.fine_supervision_valid.zero_()
            output.scan_path_valid = False
        return output

    bare.register_forward_hook(gate)
    reference.register_forward_hook(gate)
    model = DistributedDataParallel(bare, find_unused_parameters=True,
                                   broadcast_buffers=False)
    criterion = WholeCTCriterionV5()
    optimizer = torch.optim.SGD(bare.parameters(), lr=1e-4)
    serial_optimizer = torch.optim.SGD(reference.parameters(), lr=1e-4)
    modes = ['all_valid', 'one_rank_invalid', 'all_invalid',
             'other_ranks_invalid', 'one_rank_empty', 'all_empty', 'anchor_all_empty', 'all_valid_again']
    for step, mode in enumerate(modes):
        drop_semantics = (mode == 'all_invalid' or
                          (mode == 'one_rank_invalid' and rank == world-1) or
                          (mode == 'other_ranks_invalid' and rank != world-1))
        optimizer.zero_grad(set_to_none=True)
        serial_optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(700 + step * world + rank)
        rng = torch.get_rng_state()
        no_candidates = mode in ('all_empty','anchor_all_empty') or (mode == 'one_rank_empty' and rank == world - 1)
        prediction_only = mode in ('one_rank_empty', 'all_empty','anchor_all_empty')
        teacher = 0. if prediction_only else 1.
        threshold = 1. if no_candidates else 0.
        anchor_roi = torch.full((2,2,8,8,8),.1+.01*rank) if mode=='anchor_all_empty' else None
        anchor_hist = batch['semantic_histograms'][:1].expand(2,-1,-1)
        output = model(batch['image'], batch=batch, epoch=17,
                       teacher_probability=teacher, object_threshold=threshold, update_bank=False,
                       semantic_anchor_roi=anchor_roi)
        loss = criterion(output, batch,anchor=(output.anchor_probabilities,anchor_hist) if anchor_roi is not None else None)
        loss.total.backward()
        if drop_semantics:
            assert loss.counts['semantic'] == loss.counts['nodule'] == loss.counts['risk'] == 0
        assert loss.mean('missing_nodule') > 0

        torch.set_rng_state(rng)
        expected_output = reference(batch['image'], batch=batch, epoch=17,
                                    teacher_probability=teacher, object_threshold=threshold, update_bank=False,
                                    semantic_anchor_roi=anchor_roi)
        # Independent oracle: pool raw numerators with explicit global counts.
        expected_loss = criterion(expected_output, batch, synchronize=False,
            anchor=(expected_output.anchor_probabilities,anchor_hist) if anchor_roi is not None else None)
        counts = torch.tensor([expected_loss.counts[k] for k in criterion.loss_weights],dtype=torch.float64)
        dist.all_reduce(counts)
        reference_objective = sum(expected_loss.numerators[k] * weight / counts[i].clamp_min(1)
                                  for i,(k,weight) in enumerate(criterion.loss_weights.items()))
        reference_objective.backward()
        parameters = list(reference.named_parameters())
        present = torch.tensor([int(p.grad is not None) for _, p in parameters])
        gradient = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
                              for _, p in parameters])
        dist.all_reduce(present)
        dist.all_reduce(gradient)
        offset = 0
        for ((name, actual), (_, expected)), used in zip(
                zip(bare.named_parameters(), parameters), present):
            average = gradient[offset:offset+expected.numel()].reshape_as(expected)
            offset += expected.numel()
            if used:
                assert actual.grad is not None, name
                torch.testing.assert_close(actual.grad, average, rtol=1e-4, atol=1e-6,
                                           msg=lambda m: f'{mode}: {name}: {m}')
                expected.grad = average.clone()
            else:
                assert actual.grad is None, (mode, name)
                expected.grad = None
        optimizer.step()
        serial_optimizer.step()
        for actual, expected in zip(bare.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
        if rank == 0:
            print(json.dumps(dict(event='ddp_output_verified', mode=mode,
                                  world_size=world, gradients_match_serial=True)), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
