"""Regression for rank-dependent loss gating and V4's dataclass DDP output.

Run with torchrun --standalone --nproc_per_node=4 -m
back_prop.model_v4.tests.ddp_output. Uses the actual V4 forward and criterion
with small encoders; compares reduced gradients to independent serial forwards.
"""
import copy
from datetime import timedelta
import json

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from back_prop.model_v4.loss import WholeCTCriterionV4
from back_prop.model_v4.tests.test_v4 import V4Tests


def main():
    torch.set_num_threads(1)
    dist.init_process_group('gloo', timeout=timedelta(seconds=60))
    rank, world = dist.get_rank(), dist.get_world_size()
    bare, batch = V4Tests().tiny()
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
    criterion = WholeCTCriterionV4()
    optimizer = torch.optim.SGD(bare.parameters(), lr=1e-4)
    serial_optimizer = torch.optim.SGD(reference.parameters(), lr=1e-4)
    modes = ['all_valid', 'one_rank_invalid', 'all_invalid',
             'other_ranks_invalid', 'all_valid_again']
    for step, mode in enumerate(modes):
        drop_semantics = (mode == 'all_invalid' or
                          (mode == 'one_rank_invalid' and rank == world-1) or
                          (mode == 'other_ranks_invalid' and rank != world-1))
        optimizer.zero_grad(set_to_none=True)
        serial_optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(700 + step * world + rank)
        rng = torch.get_rng_state()
        output = model(batch['image'], batch=batch, epoch=17,
                       teacher_probability=1., update_bank=False)
        loss = criterion(output, batch)
        loss.total.backward()
        if drop_semantics:
            assert float(loss.semantic) == float(loss.nodule) == float(loss.risk) == 0.
        assert float(loss.missing_nodule) > 0

        torch.set_rng_state(rng)
        expected_output = reference(batch['image'], batch=batch, epoch=17,
                                    teacher_probability=1., update_bank=False)
        expected_loss = criterion(expected_output, batch)
        torch.testing.assert_close(loss.total, expected_loss.total, rtol=0, atol=0)
        expected_loss.total.backward()
        parameters = list(reference.named_parameters())
        present = torch.tensor([int(p.grad is not None) for _, p in parameters])
        gradient = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
                              for _, p in parameters])
        dist.all_reduce(present)
        dist.all_reduce(gradient)
        gradient.div_(world)
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
