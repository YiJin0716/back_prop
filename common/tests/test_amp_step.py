import math

import torch

from back_prop.common.train_utils import finish_scaled_step


def test_finite_large_gradients_are_clipped_without_norm_overflow():
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    parameter.grad = torch.full_like(parameter, 1e30)
    norm, _, _ = finish_scaled_step(optimizer, scaler, [parameter])
    assert math.isfinite(norm)
    torch.testing.assert_close(parameter.grad.norm(), torch.tensor(5.0))
    assert torch.isfinite(parameter).all()


def test_cuda_amp_overflow_skips_update_and_recovers():
    if not torch.cuda.is_available():
        return
    parameter = torch.nn.Parameter(torch.ones(1, device="cuda"))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scaler = torch.cuda.amp.GradScaler(init_scale=4.0)
    for expected_scale in (2.0, 1.0):
        optimizer.zero_grad(set_to_none=True)
        # Finite forward, but the scaled FP16 derivative exceeds 65504.
        loss = (parameter.half() * 40000).float().sum()
        assert torch.isfinite(loss)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm, _, scale = finish_scaled_step(optimizer, scaler, [parameter])
        assert not math.isfinite(norm)
        assert scale == expected_scale
        torch.testing.assert_close(parameter, torch.ones_like(parameter))
    optimizer.zero_grad(set_to_none=True)
    scaler.scale((parameter.half() * 40000).float().sum()).backward()
    scaler.unscale_(optimizer)
    norm, _, scale = finish_scaled_step(optimizer, scaler, [parameter])
    assert math.isfinite(norm)
    assert scale == 1.0
    torch.testing.assert_close(parameter, torch.full_like(parameter, 0.5))
