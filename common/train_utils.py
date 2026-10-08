"""Shared DDP, AMP and checkpoint helpers; no training entry-point imports."""
import os
from pathlib import Path
import numpy as np
import torch
from torch import Tensor, nn
import torch.distributed as dist


def distributed_setup() -> tuple[int, int, int]:
    if not torch.cuda.is_available():
        raise RuntimeError("Whole-CT training requires CUDA")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world



def reduce(values: Tensor, world: int) -> Tensor:
    if world > 1:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return values



def gradient_norm(parameters: list[nn.Parameter]) -> float:
    gradients = [
        parameter.grad.detach().float().norm()
        for parameter in parameters
        if parameter.grad is not None
    ]
    return float(torch.stack(gradients).norm()) if gradients else 0.0



def finish_scaled_step(optimizer, scaler, parameters, max_norm: float = 5.0):
    """Clip finite, already-unscaled gradients; let AMP skip overflowed steps.

    ``scaler.unscale_(optimizer)`` must have run first so GradScaler has
    recorded non-finite entries. A finite loss does not rule out FP16 backward
    overflow. Raising before scaler.update() prevents its normal recovery.
    """
    gradients = [p.grad for p in parameters if p.grad is not None]
    # Accumulate in double so the norm itself cannot overflow for finite FP32
    # gradients and be mistaken for a mixed-precision backward overflow.
    total_norm = float(torch.stack([
        g.detach().norm(dtype=torch.float64) for g in gradients
    ]).norm()) if gradients else 0.0
    if np.isfinite(total_norm):
        coefficient = min(1.0, max_norm / (total_norm + 1e-6))
        with torch.no_grad():
            for gradient in gradients:
                gradient.mul_(coefficient)
    elif not scaler.is_enabled():
        raise FloatingPointError("Non-finite gradients with AMP scaling disabled")
    scale_before = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    return total_norm, scale_before, float(scaler.get_scale())



def atomic_save(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)

