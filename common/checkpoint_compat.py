"""Preserve CUDA Philox seed/offset when moving between Torch 2.0 and 2.7.

Upstream CUDAGeneratorImpl::get_state layouts:
https://github.com/pytorch/pytorch/blob/v2.0.1/aten/src/ATen/cuda/CUDAGeneratorImpl.cpp
https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/cuda/CUDAGeneratorImpl.cpp
The old format prepends 800 bytes of unused 0xff padding to the same 16-byte
seed/offset payload. No reseeding or offset reset is performed here.
"""
import torch


def adapt_cuda_rng_state(state, expected_bytes):
    if state.dtype != torch.uint8 or state.ndim != 1 or state.device.type != 'cpu':
        raise ValueError('CUDA RNG checkpoint must be a one-dimensional CPU byte tensor')
    if state.numel() == expected_bytes:
        return state
    if state.numel() == 816 and expected_bytes == 16:
        if not bool((state[:800] == 255).all()):
            raise ValueError('Unrecognized legacy CUDA RNG padding')
        return state[800:].clone()
    if state.numel() == 16 and expected_bytes == 816:
        return torch.cat((torch.full((800,), 255, dtype=torch.uint8), state))
    raise ValueError(f'Unsupported CUDA RNG format: {state.numel()} -> {expected_bytes}')
