"""Bounded CUDA/NCCL connectivity check for a new GPU node."""
import datetime
import os
import time
import torch
import torch.distributed as dist


def main():
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    t = time.perf_counter()
    print(f'rank={rank} cuda={torch.cuda.get_device_name()} starting', flush=True)
    x = torch.randn(32, 32, device='cuda', requires_grad=True)
    (x @ x.T).sum().backward()
    torch.cuda.synchronize()
    print(f'rank={rank} CUDA forward/backward passed in {time.perf_counter()-t:.2f}s', flush=True)
    dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=90))
    value = torch.tensor([float(rank)], device='cuda')
    dist.all_reduce(value)
    torch.cuda.synchronize()
    assert value.item() == sum(range(dist.get_world_size()))
    # A scalar uses a different NCCL kernel from DDP's parameter broadcasts
    # and gradient buckets. Exercise both floating-point formats and sizes.
    for dtype in (torch.float32, torch.bfloat16):
        for count in (4096, 8 * 1024 * 1024):
            bucket = torch.full((count,), float(rank), device='cuda', dtype=dtype)
            dist.all_reduce(bucket)
            assert bool((bucket == sum(range(dist.get_world_size()))).all())
            bucket.fill_(float(rank))
            dist.broadcast(bucket, src=0)
            assert bool((bucket == 0).all())
            del bucket
    model = torch.nn.parallel.DistributedDataParallel(
        torch.nn.Conv2d(2048, 640, kernel_size=1).cuda(), device_ids=[rank])
    model(torch.randn(1, 2048, 8, 8, device='cuda')).square().mean().backward()
    assert all(bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    torch.cuda.synchronize()
    print(f'rank={rank} NCCL passed in {time.perf_counter()-t:.2f}s', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
