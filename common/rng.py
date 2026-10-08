"""Preserve training RNG streams around logging and evaluation."""
from contextlib import contextmanager
import random
import numpy as np
import torch


@contextmanager
def preserve_rng():
    py, numpy, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    # Each DDP rank owns its current device; do not initialize other GPUs.
    cuda = torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(py)
        np.random.set_state(numpy)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state(cuda)

