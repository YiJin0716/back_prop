"""FasterRisk stages 1/2 only, detached refits and frozen differentiable heads."""
from __future__ import annotations

import math
import time
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F


def fit_continuous_pool(x, y, *, sparsity=5, bound=5.0, beam=10,
                        pool_size=30, gap=0.05, attempts=50):
    from fasterrisk.sparseBeamSearch import sparseLogRegModel
    from fasterrisk.sparseDiversePool import sparseDiversePoolLogRegModel
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Non-finite FasterRisk fitting data")
    if set(np.unique(y)) != {0.0, 1.0}:
        raise ValueError("FasterRisk fitting requires both binary classes")
    mean, std = x.mean(0), x.std(0)
    varying = std > np.maximum(1e-4, np.abs(mean) * 1e-6)
    scale = np.where(varying, std, 1.0)
    normalized = np.clip((x - mean) / scale, -10.0, 10.0)
    normalized[:, ~varying] = 0.0
    signed_y = 2.0 * y - 1.0
    # An intercept-only pool is the valid degenerate case, not a random head.
    if not varying.any():
        weights = np.zeros((1, x.shape[1]))
        intercepts = np.array([np.log(y.mean() / (1.0 - y.mean()))])
    else:
        k = min(sparsity, int(varying.sum()))
        first = sparseLogRegModel(normalized, signed_y, original_lb=-bound, original_ub=bound)
        first.get_sparse_sol_via_OMP(k=k, parent_size=beam, child_size=beam)
        b0, coefficients, exp_yxb = first.get_beta0_betas_ExpyXB()
        second = sparseDiversePoolLogRegModel(normalized, signed_y, original_lb=-bound, original_ub=bound)
        second.warm_start_from_beta0_betas_ExpyXB(float(b0), coefficients.copy(), exp_yxb.copy())
        # Returned in the input feature space (our standardized space), not
        # FasterRisk's internal centered/unit-L2-column space.
        intercepts, weights = second.get_sparseDiversePool(
            gap_tolerance=gap, select_top_m=pool_size, maxAttempts=attempts)
    margins = normalized @ weights.T + intercepts
    losses = np.logaddexp(0.0, -signed_y[:, None] * margins).mean(0)
    keep = np.isfinite(losses) & (losses <= losses.min() * (1.0 + gap) + 1e-10)
    keep &= (np.abs(weights) <= bound + 1e-6).all(1)
    keep &= (np.abs(weights) > 1e-9).sum(1) <= sparsity
    selected = np.flatnonzero(keep)[np.argsort(losses[keep])]
    if not len(selected):
        raise RuntimeError("FasterRisk produced no finite feasible continuous candidate")
    weights, intercepts, losses = weights[selected], intercepts[selected], losses[selected]
    return dict(weights=weights, intercepts=intercepts, mean=mean, scale=scale,
                varying=varying, losses=losses, samples=len(y), positives=int(y.sum()),
                distinct_supports=len({tuple(np.flatnonzero(abs(w) > 1e-9)) for w in weights}))


class ContinuousRashomonBank(nn.Module):
    def __init__(self, feature_names, *, sparsity=5, bound=5.0, beam=10, pool_size=30,
                 gap=0.05, attempts=50, fraction=0.3, min_samples=32, refresh_every=1,
                 mode='sample'):
        super().__init__()
        if not (0 < fraction <= 1 and 0 < sparsity <= len(feature_names)):
            raise ValueError("Invalid sampling fraction or sparsity")
        if min(pool_size, beam, attempts) < 1 or refresh_every < 0 or bound <= 0 or gap < 0 or min_samples < 2:
            raise ValueError("Invalid FasterRisk configuration")
        if mode not in ('sample', 'single', 'all', 'best'):
            raise ValueError('Unknown Rashomon sampling mode')
        self.mode = mode
        self.feature_names = tuple(feature_names)
        self.config = dict(sparsity=sparsity, bound=bound, beam=beam, pool_size=pool_size,
                           gap=gap, attempts=attempts)
        self.fraction, self.min_samples, self.refresh_every = fraction, min_samples, refresh_every
        d = len(feature_names)
        self.register_buffer("weights", torch.zeros(pool_size, d))
        self.register_buffer("intercepts", torch.zeros(pool_size))
        self.register_buffer("mean", torch.zeros(d))
        self.register_buffer("scale", torch.ones(d))
        self.register_buffer("varying", torch.ones(d, dtype=torch.bool))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))
        self.register_buffer("fit_count", torch.zeros((), dtype=torch.long))
        self.memory = {}
        self.allowed_cases = set()
        self.updates = 0
        self.last_fit = {}

    def get_extra_state(self):
        return dict(feature_names=self.feature_names, memory=self.memory, allowed_cases=self.allowed_cases,
                    updates=self.updates, last_fit=self.last_fit, config=self.config,
                    fraction=self.fraction, min_samples=self.min_samples, refresh_every=self.refresh_every,
                    mode=self.mode)

    def set_extra_state(self, state):
        if tuple(state['feature_names']) != self.feature_names or state['config'] != self.config:
            raise ValueError("Rashomon checkpoint schema/configuration mismatch")
        for key in ('fraction', 'min_samples', 'refresh_every', 'mode'):
            if state[key] != getattr(self, key):
                raise ValueError(f"Rashomon checkpoint {key} mismatch")
        self.memory, self.allowed_cases = state['memory'], set(state['allowed_cases'])
        self.updates, self.last_fit = state['updates'], state['last_fit']

    @torch.no_grad()
    def observe_and_refit(self, features, labels, keys, *, refit=True):
        """Update one row per training nodule, then optionally refit on rank 0.

        All DDP ranks call this, even with no valid local nodules. New model
        buffers are installed before logits are computed; they never change
        between the current forward and backward.
        """
        if not self.training:
            raise RuntimeError("Evaluation must not update the training feature cache")
        rows = [(key, value, float(label)) for key, value, label in
                zip(keys, features.detach().double().cpu().numpy(), labels.detach().cpu().numpy())]
        distributed = dist.is_available() and dist.is_initialized()
        gathered = [None] * dist.get_world_size() if distributed else [rows]
        if distributed:
            dist.all_gather_object(gathered, rows)
        for group in gathered:
            for key, value, label in group:
                if key[0] not in self.allowed_cases:
                    raise ValueError(f"Non-training case in FasterRisk cache: {key[0]}")
                if not np.isfinite(value).all() or label not in (0., 1.):
                    raise ValueError("Invalid training feature/label")
                self.memory[key] = (value.copy(), label)
        self.updates += 1
        if not refit or len(self.memory) < self.min_samples:
            return
        if int(self.count) and (self.refresh_every == 0 or (self.updates - 1) % self.refresh_every):
            return
        ordered = [self.memory[key] for key in sorted(self.memory)]
        x = np.stack([row[0] for row in ordered])
        y = np.array([row[1] for row in ordered])
        if len(np.unique(y)) < 2:
            return
        package = [None]
        if not distributed or dist.get_rank() == 0:
            started = time.time()
            try:
                result = fit_continuous_pool(x, y, **self.config)
                result['seconds'] = time.time() - started
                package[0] = result
            except Exception as exc:
                # Propagate failures so other ranks do not wait indefinitely.
                package[0] = {'error': f'{type(exc).__name__}: {exc}'}
        if distributed:
            dist.broadcast_object_list(package, src=0)
        result = package[0]
        if 'error' in result:
            raise RuntimeError(result['error'])
        k = len(result['intercepts'])
        self.weights.zero_(); self.intercepts.zero_()
        self.weights[:k].copy_(torch.as_tensor(result['weights'], device=self.weights.device))
        self.intercepts[:k].copy_(torch.as_tensor(result['intercepts'], device=self.weights.device))
        for name in ('mean', 'scale', 'varying'):
            getattr(self, name).copy_(torch.as_tensor(result[name], device=self.weights.device))
        self.count.fill_(k); self.fit_count.add_(1)
        self.last_fit = dict(samples=result['samples'], positives=result['positives'], size=k,
                             distinct_supports=result['distinct_supports'], seconds=result['seconds'],
                             best_loss=float(result['losses'].min()), worst_loss=float(result['losses'].max()))

    def forward(self, features, *, all_models=False):
        k = int(self.count)
        if not k:
            raise RuntimeError("Rashomon bank is not fitted")
        with torch.autocast(device_type=features.device.type, enabled=False):
            normalized = ((features.float() - self.mean) / self.scale).clamp(-10., 10.)
            normalized = normalized * self.varying
            if self.mode == 'best' and not all_models:
                indices = torch.zeros(1, device=features.device, dtype=torch.long)
            elif self.training and not all_models and self.mode in ('sample', 'single'):
                n = 1 if self.mode == 'single' else max(1, math.ceil(k * self.fraction))
                indices = torch.randperm(k, device=features.device)[:n]
            else:
                indices = torch.arange(k, device=features.device)
            # Frozen weights, live input gradients.
            return F.linear(normalized, self.weights[indices], self.intercepts[indices]), indices


def scan_logits_per_model(nodule_logits, object_logits):
    """Stable object-weighted noisy-OR, independently for every model."""
    with torch.autocast(device_type=nodule_logits.device.type, enabled=False):
        log_event = F.logsigmoid(nodule_logits.float()) + F.logsigmoid(object_logits.float())[:, None]
        event = log_event.exp().clamp(max=1.0 - 1e-6)
        log_survival = torch.log1p(-event).sum(0)
        probability = (-torch.expm1(log_survival)).clamp(1e-6, 1.0 - 1e-6)
        return torch.logit(probability)
