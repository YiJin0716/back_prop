"""Stagewise radiomics logistic regression, then offset FasterRisk on semantics.

FasterRisk's exponential margins include a fixed per-row radiomics logit.
Its beam search, coordinate updates, and diverse-pool comparisons consequently
optimize BCE(offset + semantic score, y), never BCE on a signed residual label.
The initial negative gradient for the semantic score is exactly y - p_radiomics.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import torch
import torch.distributed as dist
import torch.nn.functional as F

from back_prop.common.features import RADIOMICS_NAMES, SEMANTIC_NAMES
from back_prop.common.rashomon import ContinuousRashomonBank


def standardize(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Expected a finite nonempty feature matrix')
    mean, std = x.mean(0), x.std(0)
    varying = std > np.maximum(1e-4, abs(mean) * 1e-6)
    scale = np.where(varying, std, 1.)
    normalized = np.clip((x - mean) / scale, -10., 10.) * varying
    return normalized, dict(mean=mean, scale=scale, varying=varying)


def fit_radiomics(x, y, *, l2=1e-3):
    x, normalization = standardize(x)
    y = np.asarray(y, dtype=np.float64)
    if y.shape != (len(x),) or set(np.unique(y)) != {0., 1.} or l2 <= 0:
        raise ValueError('Radiomics logistic fit requires both binary classes and positive L2')
    initial = np.zeros(x.shape[1] + 1)
    initial[-1] = np.log(y.mean() / (1-y.mean()))

    def objective(coef):
        z = x @ coef[:-1] + coef[-1]
        error = expit(z) - y
        loss = np.mean(np.logaddexp(0., z) - y*z) + .5*l2*np.dot(coef[:-1], coef[:-1])
        grad = np.r_[x.T @ error / len(y) + l2*coef[:-1], error.mean()]
        return loss, grad

    result = minimize(objective, initial, jac=True, method='L-BFGS-B',
                      options=dict(maxiter=500, ftol=1e-12, gtol=1e-7))
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f'Radiomics logistic fit failed: {result.message}')
    z = x @ result.x[:-1] + result.x[-1]
    return dict(**normalization, weights=result.x[:-1], intercept=result.x[-1],
                loss=float(np.mean(np.logaddexp(0., z)-y*z)), samples=len(y))


def fit_offset_pool(x, y, offset, *, sparsity=5, bound=5., beam=10,
                    pool_size=30, gap=.05, attempts=50):
    """Use FasterRisk 0.1.10 continuous stages with fixed baseline margins.

    Multiplicative exponential-margin updates preserve the offset throughout
    both searches. The numerical invariant below checks this against the
    recovered coefficients. No integerization or extra baseline feature is used.
    """
    from fasterrisk.sparseBeamSearch import sparseLogRegModel
    from fasterrisk.sparseDiversePool import sparseDiversePoolLogRegModel
    x, normalization = standardize(x)
    y, offset = np.asarray(y, dtype=np.float64), np.asarray(offset, dtype=np.float64)
    if (y.shape != (len(x),) or offset.shape != y.shape or not np.isfinite(offset).all()
            or set(np.unique(y)) != {0., 1.}):
        raise ValueError('Offset FasterRisk requires finite offsets and both binary classes')
    signed = 2*y-1
    active = np.flatnonzero(normalization['varying'])
    # Always consider no residual correction; the semantic stage need not help.
    weights, intercepts = np.zeros((1, x.shape[1])), np.zeros(1)
    if len(active):
        design = x[:, active]
        first = sparseLogRegModel(design, signed, original_lb=-bound, original_ub=bound)
        first.ExpyXB = np.exp(signed * offset)
        first.get_sparse_sol_via_OMP(k=min(sparsity, len(active)),
                                    parent_size=beam, child_size=beam)
        b0, beta, margins = first.get_beta0_betas_ExpyXB()
        recovered_b, recovered_w = first.get_original_beta0_betas()
        np.testing.assert_allclose(np.log(margins),
                                   signed*(offset + design @ recovered_w + recovered_b),
                                   atol=1e-7, rtol=1e-7,
                                   err_msg='FasterRisk discarded the radiomics offset')
        second = sparseDiversePoolLogRegModel(design, signed, original_lb=-bound, original_ub=bound)
        second.warm_start_from_beta0_betas_ExpyXB(float(b0), beta.copy(), margins.copy())
        intercepts_fit, weights_fit = second.get_sparseDiversePool(
            gap_tolerance=gap, select_top_m=pool_size, maxAttempts=attempts)
        expanded = np.zeros((len(intercepts_fit), x.shape[1]))
        expanded[:, active] = weights_fit
        weights = np.concatenate((weights, expanded))
        intercepts = np.r_[intercepts, intercepts_fit]
    z = offset[:, None] + x @ weights.T + intercepts
    losses = np.logaddexp(0., -signed[:, None]*z).mean(0)
    feasible = (np.isfinite(losses) & np.isfinite(weights).all(1)
                & (abs(weights) <= bound+1e-6).all(1)
                & ((abs(weights) > 1e-9).sum(1) <= sparsity))
    best = losses[feasible].min()
    selected = np.flatnonzero(feasible & (losses <= best*(1+gap)+1e-10))
    selected = selected[np.argsort(losses[selected])][:pool_size]
    weights, intercepts, losses = weights[selected], intercepts[selected], losses[selected]
    return dict(**normalization, weights=weights, intercepts=intercepts, losses=losses,
                samples=len(y), positives=int(y.sum()),
                baseline_loss=float(np.logaddexp(0., -signed*offset).mean()),
                residual_rmse=float(np.sqrt(np.mean((y-expit(offset))**2))),
                varying_semantics=int(len(active)),
                distinct_supports=len({tuple(np.flatnonzero(abs(w)>1e-9)) for w in weights}))


class ResidualRiskBank(ContinuousRashomonBank):
    """Two sequential fits on training-only caches; both heads are frozen buffers."""
    def __init__(self, *, baseline_l2=1e-3, **kwargs):
        super().__init__(SEMANTIC_NAMES, **kwargs)
        if baseline_l2 <= 0:
            raise ValueError('baseline_l2 must be positive')
        self.baseline_l2 = float(baseline_l2)
        self.radiomics_memory = {}
        self.refitted_baseline = False
        d = len(RADIOMICS_NAMES)
        for name, value in dict(radiomics_weights=torch.zeros(d), radiomics_intercept=torch.zeros(()),
                                radiomics_mean=torch.zeros(d), radiomics_scale=torch.ones(d),
                                radiomics_varying=torch.ones(d, dtype=torch.bool),
                                baseline_count=torch.zeros((), dtype=torch.long)).items():
            self.register_buffer(name, value)

    def get_extra_state(self):
        return dict(**super().get_extra_state(), schema='radiomics_offset_semantic_v4',
                    baseline_l2=self.baseline_l2, radiomics_memory=self.radiomics_memory)

    def set_extra_state(self, state):
        if state['schema'] != 'radiomics_offset_semantic_v4' or state['baseline_l2'] != self.baseline_l2:
            raise ValueError('V4 baseline schema/configuration mismatch')
        super().set_extra_state(state)
        self.radiomics_memory = state['radiomics_memory']
        self.refitted_baseline = False

    def _gather(self, features, labels, keys):
        if not self.training:
            raise RuntimeError('Evaluation must not update either training cache')
        if len(features) != len(labels) or len(keys) != len(labels):
            raise ValueError('Feature, label, and key counts differ')
        rows = [(key, value, float(label)) for key, value, label in
                zip(keys, features.detach().double().cpu().numpy(), labels.detach().cpu().numpy())]
        groups = [None]*dist.get_world_size() if dist.is_initialized() else [rows]
        if dist.is_initialized():
            dist.all_gather_object(groups, rows)
        rows = [row for group in groups for row in group]
        for key, value, label in rows:
            if key[0] not in self.allowed_cases:
                raise ValueError(f'Non-training case in V4 cache: {key[0]}')
            if not np.isfinite(value).all() or label not in (0., 1.):
                raise ValueError('Non-finite feature or invalid binary label')
        return rows

    @staticmethod
    def _fit_on_rank_zero(fit):
        package = [None]
        if not dist.is_initialized() or dist.get_rank() == 0:
            try:
                package[0] = fit()
            except Exception as exc:
                package[0] = dict(error=f'{type(exc).__name__}: {exc}')
        if dist.is_initialized():
            dist.broadcast_object_list(package, src=0)
        if 'error' in package[0]:
            raise RuntimeError(package[0]['error'])
        return package[0]

    @torch.no_grad()
    def observe_radiomics(self, features, labels, keys, *, refit=True):
        for key, value, label in self._gather(features, labels, keys):
            self.radiomics_memory[key] = (value.copy(), label)
        self.updates += 1
        self.refitted_baseline = False
        if not refit or len(self.radiomics_memory) < self.min_samples:
            return
        if int(self.baseline_count) and (self.refresh_every == 0 or (self.updates-1) % self.refresh_every):
            return
        rows = [self.radiomics_memory[k] for k in sorted(self.radiomics_memory)]
        x, y = np.stack([r[0] for r in rows]), np.array([r[1] for r in rows])
        if len(np.unique(y)) < 2:
            return
        started = time.perf_counter()
        result = self._fit_on_rank_zero(lambda: fit_radiomics(x, y, l2=self.baseline_l2))
        for name in ('weights', 'intercept', 'mean', 'scale', 'varying'):
            dest = getattr(self, 'radiomics_'+name)
            dest.copy_(torch.as_tensor(result[name], device=dest.device))
        self.baseline_count.add_(1)
        self.refitted_baseline = True
        self.last_fit.update(baseline_loss=result['loss'], baseline_seconds=time.perf_counter()-started,
                             baseline_samples=len(y), baseline_fits=int(self.baseline_count))

    @torch.no_grad()
    def observe_semantics(self, features, labels, keys, *, refit=True):
        for key, value, label in self._gather(features, labels, keys):
            radio, old_label = self.radiomics_memory[key]
            if label != old_label:
                raise ValueError('Stage labels disagree')
            self.memory[key] = (np.r_[radio, value], label)
        if not refit or not int(self.baseline_count) or len(self.memory) < self.min_samples:
            return
        if int(self.count) and not self.refitted_baseline:
            return
        rows = [self.memory[k] for k in sorted(self.memory)]
        x, y = np.stack([r[0] for r in rows]), np.array([r[1] for r in rows])
        if len(np.unique(y)) < 2:
            return
        # Use the installed (float32) baseline for exact fit/forward agreement.
        radio = torch.as_tensor(x[:, :len(RADIOMICS_NAMES)], dtype=torch.float32, device=self.weights.device)
        offset = self.baseline_logits(radio).double().cpu().numpy()
        started = time.perf_counter()
        result = self._fit_on_rank_zero(lambda: fit_offset_pool(x[:, len(RADIOMICS_NAMES):], y,
                                                               offset, **self.config))
        k = len(result['intercepts'])
        self.weights.zero_(); self.intercepts.zero_()
        self.weights[:k].copy_(torch.as_tensor(result['weights'], device=self.weights.device))
        self.intercepts[:k].copy_(torch.as_tensor(result['intercepts'], device=self.weights.device))
        for name in ('mean', 'scale', 'varying'):
            getattr(self, name).copy_(torch.as_tensor(result[name], device=self.weights.device))
        self.count.fill_(k); self.fit_count.add_(1)
        self.last_fit.update(samples=len(y), positives=int(y.sum()), size=k,
                             distinct_supports=result['distinct_supports'],
                             seconds=time.perf_counter()-started,
                             best_loss=float(result['losses'].min()), worst_loss=float(result['losses'].max()),
                             residual_rmse=result['residual_rmse'],
                             varying_semantics=result['varying_semantics'])

    @torch.no_grad()
    def baseline_logits(self, radiomics):
        with torch.autocast(device_type=radiomics.device.type, enabled=False):
            x = ((radiomics.float()-self.radiomics_mean)/self.radiomics_scale).clamp(-10., 10.)
            return F.linear(x*self.radiomics_varying, self.radiomics_weights[None],
                            self.radiomics_intercept[None]).squeeze(-1)

    def forward(self, semantics, baseline_logits, *, all_models=False):
        if not int(self.baseline_count):
            raise RuntimeError('Fit or load the radiomics baseline before semantic prediction')
        correction, indices = super().forward(semantics, all_models=all_models)
        # Residual supervision trains semantics against a detached radiomics baseline.
        # The baseline is fitted on cached labels; there is no radiomics loss.
        return baseline_logits.detach()[:, None] + correction, indices, correction
