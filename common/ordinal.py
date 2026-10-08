"""Trainable dense proportional-odds heads for seven LIDC ratings."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from back_prop.model_v4.compat_radiomics64 import FEATURE_NAMES
from back_prop.common.coarse_data import LABELS


DEFAULT_ORDINAL_INIT = Path(__file__).with_name("logistic_init.json")


def _softplus_inverse(value: Tensor) -> Tensor:
    value = value.clamp_min(1e-5)
    return value + torch.log(-torch.expm1(-value))


class OrdinalRadiomicsHeads(nn.Module):
    """Seven dense ordinal heads initialized without held-out-test leakage."""

    def __init__(self, initialization: str | Path = DEFAULT_ORDINAL_INIT) -> None:
        super().__init__()
        path = Path(initialization)
        # Earlier checkpoints stored the bundled initializer's former location.
        if not path.exists() and path.parts[-3:] == ('back_prop', 'model_v2', 'logistic_init.json'):
            path = DEFAULT_ORDINAL_INIT
        payload = json.loads(path.read_text())
        if tuple(payload.get("labels", ())) != LABELS:
            raise ValueError("Ordinal initialization has the wrong label order")
        if tuple(payload.get("feature_names", ())) != tuple(FEATURE_NAMES):
            raise ValueError("Ordinal initialization has the wrong radiomics order")
        records = [payload["models"][label] for label in LABELS]
        self.initialization_path = str(path.resolve())
        self.initialization_metadata = payload.get("cohort", payload.get("metadata", {}))
        self.register_buffer(
            "imputer_median",
            torch.tensor([record["imputer_median"] for record in records], dtype=torch.float32),
        )
        self.register_buffer(
            "scaler_mean",
            torch.tensor([record["scaler_mean"] for record in records], dtype=torch.float32),
        )
        self.register_buffer(
            "scaler_scale",
            torch.tensor([record["scaler_scale"] for record in records], dtype=torch.float32),
        )
        coefficients = torch.tensor(
            [record["coefficients"] for record in records], dtype=torch.float32
        )
        thresholds = torch.tensor(
            [record["thresholds"] for record in records], dtype=torch.float32
        )
        if coefficients.shape != (len(LABELS), len(FEATURE_NAMES)):
            raise ValueError(f"Unexpected coefficient shape {tuple(coefficients.shape)}")
        if thresholds.shape != (len(LABELS), 4):
            raise ValueError(f"Unexpected threshold shape {tuple(thresholds.shape)}")
        self.coefficients = nn.Parameter(coefficients.clone())
        self.threshold_base = nn.Parameter(thresholds[:, 0].clone())
        self.threshold_increments_raw = nn.Parameter(
            _softplus_inverse(thresholds[:, 1:] - thresholds[:, :-1])
        )
        self.register_buffer("initial_coefficients", coefficients.clone())
        self.register_buffer("initial_thresholds", thresholds.clone())
        self.register_buffer("classes", torch.arange(1, 6, dtype=torch.float32))

    def thresholds(self) -> Tensor:
        increments = F.softplus(self.threshold_increments_raw)
        return torch.cat(
            (
                self.threshold_base[:, None],
                self.threshold_base[:, None] + increments.cumsum(dim=1),
            ),
            dim=1,
        )

    def forward(self, raw_features: Tensor) -> tuple[Tensor, Tensor]:
        if raw_features.ndim != 2 or raw_features.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"Expected radiomics [N,{len(FEATURE_NAMES)}], got {tuple(raw_features.shape)}"
            )
        values = raw_features[:, None, :].expand(-1, len(LABELS), -1).float()
        values = torch.where(torch.isfinite(values), values, self.imputer_median[None])
        standardized = (values - self.scaler_mean[None]) / self.scaler_scale[None].clamp_min(1e-8)
        standardized = standardized.clamp(-25.0, 25.0)
        linear = torch.einsum("nlf,lf->nl", standardized, self.coefficients)
        cumulative = torch.sigmoid(self.thresholds()[None] - linear[..., None])
        probability = torch.cat(
            (
                cumulative[..., :1],
                cumulative[..., 1:] - cumulative[..., :-1],
                1.0 - cumulative[..., -1:],
            ),
            dim=-1,
        ).clamp_min(1e-7)
        probability = probability / probability.sum(dim=-1, keepdim=True)
        expected = (probability * self.classes).sum(dim=-1)
        return probability, expected

    def anchor_loss(self) -> Tensor:
        """Small optional penalty that discourages abrupt loss of initialization."""
        return (
            (self.coefficients - self.initial_coefficients).square().mean()
            + (self.thresholds() - self.initial_thresholds).square().mean()
        )


__all__ = ("DEFAULT_ORDINAL_INIT", "OrdinalRadiomicsHeads")
