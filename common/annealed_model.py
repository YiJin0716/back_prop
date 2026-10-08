"""Preserve V3 geometry, initialization and feature formulas exactly.

Only the logits entering the two diagnostic feature modules are divided by
temperature. Segmentation supervision always receives the original logits.
"""
import math

import torch

from back_prop.common.semantic_model import WholeCTJointModelV3 as BaseV3

ARCHITECTURE = "whole_ct_parallel_soft_radiomics_medicalnet_continuous_rashomon_v3_anneal"


class WholeCTJointModelV3(BaseV3):
    def __init__(self, *, mask_temperature=1.0, **kwargs):
        super().__init__(**kwargs)
        # Adding a scalar buffer consumes no RNG and changes no learned keys.
        self.register_buffer("mask_temperature", torch.tensor(1.0))
        self.set_temperature(mask_temperature)
        self.radiomics.register_forward_pre_hook(self._temper_diagnostic_mask)
        self.semantics.register_forward_pre_hook(self._temper_diagnostic_mask)

    def set_temperature(self, temperature):
        value = float(temperature)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Mask temperature must be positive and finite")
        self.mask_temperature.fill_(value)

    def _temper_diagnostic_mask(self, module, inputs):
        image, logits, valid = inputs
        return image, logits.float() / self.mask_temperature, valid
