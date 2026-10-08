"""V2 radiomics bottleneck.

Forward values are the paper-compatible 64 descriptors from the largest hard
component on the representative axial section.  Backward uses the fixed
surrogate implemented by :class:`DifferentiableRadiomics64`; the hard slice,
component and threshold operations themselves are not differentiable.
"""

from back_prop.model_v4.compat_radiomics import DifferentiableRadiomics64, FEATURE_NAMES

__all__ = ("DifferentiableRadiomics64", "FEATURE_NAMES")
