"""Paper-faithful 64-feature radiomics for one 2-D LIDC annotation.

The feature families and definitions follow Raicu et al., *Modeling Semantics
from Image Data: Opportunities from LIDC* (IJBET, 2008).  Texture details
were cross-checked against the authors' GPL-2.0 BRISC reference implementation
(https://sourceforge.net/projects/brisc/).  This is an independent NumPy/SciPy
implementation of the published mathematical operations.
"""

from __future__ import annotations

from collections import OrderedDict
import math

import numpy as np
from scipy import ndimage, signal
from skimage import measure, morphology, segmentation


SHAPE_NAMES = (
    "Circularity", "Roughness", "Elongation", "Compactness",
    "Eccentricity", "Solidity", "Extent", "RadialDistanceSD",
)
SIZE_NAMES = (
    "Area", "ConvexArea", "Perimeter", "ConvexPerimeter",
    "EquivDiameter", "MajorAxisLength", "MinorAxisLength",
)
INTENSITY_NAMES = (
    "MinIntensity", "MaxIntensity", "MeanIntensity", "SDIntensity",
    "MinIntensityBG", "MaxIntensityBG", "MeanIntensityBG", "SDIntensityBG",
    "IntensityDifference",
)
HARALICK_NAMES = (
    "contrast", "correlation", "energy", "homogeneity", "entropy",
    "thirdOrderMoment", "inverseVariance", "sumAverage", "variance",
    "clusterTendency", "maximumProbability",
)
GABOR_NAMES = tuple(
    f"Gabor{stat}_{angle}_{str(freq).replace('.', '')}"
    for angle in (0, 45, 90, 135)
    for freq in (0.3, 0.4, 0.5)
    for stat in ("Mean", "SD")
)
MARKOV_NAMES = tuple(f"Markov{i}" for i in range(5))
FEATURE_NAMES = SHAPE_NAMES + SIZE_NAMES + INTENSITY_NAMES + HARALICK_NAMES + GABOR_NAMES + MARKOV_NAMES

if len(FEATURE_NAMES) != 64:  # defensive guard against accidental edits
    raise RuntimeError(f"Expected 64 feature names, found {len(FEATURE_NAMES)}")


def _sample_std(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(values.std(ddof=1)) if values.size > 1 else 0.0


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.nonzero(mask)
    if not len(rows):
        raise ValueError("The annotation mask is empty")
    return int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1


def _crop_with_padding(image: np.ndarray, bounds: tuple[int, int, int, int], pad: int) -> np.ndarray:
    r0, r1, c0, c1 = bounds
    before_r, after_r = max(0, pad - r0), max(0, r1 + pad - image.shape[0])
    before_c, after_c = max(0, pad - c0), max(0, c1 + pad - image.shape[1])
    if before_r or after_r or before_c or after_c:
        image = np.pad(image, ((before_r, after_r), (before_c, after_c)), mode="edge")
        r0, r1, c0, c1 = r0 + before_r, r1 + before_r, c0 + before_c, c1 + before_c
    return np.asarray(image[r0 - pad:r1 + pad, c0 - pad:c1 + pad], dtype=np.float64)


def shape_and_size(mask: np.ndarray) -> OrderedDict[str, float]:
    mask = np.asarray(mask, dtype=bool)
    props = measure.regionprops(mask.astype(np.uint8))
    if not props:
        raise ValueError("The annotation mask is empty")
    # Official annotation masks are expected to contain a single outlined ROI.
    region = max(props, key=lambda item: item.area)
    component = mask if len(props) == 1 else (measure.label(mask) == region.label)
    convex = morphology.convex_hull_image(component)

    area = float(component.sum())
    convex_area = float(convex.sum())
    perimeter = float(measure.perimeter(component, neighborhood=4))
    convex_perimeter = float(measure.perimeter(convex, neighborhood=4))
    major = float(region.axis_major_length)
    minor = float(region.axis_minor_length)
    equivalent = float(region.equivalent_diameter_area)

    # The paper's circularity uses the circle induced by the convex perimeter.
    circularity = 4.0 * math.pi * area / max(convex_perimeter**2, np.finfo(float).eps)
    roughness = convex_perimeter / max(perimeter, np.finfo(float).eps)
    elongation = major / max(minor, np.finfo(float).eps)
    compactness = perimeter**2 / max(4.0 * math.pi * area, np.finfo(float).eps)

    boundary = segmentation.find_boundaries(component, connectivity=2, mode="inner")
    boundary_points = np.argwhere(boundary)
    centre = np.asarray(region.centroid, dtype=np.float64)
    radial_sd = _sample_std(np.linalg.norm(boundary_points - centre, axis=1))

    return OrderedDict(zip(
        SHAPE_NAMES + SIZE_NAMES,
        (
            circularity, roughness, elongation, compactness,
            float(region.eccentricity), area / convex_area,
            float(region.extent), radial_sd,
            area, convex_area, perimeter, convex_perimeter,
            equivalent, major, minor,
        ),
    ))


def intensity_features(image: np.ndarray, mask: np.ndarray) -> OrderedDict[str, float]:
    bounds = _bbox(mask)
    r0, r1, c0, c1 = bounds
    roi_image = np.asarray(image[r0:r1, c0:c1], dtype=np.float64)
    roi_mask = np.asarray(mask[r0:r1, c0:c1], dtype=bool)
    foreground = roi_image[roi_mask]
    background = roi_image[~roi_mask]
    if not foreground.size:
        raise ValueError("The annotation mask is empty")
    # A rectangular mask can have no within-bounding-box background. Use the
    # one-pixel surrounding ring, which preserves the intended local contrast.
    if not background.size:
        padded = _crop_with_padding(image, bounds, 1)
        ring_mask = np.pad(roi_mask, 1, constant_values=False)
        background = padded[~ring_mask]

    values = (
        float(foreground.min()), float(foreground.max()), float(foreground.mean()), _sample_std(foreground),
        float(background.min()), float(background.max()), float(background.mean()), _sample_std(background),
        float(abs(foreground.mean() - background.mean())),
    )
    return OrderedDict(zip(INTENSITY_NAMES, values))


def _cooccurrence(ranks: np.ndarray, levels: int, dr: int, dc: int) -> np.ndarray | None:
    height, width = ranks.shape
    r0, r1 = max(0, -dr), min(height, height - dr)
    c0, c1 = max(0, -dc), min(width, width - dc)
    if r0 >= r1 or c0 >= c1:
        return None
    first = ranks[r0:r1, c0:c1]
    second = ranks[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
    valid = (first >= 0) & (second >= 0)
    if not valid.any():
        return None
    flat = first[valid].astype(np.int64) * levels + second[valid].astype(np.int64)
    matrix = np.bincount(flat, minlength=levels * levels).reshape(levels, levels).astype(np.float64)
    return matrix / matrix.sum()


def _haralick_from_matrix(matrix: np.ndarray) -> np.ndarray:
    i, j = np.indices(matrix.shape, dtype=np.float64)
    imean = float((i * matrix).sum())
    jmean = float((j * matrix).sum())
    ivar = float(((i - imean) ** 2 * matrix).sum())
    jvar = float(((j - jmean) ** 2 * matrix).sum())
    delta = i - j
    nz = matrix > 0
    correlation = 0.0
    if ivar > 0 and jvar > 0:
        correlation = float(((i - imean) * (j - jmean) * matrix).sum() / math.sqrt(ivar * jvar))
    inverse = np.zeros_like(matrix)
    inverse[delta != 0] = matrix[delta != 0] / (delta[delta != 0] ** 2)
    return np.asarray((
        ((delta**2) * matrix).sum(),
        correlation,
        (matrix**2).sum(),
        (matrix / (1.0 + np.abs(delta))).sum(),
        -(matrix[nz] * np.log(matrix[nz])).sum(),
        ((delta**3) * matrix).sum(),
        inverse.sum(),
        (0.5 * (i + j) * matrix).sum(),
        (0.5 * ((i - imean) ** 2 + (j - jmean) ** 2) * matrix).sum(),
        (((i - imean + j - jmean) ** 2) * matrix).sum(),
        matrix.max(),
    ), dtype=np.float64)


def haralick_features(image: np.ndarray, mask: np.ndarray, distances: int = 5) -> OrderedDict[str, float]:
    bounds = _bbox(mask)
    r0, r1, c0, c1 = bounds
    roi_mask = np.asarray(mask[r0:r1, c0:c1], dtype=bool)
    # BRISC discretized the roughly 1496-value historical LIDC range into 64
    # bins using 1496 / 64 = 23.375, then indexed the observed ordered levels.
    quantized = np.trunc(np.asarray(image[r0:r1, c0:c1], dtype=np.float64) / 23.375).astype(np.int32)
    observed = np.unique(quantized[roi_mask])
    ranks = np.full(quantized.shape, -1, dtype=np.int32)
    ranks[roi_mask] = np.searchsorted(observed, quantized[roi_mask])

    by_distance: list[np.ndarray] = []
    for distance in range(1, distances + 1):
        by_direction = []
        for dr, dc in ((0, distance), (-distance, distance), (-distance, 0), (-distance, -distance)):
            matrix = _cooccurrence(ranks, len(observed), dr, dc)
            if matrix is not None:
                by_direction.append(_haralick_from_matrix(matrix))
        if by_direction:
            by_distance.append(np.mean(by_direction, axis=0))
    if not by_distance:
        raise ValueError("Annotation is too small to form a co-occurrence pair")
    result = np.min(np.stack(by_distance), axis=0)
    return OrderedDict(zip(HARALICK_NAMES, map(float, result)))


def _gabor_kernel(theta: float, frequency: float) -> np.ndarray:
    coords = np.arange(-4, 5, dtype=np.float64)
    x, y = np.meshgrid(coords, coords, indexing="ij")
    wavelength = 1.0 / frequency
    sigma = 0.56 * wavelength
    gamma = 0.5
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    gaussian = np.exp(-0.5 * (x_theta**2 + gamma**2 * y_theta**2) / sigma**2)
    harmonic = np.sin(2.0 * np.pi * x_theta / wavelength)
    return gaussian * harmonic


def gabor_features(image: np.ndarray, mask: np.ndarray) -> OrderedDict[str, float]:
    crop = _crop_with_padding(image, _bbox(mask), 4)
    values: list[float] = []
    for theta in (0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0):
        for frequency in (0.3, 0.4, 0.5):
            # BRISC performs correlation, not a flipped-kernel convolution;
            # the absolute response makes the odd-kernel sign immaterial.
            response = np.abs(signal.correlate2d(crop, _gabor_kernel(theta, frequency), mode="valid"))
            values.extend((float(response.mean()), _sample_std(response)))
    return OrderedDict(zip(GABOR_NAMES, values))


def _window_sum(array: np.ndarray) -> np.ndarray:
    return ndimage.uniform_filter(array, size=7, mode="constant", cval=0.0) * 49.0


def markov_features(image: np.ndarray, mask: np.ndarray) -> OrderedDict[str, float]:
    data = _crop_with_padding(image, _bbox(mask), 4)
    q = np.zeros((*data.shape, 4), dtype=np.float64)
    q[1:-1, 1:-1, 0] = data[1:-1, 2:] + data[1:-1, :-2]
    q[1:-1, 1:-1, 1] = data[2:, 1:-1] + data[:-2, 1:-1]
    q[1:-1, 1:-1, 2] = data[2:, :-2] + data[:-2, 2:]
    q[1:-1, 1:-1, 3] = data[2:, 2:] + data[:-2, :-2]

    centre = (slice(4, -4), slice(4, -4))
    corr = np.empty((data.shape[0] - 8, data.shape[1] - 8, 4, 4), dtype=np.float64)
    vector = np.empty((data.shape[0] - 8, data.shape[1] - 8, 4), dtype=np.float64)
    for a in range(4):
        vector[..., a] = _window_sum(q[..., a] * data)[centre]
        for b in range(4):
            corr[..., a, b] = _window_sum(q[..., a] * q[..., b])[centre]

    # Exact inverses in the 2006 implementation fail on constant/small ROIs.
    # A scale-relative machine ridge is equivalent on nonsingular systems and
    # gives a deterministic least-squares continuation on singular systems.
    trace = np.trace(corr, axis1=-2, axis2=-1)
    ridge = np.finfo(np.float64).eps * np.maximum(trace / 4.0, 1.0)
    corr[..., np.arange(4), np.arange(4)] += ridge[..., None]
    try:
        beta = np.linalg.solve(corr, vector)
    except np.linalg.LinAlgError:
        beta = np.einsum("...ij,...j->...i", np.linalg.pinv(corr), vector)

    sum_i2 = _window_sum(data * data)[centre]
    directional = np.empty_like(beta)
    for a in range(4):
        directional[..., a] = (
            sum_i2 - 2.0 * beta[..., a] * vector[..., a]
            + beta[..., a] ** 2 * corr[..., a, a]
        ) / 81.0
    joint = (
        sum_i2 - 2.0 * np.einsum("...i,...i->...", beta, vector)
        + np.einsum("...i,...ij,...j->...", beta, corr, beta)
    ) / 81.0
    values = [float(directional[..., i].mean()) for i in range(4)] + [float(joint.mean())]
    return OrderedDict(zip(MARKOV_NAMES, values))


def extract_radiomics64(image_hu: np.ndarray, mask: np.ndarray) -> OrderedDict[str, float]:
    """Extract all 64 features from the largest axial annotation section.

    ``image_hu`` and ``mask`` must be matching 2-D arrays. The historical BRISC
    code operated on stored CT values with air near zero; adding 1024 to HU
    reconstructs that convention for the project's NIfTI volumes.
    """
    image_hu = np.asarray(image_hu)
    mask = np.asarray(mask, dtype=bool)
    if image_hu.ndim != 2 or mask.ndim != 2 or image_hu.shape != mask.shape:
        raise ValueError(f"Expected matching 2-D image/mask, got {image_hu.shape} and {mask.shape}")
    image = np.asarray(image_hu, dtype=np.float64) + 1024.0

    features: OrderedDict[str, float] = OrderedDict()
    features.update(shape_and_size(mask))
    features.update(intensity_features(image, mask))
    features.update(haralick_features(image, mask))
    features.update(gabor_features(image, mask))
    features.update(markov_features(image, mask))
    if tuple(features) != FEATURE_NAMES:
        raise RuntimeError("Feature ordering does not match FEATURE_NAMES")
    values = np.asarray(list(features.values()))
    if not np.isfinite(values).all():
        bad = [name for name, value in features.items() if not np.isfinite(value)]
        raise FloatingPointError(f"Non-finite features: {bad}")
    return features
