"""Fixed-grid ROI utilities for the coarse-to-fine whole-CT model.

Spatial tensors in this module use the project's NIfTI order ``[X, Y, Z]``.
The input scan has already been resampled once to 1-mm isotropic spacing.  ROI
cropping is deliberately integer routed: it preserves those voxel values, but
the chosen origin is not differentiable with respect to a predicted centre.

The coarse VISTA/DETR grids represent the same full-scan field of view.  They
are sampled at the centres of the original 1-mm voxels with
``align_corners=False`` so their values remain differentiable.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor
import torch.nn.functional as F


SpatialSize = tuple[int, int, int]


def _spatial_size(value: Sequence[int], *, name: str) -> SpatialSize:
    result = tuple(int(item) for item in value)
    if len(result) != 3 or any(item <= 0 for item in result):
        raise ValueError(f"{name} must contain three positive integers, got {result}")
    return result  # type: ignore[return-value]


def _vector3(value: Tensor | Sequence[int] | Sequence[float], *, name: str) -> Tensor:
    result = torch.as_tensor(value).reshape(-1)
    if result.numel() != 3:
        raise ValueError(f"{name} must contain exactly three values, got shape {tuple(result.shape)}")
    return result


def integer_crop(
    volume: Tensor,
    center_xyz: Tensor | Sequence[float],
    size: Sequence[int] = (128, 128, 128),
    *,
    pad_value: float = 0.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Crop/pad a fixed integer ROI from a 1-mm ``[..., X, Y, Z]`` volume.

    ``center_xyz`` is expressed in continuous voxel-index coordinates: the
    centre of the first source voxel is coordinate 0.  It is rounded to the
    nearest source voxel (ties toward positive infinity), which is placed at
    local index ``size // 2``.  This explicit convention removes the inherent
    half-voxel ambiguity of an even-sized crop.

    Returns ``(patch, origin_xyz, valid_mask)``.  ``patch`` preserves all input
    leading dimensions, ``origin_xyz`` is a length-three integer tensor in the
    full scan coordinate system and may be negative, and ``valid_mask`` has
    shape ``[1, *size]`` with ``True`` only for voxels copied from the scan.
    Integer routing intentionally has no gradient with respect to the centre;
    gradients with respect to values in ``volume`` are preserved.
    """

    if volume.ndim < 3:
        raise ValueError(f"volume must have at least three spatial dimensions, got {tuple(volume.shape)}")
    crop_size = _spatial_size(size, name="size")
    scan_shape = tuple(int(item) for item in volume.shape[-3:])
    if any(item <= 0 for item in scan_shape):
        raise ValueError(f"volume has an empty spatial dimension: {scan_shape}")

    centre = _vector3(center_xyz, name="center_xyz").detach().to(dtype=torch.float64, device="cpu")
    if not torch.isfinite(centre).all():
        raise ValueError(f"center_xyz must be finite, got {centre.tolist()}")
    # floor(x + 0.5) gives deterministic round-half-up behaviour, unlike
    # torch.round's bankers rounding.
    centre_voxel = torch.floor(centre + 0.5).to(torch.long)
    origin_cpu = centre_voxel - torch.tensor(
        [item // 2 for item in crop_size], dtype=torch.long
    )
    origin_values = tuple(int(item) for item in origin_cpu.tolist())

    patch = volume.new_full((*volume.shape[:-3], *crop_size), pad_value)
    valid = torch.zeros((1, *crop_size), dtype=torch.bool, device=volume.device)

    source_start = tuple(max(0, origin) for origin in origin_values)
    source_end = tuple(
        min(scan, origin + extent)
        for scan, origin, extent in zip(scan_shape, origin_values, crop_size)
    )
    overlap = tuple(max(0, end - start) for start, end in zip(source_start, source_end))
    if all(length > 0 for length in overlap):
        destination_start = tuple(start - origin for start, origin in zip(source_start, origin_values))
        destination_end = tuple(start + length for start, length in zip(destination_start, overlap))
        source_slices = tuple(slice(start, end) for start, end in zip(source_start, source_end))
        destination_slices = tuple(slice(start, end) for start, end in zip(destination_start, destination_end))
        patch[(...,) + destination_slices] = volume[(...,) + source_slices]
        valid[(...,) + destination_slices] = True

    origin = origin_cpu.to(device=volume.device)
    return patch, origin, valid


def sample_global_at_roi(
    global_volume: Tensor,
    origin: Tensor | Sequence[int],
    scan_shape: Sequence[int],
    size: Sequence[int] = (128, 128, 128),
    *,
    mode: str = "bilinear",
) -> Tensor:
    """Sample a scan-wide coarse tensor at an integer ROI's voxel centres.

    ``global_volume`` is ``[C,GX,GY,GZ]`` or ``[B,C,GX,GY,GZ]`` and spans the
    complete original scan whose 1-mm shape is ``scan_shape``.  A grid point
    for source voxel index ``i`` is ``2 * (i + 0.5) / scan_size - 1``.  This is
    the voxel-centre convention required by ``align_corners=False``.

    Out-of-scan samples are zero.  The input rank is preserved, and gradients
    flow to ``global_volume`` (the integer ``origin`` is routing metadata and
    is intentionally detached).
    """

    if global_volume.ndim not in (4, 5):
        raise ValueError(
            "global_volume must be [C,GX,GY,GZ] or [B,C,GX,GY,GZ], "
            f"got {tuple(global_volume.shape)}"
        )
    if not global_volume.is_floating_point():
        raise TypeError("global_volume must be floating point for trilinear sampling")
    full_shape = _spatial_size(scan_shape, name="scan_shape")
    roi_size = _spatial_size(size, name="size")
    origin_tensor = _vector3(origin, name="origin").detach().to(
        device=global_volume.device, dtype=global_volume.dtype
    )

    coordinates = [
        origin_tensor[axis]
        + torch.arange(extent, device=global_volume.device, dtype=global_volume.dtype)
        + 0.5
        for axis, extent in enumerate(roi_size)
    ]
    normalized = [
        2.0 * coordinate / float(scan_extent) - 1.0
        for coordinate, scan_extent in zip(coordinates, full_shape)
    ]
    grid_x, grid_y, grid_z = torch.meshgrid(*normalized, indexing="ij")
    # grid_sample names its spatial axes D/H/W; for our [X,Y,Z] tensor, its
    # grid vector must therefore be ordered [Z,Y,X].
    grid = torch.stack((grid_z, grid_y, grid_x), dim=-1).unsqueeze(0)

    squeezed = global_volume.ndim == 4
    source = global_volume.unsqueeze(0) if squeezed else global_volume
    grid = grid.expand(source.shape[0], -1, -1, -1, -1)
    sampled = F.grid_sample(
        source,
        grid,
        mode=mode,
        padding_mode="zeros",
        align_corners=False,
    )
    return sampled[0] if squeezed else sampled


def paste_compact_mask(
    mask_crop: Tensor,
    mask_origin: Tensor | Sequence[int],
    roi_origin: Tensor | Sequence[int],
    size: Sequence[int] = (128, 128, 128),
) -> Tensor:
    """Paste one tight full-scan GT mask crop into a fixed ROI coordinate frame.

    ``mask_crop`` must be uint8 and have shape ``[X,Y,Z]`` or ``[1,X,Y,Z]``.
    Both origins are integer full-scan coordinates.  The returned uint8 tensor
    preserves whether the input had a singleton channel: its shape is
    ``[*size]`` for a 3-D input or ``[1,*size]`` for a 4-D input.  Portions
    outside the intersection are zero.
    """

    if mask_crop.dtype != torch.uint8:
        raise TypeError(f"mask_crop must be torch.uint8, got {mask_crop.dtype}")
    had_channel = mask_crop.ndim == 4
    if had_channel:
        if mask_crop.shape[0] != 1:
            raise ValueError(f"4-D mask_crop must have one channel, got {tuple(mask_crop.shape)}")
        source = mask_crop[0]
    elif mask_crop.ndim == 3:
        source = mask_crop
    else:
        raise ValueError(f"mask_crop must be [X,Y,Z] or [1,X,Y,Z], got {tuple(mask_crop.shape)}")

    roi_size = _spatial_size(size, name="size")
    source_shape = tuple(int(item) for item in source.shape)
    mask_start = tuple(int(item) for item in _vector3(mask_origin, name="mask_origin").detach().cpu())
    roi_start = tuple(int(item) for item in _vector3(roi_origin, name="roi_origin").detach().cpu())
    mask_end = tuple(start + extent for start, extent in zip(mask_start, source_shape))
    roi_end = tuple(start + extent for start, extent in zip(roi_start, roi_size))

    intersection_start = tuple(max(mask, roi) for mask, roi in zip(mask_start, roi_start))
    intersection_end = tuple(min(mask, roi) for mask, roi in zip(mask_end, roi_end))
    target = torch.zeros(roi_size, dtype=torch.uint8, device=source.device)
    if all(end > start for start, end in zip(intersection_start, intersection_end)):
        source_slices = tuple(
            slice(start - mask, end - mask)
            for start, end, mask in zip(intersection_start, intersection_end, mask_start)
        )
        target_slices = tuple(
            slice(start - roi, end - roi)
            for start, end, roi in zip(intersection_start, intersection_end, roi_start)
        )
        target[target_slices] = source[source_slices]
    return target.unsqueeze(0) if had_channel else target


def assemble_refiner_input(
    ct_patch: Tensor,
    vista_probability: Tensor,
    coarse_query: Tensor,
    valid_mask: Tensor,
    context: Tensor | None = None,
    *,
    coarse_is_logits: bool = True,
) -> Tensor:
    """Assemble the fine-refiner channel contract.

    Inputs are batched ``[B,C,X,Y,Z]`` tensors.  The channel order is CT (1),
    VISTA probability (1), coarse-query probability (1), valid mask (1), then
    optional context channels.  With the planned eight context channels the
    result has 12 channels, matching :class:`FineMaskRefiner3D`'s default.
    """

    tensors = (ct_patch, vista_probability, coarse_query, valid_mask)
    if any(value.ndim != 5 for value in tensors):
        raise ValueError("ct_patch, vista_probability, coarse_query and valid_mask must all be 5-D")
    reference = ct_patch.shape[0], ct_patch.shape[-3:]
    if any((value.shape[0], value.shape[-3:]) != reference for value in tensors[1:]):
        raise ValueError("all refiner inputs must share batch and spatial shapes")
    if any(value.shape[1] != 1 for value in tensors):
        raise ValueError("the CT, VISTA, coarse-query and valid inputs must each have one channel")
    coarse_probability = coarse_query.sigmoid() if coarse_is_logits else coarse_query
    pieces = [
        ct_patch,
        vista_probability.to(ct_patch),
        coarse_probability.to(ct_patch),
        valid_mask.to(dtype=ct_patch.dtype, device=ct_patch.device),
    ]
    if context is not None:
        if context.ndim != 5 or (context.shape[0], context.shape[-3:]) != reference:
            raise ValueError("context must share the CT batch and spatial shapes")
        pieces.append(context.to(ct_patch))
    return torch.cat(pieces, dim=1)


__all__ = (
    "assemble_refiner_input",
    "integer_crop",
    "paste_compact_mask",
    "sample_global_at_roi",
)
