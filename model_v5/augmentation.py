"""Translate both prepared semantic channels together, with zero padding."""
import torch
import torch.nn.functional as F


def translate_rois(rois, shifts_mm, spacing_mm):
    """Positive sampling offsets move content oppositely; no circular wrapping.

    Inputs are [N,2,X,Y,Z], physical shifts [N,3] in X/Y/Z order.
    The CT and mask share one interpolation grid, preserving their alignment.
    """
    if rois.ndim != 5 or rois.shape[1] != 2:
        raise ValueError('Expected [N,2,X,Y,Z] prepared semantic ROIs')
    shifts = torch.as_tensor(shifts_mm, device=rois.device, dtype=torch.float32)
    spacing = torch.as_tensor(spacing_mm, device=rois.device, dtype=torch.float32)
    if shifts.shape != (len(rois), 3) or bool((spacing <= 0).any()):
        raise ValueError('Invalid physical crop translation')
    size = shifts.new_tensor(rois.shape[-3:])
    theta = torch.eye(3, 4, device=rois.device).unsqueeze(0).repeat(len(rois), 1, 1)
    theta[:, :, 3] = (2 * shifts / spacing / size).flip(-1)
    grid = F.affine_grid(theta, rois.shape, align_corners=False)
    return F.grid_sample(rois.float(), grid, padding_mode='zeros', align_corners=False)


def augment_rois(rois, jitter_mm, spacing_mm, *, generator=None):
    # One fifth stay centered, to retain performance on the original domain.
    shifts = torch.randint(-int(jitter_mm), int(jitter_mm)+1, (len(rois),3),
                           generator=generator).float()
    shifts[torch.rand(len(rois), generator=generator) < .2] = 0
    return translate_rois(rois, shifts, spacing_mm)


def official_view(row, shift_mm):
    """Crop on the original 1-mm grid, then resize exactly once like joint ROI input.

    Store only the foreground bounding crop to avoid caching a 128-cube of zeros.
    This avoids blurring tiny nodules by translating an already downsampled ROI.
    """
    if 'compact' not in row:
        # Older diagnostic smoke caches used prepared-grid interpolation.
        return translate_rois(row['roi'][None],torch.as_tensor(shift_mm)[None],row['spacing'])[0]
    source=row['compact'];shape=tuple(row['crop_shape'])
    start=(row['compact_origin']-torch.as_tensor(shift_mm,dtype=torch.long)).tolist()
    out=source.new_zeros((1,2,*shape))
    lower=[max(0,s) for s in start]
    upper=[min(n,s+c) for n,s,c in zip(shape,start,source.shape[-3:])]
    if all(a<b for a,b in zip(lower,upper)):
        dst=tuple(slice(a,b) for a,b in zip(lower,upper))
        src=tuple(slice(a-s,b-s) for a,b,s in zip(lower,upper,start))
        out[(0,slice(None),*dst)]=source[(slice(None),*src)]
    if float(out[0,1].sum()) < .9*float(source[1].sum()):
        return row['roi']
    return F.interpolate(out,size=row['roi'].shape[-3:],mode='trilinear',align_corners=False)[0]
