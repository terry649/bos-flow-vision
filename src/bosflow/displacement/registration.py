"""Deformable image registration (DIR): multi-resolution diffeomorphic demons.

Demons registration (Thirion 1998) in its diffeomorphic form (Vercauteren et al.
2009, NeuroImage 45:S61) via SimpleITK. The reference is the fixed image and the
deflected image the moving one, so the returned displacement D satisfies
deflected(p + D(p)) ~= reference(p), the same convention as the optical flow
estimators. A coarse-to-fine pyramid handles displacements beyond the demons
force's capture range of about one pixel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom


@dataclass(frozen=True)
class DemonsParams:
    shrink_factors: tuple[int, ...] = (4, 2, 1)
    iterations: tuple[int, ...] = (60, 40, 30)
    # Gaussian regularization of the displacement field, px. Smaller keeps more of a
    # shock's peak but admits more noise; 1.5 balances both on a held-out tuning seed.
    field_sigma: float = 1.5
    update_sigma: float = 0.0  # fluid-like smoothing of each update; 0 disables it
    max_step: float = 0.5  # px per iteration


def _to_sitk(img: np.ndarray) -> sitk.Image:
    return sitk.GetImageFromArray(img.astype(np.float32))


def demons(ref: np.ndarray, defl: np.ndarray, p: DemonsParams | None = None) -> np.ndarray:
    p = p or DemonsParams()
    fixed_full = _to_sitk(ref / 255.0)
    moving_full = _to_sitk(defl / 255.0)
    rows, cols = ref.shape
    field = None  # numpy (rows_l, cols_l, 2) as (x, y) in level pixels
    for shrink, iters in zip(p.shrink_factors, p.iterations):
        if shrink > 1:
            # Anti-aliased downsampling so demons sees band-limited images.
            s = [shrink, shrink]
            fixed = sitk.Shrink(sitk.SmoothingRecursiveGaussian(fixed_full, 0.5 * shrink), s)
            moving = sitk.Shrink(sitk.SmoothingRecursiveGaussian(moving_full, 0.5 * shrink), s)
        else:
            fixed, moving = fixed_full, moving_full
        shape = sitk.GetArrayViewFromImage(fixed).shape
        if field is None:
            init = np.zeros((*shape, 2))
        else:
            factor = (shape[0] / field.shape[0], shape[1] / field.shape[1])
            init = np.stack([zoom(field[..., k], factor, order=1) * factor[1 - k]
                             for k in range(2)], axis=-1)
            init = init[: shape[0], : shape[1]]
        init_img = sitk.GetImageFromArray(init.astype(np.float64), isVector=True)
        init_img.CopyInformation(fixed)

        f = sitk.DiffeomorphicDemonsRegistrationFilter()
        f.SetNumberOfIterations(iters)
        f.SetSmoothDisplacementField(p.field_sigma > 0)
        f.SetStandardDeviations(max(p.field_sigma, 1e-3))
        f.SetSmoothUpdateField(p.update_sigma > 0)
        f.SetUpdateFieldStandardDeviations(max(p.update_sigma, 1e-3))
        f.SetMaximumUpdateStepLength(p.max_step)
        f.SetUseGradientType(f.Symmetric)
        field = sitk.GetArrayFromImage(f.Execute(fixed, moving, init_img))
    assert field.shape[:2] == (rows, cols)
    return np.moveaxis(field, -1, 0).astype(np.float32)
