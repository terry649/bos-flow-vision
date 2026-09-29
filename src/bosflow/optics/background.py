"""Random-dot BOS backgrounds rendered on a supersampled grid."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def dot_density_for_fill(fill_fraction: float, dot_diameter_px: float) -> float:
    """Dots per pixel giving mean reflectance ``fill_fraction`` (empirical fit, +-3%)."""
    return -np.log(1.0 - fill_fraction) / (1.5 * dot_diameter_px**2)


def random_dot_pattern(shape: tuple[int, int], supersample: int, dot_density: float,
                       dot_diameter_px: float, rng: np.random.Generator) -> np.ndarray:
    """Background reflectance in [0, 1] on a grid ``supersample`` times finer than pixels.

    ``dot_density`` is dots per output pixel and ``dot_diameter_px`` the dot FWHM in
    output pixels. Dots are Gaussian and overlaps saturate smoothly, as printed or
    projected dot patterns do.
    """
    rows, cols = shape
    n = rng.poisson(dot_density * rows * cols / supersample**2)
    r = rng.uniform(-0.5, rows - 0.5, n)
    c = rng.uniform(-0.5, cols - 0.5, n)
    # Bilinear splat of unit impulses at subsample positions.
    r0, c0 = np.floor(r).astype(int), np.floor(c).astype(int)
    fr, fc = r - r0, c - c0
    acc = np.zeros((rows + 1, cols + 1))
    for dr, dc, w in ((0, 0, (1 - fr) * (1 - fc)), (1, 0, fr * (1 - fc)),
                      (0, 1, (1 - fr) * fc), (1, 1, fr * fc)):
        np.add.at(acc, (np.clip(r0 + dr, 0, rows), np.clip(c0 + dc, 0, cols)), w)
    acc = acc[:rows, :cols]
    sigma = dot_diameter_px * supersample / 2.3548
    field = gaussian_filter(acc, sigma, mode="wrap") * (2.0 * np.pi * sigma**2)
    return 1.0 - np.exp(-2.0 * field)
