"""Image formation: warp the background by the displacement field and integrate pixels."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates


def block_mean(a: np.ndarray, k: int) -> np.ndarray:
    """Average over non-overlapping k x k blocks in the last two axes."""
    *lead, r, c = a.shape
    return a.reshape(*lead, r // k, k, c // k, k).mean(axis=(-3, -1))


def warp_backward(image_hr: np.ndarray, disp_hr: np.ndarray, supersample: int) -> np.ndarray:
    """I(x) = B(x - d(x)) on the supersampled grid.

    ``disp_hr`` is (2, rows, cols) in output pixels (column, row); it is scaled to
    supersampled units here. Cubic B-spline interpolation of a band-limited dot
    pattern keeps interpolation error far below sensor noise.
    """
    rows, cols = image_hr.shape
    rr, cc = np.mgrid[0:rows, 0:cols].astype(float)
    coords = np.stack([rr - supersample * disp_hr[1], cc - supersample * disp_hr[0]])
    return map_coordinates(image_hr, coords, order=3, mode="reflect", prefilter=True)


def forward_flow(disp: np.ndarray, iterations: int = 8) -> np.ndarray:
    """Optical flow from the reference to the deflected image.

    The deflected image is I(x) = B(x - d(x)), so a reference point p appears at
    p + u(p) with u(p) = d(p + u(p)). Solved by fixed-point iteration, which
    converges when |grad d| < 1.
    """
    _, rows, cols = disp.shape
    rr, cc = np.mgrid[0:rows, 0:cols].astype(float)
    u = disp.copy()
    for _ in range(iterations):
        coords = np.stack([rr + u[1], cc + u[0]])
        u = np.stack([map_coordinates(disp[i], coords, order=1, mode="nearest")
                      for i in range(2)])
    return u
