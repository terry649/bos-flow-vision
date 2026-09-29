"""Synthetic path-integrated density turbulence.

For a statistically isotropic 3D scalar field with the von Karman spectrum
Phi_3(k) ~ (k^2 + k0^2)^(-11/6), the line-of-sight integral over a depth W much
larger than the outer scale has 2D power spectrum 2 pi W Phi_3(kx, ky, 0), so the
projected field keeps the same functional form: PSD_2(k) ~ (k^2 + k0^2)^(-11/6),
with the Kolmogorov -11/3 slope in the inertial range (Hufnagel and Stanley 1964;
Andrews and Phillips, *Laser Beam Propagation through Random Media*).
"""

from __future__ import annotations

import numpy as np

VON_KARMAN_EXPONENT = -11.0 / 6.0


def von_karman_psd(k, outer_scale: float):
    k0 = 2.0 * np.pi / outer_scale
    return (np.asarray(k) ** 2 + k0**2) ** VON_KARMAN_EXPONENT


def von_karman_field(shape: tuple[int, int], dx: float, outer_scale: float,
                     rng: np.random.Generator, inner_scale: float = 0.0) -> np.ndarray:
    """Zero-mean, unit-rms Gaussian random field with a 2D von Karman spectrum.

    ``dx`` and the scales share units. A nonzero ``inner_scale`` applies the
    Tatarskii Gaussian cutoff exp(-(k l0 / 5.92)^2).
    """
    ny, nx = shape
    kx = 2.0 * np.pi * np.fft.rfftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dx)
    k = np.hypot(ky[:, None], kx[None, :])
    amp = np.sqrt(von_karman_psd(k, outer_scale))
    if inner_scale > 0.0:
        amp *= np.exp(-0.5 * (k * inner_scale / 5.92) ** 2)
    amp[0, 0] = 0.0
    noise = rng.standard_normal((ny, nx))
    field = np.fft.irfft2(np.fft.rfft2(noise) * amp, s=(ny, nx))
    field -= field.mean()
    return field / field.std()


def radial_psd(field: np.ndarray, dx: float, nbins: int = 64):
    """Azimuthally averaged power spectrum, for testing the spectral slope."""
    ny, nx = field.shape
    F = np.abs(np.fft.fft2(field)) ** 2
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dx)
    k = np.hypot(ky[:, None], kx[None, :]).ravel()
    F = F.ravel()
    edges = np.geomspace(k[k > 0].min(), np.pi / dx, nbins + 1)
    idx = np.digitize(k, edges) - 1
    ok = (idx >= 0) & (idx < nbins)
    power = np.bincount(idx[ok], weights=F[ok], minlength=nbins)
    count = np.bincount(idx[ok], minlength=nbins)
    centers = np.sqrt(edges[:-1] * edges[1:])
    good = count > 0
    return centers[good], power[good] / count[good]
