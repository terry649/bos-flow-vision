"""BOS optics: analytic displacement, projection, smearing, warp, and sign conventions."""

import numpy as np
import pytest

from bosflow.generate import object_grid, render
from bosflow.optics.background import random_dot_pattern
from bosflow.optics.camera import Camera
from bosflow.optics.deflection import Optics, displacement_from_projection, pillbox_kernel
from bosflow.optics.warp import block_mean, warp_backward
from bosflow.physics.fields import LinearGradientFlow, abel_projection_table

PATTERN = {"dot_density": 0.08, "dot_diameter_px": 2.5}
NOISELESS = Camera(psf_sigma_px=0.0, vignetting=0.0, full_well_e=1e12, read_noise_e=0.0)


def test_linear_gradient_gives_uniform_analytic_displacement():
    optics = Optics(image_size=(128, 160))
    gx, gy, W = 3.0, -5.0, 0.15  # kg/m^4, m
    flow = LinearGradientFlow(gx, gy, W)
    s = render(flow, optics, NOISELESS, PATTERN, np.random.default_rng(0), margin_px=16)
    c = optics.gain_px_per_rad * optics.gladstone_dale / optics.n0
    # d = -c grad P in image axes; rows run opposite to physical y.
    expected_col, expected_row = -c * W * gx, -c * W * (-gy)
    inner = (slice(8, -8), slice(8, -8))
    np.testing.assert_allclose(s.displacement[0][inner], expected_col, rtol=1e-6)
    np.testing.assert_allclose(s.displacement[1][inner], expected_row, rtol=1e-6)
    np.testing.assert_allclose(s.flow[0][inner], expected_col, rtol=1e-6)
    np.testing.assert_allclose(s.flow[1][inner], expected_row, rtol=1e-6)


def test_displacement_formula_by_hand():
    """Default setup, 1 kg/m^4 over 0.15 m: check against the closed form."""
    o = Optics()
    K, W = 2.26e-4, 0.15
    eps = K * W * 1.0 / (1.0 + K * 1.2)
    M_B = 0.1 / (1.7 - 0.1)
    expected_px = M_B * 0.5 * eps / 20e-6
    assert o.gain_px_per_rad * o.gladstone_dale / o.n0 * W == pytest.approx(expected_px)


def test_abel_projection_of_uniform_sphere():
    s, F = abel_projection_table(lambda r: np.where(r <= 1.0, 1.0, 0.0), 1.0)
    np.testing.assert_allclose(F, 2.0 * np.sqrt(1.0 - s**2), atol=1e-12)


def test_abel_projection_of_gaussian():
    s, F = abel_projection_table(lambda r: np.exp(-(r / 0.1) ** 2), 1.0)
    np.testing.assert_allclose(F, np.sqrt(np.pi) * 0.1 * np.exp(-(s / 0.1) ** 2), atol=1e-9)


def test_pillbox_is_normalized_disk():
    k = pillbox_kernel(9.0)
    assert k.sum() == pytest.approx(1.0)
    assert k[k.shape[0] // 2, 0] == 0.0  # corner row edge outside radius 4.5 at distance 5


def test_step_smearing_width_matches_aperture_footprint():
    """A density step is smeared into the chord profile of the aperture disk.

    The chord length of a disk of diameter D has FWHM sqrt(3)/2 D.
    """
    o = Optics()
    ds = o.object_pixel_size / 4
    n = 512
    P = np.zeros((64, n))
    P[:, n // 2:] = 1e-3
    d = displacement_from_projection(P, ds, o)
    prof = np.abs(d[0][32])
    half = prof > 0.5 * prof.max()
    fwhm_samples = half.sum()
    D_samples = o.blur_diameter / ds
    assert fwhm_samples == pytest.approx(np.sqrt(3) / 2 * D_samples, abs=2.0)


def test_pattern_moves_toward_lower_density():
    """A low-density core acts as a diverging lens: the pattern shifts toward it."""
    o = Optics()
    ds = o.object_pixel_size
    y, x = np.mgrid[-64:64, -64:64] * ds
    P = -np.exp(-(x**2 + y**2) / (20 * ds) ** 2) * 1e-3
    d = displacement_from_projection(P, ds, o)
    assert d[0][64, 64 + 15] < 0.0  # right of center moves left
    assert d[1][64 + 15, 64] < 0.0  # below center moves up


def test_warp_matches_exact_shift():
    rng = np.random.default_rng(3)
    ss = 4
    B = random_dot_pattern((256, 256), ss, 0.08, 2.5, rng)
    d = np.zeros((2, 256, 256))
    d[0] = 0.75  # output pixels = 3 supersamples
    d[1] = -0.5  # = -2 supersamples
    warped = warp_backward(B, d, ss)
    exact = np.roll(B, shift=(-2, 3), axis=(0, 1))
    inner = (slice(16, -16), slice(16, -16))
    assert np.abs(warped[inner] - exact[inner]).max() < 1e-6


def test_object_grid_scale_and_orientation():
    o = Optics(image_size=(64, 80))
    x, y = object_grid(o, 1, 0)
    assert x[0, 1] - x[0, 0] == pytest.approx(o.object_pixel_size)
    assert y[1, 0] < y[0, 0]  # rows run downward, y up
    assert x.mean() == pytest.approx(0.0, abs=1e-12)
    x4, _ = object_grid(o, 4, 0)
    assert block_mean(x4, 4) == pytest.approx(x)
