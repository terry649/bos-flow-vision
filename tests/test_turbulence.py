import numpy as np
import pytest

from bosflow.physics.turbulence import radial_psd, von_karman_field


def test_von_karman_field_has_unit_rms_and_kolmogorov_slope():
    rng = np.random.default_rng(11)
    dx, L0 = 1.0, 64.0
    f = von_karman_field((1024, 1024), dx, L0, rng)
    assert f.mean() == pytest.approx(0.0, abs=1e-12)
    assert f.std() == pytest.approx(1.0, rel=1e-12)
    k, p = radial_psd(f, dx)
    k0 = 2 * np.pi / L0
    inertial = (k > 10 * k0) & (k < 0.5 * np.pi / dx)
    slope = np.polyfit(np.log(k[inertial]), np.log(p[inertial]), 1)[0]
    assert slope == pytest.approx(-11.0 / 3.0, abs=0.15)


def test_von_karman_field_is_seeded():
    a = von_karman_field((64, 64), 1.0, 16.0, np.random.default_rng(5))
    b = von_karman_field((64, 64), 1.0, 16.0, np.random.default_rng(5))
    np.testing.assert_array_equal(a, b)
