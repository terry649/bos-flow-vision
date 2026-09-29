"""Gas-dynamics relations against published values.

References
- NACA Report 1135 (Ames Research Staff, 1953): normal-shock table, oblique-shock
  and cone charts, Prandtl-Meyer table.
- Anderson, Modern Compressible Flow, 3rd ed., Appendix A-C.
- Taylor (1950), Proc. R. Soc. A 201:175, and Sedov (1959): blast constant.
- Cone shock angles cross-checked against pygasflow 2.x
  (conical_shockwave_solver) and the NASA value 0.545 rad for M=2, 10 deg.
"""

import numpy as np
import pytest

from bosflow.physics import gasdynamics as gd

deg, rad = np.degrees, np.radians


@pytest.mark.parametrize("M, rho, p, M2", [
    (2.0, 2.6667, 4.5000, 0.5774),
    (3.0, 3.8571, 10.333, 0.4752),
    (1.5, 1.8621, 2.4583, 0.7011),
])
def test_normal_shock_table(M, rho, p, M2):
    assert gd.normal_shock_density_ratio(M) == pytest.approx(rho, abs=2e-4)
    assert gd.normal_shock_pressure_ratio(M) == pytest.approx(p, abs=2e-3)
    assert gd.normal_shock_downstream_mach(M) == pytest.approx(M2, abs=2e-4)


def test_normal_shock_strong_limit():
    assert gd.normal_shock_density_ratio(1e4) == pytest.approx(6.0, rel=1e-6)


def test_oblique_shock_M2_wedge10():
    assert deg(gd.beta_from_theta(2.0, rad(10.0))) == pytest.approx(39.31, abs=0.02)


def test_oblique_shock_M3_wedge20():
    assert deg(gd.beta_from_theta(3.0, rad(20.0))) == pytest.approx(37.76, abs=0.02)


def test_theta_beta_roundtrip_and_mach_wave_limit():
    for M in (1.5, 2.0, 4.0):
        for th in (2.0, 8.0, 15.0):
            if rad(th) >= gd.max_deflection(M)[0]:
                continue
            b = gd.beta_from_theta(M, rad(th))
            assert deg(gd.theta_from_beta(M, b)) == pytest.approx(th, abs=1e-9)
        assert gd.beta_from_theta(M, 1e-8) == pytest.approx(gd.mach_angle(M), abs=1e-5)


def test_detachment_angle_M2():
    assert deg(gd.max_deflection(2.0)[0]) == pytest.approx(22.97, abs=0.02)
    with pytest.raises(ValueError):
        gd.beta_from_theta(2.0, rad(25.0))


@pytest.mark.parametrize("M, nu", [(2.0, 26.38), (3.0, 49.76), (1.5, 11.91)])
def test_prandtl_meyer(M, nu):
    assert deg(gd.prandtl_meyer(M)) == pytest.approx(nu, abs=0.01)
    assert gd.inverse_prandtl_meyer(rad(nu)) == pytest.approx(M, abs=2e-3)


@pytest.mark.parametrize("M, cone, beta", [(2.0, 10.0, 31.21), (2.0, 20.0, 37.80),
                                           (3.0, 15.0, 25.26), (5.0, 30.0, 35.60)])
def test_taylor_maccoll_shock_angle(M, cone, beta):
    assert deg(gd.cone_shock_angle(M, rad(cone))) == pytest.approx(beta, abs=0.05)


def test_cone_shock_weaker_than_wedge_and_mach_limit():
    M = 2.5
    for th in (5.0, 15.0):
        assert gd.cone_shock_angle(M, rad(th)) < gd.beta_from_theta(M, rad(th))
    assert deg(gd.cone_shock_angle(M, rad(0.5))) == pytest.approx(
        deg(gd.mach_angle(M)), abs=0.1)


def test_taylor_maccoll_field_is_isentropic_and_monotone():
    sol = gd.taylor_maccoll(2.0, rad(15.0))
    # Density rises monotonically from the shock to the cone surface.
    assert np.all(np.diff(sol.density_ratio) >= -1e-12)
    # Just behind the shock the density jump equals the normal-shock value on Mn1.
    Mn1 = 2.0 * np.sin(sol.shock_angle)
    assert sol.density_ratio[0] == pytest.approx(gd.normal_shock_density_ratio(Mn1), rel=1e-9)
    assert sol.surface_mach < 2.0


def test_sedov_taylor_constant_and_shock_state():
    s = gd.sedov_taylor(1.4)
    assert s.xi0 == pytest.approx(1.033, abs=2e-3)
    assert s.G[-1] == pytest.approx(6.0, rel=1e-9)
    # Near the center the gas is evacuated.
    assert s.density_ratio(0.2) < 1e-2
    # R ~ t^(2/5)
    r1, r2 = s.radius(1e3, 1.2, 1e-5), s.radius(1e3, 1.2, 2e-5)
    assert r2 / r1 == pytest.approx(2.0**0.4, rel=1e-12)
