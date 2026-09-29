"""M4 physics checks on ground-truth masks: the pipeline must recover known physics."""

import numpy as np
import pytest

from bosflow import generate as G
from bosflow.analysis import physics_check as P
from bosflow.optics.camera import Camera
from bosflow.optics.deflection import Optics
from bosflow.physics import fields as F
from bosflow.physics import gasdynamics as gd

PATTERN = {"dot_density": 0.08, "dot_diameter_px": 2.5}


@pytest.mark.parametrize("M, theta", [(1.8, 8.0), (2.5, 12.0), (4.0, 15.0)])
def test_mach_from_wedge_shock_inverts_theta_beta_m(M, theta):
    beta = np.degrees(gd.beta_from_theta(M, np.radians(theta)))
    assert P.mach_from_wedge_shock(beta, theta) == pytest.approx(M, rel=1e-6)


@pytest.mark.parametrize("M, cone", [(2.0, 10.0), (3.0, 15.0)])
def test_mach_from_cone_shock_inverts_taylor_maccoll(M, cone):
    beta = np.degrees(gd.cone_shock_angle(M, np.radians(cone)))
    assert P.mach_from_cone_shock(beta, cone) == pytest.approx(M, rel=1e-4)


def test_mach_recovered_from_rendered_wedge_mask():
    """End to end on a rendered scene: GT band mask + GT magnitude -> Mach within 3%."""
    o = Optics()
    flow = F.WedgeFlow(2.5, np.radians(10.0), 0.05, 0.15,
                       F.Pose(-0.3 * o.field_of_view[1], 0.0, np.radians(12.0)))
    s = G.render(flow, o, Camera(), PATTERN, np.random.default_rng(0))
    mag = np.hypot(*s.flow)
    for inst in s.instances:
        beta = P.mask_shock_angle(inst.mask & s.valid, 12.0, weight=mag**2)
        assert P.mach_from_wedge_shock(beta, 10.0) == pytest.approx(2.5, rel=0.03)


def test_circle_fit_on_annulus():
    yy, xx = np.mgrid[0:256, 0:256]
    r = np.hypot(xx - 120.3, yy - 131.7)
    mask = np.abs(r - 60.0) < 3.0
    xc, yc, R = P.fit_circle(mask)
    assert (xc, yc) == pytest.approx((120.3, 131.7), abs=0.1)
    assert R == pytest.approx(60.0, abs=0.2)


def test_power_law_and_sedov_energy_recover_truth():
    sol = gd.sedov_taylor()
    E, rho0 = 250.0, 0.1
    t = np.linspace(2e-5, 1e-4, 8)
    R = sol.radius(E, rho0, t)
    n, _ = P.power_law_fit(t, R)
    assert n == pytest.approx(0.4, abs=1e-9)
    assert P.sedov_energy(t, R, rho0) == pytest.approx(E, rel=1e-9)


def test_thin_front_gate_separates_ring_from_filled_disk():
    yy, xx = np.mgrid[0:256, 0:256]
    r = np.hypot(xx - 128, yy - 128)
    ring = np.abs(r - 80.0) < 3.0
    disk = r < 80.0
    clipped_ring = ring & (xx < 190)  # an arc cut by the image edge is still thin
    # A band of width w = 6 px at R = 80 px: w / (sqrt(12) R) = 0.0217.
    assert P.circle_residual_ratio(ring) == pytest.approx(6 / (np.sqrt(12) * 80), rel=0.05)
    assert P.circle_residual_ratio(clipped_ring) < 0.05
    assert P.circle_residual_ratio(disk) > 0.2
