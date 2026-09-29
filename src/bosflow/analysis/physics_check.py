"""Physics from segmentation masks: shock angles to Mach number, blast radii to energy.

These close the loop from pixels back to gas dynamics:

- An attached shock on a wedge (half-angle theta) or cone at a known angle of the
  body gives the freestream Mach number by inverting theta-beta-M (or the
  Taylor-Maccoll solution) at the measured shock angle beta.
- A spherical blast front R(t) should follow the Sedov-Taylor law R = xi0 (E t^2 /
  rho0)^(1/5); fitting the exponent tests self-similarity, and the prefactor gives
  the released energy E for known ambient density.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from bosflow.analysis.shock_angle import fit_line_angle, shock_angle_relative
from bosflow.physics import gasdynamics as gd


def mask_shock_angle(mask: np.ndarray, freestream_angle_deg: float,
                     weight: np.ndarray | None = None) -> float:
    """Acute shock angle (deg) of a straight-shock mask relative to the freestream."""
    w = weight if weight is not None else mask.astype(float)
    return shock_angle_relative(fit_line_angle(w, mask, power=1.0), freestream_angle_deg)


def mach_from_wedge_shock(beta_deg: float, theta_deg: float, gamma: float = gd.GAMMA_AIR,
                          m_max: float = 20.0) -> float:
    """Freestream Mach number with weak attached shock angle beta on a wedge theta.

    theta(M) at fixed beta rises monotonically with M, so the root is unique.
    """
    beta, theta = np.radians(beta_deg), np.radians(theta_deg)
    m_min = 1.0 / np.sin(beta) + 1e-9  # the shock must be at least a Mach wave
    f = lambda M: gd.theta_from_beta(M, beta, gamma) - theta
    if f(m_max) < 0:
        raise ValueError("no weak-shock solution: beta too small for this wedge")
    return float(brentq(f, m_min, m_max, xtol=1e-10))


def mach_from_cone_shock(beta_deg: float, cone_deg: float, gamma: float = gd.GAMMA_AIR,
                         m_lo: float = 1.05, m_hi: float = 12.0) -> float:
    """Freestream Mach number giving Taylor-Maccoll shock angle beta on a cone."""
    beta, cone = np.radians(beta_deg), np.radians(cone_deg)

    def f(M):
        try:
            return gd.cone_shock_angle(M, cone, gamma) - beta
        except ValueError:  # detached at this Mach number: shock is effectively normal
            return np.pi / 2 - beta

    # A shock is never shallower than the Mach angle, so M > 1/sin(beta): at that bound
    # the cone shock is steeper than beta (f > 0), and at high Mach it is shallower.
    lo = max(m_lo, 1.0 / np.sin(beta) + 1e-6)
    if f(lo) > 0 > f(m_hi):
        return float(brentq(f, lo, m_hi, xtol=1e-8))
    # Fallback: bracket the sign change on a coarse grid.
    grid = np.geomspace(m_lo, m_hi, 40)
    vals = [f(M) for M in grid]
    for a, b, fa, fb in zip(grid[:-1], grid[1:], vals[:-1], vals[1:]):
        if fa >= 0 >= fb:
            return float(brentq(f, a, b, xtol=1e-8))
    raise ValueError("shock angle outside the attached-cone range")


def fit_circle(mask: np.ndarray, weight: np.ndarray | None = None):
    """Least-squares (Kasa) circle through mask pixels: (x_center, y_center, radius) px."""
    rows, cols = np.nonzero(mask)
    w = np.ones(rows.size) if weight is None else weight[rows, cols]
    x, y = cols.astype(float), rows.astype(float)
    A = np.stack([x, y, np.ones_like(x)], axis=1) * np.sqrt(w)[:, None]
    b = (x**2 + y**2) * np.sqrt(w)
    (c0, c1, c2), *_ = np.linalg.lstsq(A, b, rcond=None)
    xc, yc = 0.5 * c0, 0.5 * c1
    return float(xc), float(yc), float(np.sqrt(c2 + xc**2 + yc**2))


def circle_residual_ratio(mask: np.ndarray) -> float:
    """RMS distance of mask pixels from their fitted circle, over the radius.

    A shock front is a thin surface, so a valid blast-front mask has a small ratio
    (a band of width w gives about w / (sqrt(12) R)); a filled disk gives about 0.3.
    """
    xc, yc, R = fit_circle(mask)
    rows, cols = np.nonzero(mask)
    d = np.hypot(cols - xc, rows - yc) - R
    return float(np.sqrt(np.mean(d**2)) / max(R, 1e-9))


def power_law_fit(t: np.ndarray, R: np.ndarray) -> tuple[float, float]:
    """Fit R = A t^n in log space; returns (n, A)."""
    n, logA = np.polyfit(np.log(t), np.log(R), 1)
    return float(n), float(np.exp(logA))


def sedov_energy(t: np.ndarray, R: np.ndarray, rho0: float, gamma: float = gd.GAMMA_AIR) -> float:
    """Blast energy from R(t) with the exponent fixed at 2/5 (least squares on R^5 vs t^2)."""
    xi0 = gd.sedov_taylor(gamma).xi0
    # R^5 = xi0^5 E t^2 / rho0  ->  E = rho0 / xi0^5 * sum(R^5 t^2) / sum(t^4)
    return float(rho0 / xi0**5 * np.sum(R**5 * t**2) / np.sum(t**4))
