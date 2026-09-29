"""Compressible-flow relations for a calorically perfect gas.

All angles are in radians unless a name ends in ``_deg``. Relations follow
NACA Report 1135 (Ames Research Staff, 1953) and Anderson, *Modern Compressible
Flow*, 3rd ed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

GAMMA_AIR = 1.4


# ---------------------------------------------------------------------------
# Isentropic and normal-shock relations
# ---------------------------------------------------------------------------


def mach_angle(M: float) -> float:
    return float(np.arcsin(1.0 / M))


def isentropic_density_ratio(M, gamma: float = GAMMA_AIR):
    """rho / rho_0 (stagnation) at Mach number M."""
    return (1.0 + 0.5 * (gamma - 1.0) * np.asarray(M) ** 2) ** (-1.0 / (gamma - 1.0))


def normal_shock_density_ratio(M1, gamma: float = GAMMA_AIR):
    M1 = np.asarray(M1, dtype=float)
    return (gamma + 1.0) * M1**2 / ((gamma - 1.0) * M1**2 + 2.0)


def normal_shock_pressure_ratio(M1, gamma: float = GAMMA_AIR):
    M1 = np.asarray(M1, dtype=float)
    return 1.0 + 2.0 * gamma / (gamma + 1.0) * (M1**2 - 1.0)


def normal_shock_downstream_mach(M1, gamma: float = GAMMA_AIR):
    M1 = np.asarray(M1, dtype=float)
    return np.sqrt((1.0 + 0.5 * (gamma - 1.0) * M1**2) / (gamma * M1**2 - 0.5 * (gamma - 1.0)))


# ---------------------------------------------------------------------------
# Oblique shocks (theta-beta-M)
# ---------------------------------------------------------------------------


def theta_from_beta(M: float, beta: float, gamma: float = GAMMA_AIR) -> float:
    """Flow deflection angle for shock angle ``beta`` (theta-beta-M relation)."""
    num = M**2 * np.sin(beta) ** 2 - 1.0
    den = M**2 * (gamma + np.cos(2.0 * beta)) + 2.0
    return float(np.arctan(2.0 / np.tan(beta) * num / den))


def max_deflection(M: float, gamma: float = GAMMA_AIR) -> tuple[float, float]:
    """(theta_max, beta_at_theta_max) for attached oblique shocks."""
    mu = mach_angle(M)
    betas = np.linspace(mu, 0.5 * np.pi, 20001)
    thetas = np.array([theta_from_beta(M, b, gamma) for b in betas])
    i = int(np.argmax(thetas))
    return float(thetas[i]), float(betas[i])


def beta_from_theta(M: float, theta: float, gamma: float = GAMMA_AIR, weak: bool = True) -> float:
    """Shock angle for a wedge of half-angle ``theta``. Raises if the shock detaches."""
    if theta <= 0.0:
        return mach_angle(M)
    theta_max, beta_max = max_deflection(M, gamma)
    if theta > theta_max:
        raise ValueError(f"theta={np.degrees(theta):.2f} deg exceeds detachment "
                         f"limit {np.degrees(theta_max):.2f} deg at M={M}")
    f = lambda b: theta_from_beta(M, b, gamma) - theta
    if weak:
        return float(brentq(f, mach_angle(M) + 1e-12, beta_max, xtol=1e-14))
    return float(brentq(f, beta_max, 0.5 * np.pi, xtol=1e-14))


@dataclass(frozen=True)
class ObliqueShock:
    M1: float
    theta: float
    beta: float
    density_ratio: float
    pressure_ratio: float
    M2: float


def oblique_shock(M1: float, theta: float, gamma: float = GAMMA_AIR) -> ObliqueShock:
    beta = beta_from_theta(M1, theta, gamma)
    Mn1 = M1 * np.sin(beta)
    Mn2 = float(normal_shock_downstream_mach(Mn1, gamma))
    return ObliqueShock(
        M1=M1,
        theta=theta,
        beta=beta,
        density_ratio=float(normal_shock_density_ratio(Mn1, gamma)),
        pressure_ratio=float(normal_shock_pressure_ratio(Mn1, gamma)),
        M2=Mn2 / np.sin(beta - theta),
    )


# ---------------------------------------------------------------------------
# Prandtl-Meyer expansion
# ---------------------------------------------------------------------------


def prandtl_meyer(M, gamma: float = GAMMA_AIR):
    """Prandtl-Meyer function nu(M) in radians."""
    M = np.asarray(M, dtype=float)
    g = np.sqrt((gamma + 1.0) / (gamma - 1.0))
    m = np.sqrt(M**2 - 1.0)
    return g * np.arctan(m / g) - np.arctan(m)


def inverse_prandtl_meyer(nu: float, gamma: float = GAMMA_AIR) -> float:
    nu_max = 0.5 * np.pi * (np.sqrt((gamma + 1.0) / (gamma - 1.0)) - 1.0)
    if not 0.0 <= nu < nu_max:
        raise ValueError(f"nu={nu} outside [0, {nu_max})")
    if nu == 0.0:
        return 1.0
    return float(brentq(lambda M: prandtl_meyer(M, gamma) - nu, 1.0, 1e4, xtol=1e-13))


# ---------------------------------------------------------------------------
# Taylor-Maccoll conical flow
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConeFlow:
    """Solution of the Taylor-Maccoll equation for a sharp cone at zero incidence.

    ``theta_grid`` runs from the shock angle down to the cone half-angle, and
    ``density_ratio`` is rho / rho_1 (freestream) on each ray.
    """

    M1: float
    cone_half_angle: float
    shock_angle: float
    theta_grid: np.ndarray
    density_ratio: np.ndarray
    mach: np.ndarray

    @property
    def surface_mach(self) -> float:
        return float(self.mach[-1])

    def density_at(self, theta) -> np.ndarray:
        """rho / rho_1 on rays at polar angle ``theta`` (1 outside the shock)."""
        theta = np.asarray(theta, dtype=float)
        # theta_grid is decreasing; np.interp needs increasing abscissae.
        out = np.interp(theta, self.theta_grid[::-1], self.density_ratio[::-1])
        return np.where(theta > self.shock_angle, 1.0, out)


def _taylor_maccoll_rhs(theta, y, gamma):
    vr, vt = y
    a2 = 0.5 * (gamma - 1.0) * (1.0 - vr**2 - vt**2)
    dvt = (vt**2 * vr - a2 * (2.0 * vr + vt / np.tan(theta))) / (a2 - vt**2)
    return [vt, dvt]


def _integrate_from_shock(M1: float, beta: float, gamma: float):
    """Integrate Taylor-Maccoll inward from a shock at angle ``beta``.

    Velocities are normalized by the maximum velocity V_max. Returns the
    solve_ivp solution, the post-shock state, and the ray where v_theta = 0.
    """
    delta = theta_from_beta(M1, beta, gamma)
    Mn1 = M1 * np.sin(beta)
    Mn2 = float(normal_shock_downstream_mach(Mn1, gamma))
    M2 = Mn2 / np.sin(beta - delta)
    V2 = (2.0 / ((gamma - 1.0) * M2**2) + 1.0) ** -0.5
    vr0 = V2 * np.cos(beta - delta)
    vt0 = -V2 * np.sin(beta - delta)

    def hit_cone(theta, y, gamma):  # v_theta crosses zero on the cone surface
        return y[1]

    hit_cone.terminal = True
    hit_cone.direction = 1
    sol = solve_ivp(
        _taylor_maccoll_rhs, (beta, 1e-4), [vr0, vt0], args=(gamma,),
        events=hit_cone, rtol=1e-11, atol=1e-13, dense_output=True, max_step=1e-3,
    )
    theta_c = float(sol.t_events[0][0]) if sol.t_events[0].size else 0.0
    return sol, Mn1, theta_c


def cone_shock_angle(M1: float, cone_half_angle: float, gamma: float = GAMMA_AIR) -> float:
    """Shock angle for a cone of given half-angle (attached, weak branch)."""
    mu = mach_angle(M1)
    f = lambda b: _integrate_from_shock(M1, b, gamma)[2] - cone_half_angle
    # Cone angle grows monotonically with shock angle up to detachment.
    hi = mu + 1e-3
    step = np.radians(0.5)
    while f(hi) < 0.0:
        hi += step
        if hi >= 0.5 * np.pi:
            raise ValueError("cone shock detached")
    return float(brentq(f, max(mu + 1e-9, hi - step), hi, xtol=1e-12))


def taylor_maccoll(M1: float, cone_half_angle: float, gamma: float = GAMMA_AIR,
                   n: int = 400) -> ConeFlow:
    beta = cone_shock_angle(M1, cone_half_angle, gamma)
    sol, Mn1, theta_c = _integrate_from_shock(M1, beta, gamma)
    thetas = np.linspace(beta, theta_c, n)
    vr, vt = sol.sol(thetas)
    V2 = vr**2 + vt**2
    mach = np.sqrt(2.0 / (gamma - 1.0) * V2 / (1.0 - V2))
    # Post-shock flow is isentropic: rho/rho_1 = (rho_2/rho_1) * (rho/rho_02)/(rho_2/rho_02).
    rho2_rho1 = float(normal_shock_density_ratio(Mn1, gamma))
    rho_rho02 = (1.0 - V2) ** (1.0 / (gamma - 1.0))
    density = rho2_rho1 * rho_rho02 / rho_rho02[0]
    return ConeFlow(M1, cone_half_angle, beta, thetas, density, mach)


# ---------------------------------------------------------------------------
# Sedov-Taylor point blast (spherical, uniform ambient, strong shock)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SedovSolution:
    """Self-similar strong-blast profile.

    With R(t) = xi0 * (E t^2 / rho0)^(1/5), the fields behind the shock are
    rho = rho0 G(xi), u = Rdot U(xi), p = rho0 Rdot^2 P(xi), with xi = r / R.
    """

    gamma: float
    xi0: float
    xi: np.ndarray  # increasing, ends at 1
    G: np.ndarray
    U: np.ndarray
    P: np.ndarray

    def radius(self, E: float, rho0: float, t):
        return self.xi0 * (E * np.asarray(t, dtype=float) ** 2 / rho0) ** 0.2

    def time_at_radius(self, E: float, rho0: float, R):
        return np.sqrt((np.asarray(R, dtype=float) / self.xi0) ** 5 * rho0 / E)

    def density_ratio(self, xi) -> np.ndarray:
        """rho / rho0 as a function of xi = r / R (1 outside the shock)."""
        xi = np.asarray(xi, dtype=float)
        out = np.interp(xi, self.xi, self.G, left=self.G[0])
        return np.where(xi > 1.0, 1.0, out)


def _sedov_rhs(xi, y, gamma):
    G, U, P = y
    s = U - xi
    A = np.array([
        [s, G, 0.0],
        [0.0, s, 1.0 / G],
        [-gamma * P * s / G, 0.0, s],
    ])
    b = np.array([-2.0 * G * U / xi, 1.5 * U, 3.0 * P])
    return np.linalg.solve(A, b)


def sedov_taylor(gamma: float = GAMMA_AIR, xi_min: float = 1e-3) -> SedovSolution:
    """Integrate the spherical Sedov-Taylor similarity ODEs from the shock inward."""
    y0 = [(gamma + 1.0) / (gamma - 1.0), 2.0 / (gamma + 1.0), 2.0 / (gamma + 1.0)]
    sol = solve_ivp(_sedov_rhs, (1.0, xi_min), y0, args=(gamma,), method="LSODA",
                    rtol=1e-10, atol=1e-14, dense_output=True)
    xi = np.concatenate([np.geomspace(xi_min, 0.5, 400), np.linspace(0.5, 1.0, 1601)[1:]])
    G, U, P = sol.sol(xi)
    G = np.clip(G, 0.0, None)
    # Total energy E = 4 pi rho0 Rdot^2 R^3 * I with Rdot = (2/5) R / t.
    integrand = (0.5 * G * U**2 + P / (gamma - 1.0)) * xi**2
    I = float(np.trapezoid(integrand, xi))
    xi0 = (25.0 / (16.0 * np.pi * I)) ** 0.2
    return SedovSolution(gamma, xi0, xi, G, U, P)
