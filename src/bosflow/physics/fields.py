"""Density fields of canonical compressible flows, integrated along the line of sight.

Every flow returns the path-integrated density perturbation

    P(x, y) = integral of (rho - rho_ref) dz        [kg / m^2]

on object-plane coordinates (x to the right, y up, meters), which is all a BOS
measurement sees. Planar flows are extruded over a span ``W``; axisymmetric flows
use a forward Abel transform. Each flow also reports opaque bodies and the
labeled flow-feature instances (shock, expansion_fan, shear_layer).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from bosflow.physics import gasdynamics as gd
from bosflow.physics.turbulence import von_karman_field

CLASSES = ("shock", "expansion_fan", "shear_layer")


@dataclass(frozen=True)
class Pose:
    """Placement of a flow's local frame in the object plane."""

    x0: float = 0.0
    y0: float = 0.0
    angle: float = 0.0  # rotation of the local +x axis, radians, counterclockwise

    def to_local(self, x, y):
        c, s = np.cos(self.angle), np.sin(self.angle)
        dx, dy = x - self.x0, y - self.y0
        return c * dx + s * dy, -s * dx + c * dy


@dataclass
class Instance:
    cls: str
    mask: np.ndarray  # bool, same shape as the grid it was computed on
    attrs: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abel projection helper
# ---------------------------------------------------------------------------

_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(256)


def abel_projection_table(g, r_support: float, n: int = 3000):
    """Tabulate F(s) = integral of g(sqrt(s^2 + z^2)) dz over all z, for 0 <= s <= r_support.

    ``g`` is the (vectorized) radial perturbation profile, zero beyond ``r_support``.
    A jump at ``r_support`` sits at the integration endpoint, so Gauss-Legendre on
    [0, sqrt(r_support^2 - s^2)] stays accurate. Nodes cluster near the support edge
    where F has a square-root behavior.
    """
    u = np.linspace(0.0, 1.0, n)
    s = r_support * (1.0 - (1.0 - u) ** 2)
    zmax = np.sqrt(np.clip(r_support**2 - s**2, 0.0, None))
    z = 0.5 * zmax[:, None] * (_GL_NODES[None, :] + 1.0)
    r = np.sqrt(s[:, None] ** 2 + z**2)
    F = 2.0 * (0.5 * zmax) * (g(r) * _GL_WEIGHTS[None, :]).sum(axis=1)
    return s, F


def _interp_table(s_query, s_tab, F_tab):
    return np.interp(s_query, s_tab, F_tab, right=0.0)


def _band_along_ray(xl, yl, direction: float, half_width: float):
    """Pixels within ``half_width`` of the ray from the local origin at angle ``direction``."""
    ux, uy = np.cos(direction), np.sin(direction)
    along = xl * ux + yl * uy
    across = -xl * uy + yl * ux
    return (along > 0.0) & (np.abs(across) <= half_width)


# ---------------------------------------------------------------------------
# Flows
# ---------------------------------------------------------------------------


class Flow:
    flow_type: str = ""

    def projected_density(self, x, y) -> np.ndarray:
        raise NotImplementedError

    def body(self, x, y) -> np.ndarray:
        return np.zeros(np.shape(x), dtype=bool)

    def instances(self, x, y, half_width: float) -> list[Instance]:
        raise NotImplementedError

    def metadata(self) -> dict:
        raise NotImplementedError


@dataclass
class WedgeFlow(Flow):
    """Symmetric 2D wedge in uniform supersonic flow: attached oblique shocks.

    The apex is at the pose origin and the freestream points along local +x.
    """

    M1: float
    half_angle: float
    rho1: float
    span: float
    pose: Pose = Pose()
    gamma: float = gd.GAMMA_AIR
    flow_type: str = "wedge"

    def __post_init__(self):
        self.shock = gd.oblique_shock(self.M1, self.half_angle, self.gamma)

    def projected_density(self, x, y):
        xl, yl = self.pose.to_local(x, y)
        post = (xl > 0.0) & (np.abs(yl) < xl * np.tan(self.shock.beta))
        return np.where(post, self.span * self.rho1 * (self.shock.density_ratio - 1.0), 0.0)

    def body(self, x, y):
        xl, yl = self.pose.to_local(x, y)
        return (xl > 0.0) & (np.abs(yl) < xl * np.tan(self.half_angle))

    def instances(self, x, y, half_width):
        xl, yl = self.pose.to_local(x, y)
        out = []
        for side, sign in (("upper", 1.0), ("lower", -1.0)):
            m = _band_along_ray(xl, yl, sign * self.shock.beta, half_width)
            out.append(Instance("shock", m, {"side": side}))
        return out

    def metadata(self):
        return {
            "flow_type": self.flow_type, "M1": self.M1, "gamma": self.gamma,
            "wedge_half_angle_deg": float(np.degrees(self.half_angle)),
            "shock_angle_deg": float(np.degrees(self.shock.beta)),
            "density_ratio": self.shock.density_ratio, "rho1": self.rho1, "span": self.span,
            "pose": vars(self.pose),
        }


@dataclass
class ConeFlow(Flow):
    """Sharp cone at zero incidence (Taylor-Maccoll), viewed side-on, axis along local +x."""

    M1: float
    half_angle: float
    rho1: float
    pose: Pose = Pose()
    gamma: float = gd.GAMMA_AIR
    flow_type: str = "cone"

    def __post_init__(self):
        self.solution = gd.taylor_maccoll(self.M1, self.half_angle, self.gamma)
        self._table = _cone_table(self.M1, self.half_angle, self.gamma)

    def projected_density(self, x, y):
        xl, yl = self.pose.to_local(x, y)
        xp = np.where(xl > 0.0, xl, 1.0)
        F = _interp_table(np.abs(yl) / xp, *self._table)
        return np.where(xl > 0.0, self.rho1 * xp * F, 0.0)

    def body(self, x, y):
        xl, yl = self.pose.to_local(x, y)
        return (xl > 0.0) & (np.abs(yl) < xl * np.tan(self.half_angle))

    def instances(self, x, y, half_width):
        xl, yl = self.pose.to_local(x, y)
        b = self.solution.shock_angle
        m = _band_along_ray(xl, yl, b, half_width) | _band_along_ray(xl, yl, -b, half_width)
        return [Instance("shock", m, {"side": "conical"})]

    def metadata(self):
        return {
            "flow_type": self.flow_type, "M1": self.M1, "gamma": self.gamma,
            "cone_half_angle_deg": float(np.degrees(self.half_angle)),
            "shock_angle_deg": float(np.degrees(self.solution.shock_angle)),
            "surface_mach": self.solution.surface_mach, "rho1": self.rho1,
            "pose": vars(self.pose),
        }


@lru_cache(maxsize=64)
def _cone_table(M1, half_angle, gamma):
    sol = gd.taylor_maccoll(M1, half_angle, gamma)
    tan_b = np.tan(sol.shock_angle)
    tan_c = np.tan(half_angle)

    def g(eta):  # rho/rho1 - 1 on the ray with r/x = eta; inside the body use the surface value
        theta = np.arctan(np.maximum(eta, tan_c))
        return sol.density_at(theta) - 1.0

    return abel_projection_table(g, tan_b)


@dataclass
class BlastFlow(Flow):
    """Spherical Sedov-Taylor point blast at time ``t`` after release of energy ``E``."""

    E: float
    rho0: float
    t: float
    p0: float = 101325.0  # ambient pressure, used only to report the shock Mach number
    pose: Pose = Pose()
    gamma: float = gd.GAMMA_AIR
    flow_type: str = "blast"

    def __post_init__(self):
        self.solution = _sedov(self.gamma)
        self.R = float(self.solution.radius(self.E, self.rho0, self.t))
        self._table = _blast_table(self.gamma)

    def projected_density(self, x, y):
        xl, yl = self.pose.to_local(x, y)
        s = np.hypot(xl, yl) / self.R
        return self.rho0 * self.R * _interp_table(s, *self._table)

    def instances(self, x, y, half_width):
        xl, yl = self.pose.to_local(x, y)
        m = np.abs(np.hypot(xl, yl) - self.R) <= half_width
        return [Instance("shock", m, {"side": "spherical"})]

    @property
    def shock_speed(self) -> float:
        return 0.4 * self.R / self.t

    @property
    def shock_mach(self) -> float:
        return self.shock_speed / np.sqrt(self.gamma * self.p0 / self.rho0)

    def metadata(self):
        return {
            "flow_type": self.flow_type, "E": self.E, "rho0": self.rho0, "p0": self.p0,
            "t": self.t, "gamma": self.gamma, "xi0": self.solution.xi0,
            "shock_radius": self.R, "shock_speed": self.shock_speed,
            "shock_mach": self.shock_mach, "pose": vars(self.pose),
        }


@lru_cache(maxsize=8)
def _sedov(gamma):
    return gd.sedov_taylor(gamma)


@lru_cache(maxsize=8)
def _blast_table(gamma):
    sol = _sedov(gamma)
    return abel_projection_table(lambda xi: sol.density_ratio(xi) - 1.0, 1.0)


@dataclass
class ExpansionFlow(Flow):
    """Centered Prandtl-Meyer expansion around a sharp convex corner.

    Upstream wall along local y = 0, x < 0 (flow above it, along +x); downstream
    wall turned clockwise by ``turn_angle`` at the pose origin.
    """

    M1: float
    turn_angle: float
    rho1: float
    span: float
    pose: Pose = Pose()
    gamma: float = gd.GAMMA_AIR
    flow_type: str = "expansion"

    def __post_init__(self):
        g = self.gamma
        self.nu1 = float(gd.prandtl_meyer(self.M1, g))
        self.M2 = gd.inverse_prandtl_meyer(self.nu1 + self.turn_angle, g)
        self.mu1 = gd.mach_angle(self.M1)
        self.mu2 = gd.mach_angle(self.M2)
        self.head = self.mu1  # polar angle of the leading characteristic
        self.tail = self.mu2 - self.turn_angle
        M = np.linspace(self.M1, self.M2, 2000)
        phi = np.arcsin(1.0 / M) - (gd.prandtl_meyer(M, g) - self.nu1)
        rho = gd.isentropic_density_ratio(M, g) / gd.isentropic_density_ratio(self.M1, g)
        self._phi, self._rho = phi[::-1], rho[::-1]  # increasing phi

    def density_ratio_local(self, xl, yl):
        phi = np.arctan2(yl, xl)
        r = np.interp(phi, self._phi, self._rho)  # clamps to rho2 below tail, 1 above head
        return np.where(phi > self.head, 1.0, r)

    def projected_density(self, x, y):
        xl, yl = self.pose.to_local(x, y)
        return self.span * self.rho1 * (self.density_ratio_local(xl, yl) - 1.0)

    def body(self, x, y):
        xl, yl = self.pose.to_local(x, y)
        return ((xl <= 0.0) & (yl < 0.0)) | ((xl > 0.0) & (yl < -xl * np.tan(self.turn_angle)))

    def instances(self, x, y, half_width):
        xl, yl = self.pose.to_local(x, y)
        phi = np.arctan2(yl, xl)
        m = (phi <= self.head) & (phi >= self.tail)
        # Widen thin fans to the optical smearing width near the corner.
        m |= _band_along_ray(xl, yl, self.head, half_width) & (phi > self.tail)
        m &= ~self.body(x, y)
        return [Instance("expansion_fan", m, {})]

    def metadata(self):
        return {
            "flow_type": self.flow_type, "M1": self.M1, "M2": self.M2, "gamma": self.gamma,
            "turn_angle_deg": float(np.degrees(self.turn_angle)),
            "head_angle_deg": float(np.degrees(self.head)),
            "tail_angle_deg": float(np.degrees(self.tail)),
            "rho1": self.rho1, "span": self.span, "pose": vars(self.pose),
        }


@dataclass
class ShearLayerFlow(Flow):
    """Planar mixing layer between streams of different density.

    Mean profile rho = rho_m + (d_rho / 2) tanh(2 y / delta_w(x)) with vorticity
    thickness delta_w = growth * (x - x_origin), plus path-integrated von Karman
    fluctuations confined to the layer by a sech^2 envelope.
    """

    rho_upper: float
    rho_lower: float
    growth: float  # d(delta_w)/dx
    x_origin: float  # virtual origin, local frame, meters (usually off-frame upstream)
    span: float
    fluct_intensity: float  # rms path-averaged fluctuation as a fraction of |d_rho|
    outer_scale_frac: float  # outer scale as a fraction of the local thickness
    seed: int
    pose: Pose = Pose()
    flow_type: str = "shear_layer"

    def delta_w(self, xl):
        return self.growth * np.maximum(xl - self.x_origin, 1e-6)

    def projected_density(self, x, y):
        xl, yl = self.pose.to_local(x, y)
        d_rho = self.rho_upper - self.rho_lower
        dw = self.delta_w(xl)
        mean = self.rho_lower + 0.5 * d_rho * (1.0 + np.tanh(2.0 * yl / dw))
        mean_ref = 0.5 * (self.rho_upper + self.rho_lower)
        env = 1.0 - np.tanh(2.0 * yl / dw) ** 2  # sech^2 without overflow
        dx = float(np.hypot(x[0, 1] - x[0, 0], y[0, 1] - y[0, 0]))
        rng = np.random.default_rng(self.seed)
        L0 = self.outer_scale_frac * float(np.median(dw))
        turb = von_karman_field(x.shape, dx, L0, rng)
        rho = mean - mean_ref + self.fluct_intensity * abs(d_rho) * env * turb
        return self.span * rho

    def instances(self, x, y, half_width):
        xl, yl = self.pose.to_local(x, y)
        m = (xl > self.x_origin) & (np.abs(yl) <= np.maximum(0.5 * self.delta_w(xl), half_width))
        return [Instance("shear_layer", m, {})]

    def metadata(self):
        return {
            "flow_type": self.flow_type, "rho_upper": self.rho_upper,
            "rho_lower": self.rho_lower, "growth": self.growth, "x_origin": self.x_origin,
            "span": self.span, "fluct_intensity": self.fluct_intensity,
            "outer_scale_frac": self.outer_scale_frac, "pose": vars(self.pose),
        }


@dataclass
class LinearGradientFlow(Flow):
    """Uniform density gradient across a planar span: the analytic BOS calibration case."""

    grad_x: float  # kg/m^4
    grad_y: float
    span: float
    flow_type: str = "linear_gradient"

    def projected_density(self, x, y):
        return self.span * (self.grad_x * x + self.grad_y * y)

    def instances(self, x, y, half_width):
        return []

    def metadata(self):
        return {"flow_type": self.flow_type, "grad_x": self.grad_x, "grad_y": self.grad_y,
                "span": self.span}


@dataclass
class CompositeFlow(Flow):
    """Several independent flows at different depths along the line of sight.

    Path integrals are linear, so the projected density perturbations add exactly;
    bodies are the union of the members' bodies and instances are concatenated.
    """

    members: list
    flow_type: str = "composite"

    def projected_density(self, x, y):
        return sum(m.projected_density(x, y) for m in self.members)

    def body(self, x, y):
        out = np.zeros(np.shape(x), dtype=bool)
        for m in self.members:
            out |= m.body(x, y)
        return out

    def instances(self, x, y, half_width):
        out = []
        for i, m in enumerate(self.members):
            for inst in m.instances(x, y, half_width):
                inst.attrs = dict(inst.attrs, member=i, member_flow_type=m.flow_type)
                out.append(inst)
        return out

    def metadata(self):
        return {"flow_type": self.flow_type, "members": [m.metadata() for m in self.members]}
