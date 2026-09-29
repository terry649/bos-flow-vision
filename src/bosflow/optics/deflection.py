"""BOS optical geometry and the map from path-integrated density to pixel displacement.

Geometry (camera focused on the background):

    camera lens  --Z_A-->  density object  --Z_D-->  background,   Z_B = Z_A + Z_D

Gladstone-Dale: n - 1 = K rho. For paraxial rays, the deflection angle is
eps = (1 / n0) grad( integral (n - 1) dz ) = (K / n0) grad P, where P is the
path-integrated density perturbation. The apparent background displacement is
Z_D eps in the background plane and M_B Z_D eps on the sensor, with background
magnification M_B = f / (Z_B - f) (Raffel 2015, Exp. Fluids 56:60).

Because the lens is focused on the background, each pixel collects a cone of rays
whose footprint in the object plane has diameter b = D Z_D / Z_B with D = f / N.
The measured deflection is therefore the object-plane deflection convolved with
a pillbox of that diameter (Goldhahn and Seume 2007, Exp. Fluids 43:241). This
smearing is what sets the apparent width of a shock in a BOS image.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import fftconvolve

GLADSTONE_DALE_AIR = 2.26e-4  # m^3/kg, visible light (~530 nm)


@dataclass(frozen=True)
class Optics:
    focal_length: float = 0.100  # m
    f_number: float = 16.0
    sensor_pixel_pitch: float = 5e-6  # m
    binning: int = 4  # 512 px images from a 2048 px sensor
    Z_A: float = 1.2  # lens to density object, m
    Z_D: float = 0.5  # density object to background, m
    image_size: tuple[int, int] = (512, 512)  # (rows, cols)
    gladstone_dale: float = GLADSTONE_DALE_AIR
    rho_ambient: float = 1.2  # sets n0 = 1 + K rho_ambient

    @property
    def pixel_pitch(self) -> float:
        return self.sensor_pixel_pitch * self.binning

    @property
    def Z_B(self) -> float:
        return self.Z_A + self.Z_D

    @property
    def image_distance(self) -> float:
        return self.focal_length * self.Z_B / (self.Z_B - self.focal_length)

    @property
    def background_magnification(self) -> float:
        return self.focal_length / (self.Z_B - self.focal_length)

    @property
    def object_pixel_size(self) -> float:
        """Object-plane footprint of one pixel's chief ray spacing, m."""
        return self.pixel_pitch * self.Z_A / self.image_distance

    @property
    def aperture_diameter(self) -> float:
        return self.focal_length / self.f_number

    @property
    def blur_diameter(self) -> float:
        """Object-plane footprint of the ray cone collected by one pixel, m."""
        return self.aperture_diameter * self.Z_D / self.Z_B

    @property
    def blur_diameter_px(self) -> float:
        return self.blur_diameter / self.object_pixel_size

    @property
    def n0(self) -> float:
        return 1.0 + self.gladstone_dale * self.rho_ambient

    @property
    def gain_px_per_rad(self) -> float:
        """Pixel displacement per radian of deflection."""
        return self.background_magnification * self.Z_D / self.pixel_pitch

    @property
    def field_of_view(self) -> tuple[float, float]:
        s = self.object_pixel_size
        return self.image_size[0] * s, self.image_size[1] * s

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(
            pixel_pitch=self.pixel_pitch, Z_B=self.Z_B,
            background_magnification=self.background_magnification,
            object_pixel_size=self.object_pixel_size, blur_diameter=self.blur_diameter,
            blur_diameter_px=self.blur_diameter_px, gain_px_per_rad=self.gain_px_per_rad,
        )
        return d


def pillbox_kernel(diameter_samples: float, oversample: int = 8) -> np.ndarray:
    """Normalized disk kernel with anti-aliased edges."""
    r = 0.5 * diameter_samples
    n = int(np.ceil(r)) + 1
    sub = (np.arange(oversample) + 0.5) / oversample - 0.5
    ax = np.arange(-n, n + 1)[:, None] + sub[None, :]
    xx = ax.reshape(-1)[None, :]
    yy = ax.reshape(-1)[:, None]
    inside = (xx**2 + yy**2 <= r**2).astype(float)
    k = inside.reshape(2 * n + 1, oversample, 2 * n + 1, oversample).mean(axis=(1, 3))
    if k.sum() == 0.0:
        k[n, n] = 1.0
    return k / k.sum()


def displacement_from_projection(P: np.ndarray, ds: float, optics: Optics) -> np.ndarray:
    """Pixel displacement field (2, rows, cols) from path-integrated density ``P``.

    ``P`` is sampled on a grid with spacing ``ds`` meters whose rows run downward
    (image convention). Channel 0 is the column (x) displacement and channel 1 the
    row displacement, both in output pixels.
    """
    kernel = pillbox_kernel(optics.blur_diameter / ds)
    pad = kernel.shape[0] // 2
    Pp = np.pad(P, pad, mode="edge")
    Pb = fftconvolve(Pp, kernel, mode="same")[pad:-pad, pad:-pad] if pad else P
    dP_drow, dP_dcol = np.gradient(Pb, ds)
    scale = optics.gain_px_per_rad * optics.gladstone_dale / optics.n0
    # A ray traced back from pixel x bends toward higher density, so that pixel sees
    # the background point x + scale * grad P. The pattern therefore appears to move
    # by d = -scale * grad P (toward lower density, like a diverging lens for a hot
    # core), and the deflected image is I(x) = B(x - d(x)). Gradients are taken in
    # image axes, so no sign change is needed for rows running downward.
    return np.stack([-scale * dP_dcol, -scale * dP_drow])
