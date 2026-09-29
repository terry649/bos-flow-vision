"""Sensor and lens degradations applied to irradiance images."""

from __future__ import annotations

import io
from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


@dataclass(frozen=True)
class Camera:
    psf_sigma_px: float = 0.6  # residual lens blur on the (in-focus) background
    vignetting: float = 0.25  # fractional falloff at the image corner, cos^4-like
    full_well_e: float = 20000.0
    read_noise_e: float = 8.0
    exposure: float = 0.8  # mean fraction of full well for a white background
    background_level: float = 0.15  # reflectance of the space between dots
    bit_depth: int = 8
    jpeg_quality: int | None = None  # None writes lossless images

    def to_dict(self) -> dict:
        return asdict(self)


def vignetting_map(shape, strength: float) -> np.ndarray:
    rows, cols = shape
    y = (np.arange(rows) - 0.5 * (rows - 1)) / (0.5 * rows)
    x = (np.arange(cols) - 0.5 * (cols - 1)) / (0.5 * cols)
    r2 = (y[:, None] ** 2 + x[None, :] ** 2) / 2.0  # 1 at the corners
    tan2 = r2 * ((1.0 - strength) ** -0.5 - 1.0)  # chosen so the corner value is 1 - strength
    return 1.0 / (1.0 + tan2) ** 2


def capture(reflectance: np.ndarray, camera: Camera, rng: np.random.Generator,
            body: np.ndarray | None = None) -> np.ndarray:
    """Convert background reflectance at pixel resolution to a quantized image."""
    irr = camera.background_level + (1.0 - camera.background_level) * reflectance
    if body is not None:
        irr = np.where(body, 0.02, irr)  # model silhouette, lit only by stray light
    if camera.psf_sigma_px > 0:
        irr = gaussian_filter(irr, camera.psf_sigma_px, mode="nearest")
    irr = irr * vignetting_map(irr.shape, camera.vignetting)
    electrons = rng.poisson(irr * camera.exposure * camera.full_well_e).astype(float)
    electrons += rng.normal(0.0, camera.read_noise_e, irr.shape)
    levels = 2**camera.bit_depth - 1
    dn = np.clip(np.round(electrons / camera.full_well_e * levels), 0, levels)
    img = dn.astype(np.uint8 if camera.bit_depth <= 8 else np.uint16)
    if camera.jpeg_quality is not None and camera.bit_depth <= 8:
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=camera.jpeg_quality)
        img = np.asarray(Image.open(io.BytesIO(buf.getvalue())))
    return img
