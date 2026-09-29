"""Measure the orientation of a straight shock from a displacement field and a mask."""

from __future__ import annotations

import numpy as np


def fit_line_angle(magnitude: np.ndarray, mask: np.ndarray, power: float = 2.0) -> float:
    """Orientation of the ridge of ``magnitude`` inside ``mask``, degrees in [0, 180).

    Weighted principal axis of the masked pixels with weights magnitude**power.
    The displacement profile across a smeared shock is symmetric about the shock,
    so the weighted axis passes through the shock trace. The angle is measured
    counterclockwise from the image +x axis with y pointing up.
    """
    rows, cols = np.nonzero(mask)
    w = magnitude[rows, cols] ** power
    if w.sum() <= 0.0:
        raise ValueError("no signal inside mask")
    x, y = cols.astype(float), -rows.astype(float)
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.cov(np.stack([x - mx, y - my]), aweights=w)
    evals, evecs = np.linalg.eigh(cov)
    vx, vy = evecs[:, np.argmax(evals)]
    return float(np.degrees(np.arctan2(vy, vx)) % 180.0)


def shock_angle_relative(line_angle_deg: float, freestream_angle_deg: float) -> float:
    """Acute angle between a shock trace and the freestream direction, degrees."""
    d = (line_angle_deg - freestream_angle_deg) % 180.0
    return min(d, 180.0 - d)
