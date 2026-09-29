"""Displacement error metrics against synthetic ground truth."""

from __future__ import annotations

import numpy as np


def endpoint_error(est: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return np.hypot(est[0] - gt[0], est[1] - gt[1])


def summarize(est: np.ndarray, gt: np.ndarray, valid: np.ndarray,
              feature: np.ndarray | None = None) -> dict:
    """EPE statistics over valid pixels, and separately inside labeled flow features.

    ``rel_epe_feature`` normalizes by the mean GT magnitude in the features, which is
    the number that matters for extracting physics from BOS.
    """
    epe = endpoint_error(est, gt)
    mag = np.hypot(gt[0], gt[1])
    out = {
        "epe": float(epe[valid].mean()),
        "epe_rms": float(np.sqrt((epe[valid] ** 2).mean())),
        "outlier_1px": float((epe[valid] > 1.0).mean()),
        "gt_mag_mean": float(mag[valid].mean()),
    }
    if feature is not None:
        f = feature & valid
        if f.any():
            out["epe_feature"] = float(epe[f].mean())
            out["gt_mag_feature"] = float(mag[f].mean())
            out["rel_epe_feature"] = out["epe_feature"] / max(out["gt_mag_feature"], 1e-9)
            # Peak recovery: how much of the strongest displacement survives smoothing.
            top = f & (mag >= np.percentile(mag[f], 95))
            est_mag = np.hypot(est[0], est[1])
            out["peak_ratio"] = float(est_mag[top].mean() / max(mag[top].mean(), 1e-9))
        bg = valid & ~feature
        if bg.any():
            out["epe_background"] = float(epe[bg].mean())
    return out
