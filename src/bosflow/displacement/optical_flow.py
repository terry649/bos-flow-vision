"""Dense optical flow: Farneback, pyramidal Lucas-Kanade, DIS (OpenCV), and TV-L1.

All functions take 8-bit reference and deflected images and return the forward
flow (2, rows, cols) as (column, row) displacement, so that
deflected(p + u(p)) ~= reference(p).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.registration import optical_flow_tvl1


@dataclass(frozen=True)
class FarnebackParams:
    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 15
    iterations: int = 5
    poly_n: int = 5
    poly_sigma: float = 1.1
    gaussian: bool = True


def farneback(ref: np.ndarray, defl: np.ndarray, p: FarnebackParams | None = None):
    p = p or FarnebackParams()
    flags = cv2.OPTFLOW_FARNEBACK_GAUSSIAN if p.gaussian else 0
    flow = cv2.calcOpticalFlowFarneback(ref, defl, None, p.pyr_scale, p.levels, p.winsize,
                                        p.iterations, p.poly_n, p.poly_sigma, flags)
    return np.moveaxis(flow, -1, 0).astype(np.float32)


@dataclass(frozen=True)
class LucasKanadeParams:
    winsize: int = 15
    max_level: int = 3
    iterations: int = 30
    epsilon: float = 1e-3


def lucas_kanade(ref: np.ndarray, defl: np.ndarray,
                 p: LucasKanadeParams | None = None):
    """Track every pixel center; failed points are filled by a 5x5 median."""
    p = p or LucasKanadeParams()
    rows, cols = ref.shape
    yy, xx = np.mgrid[0:rows, 0:cols].astype(np.float32)
    pts = np.stack([xx.ravel(), yy.ravel()], axis=-1)[:, None, :]
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, p.iterations, p.epsilon)
    nxt, status, _ = cv2.calcOpticalFlowPyrLK(ref, defl, pts, None, winSize=(p.winsize,) * 2,
                                              maxLevel=p.max_level, criteria=crit)
    flow = (nxt - pts)[:, 0, :].reshape(rows, cols, 2)
    ok = status.reshape(rows, cols).astype(bool)
    if not ok.all():
        for k in range(2):
            med = cv2.medianBlur(np.where(ok, flow[..., k], 0).astype(np.float32), 5)
            flow[..., k] = np.where(ok, flow[..., k], med)
    return np.moveaxis(flow, -1, 0).astype(np.float32)


@dataclass(frozen=True)
class DISParams:
    preset: int = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
    patch_size: int = 8
    patch_stride: int = 3
    finest_scale: int = 0  # 0 = refine down to full resolution
    variational_iterations: int = 5


def dis(ref: np.ndarray, defl: np.ndarray, p: DISParams | None = None):
    """Dense inverse search (Kroeger et al. 2016) with variational refinement."""
    p = p or DISParams()
    d = cv2.DISOpticalFlow_create(p.preset)
    d.setPatchSize(p.patch_size)
    d.setPatchStride(p.patch_stride)
    d.setFinestScale(p.finest_scale)
    d.setVariationalRefinementIterations(p.variational_iterations)
    flow = d.calc(ref, defl, None)
    return np.moveaxis(flow, -1, 0).astype(np.float32)


@dataclass(frozen=True)
class TVL1Params:
    attachment: float = 15.0
    tightness: float = 0.3
    num_warp: int = 5
    num_iter: int = 10


def tvl1(ref: np.ndarray, defl: np.ndarray, p: TVL1Params | None = None):
    """TV-L1 variational optical flow (Zach, Pock and Bischof 2007; Wedel et al. 2009).

    scikit-image returns (row, col) components with moving(p + v) ~= reference(p).
    """
    p = p or TVL1Params()
    v = optical_flow_tvl1(ref.astype(np.float32) / 255.0, defl.astype(np.float32) / 255.0,
                          attachment=p.attachment, tightness=p.tightness,
                          num_warp=p.num_warp, num_iter=p.num_iter)
    return np.stack([v[1], v[0]]).astype(np.float32)
