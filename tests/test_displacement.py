"""Displacement estimators on analytic fields, metrics, and the benchmark harness."""

import numpy as np
import pytest

from bosflow import generate as G
from bosflow.displacement import benchmark, metrics, optical_flow, registration
from bosflow.optics.camera import Camera
from bosflow.optics.deflection import Optics
from bosflow.physics.fields import LinearGradientFlow

PATTERN = {"dot_density": 0.08, "dot_diameter_px": 2.5}
CAM = Camera(psf_sigma_px=0.6, vignetting=0.0, full_well_e=20000, read_noise_e=4.0)


@pytest.fixture(scope="module")
def uniform_shift():
    """Linear density gradient: uniform, known displacement of (1.3, -0.8) px."""
    optics = Optics(image_size=(256, 256))
    c = optics.gain_px_per_rad * optics.gladstone_dale / optics.n0
    gx, gy = -1.3 / (c * 0.15), -0.8 / (c * 0.15)  # d = -c W grad P; y is up, so d_row = +c W gy
    s = G.render(LinearGradientFlow(gx, gy, 0.15), optics, CAM, PATTERN,
                 np.random.default_rng(0), margin_px=16)
    return s


@pytest.mark.parametrize("fn, tol", [(optical_flow.farneback, 0.05),
                                     (optical_flow.lucas_kanade, 0.05),
                                     (optical_flow.dis, 0.05),
                                     (optical_flow.tvl1, 0.1),
                                     (registration.demons, 0.1)])
def test_dense_methods_recover_uniform_shift(uniform_shift, fn, tol):
    s = uniform_shift
    np.testing.assert_allclose(s.flow[:, 128, 128], [1.3, -0.8], atol=1e-6)
    est = fn(s.reference, s.deflected)
    inner = (slice(None), slice(32, -32), slice(32, -32))
    assert metrics.endpoint_error(est[inner], s.flow[inner]).mean() < tol


@pytest.mark.slow
def test_raft_recovers_uniform_shift(uniform_shift):
    from bosflow.displacement.raft import raft

    s = uniform_shift
    est = raft(s.reference, s.deflected)
    inner = (slice(None), slice(32, -32), slice(32, -32))
    assert metrics.endpoint_error(est[inner], s.flow[inner]).mean() < 0.3


def test_summarize_is_zero_for_exact_estimate():
    gt = np.random.default_rng(1).normal(size=(2, 32, 32))
    valid = np.ones((32, 32), bool)
    feature = np.zeros_like(valid)
    feature[8:16] = True
    m = metrics.summarize(gt, gt, valid, feature)
    assert m["epe"] == 0.0 and m["epe_feature"] == 0.0 and m["peak_ratio"] == pytest.approx(1.0)


def test_benchmark_harness_produces_records():
    bcfg = benchmark.load_benchmark_config()
    bcfg["noise_levels"] = {k: bcfg["noise_levels"][k] for k in ("clean", "high")}
    recs = benchmark.run(bcfg, G.load_config(), scenes_per_type=1, methods=["farneback"],
                         progress=lambda *_: None)
    assert len(recs) == len(G.FLOW_TYPES) * 2
    assert {r["flow_type"] for r in recs} == set(G.FLOW_TYPES)
    # Noise can only hurt: mean EPE at high noise exceeds the clean case.
    epe = {lv: np.mean([r["epe"] for r in recs if r["noise"] == lv]) for lv in ("clean", "high")}
    assert epe["high"] > epe["clean"]
