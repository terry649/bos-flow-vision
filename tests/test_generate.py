"""End-to-end generator checks: physics recovered from rendered data, labels, determinism."""

import hashlib

import numpy as np
import pytest

from bosflow import generate as G
from bosflow import labels
from bosflow.analysis.shock_angle import fit_line_angle, shock_angle_relative
from bosflow.optics.camera import Camera
from bosflow.optics.deflection import Optics
from bosflow.physics import fields as F
from bosflow.physics import gasdynamics as gd

PATTERN = {"dot_density": 0.08, "dot_diameter_px": 2.5}
CAMERA = Camera()


@pytest.fixture(scope="module")
def cfg():
    c = G.load_config()
    c["counts"] = {ft: 1 for ft in G.FLOW_TYPES}
    return c


@pytest.mark.parametrize("pose_deg", [0.0, 17.0])
def test_wedge_shock_angle_recovered_from_rendered_field(pose_deg):
    """M = 2, 10 deg wedge: the rendered displacement ridge sits at beta = 39.31 deg."""
    optics = Optics()
    fov = optics.field_of_view[1]
    pose = F.Pose(-0.3 * fov, 0.0, np.radians(pose_deg))
    flow = F.WedgeFlow(2.0, np.radians(10.0), 0.3, 0.15, pose)
    s = G.render(flow, optics, CAMERA, PATTERN, np.random.default_rng(1))
    mag = np.hypot(*s.flow)
    beta = np.degrees(gd.beta_from_theta(2.0, np.radians(10.0)))
    for inst in s.instances:
        line = fit_line_angle(mag, inst.mask & s.valid)
        assert shock_angle_relative(line, pose_deg) == pytest.approx(beta, abs=0.5)


def test_cone_shock_angle_recovered_from_rendered_field():
    optics = Optics()
    fov = optics.field_of_view[1]
    flow = F.ConeFlow(2.5, np.radians(12.0), 0.8, F.Pose(-0.35 * fov, 0.0, 0.0))
    s = G.render(flow, optics, CAMERA, PATTERN, np.random.default_rng(2))
    mag = np.hypot(*s.flow)
    rows = np.arange(mag.shape[0])[:, None]
    upper = s.instances[0].mask & (rows < mag.shape[0] // 2) & s.valid
    beta = np.degrees(flow.solution.shock_angle)
    assert shock_angle_relative(fit_line_angle(mag, upper), 0.0) == pytest.approx(beta, abs=0.5)


def test_normal_shock_density_jump_recovered_from_displacement_integral():
    """Integrating the displacement across a planar shock returns W * delta rho."""
    optics = Optics()
    flow = F.WedgeFlow(3.0, np.radians(15.0), 0.2, 0.15, F.Pose(-0.45 * optics.field_of_view[1]))
    s = G.render(flow, optics, CAMERA, PATTERN, np.random.default_rng(4))
    c = optics.gain_px_per_rad * optics.gladstone_dale / optics.n0
    col = 300
    rows = np.nonzero(s.instances[0].mask[:, col])[0]
    r0, r1 = rows.min() - 10, rows.max() + 10
    # Row displacement in pixels integrates to -c * dP along a column (ds = 1 px).
    dP = -s.displacement[1][r0:r1, col].sum() * optics.object_pixel_size / c
    expected = 0.15 * 0.2 * (flow.shock.density_ratio - 1.0)
    assert abs(dP) == pytest.approx(expected, rel=0.02)


def test_label_band_width_at_least_three_pixels(cfg):
    seq = np.random.SeedSequence(99).spawn(1)[0]
    s = G.generate_sample("wedge", seq, cfg)
    assert s.meta["label_band_px"] >= 3.0
    for inst in s.instances:
        # Band thickness = area / length along the band.
        rows, cols = np.nonzero(inst.mask)
        length = np.hypot(np.ptp(rows), np.ptp(cols))
        assert inst.mask.sum() / length >= 3.0 * 0.9


def test_polygon_roundtrip_keeps_annulus_hole():
    optics = Optics(image_size=(256, 256))
    flow = F.BlastFlow(500.0, 1.0, 0.0, pose=F.Pose())
    flow.R = 0.25 * optics.field_of_view[1]
    x, y = G.object_grid(optics, 1, 0)
    mask = flow.instances(x, y, 3.0 * optics.object_pixel_size)[0].mask
    polys = labels.mask_to_polygons(mask)
    back = labels.polygons_to_mask(polys, mask.shape)
    iou = (back & mask).sum() / (back | mask).sum()
    assert iou > 0.9
    assert not back[128, 128]  # the hole survives


def test_same_seed_is_byte_identical(cfg, tmp_path):
    def digest(root):
        h = hashlib.sha256()
        for p in sorted(root.rglob("*")):
            if p.is_file():
                h.update(p.relative_to(root).as_posix().encode())
                h.update(p.read_bytes())
        return h.hexdigest()

    counts = {"wedge": 1, "shear_layer": 1}
    G.generate_dataset(tmp_path / "a", cfg, counts=counts, seed=123)
    G.generate_dataset(tmp_path / "b", cfg, counts=counts, seed=123)
    G.generate_dataset(tmp_path / "c", cfg, counts=counts, seed=124)
    assert digest(tmp_path / "a") == digest(tmp_path / "b")
    assert digest(tmp_path / "a") != digest(tmp_path / "c")


def test_every_flow_type_generates_labeled_samples(cfg):
    seqs = np.random.SeedSequence(5).spawn(len(G.FLOW_TYPES))
    expected = {"wedge": "shock", "cone": "shock", "blast": "shock",
                "expansion": "expansion_fan", "shear_layer": "shear_layer"}
    lo, hi = cfg["peak_displacement_px"]
    for ft, sq in zip(G.FLOW_TYPES, seqs):
        s = G.generate_sample(ft, sq, cfg)
        assert s.reference.shape == s.deflected.shape == tuple(cfg["optics"]["image_size"])
        assert {i.cls for i in s.instances} == {expected[ft]}
        assert lo <= s.meta["peak_displacement_px"] <= hi


def test_blast_sequence_follows_sedov_radius(cfg):
    frames = G.generate_blast_sequence(np.random.SeedSequence(8), cfg, n_frames=4)
    R = np.array([f.meta["flow"]["shock_radius"] for f in frames])
    t = np.array([f.meta["flow"]["t"] for f in frames])
    assert np.all(np.diff(R) > 0)
    np.testing.assert_allclose(R[1:] / R[0], (t[1:] / t[0]) ** 0.4, rtol=1e-9)
    # Same background for every frame: reference images differ only by noise.
    diff = frames[0].reference.astype(float) - frames[1].reference.astype(float)
    assert np.abs(diff).mean() < 4.0
