"""Synthetic BOS sample generation: sample a flow and a setup, render, label, write.

Every random draw for sample ``i`` comes from ``SeedSequence(seed).spawn(n)[i]``,
so any single sample can be regenerated in isolation and repeated runs are
byte-identical.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from scipy.ndimage import binary_dilation

from bosflow import labels, lineage
from bosflow.optics.background import dot_density_for_fill, random_dot_pattern
from bosflow.optics.camera import Camera, capture
from bosflow.optics.deflection import Optics, displacement_from_projection
from bosflow.optics.warp import block_mean, forward_flow, warp_backward
from bosflow.physics import fields as F
from bosflow.physics import gasdynamics as gd

FLOW_TYPES = ("wedge", "cone", "blast", "expansion", "shear_layer")
R_AIR = 287.05

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


# ---------------------------------------------------------------------------
# Config and sampling helpers
# ---------------------------------------------------------------------------


def load_config(config_dir: Path | str = CONFIG_DIR) -> dict:
    config_dir = Path(config_dir)
    cfg = yaml.safe_load((config_dir / "dataset.yaml").read_text())
    cfg["optics"] = yaml.safe_load((config_dir / "optics.yaml").read_text())
    cfg["flows"] = {p.stem: yaml.safe_load(p.read_text())
                    for p in sorted((config_dir / "flows").glob("*.yaml"))}
    return cfg


def _u(rng, v):
    """Uniform draw from a [lo, hi] pair; scalars pass through."""
    if isinstance(v, (list, tuple)):
        return float(rng.uniform(v[0], v[1]))
    return v


def _logu(rng, v):
    if isinstance(v, (list, tuple)):
        return float(np.exp(rng.uniform(np.log(v[0]), np.log(v[1]))))
    return v


def sample_optics(rng, ocfg) -> Optics:
    return Optics(
        focal_length=ocfg["focal_length"], f_number=_u(rng, ocfg["f_number"]),
        sensor_pixel_pitch=ocfg["sensor_pixel_pitch"], binning=ocfg["binning"],
        Z_A=_u(rng, ocfg["Z_A"]), Z_D=_u(rng, ocfg["Z_D"]),
        image_size=tuple(ocfg["image_size"]), gladstone_dale=ocfg["gladstone_dale"],
        rho_ambient=ocfg["rho_ambient"],
    )


def sample_camera(rng, ccfg) -> Camera:
    jpeg = None
    if rng.uniform() < ccfg["jpeg_probability"]:
        jpeg = round(_u(rng, ccfg["jpeg_quality"]))
    return Camera(
        psf_sigma_px=_u(rng, ccfg["psf_sigma_px"]), vignetting=_u(rng, ccfg["vignetting"]),
        full_well_e=float(ccfg["full_well_e"]), read_noise_e=_u(rng, ccfg["read_noise_e"]),
        exposure=_u(rng, ccfg["exposure"]), background_level=_u(rng, ccfg["background_level"]),
        bit_depth=int(ccfg["bit_depth"]), jpeg_quality=jpeg,
    )


def sample_pattern(rng, bcfg) -> dict:
    fill, diam = _u(rng, bcfg["fill_fraction"]), _u(rng, bcfg["dot_diameter_px"])
    return {"fill_fraction": fill, "dot_diameter_px": diam,
            "dot_density": dot_density_for_fill(fill, diam)}


def _pose(rng, c, fov, xkey, ykey):
    angle = np.radians(_u(rng, c.get("angle_deg", 0.0)))
    x0, y0 = _u(rng, c[xkey]) * fov, _u(rng, c[ykey]) * fov
    if rng.uniform() < c.get("flip_probability", 0.0):
        angle += np.pi
        x0 = -x0
    return F.Pose(float(x0), float(y0), float(angle))


def sample_flow(flow_type: str, rng, fcfg: dict, fov: float) -> F.Flow:
    c = fcfg
    if flow_type == "wedge":
        M1 = _u(rng, c["M1"])
        th_max, _ = gd.max_deflection(M1)
        lo, hi = np.radians(c["half_angle_deg"])
        theta = float(rng.uniform(lo, min(hi, 0.85 * th_max)))
        return F.WedgeFlow(M1, theta, _logu(rng, c["rho1"]), c["span"],
                           _pose(rng, c, fov, "apex_x_frac", "apex_y_frac"))
    if flow_type == "cone":
        while True:
            M1 = _u(rng, c["M1"])
            theta = np.radians(_u(rng, c["half_angle_deg"]))
            try:
                gd.cone_shock_angle(M1, theta)
                break
            except ValueError:
                continue
        return F.ConeFlow(M1, float(theta), _logu(rng, c["rho1"]),
                          _pose(rng, c, fov, "apex_x_frac", "apex_y_frac"))
    if flow_type == "blast":
        sol = gd.sedov_taylor()
        while True:
            E, rho0 = _logu(rng, c["E"]), _logu(rng, c["rho0"])
            R = _u(rng, c["radius_frac"]) * fov
            t = float(sol.time_at_radius(E, rho0, R))
            p0 = rho0 * R_AIR * c["T0"]
            flow = F.BlastFlow(E, rho0, t, p0, _pose(rng, c, fov, "center_x_frac",
                                                     "center_y_frac"))
            if flow.shock_mach >= c["min_shock_mach"]:
                return flow
    if flow_type == "expansion":
        return F.ExpansionFlow(_u(rng, c["M1"]), np.radians(_u(rng, c["turn_angle_deg"])),
                               _logu(rng, c["rho1"]), c["span"],
                               _pose(rng, c, fov, "corner_x_frac", "corner_y_frac"))
    if flow_type == "shear_layer":
        rho_l = _u(rng, c["rho_lower"])
        rho_u = rho_l * _u(rng, c["density_ratio"])
        if rng.uniform() < 0.5:
            rho_l, rho_u = rho_u, rho_l
        pose_cfg = dict(c, x_frac=[0.0, 0.0])
        return F.ShearLayerFlow(
            rho_upper=rho_u, rho_lower=rho_l, growth=_u(rng, c["growth"]),
            x_origin=-_u(rng, c["origin_upstream_frac"]) * fov, span=c["span"],
            fluct_intensity=_u(rng, c["fluct_intensity"]),
            outer_scale_frac=_u(rng, c["outer_scale_frac"]),
            seed=int(rng.integers(2**31)),
            pose=_pose(rng, pose_cfg, fov, "x_frac", "center_y_frac"),
        )
    if flow_type == "composite":
        # Two distinct flow types; at most one with a solid body so bodies don't overlap.
        members, bodied = [], {"wedge", "cone", "expansion"}
        kinds = list(c.get("member_types", ["blast", "shear_layer", "wedge", "expansion"]))
        while len(members) < 2:
            ft = str(rng.choice(kinds))
            if any(m.flow_type == ft for m in members):
                continue
            if ft in bodied and any(m.flow_type in bodied for m in members):
                continue
            members.append(sample_flow(ft, rng, fcfg["_all"][ft], fov))
        return F.CompositeFlow(members)
    raise ValueError(f"unknown flow type {flow_type!r}")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def object_grid(optics: Optics, supersample: int, margin_px: int):
    """Object-plane coordinates (x right, y up, m) of sample centers, with a margin."""
    rows, cols = optics.image_size
    s = optics.object_pixel_size
    n_r, n_c = (rows + 2 * margin_px) * supersample, (cols + 2 * margin_px) * supersample
    # Sample centers in output-pixel units, pixel (0, 0) centered at 0.
    r_px = (np.arange(n_r) + 0.5) / supersample - 0.5 - margin_px
    c_px = (np.arange(n_c) + 0.5) / supersample - 0.5 - margin_px
    x = (c_px - 0.5 * (cols - 1)) * s
    y = -(r_px - 0.5 * (rows - 1)) * s
    return np.meshgrid(x, y)


@dataclass
class Sample:
    reference: np.ndarray  # uint8 (rows, cols)
    deflected: np.ndarray
    flow: np.ndarray  # float32 (2, rows, cols): forward optical flow ref -> deflected, px
    displacement: np.ndarray  # float32 (2, rows, cols): d with I_def(x) = B(x - d(x)), px
    valid: np.ndarray  # bool: background visible and unaffected by body edges
    body: np.ndarray  # bool
    instances: list[F.Instance]
    meta: dict = field(default_factory=dict)
    scene: dict = field(default_factory=dict)  # flow, optics, pattern, for re-rendering


def render(flow: F.Flow, optics: Optics, camera: Camera, pattern: dict,
           rng: np.random.Generator, supersample: int = 4, margin_px: int = 24,
           min_band_px: float = 3.0, background: np.ndarray | None = None) -> Sample:
    ss, m = supersample, margin_px
    rows, cols = optics.image_size
    x_hr, y_hr = object_grid(optics, ss, m)
    P = flow.projected_density(x_hr, y_hr)
    d_hr = displacement_from_projection(P, optics.object_pixel_size / ss, optics)
    if background is None:
        background = random_dot_pattern(x_hr.shape, ss, pattern["dot_density"],
                                        pattern["dot_diameter_px"], rng)
    warped = warp_backward(background, d_hr, ss)

    crop = (slice(m, m + rows), slice(m, m + cols))
    ref_refl = block_mean(background, ss)[crop]
    def_refl = block_mean(warped, ss)[crop]
    disp = block_mean(d_hr, ss)
    u = forward_flow(disp)[(slice(None), *crop)]
    disp = disp[(slice(None), *crop)]
    body = (block_mean(flow.body(x_hr, y_hr).astype(float), ss) > 0.5)[crop]
    r_guard = int(np.ceil(0.5 * optics.blur_diameter_px)) + 1
    guard = np.ones((2 * r_guard + 1,) * 2, bool)
    valid = ~binary_dilation(body, guard) if body.any() else np.ones_like(body)

    ref_img = capture(ref_refl, camera, rng, body)
    def_img = capture(def_refl, camera, rng, body)

    x_px, y_px = object_grid(optics, 1, 0)
    band_px = max(min_band_px, optics.blur_diameter_px)
    half_width = 0.5 * band_px * optics.object_pixel_size
    insts = []
    for inst in flow.instances(x_px, y_px, half_width):
        inst.mask = inst.mask & ~body
        insts.append(inst)

    # Largest singular value of the flow Jacobian: above 1 the image folds (caustics).
    J = np.stack([np.stack(np.gradient(u[k]), 0) for k in range(2)])  # (2 comp, 2 axis, ...)
    a, b, c, d = J[0, 1], J[0, 0], J[1, 1], J[1, 0]  # du/dx, du/dy, dv/dx, dv/dy
    fro2 = a**2 + b**2 + c**2 + d**2
    smax = np.sqrt(0.5 * (fro2 + np.sqrt(np.maximum(fro2**2 - 4 * (a * d - b * c) ** 2, 0))))
    meta = {
        # 99.5th percentile: the pillbox-smeared step has integrable edge singularities.
        "max_flow_gradient": float(np.percentile(smax[valid], 99.5)) if valid.any() else 0.0,
        "flow": flow.metadata(), "optics": optics.to_dict(), "camera": camera.to_dict(),
        "background": dict(pattern), "supersample": ss, "label_band_px": band_px,
        "peak_displacement_px": float(np.percentile(np.hypot(*u)[valid], 99.9))
        if valid.any() else 0.0,
    }
    return Sample(ref_img, def_img, u.astype(np.float32), disp.astype(np.float32),
                  valid, body, insts, meta)


def generate_sample(flow_type: str, seed_seq: np.random.SeedSequence, cfg: dict) -> Sample:
    """Draw parameters until the rendered displacement falls in the target window."""
    rng = np.random.default_rng(seed_seq)
    lo, hi = cfg["peak_displacement_px"]
    for attempt in range(cfg["max_attempts"]):
        optics = sample_optics(rng, cfg["optics"])
        camera = sample_camera(rng, cfg["optics"]["camera"])
        pattern = sample_pattern(rng, cfg["optics"]["background"])
        fcfg = cfg["flows"].get(flow_type, {})
        if flow_type == "composite":
            fcfg = dict(fcfg, _all=cfg["flows"])
        flow = sample_flow(flow_type, rng, fcfg, optics.field_of_view[1])
        sample = render(flow, optics, camera, pattern, rng, cfg["supersample"],
                        cfg["margin_px"], cfg["min_band_px"])
        sample.instances = [i for i in sample.instances
                            if i.mask.sum() >= cfg["min_instance_area_px"]]
        if (lo <= sample.meta["peak_displacement_px"] <= hi and sample.instances
                and sample.meta["max_flow_gradient"] <= cfg["max_flow_gradient"]):
            sample.meta["attempts"] = attempt + 1
            sample.scene = {"flow": flow, "optics": optics, "pattern": pattern}
            return sample
    raise RuntimeError(f"{flow_type}: no sample in the displacement window after "
                       f"{cfg['max_attempts']} attempts; widen the config ranges")


def generate_blast_sequence(seed_seq: np.random.SeedSequence, cfg: dict,
                            n_frames: int) -> list[Sample]:
    """One charge, one background, a series of times with the front crossing the frame."""
    rng = np.random.default_rng(seed_seq)
    c = cfg["flows"]["blast"]
    optics = sample_optics(rng, cfg["optics"])
    camera = sample_camera(rng, cfg["optics"]["camera"])
    pattern = sample_pattern(rng, cfg["optics"]["background"])
    fov = optics.field_of_view[1]
    first = sample_flow("blast", rng, c, fov)
    sol = gd.sedov_taylor()
    radii = np.linspace(c["radius_frac"][0], c["radius_frac"][1], n_frames) * fov
    times = sol.time_at_radius(first.E, first.rho0, radii)
    x_hr, _ = object_grid(optics, cfg["supersample"], cfg["margin_px"])
    background = random_dot_pattern(x_hr.shape, cfg["supersample"],
                                    pattern["dot_density"], pattern["dot_diameter_px"], rng)
    frames = []
    for k, t in enumerate(times):
        flow = F.BlastFlow(first.E, first.rho0, float(t), first.p0, first.pose)
        s = render(flow, optics, camera, pattern, rng, cfg["supersample"], cfg["margin_px"],
                   cfg["min_band_px"], background=background)
        s.meta["frame"] = k
        frames.append(s)
    return frames


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def displacement_magnitude_image(u: np.ndarray, valid: np.ndarray, vmax: float) -> np.ndarray:
    mag = np.where(valid, np.hypot(u[0], u[1]), 0.0)
    return np.clip(np.round(255.0 * mag / vmax), 0, 255).astype(np.uint8)


def write_sample(sample: Sample, out_dir: Path, name: str) -> dict:
    d = out_dir / name
    d.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sample.reference).save(d / "reference.png", optimize=False)
    Image.fromarray(sample.deflected).save(d / "deflected.png", optimize=False)
    masks = (np.stack([i.mask for i in sample.instances]).astype(np.uint8)
             if sample.instances else np.zeros((0, *sample.valid.shape), np.uint8))
    # The backward displacement is omitted: it differs from the flow only at second
    # order and is recomputable from the seed.
    np.savez_compressed(d / "ground_truth.npz", flow=sample.flow, valid=sample.valid,
                        body=sample.body, instance_masks=masks)
    insts = []
    for inst in sample.instances:
        polys = labels.mask_to_polygons(inst.mask)
        insts.append({"class": inst.cls, "attrs": inst.attrs, "area_px": int(inst.mask.sum()),
                      "bbox": labels.bbox_of(inst.mask), "polygons": polys})
    meta = dict(sample.meta, name=name, instances=insts)
    (d / "meta.json").write_text(json.dumps(meta, indent=1, default=float))
    return meta


def write_coco_gt(out_dir: Path, metas: list[dict], vmax: float) -> Path:
    """COCO file over ground-truth displacement-magnitude images, for inspection."""
    images, anns, ann_id = [], [], 1
    img_dir = out_dir / "dispmag_gt"
    img_dir.mkdir(exist_ok=True)
    for image_id, meta in enumerate(metas, start=1):
        gt = np.load(out_dir / meta["name"] / "ground_truth.npz")
        fname = f"{meta['name']}.png"
        Image.fromarray(displacement_magnitude_image(gt["flow"], gt["valid"], vmax)).save(
            img_dir / fname)
        rows, cols = gt["valid"].shape
        images.append({"id": image_id, "file_name": fname, "height": rows, "width": cols,
                       "flow_type": meta["flow"]["flow_type"]})
        for inst, mask in zip(meta["instances"], gt["instance_masks"]):
            a = labels.instance_annotation(ann_id, image_id, inst["class"], mask.astype(bool),
                                           inst["polygons"])
            if a:
                anns.append(a)
                ann_id += 1
    path = img_dir / "_annotations.coco.json"
    path.write_text(json.dumps(labels.build_coco(images, anns, "bosflow synthetic GT")))
    return path


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Set dotted config keys from ``key=value`` strings (values parsed as YAML)."""
    import copy

    cfg = copy.deepcopy(cfg)
    for item in overrides:
        key, _, value = item.partition("=")
        node = cfg
        *parents, leaf = key.split(".")
        for k in parents:
            node = node[k]
        if leaf not in node:
            raise KeyError(f"override {key!r} does not match an existing config key")
        node[leaf] = yaml.safe_load(value)
    return cfg


def generate_dataset(out_dir: Path | str, cfg: dict, counts: dict | None = None,
                     seed: int | None = None, prefix: str = "") -> list[dict]:
    out_dir = Path(out_dir)
    counts = counts or cfg["counts"]
    seed = cfg["seed"] if seed is None else seed
    start = lineage.manifest(cfg)
    order = [*FLOW_TYPES, *(t for t in counts if t not in FLOW_TYPES)]
    jobs = [(ft, k) for ft in order for k in range(counts.get(ft, 0))]
    children = np.random.SeedSequence(seed).spawn(len(jobs))
    metas = []
    for (ft, k), ss in zip(jobs, children):
        sample = generate_sample(ft, ss, cfg)
        metas.append(write_sample(sample, out_dir / "samples", f"{prefix}{ft}_{k:04d}"))
    with open(out_dir / "index.jsonl", "w") as fh:
        fh.writelines(
            json.dumps({"name": m["name"], "flow_type": m["flow"]["flow_type"],
                        "classes": [i["class"] for i in m["instances"]],
                        "peak_displacement_px": m["peak_displacement_px"]}) + "\n"
            for m in metas)
    write_coco_gt(out_dir / "samples", metas, vmax=cfg["peak_displacement_px"][1])
    lineage.write_manifest(out_dir / "manifest.json", cfg, start=start, seed=seed, counts=counts,
                           n_samples=len(metas))
    return metas
