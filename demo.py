"""Quickstart demo: from BOS images to gas dynamics in about a minute.

    uv sync --group train
    uv run python demo.py                                   # uses the local checkpoint if present
    uv run python demo.py --checkpoint path/to/checkpoint_best_total.pth
    uv run python demo.py --ground-truth                    # no model: use generator labels

Steps:
  1. Render a Mach 2.5 flow over a 10 degree wedge as a BOS image pair.
  2. Estimate the displacement field (DIS optical flow) and encode its magnitude.
  3. Segment flow features with RF-DETR Seg, measure the shock angle, and invert the
     theta-beta-Mach relation to recover the freestream Mach number.
  4. Render a blast-wave sequence, segment the front in each frame, keep thin fronts,
     fit circles, and recover the Sedov-Taylor exponent and the released energy.
Writes demo_output/quickstart.png and prints a summary.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bosflow import generate as G
from bosflow.analysis import physics_check as P
from bosflow.displacement.optical_flow import dis
from bosflow.optics.camera import Camera
from bosflow.optics.deflection import Optics
from bosflow.physics import fields as F
from bosflow.physics import gasdynamics as gd
from bosflow.rf.export import encode_image

CLASSES = ["shock", "expansion_fan", "shear_layer"]
DEFAULT_CKPT = Path("runs/v1_dis_medium/checkpoint_best_total.pth")
PATTERN = {"dot_density": 0.08, "dot_diameter_px": 2.5}


def segment(model, img, sample, threshold=0.5):
    """Shock masks from the model, or from the generator labels with --ground-truth."""
    if model is None:
        return [i.mask for i in sample.instances if i.cls == "shock"]
    det = model.predict(img, threshold=threshold)
    return [m for m, c in zip(det.mask, det.class_id) if CLASSES[int(c)] == "shock"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--ground-truth", action="store_true", help="skip the model; use labels")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("demo_output"))
    args = ap.parse_args()
    args.out.mkdir(exist_ok=True)
    t0 = time.time()

    model = None
    if not args.ground_truth:
        if not args.checkpoint.exists():
            raise SystemExit(f"no checkpoint at {args.checkpoint}; pass --checkpoint or --ground-truth")
        import rfdetr
        model = rfdetr.RFDETR.from_checkpoint(args.checkpoint)
    source = "ground-truth labels" if model is None else f"RF-DETR Seg ({args.checkpoint.name})"
    print(f"[demo] segmentation source: {source}")

    # 1-3: wedge
    optics, M1, theta, pose_deg = Optics(), 2.5, 10.0, 12.0
    fov = optics.field_of_view[1]
    wedge = F.WedgeFlow(M1, np.radians(theta), 0.05, 0.15,
                        F.Pose(-0.3 * fov, 0.0, np.radians(pose_deg)))
    s = G.render(wedge, optics, Camera(), PATTERN, np.random.default_rng(7))
    u = dis(s.reference, s.deflected)
    img, _ = encode_image(u, s.valid)
    shocks = segment(model, img, s)
    betas = [P.mask_shock_angle(m, pose_deg) for m in shocks if m.sum() > 50]
    machs = [P.mach_from_wedge_shock(b, theta) for b in betas]
    print(f"[demo] wedge: {len(betas)} shocks, beta = "
          + ", ".join(f"{b:.2f}" for b in betas)
          + f" deg (theory {np.degrees(wedge.shock.beta):.2f}); Mach = "
          + ", ".join(f"{m:.3f}" for m in machs) + f" (true {M1})")

    # 4: blast sequence
    cfg = G.load_config()
    frames = G.generate_blast_sequence(np.random.SeedSequence(2026), cfg, args.frames)
    t, R, R_true, last_mask = [], [], [], None
    for f in frames:
        uf = dis(f.reference, f.deflected)
        fimg, _ = encode_image(uf, f.valid)
        cands = [m for m in segment(model, fimg, f) if m.sum() > 50
                 and P.circle_residual_ratio(m) < 0.05]
        if not cands:
            continue
        mask = max(cands, key=lambda m: m.sum())
        last_mask = mask
        _, _, r_px = P.fit_circle(mask)
        s_px = f.meta["optics"]["object_pixel_size"]
        t.append(f.meta["flow"]["t"])
        R.append(r_px * s_px)
        R_true.append(f.meta["flow"]["shock_radius"])
    t, R = np.array(t), np.array(R)
    flow = frames[0].meta["flow"]
    n, _ = P.power_law_fit(t, R)
    E = P.sedov_energy(t, R, flow["rho0"])
    print(f"[demo] blast: {len(t)}/{len(frames)} frames, exponent {n:.4f} (Sedov-Taylor 0.4), "
          f"energy {E:.1f} J (true {flow['E']:.1f} J, error {100 * (E / flow['E'] - 1):+.1f}%)")

    # figure
    fig, ax = plt.subplots(2, 3, figsize=(12, 7.6))
    ax[0, 0].imshow(s.deflected, cmap="gray")
    ax[0, 0].set_title("BOS image (Mach 2.5 wedge)")
    ax[0, 1].imshow(img)
    ax[0, 1].set_title("DIS displacement magnitude")
    overlay = np.dstack([img[..., 0] / 255.0] * 3)
    for m in shocks:
        overlay[m] = 0.4 * overlay[m] + 0.6 * np.array([1.0, 0.25, 0.25])
    ax[0, 2].imshow(overlay)
    ax[0, 2].set_title("shock masks: Mach " + ", ".join(f"{m:.2f}" for m in machs))
    last = frames[-1]
    limg, _ = encode_image(dis(last.reference, last.deflected), last.valid)
    blast_overlay = np.dstack([limg[..., 0] / 255.0] * 3)
    if last_mask is not None:
        blast_overlay[last_mask] = 0.4 * blast_overlay[last_mask] + 0.6 * np.array([1.0, 0.25, 0.25])
    ax[1, 0].imshow(blast_overlay)
    ax[1, 0].set_title(f"blast front mask, frame {len(frames) - 1}")
    ax[1, 1].loglog(t * 1e6, np.array(R_true[: len(t)]) * 1e3, "k-", label="true R(t)")
    ax[1, 1].loglog(t * 1e6, R * 1e3, "o", color="tab:red", label="from masks")
    ax[1, 1].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax[1, 1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[1, 1].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax[1, 1].yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%.0f"))
    ax[1, 1].set_xlabel("t (microseconds)")
    ax[1, 1].set_ylabel("R (mm)")
    ax[1, 1].legend()
    ax[1, 1].set_title(f"exponent {n:.3f} (theory 0.4)")
    ax[1, 2].axis("off")
    ax[1, 2].text(0.0, 0.5, "\n".join([
        f"Segmentation: {source}",
        "",
        f"Wedge, theta = {theta} deg",
        f"  shock angle {np.mean(betas):.2f} deg (theory {np.degrees(wedge.shock.beta):.2f})",
        f"  recovered Mach {np.mean(machs):.3f} (true {M1})",
        "",
        "Blast wave (Sedov-Taylor)",
        f"  exponent {n:.4f} (theory 0.4)",
        f"  energy {E:.1f} J (true {flow['E']:.1f} J)",
        "",
        f"gamma = {gd.GAMMA_AIR}, runtime {time.time() - t0:.0f} s",
    ]), fontsize=10, family="monospace", va="center")
    for a in (ax[0, 0], ax[0, 1], ax[0, 2], ax[1, 0]):
        a.axis("off")
    fig.tight_layout()
    path = args.out / "quickstart.png"
    fig.savefig(path, dpi=110)
    print(f"[demo] wrote {path} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
