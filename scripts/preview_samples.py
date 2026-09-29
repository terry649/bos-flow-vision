"""Render one sample per flow type and save a preview figure.

    uv run python scripts/preview_samples.py --seed 7 --out results/m1_preview.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bosflow import generate as G

COLORS = {"shock": (1.0, 0.2, 0.2), "expansion_fan": (0.2, 0.5, 1.0),
          "shear_layer": (0.2, 0.9, 0.3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path("results/m1_preview.png"))
    args = ap.parse_args()

    cfg = G.load_config()
    seqs = np.random.SeedSequence(args.seed).spawn(len(G.FLOW_TYPES))
    fig, axes = plt.subplots(len(G.FLOW_TYPES), 4, figsize=(13, 3.3 * len(G.FLOW_TYPES)))
    for row, (ft, sq) in enumerate(zip(G.FLOW_TYPES, seqs)):
        s = G.generate_sample(ft, sq, cfg)
        mag = np.where(s.valid, np.hypot(*s.flow), np.nan)
        diff = s.deflected.astype(float) - s.reference.astype(float)
        overlay = np.dstack([s.reference / 255.0] * 3)
        for inst in s.instances:
            overlay[inst.mask] = 0.45 * overlay[inst.mask] + 0.55 * np.array(COLORS[inst.cls])
        panels = [(s.reference, "gray", "reference"), (diff, "RdBu_r", "deflected - reference"),
                  (mag, "magma", f"|u| GT, peak {s.meta['peak_displacement_px']:.1f} px"),
                  (overlay, None, "labels")]
        lim = np.percentile(np.abs(diff), 99.5)
        for ax, (img, cmap, title) in zip(axes[row], panels):
            kw = {"vmin": -lim, "vmax": lim} if cmap == "RdBu_r" else {}
            ax.imshow(img, cmap=cmap, **kw)
            ax.set_title(f"{ft}: {title}" if ax is axes[row][0] else title, fontsize=9)
            ax.axis("off")
        # Zoom on the strongest displacement to show pattern distortion at pixel scale.
        r0, c0 = np.unravel_index(np.nanargmax(np.nan_to_num(mag)), mag.shape)
        r0, c0 = np.clip(r0 - 32, 0, mag.shape[0] - 64), np.clip(c0 - 32, 0, mag.shape[1] - 64)
        inset = axes[row][1].inset_axes([0.0, 0.0, 0.35, 0.35])
        inset.imshow(s.deflected[r0:r0 + 64, c0:c0 + 64], cmap="gray")
        inset.set_xticks([]), inset.set_yticks([])
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=90)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
