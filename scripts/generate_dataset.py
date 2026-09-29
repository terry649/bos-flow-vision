"""Generate the synthetic BOS dataset.

    uv run python scripts/generate_dataset.py --per-type 4 --out data/synthetic/smoke
    uv run python scripts/generate_dataset.py --out data/synthetic/v1          # counts from config
    uv run python scripts/generate_dataset.py --sequences --out data/synthetic/v1

Output layout under --out:
    samples/<flow>_<k>/{reference.png, deflected.png, ground_truth.npz, meta.json}
    samples/dispmag_gt/*.png + _annotations.coco.json   (GT magnitude, for inspection)
    sequences/blast_<k>/frame_<j>/...                    (with --sequences)
    index.jsonl
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from bosflow import generate as G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=G.CONFIG_DIR)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--per-type", type=int, default=None,
                    help="override the per-flow-type counts in dataset.yaml")
    ap.add_argument("--sequences", action="store_true", help="also write blast sequences")
    args = ap.parse_args()

    cfg = G.load_config(args.config)
    counts = {ft: args.per_type for ft in G.FLOW_TYPES} if args.per_type else None
    t0 = time.time()
    metas = G.generate_dataset(args.out, cfg, counts=counts, seed=args.seed)
    print(f"{len(metas)} samples in {time.time() - t0:.1f} s -> {args.out}")

    if args.sequences:
        sc = cfg["sequences"]["blast"]
        n = sc["count"] if args.per_type is None else max(1, args.per_type // 2)
        seed = cfg["seed"] if args.seed is None else args.seed
        root = np.random.SeedSequence([seed, 1])  # independent of the image stream
        for k, sq in enumerate(root.spawn(n)):
            frames = G.generate_blast_sequence(sq, cfg, sc["frames"])
            for j, f in enumerate(frames):
                G.write_sample(f, args.out / "sequences" / f"blast_{k:03d}", f"frame_{j:02d}")
        print(f"{n} blast sequences x {sc['frames']} frames")


if __name__ == "__main__":
    main()
