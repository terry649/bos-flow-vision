"""M2: benchmark displacement estimators against synthetic ground truth.

    uv run python scripts/benchmark_displacement.py                    # full, config counts
    uv run python scripts/benchmark_displacement.py --scenes 2 --out results/m2_smoke

Writes records.csv, summary.md, and epe_vs_noise.png to --out.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bosflow import generate as G
from bosflow import lineage
from bosflow.displacement import benchmark as B


def table(records, metric, noise):
    methods = list(dict.fromkeys(r["method"] for r in records))
    lines = ["| flow type | " + " | ".join(methods) + " |",
             "|---|" + "---:|" * len(methods)]
    for ft in [*G.FLOW_TYPES, "all"]:
        row = [ft]
        for m in methods:
            v = [r[metric] for r in records if r["noise"] == noise and r["method"] == m
                 and (ft == "all" or r["flow_type"] == ft) and metric in r]
            row.append(f"{np.mean(v):.3f}" if v else "n/a")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=None, help="scenes per flow type")
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=Path("results/m2"))
    args = ap.parse_args()

    bcfg, gcfg = B.load_benchmark_config(), G.load_config()
    recs = B.run(bcfg, gcfg, args.scenes, args.methods)
    args.out.mkdir(parents=True, exist_ok=True)
    lineage.write_manifest(args.out / "manifest.json", {"benchmark": bcfg, "generator": gcfg},
                           scenes_per_type=args.scenes or bcfg["scenes_per_flow_type"])
    keys = sorted({k for r in recs for k in r})
    with open(args.out / "records.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(recs)

    levels = list(bcfg["noise_levels"])
    methods = list(dict.fromkeys(r["method"] for r in recs))
    n = len({(r["flow_type"], r["scene"]) for r in recs})
    intro = (f"# M2 displacement benchmark\n\n{n} scenes, noise levels {', '.join(levels)}. "
             "EPE in pixels over valid pixels; feature EPE inside labeled flow features; "
             "peak ratio is estimated over true magnitude in the top 5% of feature pixels.\n")
    md = [intro]
    for metric, title in [("epe", "EPE, all valid pixels"),
                          ("epe_feature", "EPE inside flow features"),
                          ("rel_epe_feature", "Relative EPE inside flow features"),
                          ("peak_ratio", "Peak displacement recovery (1 is perfect)")]:
        for level in levels:
            md.append(f"\n## {title}, noise = {level}\n\n{table(recs, metric, level)}\n")
    md.append("\n## Runtime per image pair, seconds (mean)\n")
    for m in methods:
        md.append(f"- {m}: {np.mean([r['seconds'] for r in recs if r['method'] == m]):.3f}")
    (args.out / "summary.md").write_text("\n".join(md) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ["epe_feature", "peak_ratio"]):
        for m in methods:
            ax.plot(levels, [np.mean([r[metric] for r in recs if r["method"] == m
                                      and r["noise"] == lv and metric in r]) for lv in levels],
                    marker="o", label=m)
        ax.set_title(metric)
        ax.set_xlabel("noise level")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("px")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(args.out / "epe_vs_noise.png", dpi=110)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
