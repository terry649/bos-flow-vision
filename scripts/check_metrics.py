"""Fail if benchmark metrics regress past the thresholds in configs/gates.yaml.

    uv run python scripts/check_metrics.py results/m2/records.csv [--summary FILE]

Gates are means over all scenes for one (method, noise level). ``max`` gates are
upper bounds (errors) and ``min`` gates lower bounds (peak recovery).
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import yaml

GATES = Path(__file__).resolve().parents[1] / "configs" / "gates.yaml"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("records", type=Path)
    ap.add_argument("--gates", type=Path, default=GATES)
    ap.add_argument("--summary", type=Path, default=None, help="append a markdown table here")
    args = ap.parse_args()

    groups: dict[tuple, list[dict]] = defaultdict(list)
    with open(args.records) as fh:
        for r in csv.DictReader(fh):
            groups[(r["method"], r["noise"])].append(r)

    gates = yaml.safe_load(args.gates.read_text())["gates"]
    rows, failed = [], False
    for g in gates:
        recs = groups.get((g["method"], g["noise"]), [])
        vals = [float(r[g["metric"]]) for r in recs if r.get(g["metric"])]
        if not vals:
            rows.append((g, None, "missing"))
            failed = True
            continue
        mean = sum(vals) / len(vals)
        ok = mean <= g["max"] if "max" in g else mean >= g["min"]
        failed |= not ok
        rows.append((g, mean, "pass" if ok else "FAIL"))

    lines = ["| method | noise | metric | value | gate | result |", "|---|---|---|---:|---:|---|"]
    for g, v, res in rows:
        bound = f"<= {g['max']}" if "max" in g else f">= {g['min']}"
        val = "n/a" if v is None else f"{v:.4f}"
        lines.append(f"| {g['method']} | {g['noise']} | {g['metric']} | {val} | {bound} | {res} |")
    table = "\n".join(lines)
    print(table)
    if args.summary:
        with open(args.summary, "a") as fh:
            fh.write("## Displacement benchmark gates\n\n" + table + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
