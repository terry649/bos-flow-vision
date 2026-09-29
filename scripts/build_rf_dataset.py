"""M3 step 1: export displacement-magnitude images with COCO labels (local, no network).

    uv run python scripts/build_rf_dataset.py --dataset data/synthetic/v1 --estimator dis \
        --out data/synthetic/v1_rf_dis
"""

import argparse
from pathlib import Path

from bosflow.rf.export import export

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", type=Path, required=True)
ap.add_argument("--estimator", default="dis",
                help="farneback | lucas_kanade | dis | tvl1 | demons | raft | gt")
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--limit-per-type", type=int, default=None)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--encoding", default="mag", choices=["mag", "vec"])
args = ap.parse_args()
print(export(args.dataset, args.out, args.estimator, args.seed, args.limit_per_type,
             encoding=args.encoding))
