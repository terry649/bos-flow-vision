"""M3 step 2: upload an export to Roboflow and generate a dataset version.

Requires ROBOFLOW_API_KEY in the environment. Start with a small trial:

    uv run python scripts/rf_upload.py --export data/synthetic/v1_rf_dis --project bos-flow-features --limit 3
    uv run python scripts/rf_upload.py --export data/synthetic/v1_rf_dis --project bos-flow-features --version

Only synthetic exports are accepted: every image must carry synthetic provenance.
"""

import argparse
import json
from pathlib import Path

from bosflow.rf import client

ap = argparse.ArgumentParser()
ap.add_argument("--export", type=Path, required=True)
ap.add_argument("--project", required=True)
ap.add_argument("--batch", default=None, help="upload batch name (default: export dir name)")
ap.add_argument("--limit", type=int, default=None, help="upload only the first N images")
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--version", action="store_true", help="generate a version after upload")
args = ap.parse_args()

# Guard against ever pushing non-synthetic data: exports are built from generated samples.
manifest = json.loads((args.export / "manifest.json").read_text())
if "synthetic" not in manifest.get("source_dataset", ""):
    raise SystemExit("refusing to upload: export does not come from data/synthetic/")

ws = None if args.dry_run else client.connect()
project = None if args.dry_run else client.get_or_create_project(ws, args.project)
n = client.upload_export(project, args.export, args.batch or args.export.name,
                         limit=args.limit, dry_run=args.dry_run)
print(f"uploaded {n} images")
if args.version and not args.dry_run:
    v = client.generate_version(project)
    print(f"generated version {v}")
