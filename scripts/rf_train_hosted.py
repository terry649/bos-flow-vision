"""M3 step 3a: queue hosted RF-DETR Seg training on a Roboflow version (costs credits:
1 credit per 30 GPU-minutes), then later fetch per-class evaluation results.

    uv run python scripts/rf_train_hosted.py --project bos-flow-features --version 1 --model rfdetr-seg-medium
    uv run python scripts/rf_train_hosted.py --project bos-flow-features --version 1 --results
"""

import argparse
import json

from bosflow.rf import client

ap = argparse.ArgumentParser()
ap.add_argument("--project", required=True)
ap.add_argument("--version", type=int, required=True)
ap.add_argument("--model", default="rfdetr-seg-medium")
ap.add_argument("--epochs", type=int, default=None)
ap.add_argument("--results", action="store_true", help="fetch evaluations instead of training")
args = ap.parse_args()

ws = client.connect()
if args.results:
    print(json.dumps(client.per_class_evals(ws, args.project, args.version), indent=1,
                     default=str))
else:
    project = ws.project(args.project)
    training = client.start_hosted_training(project, args.version, args.model, args.epochs)
    print(f"queued: {training}")
