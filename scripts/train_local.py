"""M3 step 3b: train RF-DETR Seg locally (MPS on Apple silicon, CUDA on Colab, or CPU),
then report per-class mask mAP on the test split. Free, no credits.

    uv sync --group train
    uv run python scripts/train_local.py --data data/synthetic/v1_rf_dis --size medium --epochs 50
    uv run python scripts/train_local.py --data data/synthetic/smoke_export --size nano --epochs 1 --batch 2
    # replicate Roboflow's hosted default recipe for rfdetr-seg-medium
    uv run python scripts/train_local.py --data ... --epochs 100 --set cls_loss_coef=5 \
        --set early_stopping=true --set early_stopping_patience=15 --set early_stopping_min_delta=0.0005

Extra --set values are passed to rfdetr's TrainConfig, which rejects unknown keys.
"""

import argparse
import json
from pathlib import Path

import rfdetr

from bosflow.rf.evaluate import evaluate_model

SIZES = {"nano": "RFDETRSegNano", "small": "RFDETRSegSmall", "medium": "RFDETRSegMedium",
         "large": "RFDETRSegLarge"}


def main():
    # DataLoader workers use spawn on macOS and re-import this module, so no
    # top-level side effects.
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--size", default="medium", choices=SIZES)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4, help="grad accumulation; batch x accum ~ 16")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default=None, help="mps | cuda | cpu (default: auto)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="extra rfdetr TrainConfig field (value parsed as JSON when possible)")
    args = ap.parse_args()

    out = args.out or Path("runs") / f"{args.data.name}_{args.size}"
    model = getattr(rfdetr, SIZES[args.size])()
    kw = {"dataset_dir": str(args.data), "epochs": args.epochs, "batch_size": args.batch,
          "grad_accum_steps": args.accum, "lr": args.lr, "output_dir": str(out), "seed": args.seed,
          "log_per_class_metrics": True}
    if args.device:
        kw["device"] = args.device
    for item in args.set:
        key, _, value = item.partition("=")
        try:
            kw[key] = json.loads(value)
        except json.JSONDecodeError:
            kw[key] = value
    model.train(**kw)

    best = rfdetr.RFDETR.from_checkpoint(out / "checkpoint_best_total.pth")
    report = evaluate_model(best, args.data / "test")
    (out / "mask_map_test.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
