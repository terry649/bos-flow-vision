"""Check that the serving path (deploy/predict.py, Roboflow inference) reproduces
rfdetr's own predictions for the same checkpoint: same classes, near-identical masks.

    uv run --group train python scripts/check_serving_parity.py \
        --checkpoint runs/x/checkpoint_best_total.pth --images DIR --serving DIR/predictions_valid.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import rfdetr
from PIL import Image
from pycocotools import mask as mask_utils

CLASSES = ["shock", "expansion_fan", "shear_layer"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--serving", type=Path, required=True)
    ap.add_argument("--confidence", type=float, default=0.3)
    args = ap.parse_args()

    model = rfdetr.RFDETR.from_checkpoint(args.checkpoint)
    serving = {im["file_name"]: im["detections"]
               for im in json.loads(args.serving.read_text())["images"]}
    worst, n_ref, n_srv, class_mismatch = 1.0, 0, 0, 0
    for fname, srv in sorted(serving.items()):
        rgb = np.asarray(Image.open(args.images / fname).convert("RGB"))
        ref = model.predict(rgb, threshold=args.confidence)
        ref_items = [(CLASSES[int(c)], m) for c, m in zip(ref.class_id, ref.mask)]
        srv_items = [(d["class_name"], mask_utils.decode(d["rle"]).astype(bool)) for d in srv]
        n_ref += len(ref_items)
        n_srv += len(srv_items)
        for cls, m in ref_items:
            ious = [((m & s).sum() / max((m | s).sum(), 1), c) for c, s in srv_items]
            best_iou, best_cls = max(ious, default=(0.0, None))
            worst = min(worst, best_iou)
            class_mismatch += best_cls != cls
            print(f"{fname}: rfdetr {cls:14s} -> serving {best_cls!s:14s} IoU {best_iou:.3f}")
    print(f"detections rfdetr={n_ref} serving={n_srv}, class mismatches={class_mismatch}, "
          f"worst matched IoU={worst:.3f}")
    ok = class_mismatch == 0 and n_ref == n_srv and worst > 0.9
    print("PARITY OK" if ok else "PARITY FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
