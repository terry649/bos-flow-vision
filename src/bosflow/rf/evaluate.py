"""Per-class mask mAP with pycocotools.

rfdetr 1.11 logs per-class AP from the box evaluator even for segmentation models
(training/callbacks/coco_eval.py), so per-class *mask* AP is computed here from
model predictions on the held-out split.
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def detections_to_coco(detections, image_id: int, class_ids: list[int]) -> list[dict]:
    """supervision Detections with masks -> COCO result records (RLE masks)."""
    out = []
    if detections.mask is None:
        return out
    for m, cls, score in zip(detections.mask, detections.class_id, detections.confidence):
        rle = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("ascii")
        out.append({"image_id": image_id, "category_id": class_ids[int(cls)],
                    "segmentation": rle, "score": float(score)})
    return out


def coco_mask_map(gt_json: Path | str, results: list[dict]) -> dict:
    """Overall and per-class mask AP (AP50-95, AP50) from COCO results."""
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(gt_json))
        if not results:
            return {"mAP50_95": 0.0, "mAP50": 0.0, "per_class": {}}
        dt = gt.loadRes(results)
        ev = COCOeval(gt, dt, iouType="segm")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    prec = ev.eval["precision"]  # [T iou, R recall, K class, A area, M maxdets]
    per_class = {}
    for k, cat_id in enumerate(ev.params.catIds):
        name = gt.loadCats(cat_id)[0]["name"]
        p = prec[:, :, k, 0, -1]
        p50 = prec[0, :, k, 0, -1]
        per_class[name] = {
            "AP50_95": float(p[p > -1].mean()) if (p > -1).any() else float("nan"),
            "AP50": float(p50[p50 > -1].mean()) if (p50 > -1).any() else float("nan"),
            "n_gt": len(gt.getAnnIds(catIds=[cat_id])),
        }
    return {"mAP50_95": float(ev.stats[0]), "mAP50": float(ev.stats[1]), "per_class": per_class}


def evaluate_model(model, split_dir: Path | str, threshold: float = 0.3) -> dict:
    """Run an RF-DETR model over a COCO split directory and score its masks."""
    split_dir = Path(split_dir)
    gt_json = split_dir / "_annotations.coco.json"
    doc = json.loads(gt_json.read_text())
    # rfdetr class indices follow the sorted category ids of the training annotations.
    class_ids = sorted(c["id"] for c in doc["categories"])
    results = []
    for img in doc["images"]:
        rgb = np.asarray(Image.open(split_dir / img["file_name"]).convert("RGB"))
        det = model.predict(rgb, threshold=threshold)
        results += detections_to_coco(det, img["id"], class_ids)
    return coco_mask_map(gt_json, results)
