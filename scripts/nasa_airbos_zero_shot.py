"""Qualitative zero-shot test on NASA AirBOS 4 images (public NASA content, see
data/public/nasa_airbos/SOURCES.md). Not a metric: no ground truth exists, and the
published images are processed composites, not raw BOS pairs.

The grayscale TIFFs show one signed displacement component on a mid-gray background.
They are mapped to the model's input domain by removing a smooth local background,
taking the absolute value as a magnitude proxy, and applying the training encoding's
robust scale and gamma curve.

    uv run python scripts/nasa_airbos_zero_shot.py \
        --checkpoint runs/v1_dis_medium/checkpoint_best_total.pth --out results/m4_experiments/nasa
"""

import argparse
from pathlib import Path

import numpy as np
import rfdetr
import supervision as sv
from PIL import Image
from scipy.ndimage import median_filter, zoom

SRC = Path(__file__).resolve().parents[1] / "data" / "public" / "nasa_airbos"
CLASSES = ["shock", "expansion_fan", "shear_layer"]


def to_model_domain(gray: np.ndarray, long_side: int) -> np.ndarray:
    f = long_side / max(gray.shape)
    g = zoom(gray.astype(float), f, order=1)
    background = median_filter(g, size=31, mode="nearest")
    mag = np.abs(g - background)
    scale = max(float(np.percentile(mag, 99.5)), 1e-6)
    img = np.round(255.0 * np.clip(mag / scale, 0, 1) ** 0.5).astype(np.uint8)
    return np.dstack([img] * 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--long-side", type=int, default=768)
    ap.add_argument("--threshold", type=float, default=0.3)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    model = rfdetr.RFDETR.from_checkpoint(args.checkpoint)
    mask_ann = sv.MaskAnnotator(opacity=0.45)
    label_ann = sv.LabelAnnotator(text_scale=0.5, text_padding=4)
    lines = ["| image | detections (class, confidence) |", "|---|---|"]
    for tif in sorted(SRC.glob("AD21-*.tif")):
        gray = np.asarray(Image.open(tif).convert("L"))
        rgb = to_model_domain(gray, args.long_side)
        det = model.predict(rgb, threshold=args.threshold)
        labels = [f"{CLASSES[int(c)]} {p:.2f}" for c, p in zip(det.class_id, det.confidence)]
        scene = label_ann.annotate(mask_ann.annotate(rgb.copy(), det), det, labels=labels)
        side = np.hstack([rgb, scene])
        Image.fromarray(side).save(args.out / f"{tif.stem}_zero_shot.png")
        lines.append(f"| {tif.stem} | {', '.join(labels) or 'none'} |")
        print(tif.stem, labels)
    (args.out / "README.md").write_text(
        "# NASA AirBOS 4, qualitative zero-shot\n\n"
        "Images: NASA (NASA Ames Research Center; JT Heineck; schlieren processing Neal "
        "Smith). Used for qualitative illustration only; no NASA endorsement implied. "
        "See data/public/nasa_airbos/SOURCES.md.\n\n"
        "Left: magnitude proxy fed to the model. Right: predicted masks. The aircraft are "
        "photographs pasted over the BOS field in NASA's processing, so predictions on "
        "them are not meaningful.\n\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
