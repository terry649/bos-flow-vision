"""M4 serving: run the segmentation model with Roboflow `inference`, annotate with
`supervision`, and track blast fronts across sequence frames with ByteTrack.

Runs in the serving environment (deploy/). Two model sources:

    # the Roboflow-trained model (weights downloaded once, inference runs locally)
    uv run --project deploy python deploy/predict.py --model-id bos-flow-features/1 ...
    # a locally trained RF-DETR checkpoint, loaded through inference-models
    uv run --project deploy python deploy/predict.py --checkpoint runs/x/checkpoint_best_total.pth \
        --model-type rfdetr-seg-medium --resolution 432 ...

Inputs: --images DIR (a COCO split directory) and/or --sequences DIR (one folder of
frame_XX.png per sequence). Outputs under --out: predictions JSON with RLE masks and,
for sequences, tracker ids plus an annotated MP4 per sequence.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from pycocotools import mask as mask_utils
from trackers import ByteTrackTracker

CLASSES = ["shock", "expansion_fan", "shear_layer"]
REPO = Path(__file__).resolve().parents[1]


def api_key() -> str | None:
    key = os.environ.get("ROBOFLOW_API_KEY")
    env = REPO / ".env"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "ROBOFLOW_API_KEY":
                key = value.strip().strip("'\"")
    return key


class Model:
    """Uniform predict(image_bgr) -> sv.Detections with masks and data['class_name']."""

    def __init__(self, args):
        self.confidence = args.confidence
        self.id_to_name: dict[int, str] = {}
        if args.model_id:
            from inference import get_model

            self.name = f"roboflow:{args.model_id}"
            self._m = get_model(model_id=args.model_id, api_key=api_key())
            self._local = False
        else:
            import torch
            from inference_models import AutoModel

            # rfdetr 1.11 checkpoints carry a keypoint buffer inference-models does not
            # expect; strip it into a temporary copy.
            ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            ck["model"].pop("_kp_active_mask", None)
            tmpdir = tempfile.mkdtemp(prefix="bosflow-ckpt-")
            atexit.register(shutil.rmtree, tmpdir, ignore_errors=True)
            stripped = Path(tmpdir) / "checkpoint.pth"
            torch.save(ck, stripped)
            # rfdetr heads have num_classes + 1 outputs. Locally trained checkpoints put the
            # k-th training category at index k (shock = 0) with an unused last slot
            # (verified by scripts/check_serving_parity.py). Checkpoints trained from a
            # Roboflow export put the annotation group at index 0 and the classes in
            # alphabetical order after it; pass --labels for those.
            self.labels = args.labels.split(",") if args.labels else [*CLASSES, "_unused"]
            self.name = f"local:{args.checkpoint}"
            self._m = AutoModel.from_pretrained(
                str(stripped), model_type=args.model_type, task_type="instance-segmentation",
                device=args.device, labels=self.labels, resolution=args.resolution)
            self._local = True

    def predict(self, image_bgr: np.ndarray) -> sv.Detections:
        if self._local:
            det = self._m(image_bgr, confidence=self.confidence)[0].to_supervision()
        else:
            # Polygon masks (the default) cannot represent holes, so a blast-front ring
            # comes back as a filled disk. RLE is lossless.
            res = self._m.infer(image_bgr, confidence=self.confidence,
                                response_mask_format="rle")[0]
            det = sv.Detections.from_inference(res)
            # Remember the hosted model's id -> name mapping from its own responses.
            for cid, name in zip(det.class_id, det.data.get("class_name", [])):
                self.id_to_name[int(cid)] = str(name)
        return det

    def class_names(self, det: sv.Detections) -> list[str]:
        """Names from class ids; trackers may drop det.data on empty frames."""
        if self._local:
            return [self.labels[int(c)] for c in det.class_id]
        return [self.id_to_name.get(int(c), str(c)) for c in det.class_id]


def to_records(det: sv.Detections, names: list[str]) -> list[dict]:
    out = []
    for i in range(len(det)):
        rle = mask_utils.encode(np.asfortranarray(det.mask[i].astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("ascii")
        rec = {"class_name": names[i],
               "confidence": float(det.confidence[i]),
               "bbox_xyxy": [float(v) for v in det.xyxy[i]], "rle": rle}
        if det.tracker_id is not None:
            rec["tracker_id"] = int(det.tracker_id[i])
        out.append(rec)
    return out


def predict_images(model: Model, image_dir: Path, out: Path) -> None:
    doc = json.loads((image_dir / "_annotations.coco.json").read_text())
    images = []
    for img in doc["images"]:
        frame = cv2.imread(str(image_dir / img["file_name"]))
        det = model.predict(frame)
        images.append({"file_name": img["file_name"],
                       "detections": to_records(det, model.class_names(det))})
    (out / f"predictions_{image_dir.name}.json").write_text(
        json.dumps({"model": model.name, "images": images}))


def track_sequences(model: Model, seq_root: Path, out: Path, fps: int) -> None:
    mask_ann = sv.MaskAnnotator(opacity=0.45)
    label_ann = sv.LabelAnnotator(text_scale=0.45, text_padding=4)
    trace_ann = sv.TraceAnnotator(trace_length=12, thickness=2)
    results = {}
    for seq in sorted(p for p in seq_root.iterdir() if p.is_dir()):
        # 12 frames per sequence: confirm tracks immediately and never drop them.
        tracker = ByteTrackTracker(lost_track_buffer=12, frame_rate=fps,
                                   track_activation_threshold=0.3,
                                   minimum_consecutive_frames=1, minimum_iou_threshold=0.1,
                                   high_conf_det_threshold=0.5)
        frames = sorted(seq.glob("frame_*.png"))
        first = cv2.imread(str(frames[0]))
        info = sv.VideoInfo(width=first.shape[1], height=first.shape[0], fps=fps)
        records = []
        with sv.VideoSink(str(out / f"{seq.name}.mp4"), info) as sink:
            for f in frames:
                frame = cv2.imread(str(f))
                det = model.predict(frame)
                det = tracker.update(det)
                # ByteTrack confirms a new track only on its second match, so first-frame
                # detections carry tracker_id -1. Keep them in the records (the analysis
                # backfills them); draw only confirmed tracks.
                shown = det[det.tracker_id >= 0] if det.tracker_id is not None else det
                names = model.class_names(shown)
                labels = [f"#{t} {c} {p:.2f}" for t, c, p in
                          zip(shown.tracker_id, names, shown.confidence)]
                scene = mask_ann.annotate(frame.copy(), shown)
                scene = label_ann.annotate(scene, shown, labels=labels)
                scene = trace_ann.annotate(scene, shown)
                sink.write_frame(scene)
                records.append({"file_name": f.name,
                                "detections": to_records(det, model.class_names(det))})
        results[seq.name] = records
    (out / "predictions_sequences.json").write_text(
        json.dumps({"model": model.name, "sequences": results}))


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--model-id", help="Roboflow model id, e.g. bos-flow-features/1")
    src.add_argument("--checkpoint", type=Path, help="local rfdetr checkpoint")
    ap.add_argument("--model-type", default="rfdetr-seg-medium")
    ap.add_argument("--resolution", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--confidence", type=float, default=0.3)
    ap.add_argument("--labels", default=None,
                    help="comma-separated class name per output index (local checkpoints only)")
    ap.add_argument("--images", type=Path, action="append", default=[])
    ap.add_argument("--sequences", type=Path, default=None)
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    model = Model(args)
    for d in args.images:
        predict_images(model, d, args.out)
        print(f"predicted {d}")
    if args.sequences:
        track_sequences(model, args.sequences, args.out, args.fps)
        print(f"tracked sequences in {args.sequences}")


if __name__ == "__main__":
    main()
