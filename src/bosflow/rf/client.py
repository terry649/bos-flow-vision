"""Roboflow platform operations: project, upload, version, hosted training, evaluations.

API per roboflow 1.5.1 (source-verified 2026-09-28). The key is read from the
ROBOFLOW_API_KEY environment variable and never logged or written anywhere.
Publishing to Universe has no API; it is a web UI action (Projects, "Make Public").
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

PROJECT_TYPE = "instance-segmentation"
LICENSE = "MIT"
ANNOTATION_GROUP = "flow-features"
# Keep the images exactly as rendered: no resize, no auto-orient, no augmentation.
VERSION_SETTINGS = {"preprocessing": {"auto-orient": False}, "augmentation": {}}


def connect():
    import roboflow

    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise RuntimeError("Set ROBOFLOW_API_KEY in the environment (see .env.example).")
    return roboflow.Roboflow(api_key=key).workspace()


def get_or_create_project(workspace, name: str):
    """``name`` should be a slug (lowercase, hyphens), which Roboflow uses as the project id."""
    existing = {pid.split("/")[-1] for pid in workspace.projects()}
    if name in existing:
        return workspace.project(name)
    return workspace.create_project(name, PROJECT_TYPE, LICENSE, ANNOTATION_GROUP)


def single_image_coco(split_coco: dict, file_name: str) -> dict:
    """COCO document restricted to one image, with its annotations renumbered."""
    img = next(i for i in split_coco["images"] if i["file_name"] == file_name)
    anns = [dict(a, image_id=1, id=k + 1) for k, a in
            enumerate(a for a in split_coco["annotations"] if a["image_id"] == img["id"])]
    return dict(split_coco, images=[dict(img, id=1)], annotations=anns)


def iter_export(export_dir: Path):
    """Yield (split, image path, single-image COCO dict, metadata) for an export."""
    meta = {}
    for line in (export_dir / "metadata.jsonl").read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            meta[rec["file_name"]] = rec
    for split in ("train", "valid", "test"):
        coco = json.loads((export_dir / split / "_annotations.coco.json").read_text())
        for img in coco["images"]:
            fn = img["file_name"]
            yield split, export_dir / split / fn, single_image_coco(coco, fn), meta[fn]


def upload_export(project, export_dir: Path | str, batch_name: str, limit: int | None = None,
                  dry_run: bool = False, progress=print) -> int:
    """Upload images with annotations, split, tags, and flow metadata. Returns the count."""
    export_dir = Path(export_dir)
    n = 0
    with tempfile.TemporaryDirectory() as tmp:
        for split, image_path, coco, meta in iter_export(export_dir):
            if limit is not None and n >= limit:
                break
            ann_path = Path(tmp) / f"{image_path.stem}.json"
            ann_path.write_text(json.dumps(coco))
            tags = [f"flow_type:{meta['flow_type']}", f"estimator:{meta['estimator']}"]
            metadata = {
                "flow_type": meta["flow_type"], "estimator": meta["estimator"],
                "magnitude_scale_px": round(meta["magnitude_scale_px"], 4),
                "peak_displacement_px": round(meta["peak_displacement_px"], 4),
                "synthetic": "true",
            }
            for key in ("M1", "shock_angle_deg", "shock_radius", "turn_angle_deg"):
                if key in meta["flow"]:
                    metadata[key] = round(float(meta["flow"][key]), 5)
            if dry_run:
                progress(f"[dry run] {split} {image_path.name} tags={tags}")
            else:
                project.upload(str(image_path), annotation_path=str(ann_path), split=split,
                               batch_name=batch_name, tag_names=tags, metadata=metadata,
                               num_retry_uploads=2)
            n += 1
            if n % 50 == 0:
                progress(f"uploaded {n}")
    return n


def generate_version(project) -> int:
    return int(project.generate_version(VERSION_SETTINGS))


def start_hosted_training(project, version: int, model_type: str, epochs: int | None = None):
    """Queue hosted training without blocking; returns the SDK Training object."""
    return project.version(version).create_training(model_type=model_type, epochs=epochs)


def per_class_evals(workspace, project_name: str, version: int) -> list[dict]:
    """Per-class metrics from Roboflow's automatic evaluation on the test split."""
    out = []
    for ev in workspace.evals(project=project_name, version=version, status="done"):
        rows = ev.performance_by_class()
        out.append({"summary": ev.summary, "per_class": rows})
    return out
