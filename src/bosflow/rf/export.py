"""Build the segmentation dataset: displacement-magnitude images with COCO polygons.

Images come from an *estimated* displacement field (a dense optical flow or DIR
method from M2), not the ground truth, so the segmentation model trains on what a
BOS practitioner actually has. Labels come from the physics, so they are exact.

Layout (the standard COCO export layout, one annotation file per split):

    <out>/train/_annotations.coco.json + images
    <out>/valid/...
    <out>/test/...
    <out>/metadata.jsonl      per-image flow type and physics, keyed by file name
    <out>/manifest.json       lineage
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from bosflow import labels, lineage

SPLITS = {"train": 0.7, "valid": 0.2, "test": 0.1}


def encode_magnitude(u: np.ndarray, valid: np.ndarray, gamma: float = 0.5,
                     pct: float = 99.5) -> tuple[np.ndarray, float]:
    """8-bit displacement magnitude with robust per-image scaling and a gamma curve.

    Peak displacements span 0.3 to 12 px across the dataset, so a fixed scale would
    leave weak features invisible. The scale (px at full white) is returned and
    stored as metadata, keeping the physical magnitude recoverable.
    """
    mag = np.where(valid, np.hypot(u[0], u[1]), 0.0)
    scale = float(np.percentile(mag[valid], pct)) if valid.any() else 1.0
    scale = max(scale, 1e-3)
    img = np.clip(mag / scale, 0.0, 1.0) ** gamma
    return np.round(255.0 * img).astype(np.uint8), scale


def assign_splits(names_by_type: dict[str, list[str]], seed: int) -> dict[str, str]:
    """Stratified split by flow type, deterministic in ``seed``."""
    rng = np.random.default_rng(seed)
    out = {}
    for ft in sorted(names_by_type):
        names = sorted(names_by_type[ft])
        rng.shuffle(names)
        n = len(names)
        n_train = round(SPLITS["train"] * n)
        n_valid = round(SPLITS["valid"] * n)
        for i, name in enumerate(names):
            out[name] = "train" if i < n_train else "valid" if i < n_train + n_valid else "test"
    return out


def export(dataset_dir: Path | str, out_dir: Path | str, estimator: str,
           seed: int = 0, limit_per_type: int | None = None, progress=print) -> dict:
    from bosflow.displacement.benchmark import _method

    dataset_dir, out_dir = Path(dataset_dir), Path(out_dir)
    index = [json.loads(line) for line in (dataset_dir / "index.jsonl").read_text().splitlines()]
    by_type: dict[str, list[str]] = {}
    for rec in index:
        by_type.setdefault(rec["flow_type"], []).append(rec["name"])
    if limit_per_type:
        by_type = {k: sorted(v)[:limit_per_type] for k, v in by_type.items()}
    split_of = assign_splits(by_type, seed)
    estimate = None if estimator == "gt" else _method(estimator)

    coco = {s: {"images": [], "annotations": []} for s in SPLITS}
    for s in SPLITS:
        (out_dir / s).mkdir(parents=True, exist_ok=True)
    meta_lines = []
    for k, name in enumerate(sorted(split_of)):
        split = split_of[name]
        sdir = dataset_dir / "samples" / name
        meta = json.loads((sdir / "meta.json").read_text())
        gt = np.load(sdir / "ground_truth.npz")
        if estimate is None:
            u = gt["flow"]
        else:
            ref = np.asarray(Image.open(sdir / "reference.png"))
            defl = np.asarray(Image.open(sdir / "deflected.png"))
            u = estimate(ref, defl)
        img, scale = encode_magnitude(u, gt["valid"])
        fname = f"{name}.png"
        Image.fromarray(np.dstack([img] * 3)).save(out_dir / split / fname)

        c = coco[split]
        image_id = len(c["images"]) + 1
        c["images"].append({"id": image_id, "file_name": fname,
                            "height": img.shape[0], "width": img.shape[1]})
        for inst, mask in zip(meta["instances"], gt["instance_masks"]):
            a = labels.instance_annotation(len(c["annotations"]) + 1, image_id, inst["class"],
                                           mask.astype(bool), inst["polygons"])
            if a:
                c["annotations"].append(a)
        meta_lines.append(json.dumps({
            "file_name": fname, "split": split, "flow_type": meta["flow"]["flow_type"],
            "estimator": estimator, "magnitude_scale_px": scale,
            "peak_displacement_px": meta["peak_displacement_px"], "flow": meta["flow"],
        }, default=float))
        if (k + 1) % 50 == 0:
            progress(f"exported {k + 1}/{len(split_of)}")

    for split, c in coco.items():
        doc = labels.build_coco(c["images"], c["annotations"],
                                f"bosflow synthetic BOS, {estimator} displacement magnitude")
        (out_dir / split / "_annotations.coco.json").write_text(json.dumps(doc))
    (out_dir / "metadata.jsonl").write_text("\n".join(meta_lines) + "\n")
    source_manifest = json.loads((dataset_dir / "manifest.json").read_text())
    counts = {s: len(c["images"]) for s, c in coco.items()}
    lineage.write_manifest(out_dir / "manifest.json",
                           {"estimator": estimator, "split_seed": seed, "splits": SPLITS},
                           source_dataset=str(dataset_dir),
                           source_config_sha256=source_manifest["config_sha256"],
                           source_git_commit=source_manifest["git_commit"], counts=counts)
    return counts


def export_sequences(seq_root: Path | str, out_dir: Path | str, estimator: str) -> int:
    """Blast sequences -> displacement-magnitude frames plus per-frame ground truth.

    Writes ``<out>/<sequence>/frame_XX.png`` and ``<out>/<sequence>/truth.json`` with
    each frame's time, true shock radius (px and m), and physics metadata.
    """
    from bosflow.displacement.benchmark import _method

    seq_root, out_dir = Path(seq_root), Path(out_dir)
    estimate = None if estimator == "gt" else _method(estimator)
    n = 0
    for seq in sorted(p for p in seq_root.iterdir() if p.is_dir()):
        (out_dir / seq.name).mkdir(parents=True, exist_ok=True)
        truth = []
        for frame in sorted(p for p in seq.iterdir() if p.is_dir()):
            meta = json.loads((frame / "meta.json").read_text())
            gt = np.load(frame / "ground_truth.npz")
            if estimate is None:
                u = gt["flow"]
            else:
                u = estimate(np.asarray(Image.open(frame / "reference.png")),
                             np.asarray(Image.open(frame / "deflected.png")))
            img, scale = encode_magnitude(u, gt["valid"])
            fname = f"{frame.name}.png"
            Image.fromarray(np.dstack([img] * 3)).save(out_dir / seq.name / fname)
            s = meta["optics"]["object_pixel_size"]
            pose = meta["flow"]["pose"]
            rows, cols = img.shape
            truth.append({
                "file_name": fname, "t": meta["flow"]["t"],
                "shock_radius_m": meta["flow"]["shock_radius"],
                "shock_radius_px": meta["flow"]["shock_radius"] / s,
                # Object-plane origin is the image center, y up.
                "center_px": [0.5 * (cols - 1) + pose["x0"] / s, 0.5 * (rows - 1) - pose["y0"] / s],
                "object_pixel_size_m": s, "E": meta["flow"]["E"], "rho0": meta["flow"]["rho0"],
                "magnitude_scale_px": scale,
            })
            n += 1
        (out_dir / seq.name / "truth.json").write_text(json.dumps(truth, indent=1))
    return n
