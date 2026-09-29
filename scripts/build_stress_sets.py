"""M4: generate the synthetic stress-test sets defined in configs/stress.yaml and
export each as a COCO test split (both input encodings).

    uv run python scripts/build_stress_sets.py --out data/synthetic/stress

Layout: <out>/<set>/raw (generated samples) and <out>/<set>/<encoding>/test/...
Sets with ``base`` reuse another set's raw samples and change only the estimator.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml

from bosflow import generate as G
from bosflow.rf.export import export


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/synthetic/stress"))
    ap.add_argument("--config", type=Path, default=G.CONFIG_DIR / "stress.yaml")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    scfg = yaml.safe_load(args.config.read_text())
    base_cfg = G.load_config()
    for i, (name, spec) in enumerate(scfg["sets"].items()):
        if args.only and name not in args.only:
            continue
        set_dir = args.out / name
        raw = args.out / spec.get("base", name) / "raw"
        if "base" not in spec:
            cfg = G.apply_overrides(base_cfg, spec.get("overrides", []))
            types = spec.get("types", list(G.FLOW_TYPES))
            n = spec.get("per_type", scfg["per_type"])
            G.generate_dataset(raw, cfg, counts={t: n for t in types}, seed=scfg["seed"] + i)
        for encoding in ("mag", "vec"):
            tmp = set_dir / f"_{encoding}_all"
            export(raw, tmp, spec.get("estimator", "dis"), encoding=encoding,
                   progress=lambda *_: None)
            # Stress sets are evaluation-only: merge the three splits into one test split.
            merge_to_test(tmp, set_dir / encoding)
            shutil.rmtree(tmp)
        (set_dir / "spec.json").write_text(json.dumps(spec, indent=1))
        print(f"{name}: done")


def merge_to_test(src: Path, dst: Path) -> None:
    from bosflow import labels

    (dst / "test").mkdir(parents=True, exist_ok=True)
    images, anns = [], []
    for split in ("train", "valid", "test"):
        doc = json.loads((src / split / "_annotations.coco.json").read_text())
        remap = {}
        for im in doc["images"]:
            remap[im["id"]] = len(images) + 1
            images.append(dict(im, id=remap[im["id"]]))
            shutil.copy(src / split / im["file_name"], dst / "test" / im["file_name"])
        for a in doc["annotations"]:
            anns.append(dict(a, id=len(anns) + 1, image_id=remap[a["image_id"]]))
    (dst / "test" / "_annotations.coco.json").write_text(
        json.dumps(labels.build_coco(images, anns, f"stress set {dst.parent.name}")))
    shutil.copy(src / "metadata.jsonl", dst / "metadata.jsonl")
    shutil.copy(src / "manifest.json", dst / "manifest.json")


if __name__ == "__main__":
    main()
