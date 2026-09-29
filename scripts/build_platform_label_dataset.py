"""Arm D dataset: our own PNG images with the labels exactly as the dataset-hosting
platform stored them (read back from its COCO export).

Isolates the effect of the platform's label handling from its JPEG re-encoding: the
images and splits are ours; only the annotations come from the export.

    uv run python scripts/build_platform_label_dataset.py --local data/synthetic/v2_rf_dis \
        --platform runs/rf_v2_export --out data/synthetic/v2_platform_labels
"""

import argparse
import json
import re
import shutil
from pathlib import Path

from bosflow import labels

EXPORT_NAME = re.compile(r"_png\.rf\.[0-9a-f]+\.jpg$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", type=Path, required=True)
    ap.add_argument("--platform", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)
    for split in ("train", "valid", "test"):
        (args.out / split).mkdir(parents=True)
        plat = json.loads((args.platform / split / "_annotations.coco.json").read_text())
        local = json.loads((args.local / split / "_annotations.coco.json").read_text())
        local_cat = {c["name"]: c["id"] for c in local["categories"]}
        plat_name = {c["id"]: c["name"] for c in plat["categories"]}
        by_name = {im["file_name"]: im for im in local["images"]}
        images, anns = [], []
        for im in plat["images"]:
            fname = EXPORT_NAME.sub(".png", im["file_name"])
            if fname not in by_name:
                continue
            new_id = len(images) + 1
            images.append(dict(by_name[fname], id=new_id))
            shutil.copy(args.local / split / fname, args.out / split / fname)
            for a in plat["annotations"]:
                name = plat_name[a["category_id"]]
                if a["image_id"] == im["id"] and name in local_cat:
                    anns.append(dict(a, id=len(anns) + 1, image_id=new_id,
                                     category_id=local_cat[name]))
        doc = labels.build_coco(images, anns, "local images, platform-stored labels")
        (args.out / split / "_annotations.coco.json").write_text(json.dumps(doc))
        print(split, len(images), "images", len(anns), "annotations")
    shutil.copy(args.local / "metadata.jsonl", args.out / "metadata.jsonl")


if __name__ == "__main__":
    main()
