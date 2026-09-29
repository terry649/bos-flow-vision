"""M4 data arm: v2 = v1 export + extra blast images from a shifted distribution
(large, off-center fronts that the image edge clips).

The extras go only to train (80%) and valid (20%); the test split is v1's, unchanged,
so every model is scored on identical test images. The extras are also kept as their
own export (<out>_extras, no test split) for uploading as a new Roboflow batch.

    uv run python scripts/build_v2_dataset.py --v1 data/synthetic/v1_rf_dis \
        --extra data/synthetic/v2_extra_blast --out data/synthetic/v2_rf_dis
"""

import argparse
import json
import shutil
from pathlib import Path

from bosflow import labels, lineage
from bosflow.rf.export import export


def fold_test_into_train(d: Path) -> None:
    """Move an export's test images into train (evaluation stays on v1's test)."""
    train_p, test_p = d / "train" / "_annotations.coco.json", d / "test" / "_annotations.coco.json"
    train, test = json.loads(train_p.read_text()), json.loads(test_p.read_text())
    next_img = max((im["id"] for im in train["images"]), default=0) + 1
    next_ann = max((a["id"] for a in train["annotations"]), default=0) + 1
    remap = {}
    for im in test["images"]:
        remap[im["id"]] = next_img
        train["images"].append(dict(im, id=next_img))
        next_img += 1
        shutil.move(d / "test" / im["file_name"], d / "train" / im["file_name"])
    for a in test["annotations"]:
        train["annotations"].append(dict(a, id=next_ann, image_id=remap[a["image_id"]]))
        next_ann += 1
    train_p.write_text(json.dumps(train))
    test_p.write_text(json.dumps(dict(test, images=[], annotations=[])))
    lines = [json.loads(line) for line in (d / "metadata.jsonl").read_text().splitlines() if line]
    for m in lines:
        if m["split"] == "test":
            m["split"] = "train"
    (d / "metadata.jsonl").write_text("".join(json.dumps(m, default=float) + "\n" for m in lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", type=Path, required=True)
    ap.add_argument("--extra", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--estimator", default="dis")
    ap.add_argument("--encoding", default="mag")
    args = ap.parse_args()

    extras = args.out.parent / f"{args.out.name}_extras"
    for d in (args.out, extras):
        if d.exists():
            shutil.rmtree(d)
    shutil.copytree(args.v1, args.out)
    export(args.extra, extras, args.estimator, encoding=args.encoding, progress=lambda *_: None)
    fold_test_into_train(extras)
    tmp = extras

    extra_meta = {json.loads(line)["file_name"]: json.loads(line)
                  for line in (tmp / "metadata.jsonl").read_text().splitlines() if line.strip()}
    added = {"train": 0, "valid": 0}
    for src_split in ("train", "valid"):
        dst_split = src_split
        src = json.loads((tmp / src_split / "_annotations.coco.json").read_text())
        dst_path = args.out / dst_split / "_annotations.coco.json"
        dst = json.loads(dst_path.read_text())
        next_img = max((im["id"] for im in dst["images"]), default=0) + 1
        next_ann = max((a["id"] for a in dst["annotations"]), default=0) + 1
        remap = {}
        for im in src["images"]:
            remap[im["id"]] = next_img
            dst["images"].append(dict(im, id=next_img))
            next_img += 1
            shutil.copy(tmp / src_split / im["file_name"], args.out / dst_split / im["file_name"])
            extra_meta[im["file_name"]]["split"] = dst_split
            added[dst_split] += 1
        for a in src["annotations"]:
            dst["annotations"].append(dict(a, id=next_ann, image_id=remap[a["image_id"]]))
            next_ann += 1
        dst_path.write_text(json.dumps(labels.build_coco(dst["images"], dst["annotations"],
                                                         dst["info"]["description"] + " + v2 extras")))
    with open(args.out / "metadata.jsonl", "a") as fh:
        fh.writelines(json.dumps(m, default=float) + "\n" for m in extra_meta.values())
    lineage.write_manifest(args.out / "manifest.json",
                           {"base": str(args.v1), "extra": str(args.extra),
                            "estimator": args.estimator, "encoding": args.encoding},
                           added=added)
    print(f"v2 = v1 + {added}")


if __name__ == "__main__":
    main()
