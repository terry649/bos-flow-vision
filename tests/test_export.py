import json

import numpy as np

from bosflow import generate as G
from bosflow.rf import export as E


def test_encode_magnitude_scale_and_range():
    u = np.zeros((2, 64, 64))
    u[0, 10:20] = 3.0
    valid = np.ones((64, 64), bool)
    img, scale = E.encode_magnitude(u, valid, pct=99.9)
    assert img.dtype == np.uint8 and img.max() == 255 and img[0, 0] == 0
    assert scale == 3.0


def test_splits_are_stratified_and_deterministic():
    names = {ft: [f"{ft}_{i:04d}" for i in range(20)] for ft in G.FLOW_TYPES}
    a, b = E.assign_splits(names, 1), E.assign_splits(names, 1)
    assert a == b
    for ft in G.FLOW_TYPES:
        counts = {s: sum(1 for n in names[ft] if a[n] == s) for s in E.SPLITS}
        assert counts == {"train": 14, "valid": 4, "test": 2}


def test_export_writes_coco_splits(tmp_path):
    cfg = G.load_config()
    G.generate_dataset(tmp_path / "ds", cfg, counts={"wedge": 2, "blast": 2}, seed=3)
    counts = E.export(tmp_path / "ds", tmp_path / "out", "gt", progress=lambda *_: None)
    assert sum(counts.values()) == 4
    for split in E.SPLITS:
        doc = json.loads((tmp_path / "out" / split / "_annotations.coco.json").read_text())
        assert [c["name"] for c in doc["categories"]] == ["shock", "expansion_fan",
                                                           "shear_layer"]
        ids = {im["id"] for im in doc["images"]}
        assert all(a["image_id"] in ids for a in doc["annotations"])
        for im in doc["images"]:
            assert (tmp_path / "out" / split / im["file_name"]).exists()
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert manifest["config"]["estimator"] == "gt"


def test_mask_map_is_one_for_perfect_predictions(tmp_path):
    from pycocotools import mask as mask_utils

    from bosflow import labels
    from bosflow.rf.evaluate import coco_mask_map

    cfg = G.load_config()
    G.generate_dataset(tmp_path / "ds", cfg, counts={"wedge": 1, "expansion": 1,
                                                    "shear_layer": 1}, seed=5)
    E.export(tmp_path / "ds", tmp_path / "out", "gt", progress=lambda *_: None)
    gt_json = tmp_path / "out" / "train" / "_annotations.coco.json"
    doc = json.loads(gt_json.read_text())
    results = []
    for a in doc["annotations"]:
        img = next(i for i in doc["images"] if i["id"] == a["image_id"])
        m = labels.polygons_to_mask(a["segmentation"], (img["height"], img["width"]))
        rle = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("ascii")
        results.append({"image_id": a["image_id"], "category_id": a["category_id"],
                        "segmentation": rle, "score": 0.9})
    report = coco_mask_map(gt_json, results)
    present = {c for c, v in report["per_class"].items() if v["n_gt"] > 0}
    assert present
    for c in present:
        assert report["per_class"][c]["AP50"] > 0.99
