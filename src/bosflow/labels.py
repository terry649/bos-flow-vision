"""Instance masks to COCO polygons, and COCO dataset assembly.

COCO polygons cannot carry holes, but an expanding blast front labeled as a band
is an annulus. Holes are merged into their outer contour with a zero-width
"keyhole" bridge, which even-odd and nonzero polygon fills both render correctly.

An instance can also have several disconnected pieces, for example a blast front
cut into arcs by the image edges. COCO allows a list of polygons per annotation,
but some annotation platforms store one polygon per instance and concatenate the
list into a single filled shape. ``instance_annotations`` therefore emits one
annotation per piece by default; their union is the original mask.
"""

from __future__ import annotations

import cv2
import numpy as np

from bosflow.physics.fields import CLASSES


def _bridge(outer: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """Splice ``hole`` into ``outer`` at their closest vertex pair."""
    d = ((outer[:, None, :] - hole[None, :, :]) ** 2).sum(-1)
    i, j = np.unravel_index(int(np.argmin(d)), d.shape)
    hole_loop = np.concatenate([hole[j:], hole[: j + 1]])
    return np.concatenate([outer[: i + 1], hole_loop, outer[i:]])


def mask_to_polygons(mask: np.ndarray, epsilon_px: float = 0.75,
                     min_area_px: float = 12.0) -> list[list[float]]:
    """Flattened [x0, y0, x1, y1, ...] polygons in pixel coordinates."""
    m = np.ascontiguousarray(mask.astype(np.uint8))
    contours, hierarchy = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]
    polys = []
    for idx, (_, _, child, parent) in enumerate(hierarchy):
        if parent != -1:
            continue
        outer = contours[idx]
        if cv2.contourArea(outer) < min_area_px:
            continue
        outer = cv2.approxPolyDP(outer, epsilon_px, True)[:, 0, :].astype(float)
        while child != -1:
            hole = contours[child]
            if cv2.contourArea(hole) >= min_area_px:
                hole = cv2.approxPolyDP(hole, epsilon_px, True)[:, 0, :].astype(float)
                outer = _bridge(outer, hole)
            child = hierarchy[child][0]
        if len(outer) >= 3:
            polys.append(outer.reshape(-1).round(2).tolist())
    return polys


def polygons_to_mask(polys: list[list[float]], shape: tuple[int, int]) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    for p in polys:
        pts = np.round(np.asarray(p).reshape(-1, 2)).astype(np.int32)
        cv2.fillPoly(m, [pts], 1)  # even-odd per polygon via keyhole geometry
    return m.astype(bool)


def bbox_of(mask: np.ndarray) -> list[float]:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x0, y0 = float(cols.min()), float(rows.min())
    return [x0, y0, float(cols.max()) - x0 + 1.0, float(rows.max()) - y0 + 1.0]


def coco_categories() -> list[dict]:
    return [{"id": i + 1, "name": c, "supercategory": "flow_feature"}
            for i, c in enumerate(CLASSES)]


def build_coco(images: list[dict], annotations: list[dict], description: str) -> dict:
    return {
        # No timestamps, so identical seeds give byte-identical files.
        "info": {"description": description, "version": "1.0",
                 "contributor": "bosflow (synthetic)"},
        "licenses": [{"id": 1, "name": "MIT", "url": "https://opensource.org/licenses/MIT"}],
        "categories": coco_categories(),
        "images": images,
        "annotations": annotations,
    }


def instance_annotations(first_id: int, image_id: int, cls: str, mask: np.ndarray,
                         polys: list[list[float]] | None = None,
                         split_parts: bool = True) -> list[dict]:
    """COCO annotations for one instance: one per disconnected piece when ``split_parts``."""
    polys = mask_to_polygons(mask) if polys is None else polys
    if not polys:
        return []
    if not split_parts or len(polys) == 1:
        a = instance_annotation(first_id, image_id, cls, mask, polys)
        return [a] if a else []
    out = []
    for p in polys:
        piece = polygons_to_mask([p], mask.shape) & mask
        if piece.any():
            out.append({"id": first_id + len(out), "image_id": image_id,
                        "category_id": CLASSES.index(cls) + 1, "segmentation": [p],
                        "area": float(piece.sum()), "bbox": bbox_of(piece), "iscrowd": 0})
    return out


def instance_annotation(ann_id: int, image_id: int, cls: str, mask: np.ndarray,
                        polys: list[list[float]] | None = None) -> dict | None:
    polys = mask_to_polygons(mask) if polys is None else polys
    if not polys:
        return None
    return {
        "id": ann_id, "image_id": image_id,
        "category_id": CLASSES.index(cls) + 1,
        "segmentation": polys, "area": float(mask.sum()),
        "bbox": bbox_of(mask), "iscrowd": 0,
    }
