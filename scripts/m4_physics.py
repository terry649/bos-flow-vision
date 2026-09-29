"""M4 analysis (research env): score predictions and test them against gas dynamics.

Reads predictions written by deploy/predict.py and ground truth from the export:

    uv run python scripts/m4_physics.py --export data/synthetic/v1_rf_dis \
        --sequences data/synthetic/v1_seq_dis --pred results/m4/local --out results/m4/local

Reports (1) per-class mask mAP on the test split, (2) Mach number inferred from
predicted wedge and cone shock masks vs the true freestream Mach, and (3) blast
front R(t) from tracked masks: power-law exponent vs 2/5 and energy vs truth.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

from bosflow.analysis import physics_check as P
from bosflow.rf.evaluate import coco_mask_map

CLASSES = ["shock", "expansion_fan", "shear_layer"]


def decode(det):
    return mask_utils.decode(det["rle"]).astype(bool)


def mask_map(export: Path, pred: dict) -> dict:
    gt_json = export / "test" / "_annotations.coco.json"
    doc = json.loads(gt_json.read_text())
    id_of = {im["file_name"]: im["id"] for im in doc["images"]}
    cat_of = {c["name"]: c["id"] for c in doc["categories"]}
    results = [{"image_id": id_of[im["file_name"]], "category_id": cat_of[d["class_name"]],
                "segmentation": d["rle"], "score": d["confidence"]}
               for im in pred["images"] for d in im["detections"] if d["class_name"] in cat_of]
    return coco_mask_map(gt_json, results)


def shock_mach(export: Path, pred: dict, min_conf: float) -> list[dict]:
    meta = {}
    for line in (export / "metadata.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            meta[r["file_name"]] = r
    rows = []
    for im in pred["images"]:
        m = meta[im["file_name"]]
        ft, flow = m["flow_type"], m["flow"]
        if ft not in ("wedge", "cone"):
            continue
        freestream = np.degrees(flow["pose"]["angle"])
        body = flow["wedge_half_angle_deg"] if ft == "wedge" else flow["cone_half_angle_deg"]
        for d in im["detections"]:
            if d["class_name"] != "shock" or d["confidence"] < min_conf:
                continue
            mask = decode(d)
            halves = [mask]
            if ft == "cone":  # the V-shaped trace: fit each side of the axis separately
                rows_, cols_ = np.indices(mask.shape)
                x = cols_ - 0.5 * (mask.shape[1] - 1)
                y = 0.5 * (mask.shape[0] - 1) - rows_
                a = flow["pose"]["angle"]
                across = -np.sin(a) * x + np.cos(a) * y  # sign splits the V by the axis
                halves = [mask & (across > 0), mask & (across < 0)]
            for h in halves:
                if h.sum() < 50:
                    continue
                beta = P.mask_shock_angle(h, freestream)
                try:
                    M = (P.mach_from_wedge_shock(beta, body) if ft == "wedge"
                         else P.mach_from_cone_shock(beta, body))
                except ValueError:
                    M = float("nan")
                rows.append({"file_name": im["file_name"], "flow_type": ft,
                             "beta_pred_deg": beta, "beta_true_deg": flow["shock_angle_deg"],
                             "M_pred": M, "M_true": flow["M1"], "confidence": d["confidence"]})
    return rows


def box_iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter + 1e-9)


def blast_tracks(seq_root: Path, pred: dict, max_residual: float = 0.05) -> list[dict]:
    out = []
    for name, frames in pred["sequences"].items():
        truth = {t["file_name"]: t for t in json.loads((seq_root / name / "truth.json").read_text())}
        ids = Counter(d["tracker_id"] for f in frames for d in f["detections"]
                      if d["class_name"] == "shock")
        if not ids:
            out.append({"sequence": name, "frames_tracked": 0})
            continue
        ids.pop(-1, None)
        if not ids:
            out.append({"sequence": name, "frames_tracked": 0})
            continue
        tid = ids.most_common(1)[0][0]
        # Backfill ByteTrack's activation delay: an unconfirmed (-1) shock in the frame
        # just before the track's first confirmed frame belongs to it if the boxes overlap.
        first = next(i for i, f in enumerate(frames)
                     if any(d.get("tracker_id") == tid for d in f["detections"]))
        if first > 0:
            anchor = next(d for d in frames[first]["detections"] if d.get("tracker_id") == tid)
            for d in frames[first - 1]["detections"]:
                if (d.get("tracker_id") == -1 and d["class_name"] == "shock"
                        and box_iou(d["bbox_xyxy"], anchor["bbox_xyxy"]) > 0.3):
                    d["tracker_id"] = tid
                    break
        tr0 = next(iter(truth.values()))
        rec = {"sequence": name, "frames_total": len(frames), "track_ids_seen": len(ids)}
        for variant in ("raw", "gated"):
            t, R, R_true, changed = [], [], [], 0
            for f in frames:
                on_track = [d for d in f["detections"] if d.get("tracker_id") == tid]
                if not on_track:
                    continue
                chosen = on_track[0]
                if variant == "gated":
                    # Thin-front gate: a shock front is a thin surface. Among shock masks
                    # overlapping the track's box (confirmed or not), keep thin fronts
                    # and take the most confident; drop the frame if none is thin.
                    box = on_track[0]["bbox_xyxy"]
                    cands = [d for d in f["detections"] if d["class_name"] == "shock"
                             and box_iou(d["bbox_xyxy"], box) > 0.3
                             and P.circle_residual_ratio(decode(d)) < max_residual]
                    if not cands:
                        changed += 1
                        continue
                    chosen = max(cands, key=lambda d: d["confidence"])
                    changed += chosen is not on_track[0]
                tr = truth[f["file_name"]]
                _, _, r_px = P.fit_circle(decode(chosen))
                t.append(tr["t"])
                R.append(r_px * tr["object_pixel_size_m"])
                R_true.append(tr["shock_radius_m"])
            t, R, R_true = map(np.asarray, (t, R, R_true))
            v = {"frames_used": len(t), "frames_replaced_or_dropped": changed}
            if len(t):
                v["radius_err_px_median"] = float(np.median(np.abs(R - R_true)) /
                                                  tr0["object_pixel_size_m"])
            if len(t) >= 3:
                n, _ = P.power_law_fit(t, R)
                E = P.sedov_energy(t, R, tr0["rho0"])
                v.update(exponent=n, E_pred=E, E_true=tr0["E"], E_rel_err=E / tr0["E"] - 1.0)
            rec[variant] = v
        out.append(rec)
    return out


def summarize(mm: dict, mach: list[dict], blast: list[dict]) -> str:
    lines = ["# M4 physics check\n", "## Mask mAP on the test split (pycocotools, segm)\n",
             f"mAP50-95 = {mm['mAP50_95']:.3f}, mAP50 = {mm['mAP50']:.3f}\n",
             "| class | AP50-95 | AP50 | test instances |", "|---|---:|---:|---:|"]
    for c, v in mm["per_class"].items():
        lines.append(f"| {c} | {v['AP50_95']:.3f} | {v['AP50']:.3f} | {v['n_gt']} |")
    lines.append("\n## Mach number from predicted shock masks\n")
    lines.append("| flow | shocks | median abs beta error (deg) | median abs Mach error | "
                 "median rel Mach error | unsolvable |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for ft in ("wedge", "cone"):
        rs = [r for r in mach if r["flow_type"] == ft]
        if not rs:
            continue
        db = np.abs([r["beta_pred_deg"] - r["beta_true_deg"] for r in rs])
        ok = [r for r in rs if np.isfinite(r["M_pred"])]
        dm = np.abs([r["M_pred"] - r["M_true"] for r in ok])
        rel = np.abs([r["M_pred"] / r["M_true"] - 1 for r in ok])
        lines.append(f"| {ft} | {len(rs)} | {np.median(db):.2f} | "
                     f"{np.median(dm) if ok else float('nan'):.3f} | "
                     f"{100 * np.median(rel) if ok else float('nan'):.1f}% | {len(rs) - len(ok)} |")
    lines.append("\n## Blast fronts tracked through sequences\n")
    lines.append("raw: the tracker's detection in each frame. gated: thin-front check "
                 "(circle-fit residual < 5% of radius) among shock masks overlapping the "
                 "track, most confident kept.\n")
    lines.append("| variant | sequences | frames used | frames replaced or dropped | "
                 "exponent n (theory 0.4) | median abs E error | worst E error | "
                 "median radius error (px) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    total = sum(b["frames_total"] for b in blast)
    for variant in ("raw", "gated"):
        done = [b[variant] for b in blast if "exponent" in b.get(variant, {})]
        if not done:
            continue
        ex = np.array([d["exponent"] for d in done])
        er = np.abs([d["E_rel_err"] for d in done])
        lines.append(
            f"| {variant} | {len(done)}/{len(blast)} | "
            f"{sum(d['frames_used'] for d in done)}/{total} | "
            f"{sum(d['frames_replaced_or_dropped'] for d in done)} | "
            f"{ex.mean():.4f} +- {ex.std():.4f} | {100 * np.median(er):.1f}% | "
            f"{100 * er.max():.1f}% | "
            f"{np.median([d['radius_err_px_median'] for d in done]):.2f} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", type=Path, required=True)
    ap.add_argument("--sequences", type=Path, required=True)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--reuse", action="store_true",
                    help="reuse mask_map.json and shock_mach.json in --out (blast analysis only)")
    args = ap.parse_args()

    pred_test = json.loads((args.pred / "predictions_test.json").read_text())
    pred_seq = json.loads((args.pred / "predictions_sequences.json").read_text())
    if args.reuse:
        mm = json.loads((args.out / "mask_map.json").read_text())
        mach = json.loads((args.out / "shock_mach.json").read_text())
    else:
        mm = mask_map(args.export, pred_test)
        mach = shock_mach(args.export, pred_test, args.min_conf)
    blast = blast_tracks(args.sequences, pred_seq)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "mask_map.json").write_text(json.dumps(mm, indent=1))
    (args.out / "shock_mach.json").write_text(json.dumps(mach, indent=1))
    (args.out / "blast_tracks.json").write_text(json.dumps(blast, indent=1))
    text = summarize(mm, mach, blast)
    (args.out / "summary.md").write_text(f"Model: {pred_test['model']}\n\n" + text)
    print(text)


if __name__ == "__main__":
    main()
