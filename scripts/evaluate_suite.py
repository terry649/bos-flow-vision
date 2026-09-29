"""M4: evaluate every model on the v1 test split, the blast sequences, and every
stress set, with mask mAP and physics metrics.

Serving runs in the serving environment (deploy/predict.py, one subprocess per
model and set); scoring runs here. Results: <out>/<model>/<set>/ plus <out>/table.md.

    uv run python scripts/evaluate_suite.py --out results/m4_experiments/eval
    uv run python scripts/evaluate_suite.py --models m3_local arm_A --sets v1_test control
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from m4_physics import blast_tracks, mask_map, shock_mach  # scripts/ is on sys.path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic"

# name -> (source args for deploy/predict.py, input encoding)
MODELS = {
    "m3_hosted": (["--model-id", "bos-flow-features/1"], "mag"),
    "m3_local": (["--checkpoint", "runs/v1_dis_medium/checkpoint_best_total.pth"], "mag"),
    "arm_A_hosted_recipe": (["--checkpoint", "runs/m4_A_hosted_recipe/checkpoint_best_total.pth"],
                            "mag"),
    "arm_B_v2_data": (["--checkpoint", "runs/m4_B_v2_data/checkpoint_best_total.pth"], "mag"),
    "arm_C_vec_encoding": (["--checkpoint", "runs/m4_C_vec_encoding/checkpoint_best_total.pth"],
                           "vec"),
    "v2_hosted": (["--model-id", "bos-flow-features/2"], "mag"),
    # Same hosted-trained weights, downloaded and served in PyTorch like the local models.
    "v2_hosted_weights_pytorch": (["--checkpoint", "runs/hosted_v2_weights/weights.pt",
                                   "--labels", "_group,expansion_fan,shear_layer,shock"], "mag"),
}
LOCAL = ["--model-type", "rfdetr-seg-medium", "--resolution", "432"]


def set_dirs(encoding: str) -> dict[str, Path]:
    """Export directories (each with test/ and metadata.jsonl) per evaluation set."""
    sets = {"v1_test": DATA / ("v1_rf_dis" if encoding == "mag" else "v1_rf_dis_vec")}
    for d in sorted((DATA / "stress").glob("*/")):
        if (d / encoding / "test").exists():
            sets[d.name] = d / encoding
    return sets


def run_serving(model: str, images: Path | None, sequences: Path | None, out: Path) -> None:
    src, _ = MODELS[model]
    cmd = ["uv", "run", "--project", "deploy", "python", "deploy/predict.py", *src, "--out",
           str(out)]
    if "--checkpoint" in src:
        cmd += LOCAL
    if images:
        cmd += ["--images", str(images)]
    if sequences:
        cmd += ["--sequences", str(sequences)]
    # Hosted models call the Roboflow API at load time; retry transient network errors.
    for attempt in range(3):
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
        if r.returncode == 0:
            return
        print(f"serving {model} failed (attempt {attempt + 1}): {r.stderr.strip()[-300:]}")
        time.sleep(30)
    raise RuntimeError(f"serving {model} failed after 3 attempts")


def summarize_mach(rows: list[dict]) -> dict:
    out = {}
    for ft in ("wedge", "cone"):
        ok = [r for r in rows if r["flow_type"] == ft and np.isfinite(r["M_pred"])]
        if ok:
            out[f"{ft}_mach_rel_err_median"] = float(np.median(
                [abs(r["M_pred"] / r["M_true"] - 1) for r in ok]))
            out[f"{ft}_n"] = len(ok)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "m4_experiments" / "eval")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--sets", nargs="*", default=None)
    ap.add_argument("--sequences", action="store_true", help="also run the blast sequences")
    args = ap.parse_args()

    results = {}
    for model in args.models or list(MODELS):
        src, enc = MODELS[model]
        if "--checkpoint" in src and not (ROOT / src[1]).exists():
            print(f"skip {model}: no checkpoint yet")
            continue
        for name, export_dir in set_dirs(enc).items():
            if args.sets and name not in args.sets:
                continue
            out = args.out / model / name
            if not (out / "predictions_test.json").exists():
                run_serving(model, export_dir / "test", None, out)
            pred = json.loads((out / "predictions_test.json").read_text())
            mm = mask_map(export_dir, pred)
            rec = {"mAP50_95": mm["mAP50_95"], "mAP50": mm["mAP50"],
                   **{f"AP50_95_{c}": v["AP50_95"] for c, v in mm["per_class"].items()},
                   **summarize_mach(shock_mach(export_dir, pred, 0.5))}
            (out / "metrics.json").write_text(json.dumps(rec, indent=1))
            results.setdefault(model, {})[name] = rec
            print(model, name, {k: round(v, 3) for k, v in rec.items() if isinstance(v, float)})
        if args.sequences:
            seq = DATA / ("v1_seq_dis" if enc == "mag" else "v1_seq_dis_vec")
            out = args.out / model / "sequences"
            if not (out / "predictions_sequences.json").exists():
                run_serving(model, None, seq, out)
            b = blast_tracks(seq, json.loads((out / "predictions_sequences.json").read_text()))
            (out / "blast_tracks.json").write_text(json.dumps(b, indent=1))
            for variant in ("raw", "gated"):
                done = [x[variant] for x in b if "exponent" in x.get(variant, {})]
                if done:
                    results.setdefault(model, {})[f"blast_{variant}"] = {
                        "exponent_mean": float(np.mean([d["exponent"] for d in done])),
                        "exponent_std": float(np.std([d["exponent"] for d in done])),
                        "E_err_worst": float(max(abs(d["E_rel_err"]) for d in done))}

    args.out.mkdir(parents=True, exist_ok=True)
    prev = args.out / "results.json"
    merged = json.loads(prev.read_text()) if prev.exists() else {}
    for m, sets in results.items():
        merged.setdefault(m, {}).update(sets)
    prev.write_text(json.dumps(merged, indent=1))
    print(f"wrote {prev}")


if __name__ == "__main__":
    main()
