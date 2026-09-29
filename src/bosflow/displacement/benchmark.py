"""M2 benchmark: endpoint error of displacement estimators by flow type and noise level."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import yaml

from bosflow import generate as G
from bosflow.displacement import metrics, optical_flow, registration
from bosflow.optics.background import random_dot_pattern
from bosflow.optics.camera import Camera

METHODS = {
    "farneback": optical_flow.farneback,
    "lucas_kanade": optical_flow.lucas_kanade,
    "dis": optical_flow.dis,
    "tvl1": optical_flow.tvl1,
    "demons": registration.demons,
}


def _method(name):
    if name == "raft":  # imported lazily so torch is only loaded when needed
        from bosflow.displacement.raft import raft
        return raft
    return METHODS[name]


def load_benchmark_config(path: Path | str = G.CONFIG_DIR / "benchmark.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


def run(bcfg: dict, gcfg: dict, scenes_per_type: int | None = None,
        methods: list[str] | None = None, progress=print) -> list[dict]:
    """Return one record per (scene, noise level, method)."""
    n = scenes_per_type or bcfg["scenes_per_flow_type"]
    methods = methods or bcfg["methods"]
    cams = {k: Camera(**v) for k, v in bcfg["noise_levels"].items()}
    records = []
    root = np.random.SeedSequence(bcfg["seed"])
    for ft, ft_seq in zip(G.FLOW_TYPES, root.spawn(len(G.FLOW_TYPES))):
        for i, sq in enumerate(ft_seq.spawn(n)):
            scene_seq, bg_seq, noise_seq = sq.spawn(3)
            base = G.generate_sample(ft, scene_seq, gcfg)
            flow, optics, pattern = (base.scene[k] for k in ("flow", "optics", "pattern"))
            x_hr, _ = G.object_grid(optics, gcfg["supersample"], gcfg["margin_px"])
            bg = random_dot_pattern(x_hr.shape, gcfg["supersample"], pattern["dot_density"],
                                    pattern["dot_diameter_px"], np.random.default_rng(bg_seq))
            for (level, cam), nseq in zip(cams.items(), noise_seq.spawn(len(cams))):
                s = G.render(flow, optics, cam, pattern, np.random.default_rng(nseq),
                             gcfg["supersample"], gcfg["margin_px"], gcfg["min_band_px"],
                             background=bg)
                feature = np.any([m.mask for m in base.instances], axis=0)
                for name in methods:
                    t0 = time.perf_counter()
                    est = _method(name)(s.reference, s.deflected)
                    dt = time.perf_counter() - t0
                    rec = {"flow_type": ft, "scene": i, "noise": level, "method": name,
                           "seconds": dt, "peak_gt_px": s.meta["peak_displacement_px"]}
                    rec.update(metrics.summarize(est, s.flow, s.valid, feature))
                    records.append(rec)
            progress(f"{ft} scene {i + 1}/{n}")
    return records
