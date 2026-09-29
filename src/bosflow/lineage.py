"""Provenance for generated datasets and benchmark runs.

A manifest records exactly which code, configuration, and environment produced an
artifact, so any dataset version uploaded to Roboflow or any reported number can be
traced back and regenerated. It holds no timestamps, so reruns stay byte-identical.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from importlib import metadata
from pathlib import Path

PACKAGES = ("bosflow", "numpy", "scipy", "opencv-python-headless", "scikit-image",
            "simpleitk", "torch", "torchvision", "pillow")


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True,
                              cwd=Path(__file__).resolve().parent).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def config_digest(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()


def manifest(cfg: dict, **extra) -> dict:
    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "python": platform.python_version(),
        "platform": f"{platform.system()}-{platform.machine()}",
        "packages": versions,
        "config_sha256": config_digest(cfg),
        "config": cfg,
        **extra,
    }


def write_manifest(path: Path, cfg: dict, start: dict | None = None, **extra) -> dict:
    """Write a manifest. Pass ``start=manifest(cfg)`` captured when a long run began, so
    commits made while it ran are not misattributed; the end state is recorded too."""
    m = manifest(cfg, **extra)
    if start is not None:
        m["git_commit_at_end"], m["git_dirty_at_end"] = m["git_commit"], m["git_dirty"]
        m["git_commit"], m["git_dirty"] = start["git_commit"], start["git_dirty"]
    Path(path).write_text(json.dumps(m, indent=1, sort_keys=True, default=str))
    return m
