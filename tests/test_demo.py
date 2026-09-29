"""The quickstart demo runs end to end without a model and recovers the physics."""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_demo_ground_truth_mode(tmp_path):
    r = subprocess.run([sys.executable, "demo.py", "--ground-truth", "--frames", "5",
                        "--out", str(tmp_path)], cwd=ROOT, capture_output=True, text=True,
                       check=True)
    assert (tmp_path / "quickstart.png").exists()
    machs = [float(m) for m in re.search(r"Mach = ([\d., ]+) \(true", r.stdout).group(1).split(",")]
    assert all(abs(m - 2.5) < 0.02 for m in machs)
    exponent = float(re.search(r"exponent ([\d.]+)", r.stdout).group(1))
    assert abs(exponent - 0.4) < 0.005
