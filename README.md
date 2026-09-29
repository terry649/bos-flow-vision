# bosflow

Physics-based synthetic background-oriented schlieren (BOS) data, displacement
estimation baselines, and flow-feature instance segmentation.

This document is a placeholder. The full technical write-up is milestone M5.

## Setup

```
uv sync
git config core.hooksPath .githooks
uv run pytest
```

## Development workflow

- `uv run pytest` runs the fast suite; `uv run pytest -m slow` adds tests that download
  pretrained RAFT weights.
- `scripts/check_data_policy.py` enforces the data-release policy. The pre-commit hook
  runs it on staged files, and CI runs it on the whole tree along with a gitleaks scan.
- CI (`.github/workflows/ci.yml`) runs policy, lint, and tests on every push and pull
  request. Linux runners use CPU-only PyTorch wheels (see `[tool.uv.sources]`).
- The benchmark workflow (`.github/workflows/benchmark.yml`) reruns a reduced
  displacement benchmark whenever physics, optics, estimator, or config files change,
  and fails if any metric crosses its gate in `configs/gates.yaml`.
- Every generated dataset and benchmark run writes `manifest.json` with the git commit,
  dirty flag, package versions, and the full configuration with its SHA-256.
