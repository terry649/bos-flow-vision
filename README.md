# bos-flow-vision

A physics-informed computer vision study of compressible flows, extending the
author's doctoral research on background-oriented schlieren (BOS).

## Motivation

Background-oriented schlieren measures the refractive-index gradients of a flow by
tracking the apparent displacement of a background pattern. The displacement field
carries the physics: shock waves, expansion fans, and shear layers each leave a
distinct signature. Extracting those features, and quantities such as shock angles
or blast-wave radii, is still largely manual.

This project asks whether modern instance segmentation can locate these flow
features reliably enough that physical quantities can be recovered from the model's
output alone, and where such models fail. Because experimental BOS data with full
ground truth is scarce, and some is subject to distribution restrictions, the study
uses a physics-based synthetic data generator whose every image carries exact labels
and known flow parameters.

## Approach

1. **Synthetic BOS generator (M1).** Canonical compressible flows (attached oblique
   shocks, Taylor-Maccoll conical flow, Sedov-Taylor blast waves, Prandtl-Meyer
   expansions, and turbulent shear layers) are projected along the line of sight
   through the Gladstone-Dale relation and rendered with a BOS optical model that
   includes aperture blur, a random-dot background, and sensor noise. Physics unit
   tests check the generator against published values.
2. **Displacement estimation (M2).** Dense optical flow and deformable image
   registration methods are benchmarked against ground truth by flow type and noise
   level.
3. **Segmentation and physics recovery (M3).** RF-DETR instance segmentation models
   are trained on displacement-magnitude images. Predicted masks are converted back
   to physics: freestream Mach number from shock angles, and blast energy and
   self-similarity from tracked blast fronts.
4. **Controlled experiments (M4).** Training recipe versus data, input encoding, and
   robustness on held-out synthetic distribution shifts, plus a qualitative zero-shot
   test on published NASA AirBOS imagery.
5. **Write-up and quickstart demo (M5).**

## Status

| Milestone | Scope | Status |
|---|---|---|
| M1 | Synthetic generator and physics tests | Complete |
| M2 | Displacement estimation benchmark | Complete (`results/m2`) |
| M3 | Dataset, segmentation models, inference, tracking, physics checks | Complete (`results/m3`, `results/m4`) |
| M4 | Controlled experiments and stress tests | In progress (`results/m4_experiments`) |
| M5 | Technical write-up and quickstart demo | Planned |

Note on history: earlier commits label the inference, tracking, and physics-check
work as "M4"; that work is now part of M3, and M4 refers to the controlled
experiments.

## Selected results to date

On a held-out synthetic test set of 100 images, RF-DETR Seg Medium reaches a mask
mAP50-95 of 0.83 to 0.84 (mAP50 of 1.00). From the predicted masks alone, the
freestream Mach number is recovered with a median error of 0.2% for wedges and 1.9%
for cones, and tracked blast fronts give a Sedov-Taylor exponent of 0.399 +- 0.001
(theory: 0.4) with blast energy within 3%. Details, including a failure mode on
blast fronts clipped by the image boundary and a physically motivated correction,
are in `results/m4/README.md`. All results are on synthetic data.

## Repository layout

| Path | Contents |
|---|---|
| `src/bosflow/physics` | Gas dynamics, flow fields, turbulence |
| `src/bosflow/optics` | BOS optical model, background, warping, camera |
| `src/bosflow/displacement` | Displacement estimators and metrics |
| `src/bosflow/rf` | Dataset export, Roboflow client, mask evaluation |
| `src/bosflow/analysis` | Shock-angle and physics checks from masks |
| `deploy/` | Serving environment (Roboflow `inference`, `supervision`, `trackers`) |
| `configs/` | Generator, optics, benchmark, gate, and stress-test configuration |
| `scripts/` | Command-line entry points for each stage |
| `results/` | Committed metrics, summaries, and figures |

## Tooling

The dataset is hosted and versioned on the Roboflow platform (project
`bos-flow-features`). Models are RF-DETR Seg, trained both on Roboflow and locally
with the open-source `rfdetr` package, served with Roboflow `inference`, annotated
with `supervision`, and tracked with ByteTrack from `trackers`.

## Setup

```
uv sync
git config core.hooksPath .githooks
uv run pytest
```

The serving environment is separate: `uv sync --project deploy`. Local training
needs the optional group: `uv sync --group train`. Roboflow operations read
`ROBOFLOW_API_KEY` from the environment or from a local, gitignored `.env`.

## Data policy

The repository contains synthetic data and code only. Files under `data/real/` are
rejected by the pre-commit hook and by CI unless they are listed with a matching
SHA-256 in `data/real/CLEARED.md`. NASA AirBOS reference images are not committed;
`scripts/fetch_nasa_airbos.py` downloads them and verifies the hashes recorded in
`data/public/nasa_airbos/SOURCES.md`. Images courtesy of NASA; their use here implies
no endorsement by NASA.

## Development workflow

- `uv run pytest` runs the fast suite; `uv run pytest -m slow` adds tests that
  download pretrained RAFT weights.
- `scripts/check_data_policy.py` enforces the data policy. The pre-commit hook runs
  it on staged files, and CI runs it on the whole tree along with a gitleaks scan.
- CI (`.github/workflows/ci.yml`) runs the policy check, lint, and tests on every
  push and pull request. Linux runners use CPU-only PyTorch wheels (see
  `[tool.uv.sources]`).
- The benchmark workflow (`.github/workflows/benchmark.yml`) reruns a reduced
  displacement benchmark whenever physics, optics, estimator, or configuration files
  change, and fails if any metric crosses its gate in `configs/gates.yaml`.
- Every generated dataset and benchmark run writes `manifest.json` with the git
  commit, dirty flag, package versions, and the full configuration with its SHA-256.

## License

MIT. See `LICENSE`.
