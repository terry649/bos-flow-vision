# bos-flow-vision

A physics-informed computer vision study of compressible flows, extending the
author's doctoral research on background-oriented schlieren (BOS).

## Abstract

Background-oriented schlieren recovers the refractive-index gradients of a flow from
the apparent displacement of a background pattern. This project asks whether modern
instance segmentation can locate shock waves, expansion fans, and shear layers in BOS
displacement fields accurately enough that physical quantities can be recovered from
the predicted masks alone, and where such models fail.

A physics-based generator renders canonical compressible flows through a BOS optical
model with exact labels and known flow parameters. Six dense displacement estimators
are benchmarked on its output. RF-DETR Seg models trained on estimated displacement
magnitude reach a mask mAP50-95 of 0.83 to 0.84 (mAP50 of 1.00) on held-out scenes.
From the predicted masks alone, the freestream Mach number is recovered with a median
error of 0.1 to 0.2% for wedges and 1.3 to 2.1% for cones, and tracked blast fronts
follow the Sedov-Taylor law with a fitted exponent of 0.399 +- 0.001 (theory 0.4) and
blast energy within 3.5%. Controlled experiments trace a failure on blast fronts
clipped by the image boundary to label integrity rather than to the training recipe,
data volume, or serving path, and show that adding flow-direction channels to the input
trades robustness to estimator changes for robustness to overlapping features and
camera noise. All quantitative results are on synthetic data.

## 1. Physics-based synthetic BOS (M1)

**Flows.** Five canonical flows with analytic or numerically exact solutions:

| Flow | Model | Label |
|---|---|---|
| Wedge | attached oblique shocks from the theta-beta-Mach relation | shock |
| Cone | Taylor-Maccoll conical flow | shock |
| Blast wave | Sedov-Taylor similarity solution (strong shock, spherical) | shock |
| Convex corner | centered Prandtl-Meyer expansion | expansion fan |
| Mixing layer | tanh mean profile with von Karman path-integrated turbulence | shear layer |

**Optics.** Density perturbations are integrated along the line of sight (planar
extrusion, or a forward Abel transform for axisymmetric flows), converted to deflection
angles with the Gladstone-Dale relation, and mapped to background displacement for a
camera focused on the background (Raffel 2015). The object-plane deflection is smeared
by the footprint of the ray cone collected by each pixel (Goldhahn and Seume 2007),
which sets the apparent width of a shock. The background is a random-dot pattern
warped by the displacement field on a supersampled grid, followed by a camera model
with lens blur, vignetting, shot and read noise, quantization, and optional JPEG
compression. Scenes whose displacement gradient would fold the pattern (caustics) are
rejected, as practical BOS setups avoid them.

**Labels.** Instance masks are bands around each shock whose width follows the optical
smearing (at least 3 px), the full extent of each expansion fan, and the vorticity
thickness of each mixing layer, exported as COCO polygons. Rings are encoded with a
zero-width bridge to their hole, and an instance split into several pieces by the image
boundary is exported as one annotation per piece (Section 4).

**Verification.** The test suite checks the gas dynamics against published values
(NACA Report 1135; Anderson, *Modern Compressible Flow*): the weak oblique shock of
39.31 deg for Mach 2 and a 10 deg wedge, Prandtl-Meyer angles, normal-shock ratios,
cone shock angles, and the Sedov-Taylor constant xi0 = 1.033 for gamma = 1.4. It also
checks the optics against analytic cases: a linear density gradient gives the exact
uniform displacement, Abel projections match closed forms, and the smeared shock width
matches the aperture footprint. Shock angles measured from rendered images agree with
theory to within 0.05 deg, and identical seeds produce byte-identical datasets.

## 2. Displacement estimation (M2)

Six dense estimators were scored against ground truth on 100 scenes at four noise
levels (endpoint error, EPE, in pixels; `results/m2`):

| Method | EPE, clean | EPE, medium noise | EPE in flow features (medium) | Peak displacement recovered (medium) | Time per pair |
|---|---:|---:|---:|---:|---:|
| DIS (Kroeger et al. 2016) | 0.010 | 0.029 | 0.176 | 0.68 | 0.02 s |
| Farneback (2003) | 0.013 | 0.043 | 0.178 | 0.74 | 0.03 s |
| Lucas-Kanade, pyramidal | 0.018 | 0.035 | 0.234 | 0.60 | 0.08 s |
| TV-L1 (Zach et al. 2007) | 0.016 | 0.064 | 0.190 | 0.69 | 1.25 s |
| Diffeomorphic demons (Vercauteren et al. 2009) | 0.017 | 0.100 | 0.182 | 0.82 | 0.24 s |
| RAFT, pretrained (Teed and Deng 2020) | 0.050 | 0.049 | 0.510 | 0.10 | 0.12 s |

DIS has the lowest error at every noise level and is used to build the segmentation
dataset. Deformable registration (demons) preserves the most shock peak amplitude at
the cost of background noise, a direct resolution-noise trade-off. Pretrained RAFT
recovers only about 10% of sub-pixel shock displacement, which motivates
physics-specific training data.

## 3. Segmentation and physics recovery (M3)

**Data.** 1,000 images (200 per flow type) of DIS displacement magnitude, split
700/200/100 within each flow type, and 20 blast-wave sequences of 12 frames for
tracking. The dataset is versioned on a dataset-hosting platform.

**Models.** RF-DETR Seg Medium, trained locally for 40 epochs on an Apple M5 Pro GPU,
and trained on the hosting platform with its default recipe. Inference runs through
`inference`, annotation through `supervision`, and tracking through ByteTrack. A parity
check confirms that the serving path reproduces the training framework's predictions
(mask IoU 0.955 to 0.998).

**Results.** Mask mAP on the 100-image test split (pycocotools):

| Model | mAP50-95 | mAP50 | Shock | Expansion fan | Shear layer |
|---|---:|---:|---:|---:|---:|
| Local | 0.826 | 1.000 | 0.573 | 0.977 | 0.928 |
| Hosted | 0.840 | 1.000 | 0.618 | 0.989 | 0.912 |

Shock AP is lowest at strict IoU because the labeled bands are 3 to 13 px wide, so a
one-pixel offset costs a large share of the overlap; every shock is found at IoU 0.5.

**Physics from masks.** Each predicted wedge or cone shock mask gives a shock angle
relative to the known flow direction; inverting theta-beta-Mach (wedge) or the
Taylor-Maccoll solution (cone) recovers the freestream Mach number with a median error
of 0.1 to 0.2% (wedge) and about 1.9% (cone). For blast waves, a circle fitted to the
tracked front in each frame gives R(t); for the local model the fitted exponent is
0.3988 +- 0.0012 and the recovered energy is within 2.8% in all 20 sequences.

## 4. Controlled experiments (M4)

The hosted model failed on blast fronts clipped by the image boundary (worst energy
error 79%), while the local model did not. Single-variable training arms isolated the
cause, and every model was scored on seven held-out stress sets
(`results/m4_experiments`).

**Label integrity.** A clipped blast front can split into several arcs, exported as a
COCO annotation with a list of polygons. The hosting platform stores one polygon per
instance; after upload, 107 of 108 such annotations came back as filled regions. Only
models trained on those stored labels fail:

| Model | Training labels | Sequences with energy error above 10% |
|---|---|---:|
| Local (M3), A: hosted recipe, B: v2 data, C: vector encoding | exported locally | 0 of 20 each |
| D: platform labels, trained locally | as stored by the platform | 2 of 20 |
| Hosted, v2 data | as stored by the platform | 7 of 20 |

The same hosted weights fail identically when served in PyTorch, so the serving path is
not the cause, and adding 200 more clipped fronts made the hosted model worse, because
those are exactly the corrupted shapes. Exporting each piece of a disconnected instance
as its own simple polygon fixes the problem; a round trip through the platform confirmed
that such polygons are stored exactly. At inference time, a thin-front gate (keep masks
whose circle-fit residual is below 5% of the radius) restores every model to an exponent
within 0.002 of 0.4.

**Recipe, data, and encoding.** Under a fixed 40-epoch budget, the platform's default
settings train to lower mask quality than the local settings (0.783 against 0.826).
Adding 200 edge-clipped fronts costs nothing on the test split and gives the most
robust model on several shifts. Adding signed displacement components to the input
(magnitude, u_x, u_y) gives the best results on composite scenes (mAP 0.474) and under
harsh camera noise, but degrades most when a different displacement estimator produces
the images (0.652 with demons registration, against 0.720 to 0.730 for magnitude-only
models trained locally on correct labels).

**Robustness.** Composite scenes with two flows (mAP 0.35 to 0.47, shock AP about 0.1)
and camera noise beyond the training range (0.52 to 0.62) are the largest gaps for
every model. Changing the displacement estimator costs 0.12 to 0.28 in mAP.

**Real imagery.** On published NASA AirBOS images (used for qualitative illustration
only; no NASA endorsement implied), the model outlines the outer shock envelope of a
supersonic T-38 and misses nested and interacting shocks, which the generator does not
produce.

## 5. Limitations and next steps

- All quantitative results are on synthetic data from one generator.
- Each training arm is a single run; mAP differences of about 0.01 are within
  run-to-run variation.
- The optical model is paraxial: rays are straight through the flow and parallel, and
  diffraction is neglected. A numerical ray-tracing check would bound its error.
- Next steps: multi-shock and shock-interaction scenes, training across displacement
  estimators, and a hosted retrain on the corrected labels.

## Reproducing the results

`notebooks/reproduce.ipynb` recomputes the headline results in minutes: it runs the
generator and physics checks live, reruns a reduced displacement benchmark, and
re-scores committed model predictions against committed reference ground truth
(`results/reference`), with no model weights or generated images required.
`docs/quickstart.md` describes `demo.py`, which runs the full pipeline on fresh scenes in
under a minute. Full regeneration from seeds:

```
uv sync --group train
uv run python scripts/generate_dataset.py --sequences --out data/synthetic/v1
uv run python scripts/benchmark_displacement.py --out results/m2
uv run python scripts/build_rf_dataset.py --dataset data/synthetic/v1 --estimator dis --out data/synthetic/v1_rf_dis
uv run python scripts/train_local.py --data data/synthetic/v1_rf_dis --size medium --epochs 40
uv run python scripts/build_stress_sets.py --out data/synthetic/stress
uv run python scripts/evaluate_suite.py --models m3_local --sequences
uv run python scripts/m4_tables.py
```

Note on history: earlier commits label the inference, tracking, and physics-check work
as "M4"; that work is part of M3, and M4 refers to the controlled experiments.

## Repository layout

| Path | Contents |
|---|---|
| `src/bosflow/physics` | Gas dynamics, flow fields, turbulence |
| `src/bosflow/optics` | BOS optical model, background, warping, camera |
| `src/bosflow/displacement` | Displacement estimators, metrics, benchmark |
| `src/bosflow/rf` | Dataset export, dataset hosting client, mask evaluation |
| `src/bosflow/analysis` | Shock-angle and physics checks from masks |
| `deploy/` | Serving environment (`inference`, `supervision`, `trackers`) |
| `configs/` | Generator, optics, benchmark, gate, and stress-test configuration |
| `scripts/` | Command-line entry points for each stage |
| `notebooks/`, `docs/`, `demo.py` | Reproduction notebook and quickstart demo |
| `results/` | Committed metrics, tables, figures, and reference ground truth |

## Dependencies

NumPy, SciPy, OpenCV, scikit-image, and SimpleITK for the physics, optics, and
displacement estimation; PyTorch, torchvision (RAFT), and `rfdetr` (RF-DETR Seg) for
learning; `inference`, `supervision`, and `trackers` (ByteTrack) for serving,
annotation, and tracking; pycocotools for evaluation. Exact versions are pinned in
`uv.lock` and `deploy/uv.lock`.

## Setup

```
uv sync
git config core.hooksPath .githooks
uv run pytest
```

The serving environment is separate: `uv sync --project deploy`. Local training needs
the optional group `uv sync --group train`, and the notebook the group `notebook`.
Dataset hosting and hosted training read an API key from the environment or from a
local, gitignored `.env` (see `.env.example`).

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
- `scripts/check_data_policy.py` enforces the data policy. The pre-commit hook runs it
  on staged files, and CI runs it on the whole tree along with a gitleaks scan.
- CI (`.github/workflows/ci.yml`) runs the policy check, lint, and tests on every push
  and pull request. Linux runners use CPU-only PyTorch wheels.
- The benchmark workflow (`.github/workflows/benchmark.yml`) reruns a reduced
  displacement benchmark whenever physics, optics, estimator, or configuration files
  change, and fails if any metric crosses its gate in `configs/gates.yaml`.
- Every generated dataset and benchmark run writes `manifest.json` with the git
  commit, dirty flag, package versions, and the full configuration with its SHA-256.

## References

- Ames Research Staff (1953). Equations, tables, and charts for compressible flow. NACA
  Report 1135.
- Anderson, J. D. *Modern Compressible Flow*, 3rd ed. McGraw-Hill.
- Farneback, G. (2003). Two-frame motion estimation based on polynomial expansion. SCIA.
- Goldhahn, E., and Seume, J. (2007). The background oriented schlieren technique:
  sensitivity, accuracy, resolution and application to a three-dimensional density
  field. *Experiments in Fluids* 43, 241-249.
- Kroeger, T., Timofte, R., Dai, D., and Van Gool, L. (2016). Fast optical flow using
  dense inverse search. ECCV.
- Raffel, M. (2015). Background-oriented schlieren (BOS) techniques. *Experiments in
  Fluids* 56, 60.
- Sedov, L. I. (1959). *Similarity and Dimensional Methods in Mechanics*. Academic Press.
- Taylor, G. I. (1950). The formation of a blast wave by a very intense explosion.
  *Proceedings of the Royal Society A* 201, 159-174.
- Teed, Z., and Deng, J. (2020). RAFT: Recurrent all-pairs field transforms for optical
  flow. ECCV.
- Vercauteren, T., Pennec, X., Perchant, A., and Ayache, N. (2009). Diffeomorphic
  demons: efficient non-parametric image registration. *NeuroImage* 45, S61-S72.
- Zach, C., Pock, T., and Bischof, H. (2007). A duality based approach for realtime
  TV-L1 optical flow. DAGM.

## License

MIT. See `LICENSE`.
