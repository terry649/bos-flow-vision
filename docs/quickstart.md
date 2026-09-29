# Quickstart demo

`demo.py` runs the full pipeline on two freshly rendered scenes in under a minute and
writes one summary figure. It is the fastest way to see what the project does.

## Run it

```
uv sync --group train
uv run python demo.py                      # uses runs/v1_dis_medium/checkpoint_best_total.pth
uv run python demo.py --checkpoint PATH    # any RF-DETR Seg checkpoint trained by this project
uv run python demo.py --ground-truth       # no model: generator labels stand in for predictions
```

Output: `demo_output/quickstart.png` and a short printed summary. Typical output with the
trained model:

```
[demo] wedge: 2 shocks, beta = 31.78, 31.79 deg (theory 31.85); Mach = 2.507, 2.505 (true 2.5)
[demo] blast: 8/8 frames, exponent 0.3972 (Sedov-Taylor 0.4), energy 1678.5 J (true 1717.7 J, error -2.3%)
```

## What each step shows (about five minutes of explanation)

1. **A physics-based BOS image pair.** A Mach 2.5 flow over a 10 degree wedge is
   integrated along the line of sight, converted to light deflection with the
   Gladstone-Dale relation, smeared by the camera aperture, and used to warp a random
   dot background. The two images differ only where the shocks bend the light.
2. **Displacement estimation.** Dense optical flow (DIS) recovers the apparent motion
   of the dots. Its magnitude makes the shocks visible as thin bright bands. In the
   benchmark (M2) this estimator had the lowest error at every noise level.
3. **Segmentation.** An RF-DETR Seg instance segmentation model, trained only on
   synthetic data, outlines each shock.
4. **Physics from the masks.** The angle of each shock mask, relative to the known
   flow direction, is inserted into the theta-beta-Mach relation and solved for the
   freestream Mach number. The model's masks recover Mach 2.5 to within 0.3%.
5. **A blast wave in time.** In each frame of a rendered blast sequence, the model
   segments the front, a thin-front check discards any filled or malformed mask, and a
   circle fit gives the radius. The radii follow the Sedov-Taylor law R ~ t^(2/5), and
   the fitted prefactor returns the released energy to within a few percent.

## Points worth discussing

- **Why synthetic data.** Every image carries exact labels and known flow parameters,
  so the pipeline can be checked against gas dynamics rather than only against
  annotations. Pretrained optical flow (RAFT) recovers only about 10% of these
  sub-pixel shock displacements, which motivates physics-specific data.
- **What can go wrong.** The M4 experiments found that instances split into several
  pieces by the image boundary lose their shape when a multi-polygon annotation is
  stored as a single polygon, and that a model trained on those labels mis-segments
  clipped blast fronts. `results/m4_experiments/README.md` has the full analysis.
- **Where it stops.** On published NASA AirBOS images, the model finds the outer shock
  envelope of a supersonic aircraft but misses nested and interacting shocks, which the
  generator does not yet produce.
