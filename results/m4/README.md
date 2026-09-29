# M4: inference, tracking, and physics checks

Two RF-DETR Seg Medium models trained on the same Roboflow dataset version
(`terry-zhou/bos-flow-features/1`, 700 train, 200 valid, 100 test images of DIS
displacement magnitude):

- **hosted**: trained on Roboflow with the default recipe (early stopping), served
  with Roboflow `inference` 1.7.2 via `get_model("bos-flow-features/1")`.
- **local**: trained with `rfdetr` 1.11.0 on Apple M5 Pro (MPS) for 40 epochs,
  served through `inference-models` in the same serving environment.

Both run through `deploy/predict.py` (serving environment) and are scored by
`scripts/m4_physics.py` (research environment). Test split: 100 images. Blast
sequences: 20 sequences of 12 frames, tracked with ByteTrack (`trackers` 2.4).

## Segmentation (pycocotools mask mAP, test split)

| model | mAP50-95 | mAP50 | shock | expansion_fan | shear_layer |
|---|---:|---:|---:|---:|---:|
| hosted | 0.840 | 1.000 | 0.618 | 0.989 | 0.912 |
| local | 0.826 | 1.000 | 0.573 | 0.977 | 0.928 |

Per-class values are mask AP50-95. Shocks score lowest at strict IoU because the
labeled bands are 3 to 13 px wide, so a 1 px offset costs a large share of IoU; at
IoU 0.5 every shock is found (AP50 = 1.000). Roboflow's own test evaluation of the
hosted model reports 0.844 overall, labeled `object-detection-like` (box metrics).
Evaluated directly with `rfdetr`, the local checkpoint scores 0.831; the 0.005 gap
to the served number matches the serving/rfdetr parity check (mask IoU 0.955 to 0.998).

## Physics from predicted masks

**Mach number from shock angle.** Each predicted wedge or cone shock mask gives a
shock angle beta relative to the known freestream direction. Inverting theta-beta-M
(wedge) or the Taylor-Maccoll solution (cone) at the known body angle gives the
freestream Mach number.

| model | flow | shocks | median abs beta error | median rel Mach error |
|---|---|---:|---:|---:|
| hosted | wedge | 40 | 0.04 deg | 0.1% |
| hosted | cone | 40 | 0.24 deg | 1.9% |
| local | wedge | 40 | 0.04 deg | 0.2% |
| local | cone | 40 | 0.22 deg | 1.9% |

**Blast energy from R(t).** A circle is fitted to the tracked shock mask in every
frame; R(t) is fitted with a power law (Sedov-Taylor predicts n = 2/5) and with the
exponent fixed at 2/5 to recover the energy E for the known ambient density.

| model | variant | frames changed | exponent n | median abs E error | worst E error | median radius error |
|---|---|---:|---:|---:|---:|---:|
| hosted | raw | 0 | 0.383 +- 0.031 | 0.5% | 79.4% | 0.15 px |
| hosted | gated | 13 of 240 | 0.399 +- 0.001 | 0.4% | 2.4% | 0.14 px |
| local | raw | 0 | 0.399 +- 0.001 | 0.3% | 2.8% | 0.13 px |
| local | gated | 3 of 240 | 0.399 +- 0.001 | 0.3% | 2.8% | 0.13 px |

## Failure mode found: clipped blast fronts

Once a blast front grows past the image edges, the visible shock is an open arc.
On 5 of 20 sequences the hosted model then predicts a filled region (up to 140k px,
circle-fit radius about 137 px against a true 200 px) with higher confidence than
the correct thin arc, which it still detects at lower confidence. ByteTrack
associates by box IoU, and the filled region has the same box as the ring, so the
track follows the wrong mask (`blast_014.mp4`, frames 9 to 11). The local model
does not show this on any sequence.

The **gated** variant applies a physical prior rather than a tuned threshold: a
shock front is a thin surface, so a valid mask must have a circle-fit residual
below 5% of its radius (a band of width w gives w / (sqrt(12) R), about 0.02 here;
a filled disk gives about 0.3). Among shock masks overlapping the track it keeps
the most confident thin one. It changes 3 of 240 frames for the local model with
no effect on its results, and restores the hosted model's exponent to
0.399 +- 0.001 and worst-case energy error to 2.4%.

Two engineering lessons from the same analysis:

1. `inference` returns polygon masks by default, which cannot represent holes, so
   every blast ring first came back as a filled disk (shock AP50 0.678, radius error
   27 px). Requesting `response_mask_format="rle"` fixes it.
2. ByteTrack confirms a track only on its second match, so first-frame detections
   carry tracker id -1; they are kept and attached to the track they precede.

## Files

- `hosted/`, `local/`: `summary.md`, `mask_map.json`, `shock_mach.json`,
  `blast_tracks.json`, predictions with RLE masks, and annotated videos for
  `blast_003` (typical) and `blast_014` (the clipped-front failure case).

Caveat: all results are on synthetic data from the same generator as the training
set, so they bound what real BOS images would give; the sim-to-real test is
pending written clearance of real images.
