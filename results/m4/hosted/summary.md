Model: roboflow:bos-flow-features/1

# M4 physics check

## Mask mAP on the test split (pycocotools, segm)

mAP50-95 = 0.840, mAP50 = 1.000

| class | AP50-95 | AP50 | test instances |
|---|---:|---:|---:|
| shock | 0.618 | 1.000 | 80 |
| expansion_fan | 0.989 | 1.000 | 20 |
| shear_layer | 0.912 | 1.000 | 20 |

## Mach number from predicted shock masks

| flow | shocks | median abs beta error (deg) | median abs Mach error | median rel Mach error | unsolvable |
|---|---:|---:|---:|---:|---:|
| wedge | 40 | 0.04 | 0.004 | 0.1% | 0 |
| cone | 40 | 0.24 | 0.046 | 1.9% | 0 |

## Blast fronts tracked through sequences

raw: the tracker's detection in each frame. gated: thin-front check (circle-fit residual < 5% of radius) among shock masks overlapping the track, most confident kept.

| variant | sequences | frames used | frames replaced or dropped | exponent n (theory 0.4) | median abs E error | worst E error | median radius error (px) |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 20/20 | 240/240 | 0 | 0.3831 +- 0.0307 | 0.5% | 79.4% | 0.15 |
| gated | 20/20 | 231/240 | 13 | 0.3990 +- 0.0013 | 0.4% | 2.4% | 0.14 |
