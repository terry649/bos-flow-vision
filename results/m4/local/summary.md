Model: local:runs/v1_dis_medium/checkpoint_best_total.pth

# M4 physics check

## Mask mAP on the test split (pycocotools, segm)

mAP50-95 = 0.826, mAP50 = 1.000

| class | AP50-95 | AP50 | test instances |
|---|---:|---:|---:|
| shock | 0.573 | 1.000 | 80 |
| expansion_fan | 0.977 | 1.000 | 20 |
| shear_layer | 0.928 | 1.000 | 20 |

## Mach number from predicted shock masks

| flow | shocks | median abs beta error (deg) | median abs Mach error | median rel Mach error | unsolvable |
|---|---:|---:|---:|---:|---:|
| wedge | 40 | 0.04 | 0.004 | 0.2% | 0 |
| cone | 40 | 0.22 | 0.050 | 1.9% | 0 |

## Blast fronts tracked through sequences

raw: the tracker's detection in each frame. gated: thin-front check (circle-fit residual < 5% of radius) among shock masks overlapping the track, most confident kept.

| variant | sequences | frames used | frames replaced or dropped | exponent n (theory 0.4) | median abs E error | worst E error | median radius error (px) |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 20/20 | 240/240 | 0 | 0.3988 +- 0.0012 | 0.3% | 2.8% | 0.13 |
| gated | 20/20 | 238/240 | 3 | 0.3989 +- 0.0011 | 0.3% | 2.8% | 0.13 |
