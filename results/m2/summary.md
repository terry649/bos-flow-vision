# M2 displacement benchmark

100 scenes, noise levels clean, low, medium, high. EPE in pixels over valid pixels; feature EPE inside labeled flow features; peak ratio is estimated over true magnitude in the top 5% of feature pixels.


## EPE, all valid pixels, noise = clean

| flow type | farneback | lucas_kanade | dis | tvl1 | demons | raft |
|---|---:|---:|---:|---:|---:|---:|
| wedge | 0.010 | 0.014 | 0.008 | 0.012 | 0.011 | 0.025 |
| cone | 0.009 | 0.012 | 0.007 | 0.013 | 0.015 | 0.042 |
| blast | 0.013 | 0.018 | 0.010 | 0.017 | 0.018 | 0.044 |
| expansion | 0.011 | 0.015 | 0.007 | 0.013 | 0.016 | 0.054 |
| shear_layer | 0.021 | 0.028 | 0.019 | 0.025 | 0.024 | 0.085 |
| all | 0.013 | 0.018 | 0.010 | 0.016 | 0.017 | 0.050 |


## EPE, all valid pixels, noise = low

| flow type | farneback | lucas_kanade | dis | tvl1 | demons | raft |
|---|---:|---:|---:|---:|---:|---:|
| wedge | 0.022 | 0.021 | 0.016 | 0.042 | 0.056 | 0.025 |
| cone | 0.020 | 0.019 | 0.015 | 0.041 | 0.057 | 0.042 |
| blast | 0.023 | 0.024 | 0.017 | 0.043 | 0.055 | 0.044 |
| expansion | 0.022 | 0.021 | 0.014 | 0.041 | 0.061 | 0.055 |
| shear_layer | 0.029 | 0.033 | 0.026 | 0.049 | 0.061 | 0.085 |
| all | 0.023 | 0.024 | 0.018 | 0.043 | 0.058 | 0.050 |


## EPE, all valid pixels, noise = medium

| flow type | farneback | lucas_kanade | dis | tvl1 | demons | raft |
|---|---:|---:|---:|---:|---:|---:|
| wedge | 0.042 | 0.033 | 0.028 | 0.064 | 0.100 | 0.024 |
| cone | 0.040 | 0.031 | 0.026 | 0.062 | 0.099 | 0.041 |
| blast | 0.043 | 0.035 | 0.028 | 0.065 | 0.097 | 0.044 |
| expansion | 0.042 | 0.033 | 0.026 | 0.062 | 0.104 | 0.053 |
| shear_layer | 0.048 | 0.043 | 0.037 | 0.069 | 0.102 | 0.083 |
| all | 0.043 | 0.035 | 0.029 | 0.064 | 0.100 | 0.049 |


## EPE, all valid pixels, noise = high

| flow type | farneback | lucas_kanade | dis | tvl1 | demons | raft |
|---|---:|---:|---:|---:|---:|---:|
| wedge | 0.144 | 0.095 | 0.094 | 0.121 | 0.230 | 0.027 |
| cone | 0.143 | 0.092 | 0.090 | 0.118 | 0.229 | 0.049 |
| blast | 0.143 | 0.095 | 0.090 | 0.122 | 0.223 | 0.047 |
| expansion | 0.145 | 0.095 | 0.091 | 0.119 | 0.234 | 0.055 |
| shear_layer | 0.148 | 0.101 | 0.099 | 0.124 | 0.230 | 0.089 |
| all | 0.145 | 0.096 | 0.093 | 0.121 | 0.229 | 0.053 |


## EPE inside flow features, noise = clean

| flow type | farneback | lucas_kanade | dis | tvl1 | demons | raft |
|---|---:|---:|---:|---:|---:|---:|
| wedge | 0.251 | 0.329 | 0.257 | 0.279 | 0.208 | 0.592 |
| cone | 0.153 | 0.195 | 0.124 | 0.134 | 0.137 | 0.449 |
| blast | 0.259 | 0.377 | 0.244 | 0.255 | 0.209 | 0.669 |
| expansion | 0.072 | 0.091 | 0.051 | 0.083 | 0.097 | 0.369 |
| shear_layer | 0.115 | 0.162 | 0.109 | 0.121 | 0.112 | 0.486 |
| all | 0.170 | 0.231 | 0.157 | 0.174 | 0.152 | 0.513 |


## EPE inside flow features, noise = low

| flow type | farneback | lucas_kanade | dis | tvl1 | demons | raft |
|---|---:|---:|---:|---:|---:|---:|
| wedge | 0.251 | 0.329 | 0.261 | 0.267 | 0.210 | 0.593 |
| cone | 0.154 | 0.195 | 0.126 | 0.136 | 0.141 | 0.450 |
| blast | 0.259 | 0.377 | 0.248 | 0.251 | 0.212 | 0.669 |
| expansion | 0.074 | 0.092 | 0.053 | 0.089 | 0.109 | 0.376 |
| shear_layer | 0.116 | 0.162 | 0.113 | 0.122 | 0.119 | 0.485 |
| all | 0.171 | 0.231 | 0.160 | 0.173 | 0.158 | 0.514 |


## EPE inside flow features, noise = medium

| flow type | farneback | lucas_kanade | dis | tvl1 | demons | raft |
|---|---:|---:|---:|---:|---:|---:|
| wedge | 0.256 | 0.332 | 0.279 | 0.281 | 0.234 | 0.593 |
| cone | 0.160 | 0.197 | 0.139 | 0.149 | 0.161 | 0.451 |
| blast | 0.265 | 0.380 | 0.270 | 0.281 | 0.235 | 0.668 |
| expansion | 0.087 | 0.098 | 0.066 | 0.105 | 0.138 | 0.368 |
| shear_layer | 0.123 | 0.165 | 0.125 | 0.132 | 0.144 | 0.468 |
| all | 0.178 | 0.234 | 0.176 | 0.190 | 0.182 | 0.510 |


## EPE inside flow features, noise = high

| flow type | farneback | lucas_kanade | dis | tvl1 | demons | raft |
|---|---:|---:|---:|---:|---:|---:|
| wedge | 0.294 | 0.343 | 0.319 | 0.338 | 0.318 | 0.595 |
| cone | 0.215 | 0.218 | 0.186 | 0.208 | 0.257 | 0.473 |
| blast | 0.305 | 0.393 | 0.332 | 0.379 | 0.327 | 0.668 |
| expansion | 0.177 | 0.145 | 0.121 | 0.167 | 0.257 | 0.363 |
| shear_layer | 0.184 | 0.186 | 0.166 | 0.185 | 0.249 | 0.496 |
| all | 0.235 | 0.257 | 0.225 | 0.256 | 0.281 | 0.519 |


## Relative EPE inside flow features, noise = clean

| flow type | farneback | lucas_kanade | dis | tvl1 | demons | raft |
|---|---:|---:|---:|---:|---:|---:|
| wedge | 0.435 | 0.563 | 0.457 | 0.495 | 0.363 | 0.986 |
| cone | 0.284 | 0.355 | 0.241 | 0.270 | 0.279 | 0.798 |
| blast | 0.391 | 0.563 | 0.378 | 0.413 | 0.323 | 1.008 |
| expansion | 0.115 | 0.142 | 0.084 | 0.157 | 0.194 | 0.662 |
| shear_layer | 0.221 | 0.305 | 0.213 | 0.244 | 0.217 | 0.758 |
| all | 0.289 | 0.386 | 0.275 | 0.316 | 0.275 | 0.842 |


## Relative EPE inside flow features, noise = low

| flow type | farneback | lucas_kanade | dis | tvl1 | demons | raft |
|---|---:|---:|---:|---:|---:|---:|
| wedge | 0.436 | 0.563 | 0.465 | 0.469 | 0.368 | 0.986 |
| cone | 0.286 | 0.355 | 0.247 | 0.275 | 0.290 | 0.798 |
| blast | 0.391 | 0.563 | 0.387 | 0.400 | 0.333 | 1.007 |
| expansion | 0.123 | 0.145 | 0.089 | 0.174 | 0.231 | 0.672 |
| shear_layer | 0.224 | 0.306 | 0.221 | 0.246 | 0.235 | 0.759 |
| all | 0.292 | 0.387 | 0.282 | 0.313 | 0.292 | 0.844 |


## Relative EPE inside flow features, noise = medium

| flow type | farneback | lucas_kanade | dis | tvl1 | demons | raft |
|---|---:|---:|---:|---:|---:|---:|
| wedge | 0.445 | 0.568 | 0.495 | 0.488 | 0.416 | 0.988 |
| cone | 0.303 | 0.362 | 0.276 | 0.302 | 0.344 | 0.807 |
| blast | 0.402 | 0.568 | 0.426 | 0.434 | 0.377 | 1.007 |
| expansion | 0.161 | 0.164 | 0.117 | 0.213 | 0.313 | 0.684 |
| shear_layer | 0.240 | 0.311 | 0.248 | 0.266 | 0.293 | 0.759 |
| all | 0.310 | 0.395 | 0.312 | 0.341 | 0.349 | 0.849 |


## Relative EPE inside flow features, noise = high

| flow type | farneback | lucas_kanade | dis | tvl1 | demons | raft |
|---|---:|---:|---:|---:|---:|---:|
| wedge | 0.531 | 0.596 | 0.562 | 0.589 | 0.595 | 0.989 |
| cone | 0.452 | 0.423 | 0.378 | 0.430 | 0.585 | 0.850 |
| blast | 0.497 | 0.602 | 0.526 | 0.581 | 0.575 | 1.007 |
| expansion | 0.408 | 0.300 | 0.262 | 0.359 | 0.631 | 0.702 |
| shear_layer | 0.385 | 0.364 | 0.334 | 0.368 | 0.532 | 0.810 |
| all | 0.454 | 0.457 | 0.412 | 0.466 | 0.584 | 0.872 |


## Peak displacement recovery (1 is perfect), noise = clean

| flow type | farneback | lucas_kanade | dis | tvl1 | demons | raft |
|---|---:|---:|---:|---:|---:|---:|
| wedge | 0.572 | 0.383 | 0.496 | 0.497 | 0.712 | 0.019 |
| cone | 0.792 | 0.663 | 0.752 | 0.762 | 0.866 | 0.198 |
| blast | 0.621 | 0.388 | 0.568 | 0.568 | 0.743 | 0.013 |
| expansion | 0.839 | 0.732 | 0.835 | 0.803 | 0.866 | 0.093 |
| shear_layer | 0.903 | 0.823 | 0.882 | 0.873 | 0.946 | 0.162 |
| all | 0.746 | 0.598 | 0.707 | 0.701 | 0.827 | 0.097 |


## Peak displacement recovery (1 is perfect), noise = low

| flow type | farneback | lucas_kanade | dis | tvl1 | demons | raft |
|---|---:|---:|---:|---:|---:|---:|
| wedge | 0.572 | 0.383 | 0.487 | 0.522 | 0.707 | 0.019 |
| cone | 0.793 | 0.664 | 0.747 | 0.766 | 0.867 | 0.196 |
| blast | 0.622 | 0.388 | 0.559 | 0.588 | 0.742 | 0.013 |
| expansion | 0.839 | 0.732 | 0.834 | 0.805 | 0.869 | 0.083 |
| shear_layer | 0.903 | 0.823 | 0.877 | 0.875 | 0.947 | 0.164 |
| all | 0.746 | 0.598 | 0.701 | 0.711 | 0.826 | 0.095 |


## Peak displacement recovery (1 is perfect), noise = medium

| flow type | farneback | lucas_kanade | dis | tvl1 | demons | raft |
|---|---:|---:|---:|---:|---:|---:|
| wedge | 0.570 | 0.382 | 0.457 | 0.509 | 0.692 | 0.017 |
| cone | 0.793 | 0.662 | 0.720 | 0.744 | 0.858 | 0.193 |
| blast | 0.620 | 0.388 | 0.522 | 0.561 | 0.731 | 0.011 |
| expansion | 0.837 | 0.732 | 0.816 | 0.770 | 0.868 | 0.099 |
| shear_layer | 0.902 | 0.822 | 0.861 | 0.864 | 0.944 | 0.175 |
| all | 0.744 | 0.597 | 0.675 | 0.690 | 0.819 | 0.099 |


## Peak displacement recovery (1 is perfect), noise = high

| flow type | farneback | lucas_kanade | dis | tvl1 | demons | raft |
|---|---:|---:|---:|---:|---:|---:|
| wedge | 0.605 | 0.405 | 0.430 | 0.452 | 0.715 | 0.019 |
| cone | 0.809 | 0.677 | 0.696 | 0.668 | 0.872 | 0.157 |
| blast | 0.640 | 0.401 | 0.467 | 0.455 | 0.743 | 0.013 |
| expansion | 0.841 | 0.739 | 0.790 | 0.685 | 0.872 | 0.135 |
| shear_layer | 0.905 | 0.823 | 0.827 | 0.797 | 0.951 | 0.136 |
| all | 0.760 | 0.609 | 0.642 | 0.611 | 0.831 | 0.092 |


## Runtime per image pair, seconds (mean)

- farneback: 0.027
- lucas_kanade: 0.077
- dis: 0.025
- tvl1: 1.248
- demons: 0.243
- raft: 0.124
