### Dose 25% — all metrics
232 slices from 4 held-out patients; mean ± s.d. **across patients**. Bold = best in column.

**Primary — vs FBP(clean\_sinogram), and the sinogram itself**

| Method | PSNR$_{op}$ | SSIM$_{op}$ | sino NRMSE |
|---|---|---|---|
| FBP (no denoising) | 33.44 ± 2.70 | 0.7448 ± 0.0877 | 0.0183 ± 0.0045 |
| B0 source-only | 40.53 ± 0.58 | 0.9619 ± 0.0037 | 0.0070 ± 0.0003 |
| AdaBN | 41.60 ± 0.71 | 0.9684 ± 0.0042 | 0.0064 ± 0.0003 |
| Noise2Inverse (K=2) | 41.47 ± 2.59 | 0.9464 ± 0.0341 | -- |
| Noise2Inverse (K=4) | 36.45 ± 1.77 | 0.9250 ± 0.0385 | -- |
| Masked SSL-TTA | 40.45 ± 2.11 | 0.9232 ± 0.0347 | 0.0076 ± 0.0015 |
| Global-variance TTA | 43.34 ± 1.93 | 0.9647 ± 0.0164 | 0.0060 ± 0.0010 |
| Global-variance TTA $+$anchor | 43.03 ± 1.97 | 0.9627 ± 0.0182 | 0.0067 ± 0.0008 |
| NoLA (ours, $L_{mom}$ only) | 43.50 ± 1.83 | 0.9665 ± 0.0151 | 0.0054 ± 0.0012 |
|   $+$whiteness | 43.32 ± 1.69 | 0.9658 ± 0.0146 | 0.0055 ± 0.0013 |
|   $+$anchor | 43.59 ± 1.69 | 0.9676 ± 0.0127 | 0.0055 ± 0.0013 |
|   $+$whiteness $+$anchor | 43.33 ± 1.69 | 0.9659 ± 0.0145 | 0.0054 ± 0.0012 |
|   whiteness$+$anchor, no moment | 42.58 ± 0.93 | 0.9578 ± 0.0071 | 0.0065 ± 0.0005 |
| NoLA, oracle law | 43.48 ± 1.61 | 0.9672 ± 0.0131 | 0.0054 ± 0.0012 |
| Supervised FT | 45.06 ± 0.94 | 0.9775 ± 0.0036 | **0.0040 ± 0.0001** |
| RED-CNN (retrained) | **45.96 ± 1.15** | **0.9795 ± 0.0032** | -- |
| CTformer (retrained) | 38.36 ± 1.47 | 0.8962 ± 0.0197 | -- |
|   $-$orthogonality (old full) | 43.28 ± 1.88 | 0.9643 ± 0.0177 | 0.0053 ± 0.0012 |

**Secondary — vs image\_hu (the scanner reconstruction)**

| Method | PSNR | SSIM | MS-SSIM | VIF | GMSD | LPIPS | DISTS |
|---|---|---|---|---|---|---|---|
| FBP (no denoising) | 33.12 ± 2.59 | 0.7189 ± 0.0868 | 0.9461 ± 0.0262 | 0.5795 ± 0.0465 | 0.0607 ± 0.0291 | 0.3674 ± 0.0913 | 0.2624 ± 0.0388 |
| B0 source-only | 38.49 ± 0.47 | 0.9284 ± 0.0041 | 0.9800 ± 0.0032 | 0.3971 ± 0.0244 | 0.0458 ± 0.0038 | 0.2323 ± 0.0441 | 0.2213 ± 0.0109 |
| AdaBN | 39.32 ± 0.54 | 0.9366 ± 0.0048 | 0.9842 ± 0.0030 | 0.4393 ± 0.0257 | 0.0354 ± 0.0036 | 0.1950 ± 0.0407 | 0.1917 ± 0.0141 |
| Noise2Inverse (K=2) | 39.52 ± 1.98 | 0.9199 ± 0.0327 | 0.9828 ± 0.0097 | 0.4855 ± 0.0271 | 0.0317 ± 0.0127 | 0.2314 ± 0.0675 | 0.1811 ± 0.0260 |
| Noise2Inverse (K=4) | 35.52 ± 1.54 | 0.8968 ± 0.0378 | 0.9721 ± 0.0128 | 0.3789 ± 0.0188 | 0.0648 ± 0.0141 | 0.2337 ± 0.0661 | 0.1746 ± 0.0255 |
| Masked SSL-TTA | 39.41 ± 1.85 | 0.8992 ± 0.0345 | 0.9848 ± 0.0061 | **0.5814 ± 0.0350** | 0.0184 ± 0.0055 | 0.2030 ± 0.0783 | 0.1794 ± 0.0321 |
| Global-variance TTA | 40.82 ± 1.36 | 0.9375 ± 0.0162 | 0.9871 ± 0.0059 | 0.5519 ± 0.0273 | 0.0241 ± 0.0115 | 0.1358 ± 0.0409 | **0.1271 ± 0.0199** |
| Global-variance TTA $+$anchor | 40.58 ± 1.44 | 0.9339 ± 0.0178 | 0.9866 ± 0.0064 | 0.5387 ± 0.0288 | 0.0259 ± 0.0123 | 0.1464 ± 0.0450 | 0.1357 ± 0.0216 |
| NoLA (ours, $L_{mom}$ only) | 40.75 ± 1.21 | 0.9384 ± 0.0149 | 0.9871 ± 0.0055 | 0.5361 ± 0.0273 | 0.0244 ± 0.0104 | 0.1398 ± 0.0416 | 0.1390 ± 0.0170 |
|   $+$whiteness | 40.65 ± 1.11 | 0.9372 ± 0.0143 | 0.9871 ± 0.0049 | 0.5195 ± 0.0270 | 0.0243 ± 0.0084 | 0.1604 ± 0.0482 | 0.1570 ± 0.0180 |
|   $+$anchor | 40.86 ± 1.10 | 0.9397 ± 0.0125 | 0.9873 ± 0.0051 | 0.5403 ± 0.0270 | 0.0239 ± 0.0096 | 0.1355 ± 0.0387 | 0.1353 ± 0.0155 |
|   $+$whiteness $+$anchor | 40.65 ± 1.11 | 0.9374 ± 0.0142 | 0.9870 ± 0.0049 | 0.5189 ± 0.0270 | 0.0245 ± 0.0084 | 0.1609 ± 0.0481 | 0.1573 ± 0.0181 |
|   whiteness$+$anchor, no moment | 40.35 ± 0.65 | 0.9291 ± 0.0072 | 0.9870 ± 0.0026 | 0.5240 ± 0.0274 | 0.0189 ± 0.0033 | 0.1381 ± 0.0391 | 0.1466 ± 0.0187 |
| NoLA, oracle law | 40.75 ± 1.04 | 0.9387 ± 0.0129 | 0.9875 ± 0.0045 | 0.5212 ± 0.0269 | 0.0233 ± 0.0074 | 0.1583 ± 0.0469 | 0.1553 ± 0.0177 |
| Supervised FT | 41.78 ± 0.59 | 0.9488 ± 0.0046 | 0.9905 ± 0.0020 | 0.5465 ± 0.0288 | 0.0167 ± 0.0030 | **0.1252 ± 0.0306** | 0.1306 ± 0.0147 |
| RED-CNN (retrained) | **42.56 ± 0.74** | **0.9539 ± 0.0065** | **0.9911 ± 0.0020** | 0.5420 ± 0.0255 | **0.0155 ± 0.0028** | 0.1821 ± 0.0285 | 0.1405 ± 0.0050 |
| CTformer (retrained) | 37.13 ± 1.22 | 0.8593 ± 0.0192 | 0.9765 ± 0.0065 | 0.4938 ± 0.0103 | 0.0298 ± 0.0108 | 0.2685 ± 0.0502 | 0.2011 ± 0.0171 |
|   $-$orthogonality (old full) | 40.59 ± 1.27 | 0.9358 ± 0.0174 | 0.9869 ± 0.0055 | 0.5192 ± 0.0273 | 0.0246 ± 0.0095 | 0.1607 ± 0.0497 | 0.1564 ± 0.0200 |

**Texture — vs a full-dose acquisition through this operator**

| Method | NPS shape | NPS power |
|---|---|---|
| FBP (no denoising) | 0.452 ± 0.032 | 4.857 ± 1.570 |
| B0 source-only | 0.477 ± 0.063 | 0.077 ± 0.024 |
| AdaBN | 0.652 ± 0.107 | 0.131 ± 0.049 |
| Noise2Inverse (K=2) | 1.063 ± 0.202 | 0.650 ± 0.439 |
| Noise2Inverse (K=4) | 1.218 ± 0.232 | 0.780 ± 0.600 |
| Masked SSL-TTA | **0.422 ± 0.061** | 0.714 ± 0.068 |
| Global-variance TTA | 1.087 ± 0.089 | 0.507 ± 0.128 |
| Global-variance TTA $+$anchor | 0.965 ± 0.095 | 0.454 ± 0.169 |
| NoLA (ours, $L_{mom}$ only) | 1.047 ± 0.060 | 0.396 ± 0.120 |
|   $+$whiteness | 0.891 ± 0.084 | 0.330 ± 0.087 |
|   $+$anchor | 1.082 ± 0.057 | 0.402 ± 0.091 |
|   $+$whiteness $+$anchor | 0.888 ± 0.084 | 0.327 ± 0.087 |
|   whiteness$+$anchor, no moment | 0.846 ± 0.037 | 0.371 ± 0.079 |
| NoLA, oracle law | 0.891 ± 0.087 | 0.310 ± 0.071 |
| Supervised FT | 1.023 ± 0.085 | 0.246 ± 0.086 |
| RED-CNN (retrained) | 2.431 ± 0.309 | 0.173 ± 0.065 |
| CTformer (retrained) | 0.471 ± 0.059 | 1.191 ± 0.176 |
|   $-$orthogonality (old full) | 0.871 ± 0.091 | 0.346 ± 0.128 |

Notes. The three blocks use **different references** and must not be read across: PSNR$_{op}$/SSIM$_{op}$ isolate denoising error; the secondary block is comparable with published numbers but also charges every method for the B30-versus-ramp kernel mismatch; NPS power is a magnitude, not a quality score — the ideal output is the clean image, whose ratio is near 0, while 1.0 means as noisy as a full-dose scan, so no 'best' is marked for it.
GMSD, LPIPS and DISTS are distances (lower is better); VIF and the SSIM family are similarities (higher is better).
