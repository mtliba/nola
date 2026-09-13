### Dose 10% — all metrics
232 slices from 4 held-out patients; mean ± s.d. **across patients**. Bold = best in column.

**Primary — vs FBP(clean\_sinogram), and the sinogram itself**

| Method | PSNR$_{op}$ | SSIM$_{op}$ | sino NRMSE |
|---|---|---|---|
| FBP (no denoising) | 26.75 ± 3.09 | 0.5173 ± 0.1166 | 0.0426 ± 0.0121 |
| B0 source-only | 40.55 ± 0.77 | 0.9588 ± 0.0069 | 0.0074 ± 0.0002 |
| AdaBN | 38.59 ± 3.45 | 0.9248 ± 0.0571 | 0.0115 ± 0.0065 |
| Noise2Inverse (K=2) | 36.96 ± 3.94 | 0.8870 ± 0.0691 | -- |
| Masked SSL-TTA | 36.38 ± 2.63 | 0.8392 ± 0.0724 | 0.0148 ± 0.0045 |
| Global-variance TTA | 37.40 ± 2.62 | 0.8892 ± 0.0539 | 0.0155 ± 0.0024 |
| NoLA (ours, $L_{mom}$ only) | 38.91 ± 2.73 | 0.9152 ± 0.0412 | 0.0131 ± 0.0047 |
|   $+$whiteness | 38.31 ± 2.52 | 0.9076 ± 0.0428 | 0.0139 ± 0.0043 |
|   $+$anchor | 38.29 ± 2.57 | 0.9130 ± 0.0376 | 0.0143 ± 0.0045 |
|   $+$whiteness $+$anchor | 36.68 ± 3.24 | 0.8506 ± 0.0844 | 0.0150 ± 0.0049 |
|   whiteness$+$anchor, no moment | 40.40 ± 1.75 | 0.9397 ± 0.0212 | 0.0116 ± 0.0034 |
| NoLA, oracle law | 38.35 ± 2.50 | 0.9080 ± 0.0425 | 0.0136 ± 0.0041 |
| Supervised FT | **42.83 ± 1.06** | **0.9691 ± 0.0059** | **0.0055 ± 0.0004** |
|   $-$orthogonality (old full) | 36.08 ± 3.73 | 0.8386 ± 0.0967 | 0.0165 ± 0.0064 |

**Secondary — vs image\_hu (the scanner reconstruction)**

| Method | PSNR | SSIM | MS-SSIM | VIF | GMSD | LPIPS | DISTS |
|---|---|---|---|---|---|---|---|
| FBP (no denoising) | 26.63 ± 3.05 | 0.4895 ± 0.1120 | 0.8539 ± 0.0633 | 0.4340 ± 0.0589 | 0.1466 ± 0.0469 | 0.4939 ± 0.0952 | 0.3491 ± 0.0389 |
| B0 source-only | 38.55 ± 0.58 | 0.9260 ± 0.0071 | 0.9794 ± 0.0044 | 0.4028 ± 0.0251 | 0.0425 ± 0.0039 | 0.2222 ± 0.0445 | 0.2060 ± 0.0095 |
| AdaBN | 36.80 ± 2.95 | 0.8940 ± 0.0553 | 0.9668 ± 0.0243 | 0.4100 ± 0.0307 | 0.0600 ± 0.0326 | 0.2273 ± 0.0644 | 0.2039 ± 0.0256 |
| Noise2Inverse (K=2) | 35.68 ± 3.38 | 0.8612 ± 0.0654 | 0.9568 ± 0.0275 | 0.4133 ± 0.0297 | 0.0707 ± 0.0341 | 0.2631 ± 0.0743 | 0.2121 ± 0.0318 |
| Masked SSL-TTA | 35.72 ± 2.42 | 0.8147 ± 0.0716 | 0.9669 ± 0.0154 | **0.4919 ± 0.0403** | 0.0361 ± 0.0111 | 0.3052 ± 0.0961 | 0.2405 ± 0.0369 |
| Global-variance TTA | 35.98 ± 2.09 | 0.8599 ± 0.0507 | 0.9615 ± 0.0191 | 0.4119 ± 0.0252 | 0.0654 ± 0.0234 | 0.2547 ± 0.0728 | 0.2099 ± 0.0297 |
| NoLA (ours, $L_{mom}$ only) | 37.43 ± 2.16 | 0.8909 ± 0.0386 | 0.9658 ± 0.0168 | 0.4724 ± 0.0339 | 0.0592 ± 0.0220 | 0.2112 ± 0.0686 | 0.1806 ± 0.0293 |
|   $+$whiteness | 36.99 ± 2.04 | 0.8826 ± 0.0411 | 0.9633 ± 0.0171 | 0.4576 ± 0.0324 | 0.0621 ± 0.0199 | 0.2334 ± 0.0684 | 0.2004 ± 0.0264 |
|   $+$anchor | 37.01 ± 2.10 | 0.8889 ± 0.0356 | 0.9609 ± 0.0185 | 0.4774 ± 0.0335 | 0.0687 ± 0.0233 | 0.2073 ± 0.0605 | 0.1801 ± 0.0279 |
|   $+$whiteness $+$anchor | 35.49 ± 2.76 | 0.8250 ± 0.0828 | 0.9569 ± 0.0231 | 0.4500 ± 0.0378 | 0.0590 ± 0.0200 | 0.2667 ± 0.0861 | 0.2109 ± 0.0333 |
|   whiteness$+$anchor, no moment | 38.72 ± 1.32 | 0.9133 ± 0.0196 | 0.9776 ± 0.0083 | 0.4668 ± 0.0327 | 0.0339 ± 0.0089 | 0.1865 ± 0.0566 | 0.1716 ± 0.0211 |
| NoLA, oracle law | 37.02 ± 2.02 | 0.8828 ± 0.0407 | 0.9638 ± 0.0170 | 0.4592 ± 0.0328 | 0.0620 ± 0.0203 | 0.2331 ± 0.0679 | 0.2005 ± 0.0262 |
| Supervised FT | **40.33 ± 0.73** | **0.9392 ± 0.0066** | **0.9853 ± 0.0036** | 0.4789 ± 0.0308 | **0.0274 ± 0.0047** | **0.1615 ± 0.0408** | **0.1601 ± 0.0158** |
|   $-$orthogonality (old full) | 34.92 ± 3.22 | 0.8135 ± 0.0944 | 0.9518 ± 0.0280 | 0.4471 ± 0.0402 | 0.0657 ± 0.0251 | 0.2707 ± 0.0911 | 0.2140 ± 0.0364 |

**Texture — vs a full-dose acquisition through this operator**

| Method | NPS shape | NPS power |
|---|---|---|
| FBP (no denoising) | 0.584 ± 0.082 | 25.845 ± 10.375 |
| B0 source-only | **0.584 ± 0.065** | 0.135 ± 0.023 |
| AdaBN | 0.714 ± 0.096 | 1.953 ± 2.779 |
| Noise2Inverse (K=2) | 0.786 ± 0.091 | 3.462 ± 2.597 |
| Masked SSL-TTA | 0.693 ± 0.065 | 1.668 ± 0.484 |
| Global-variance TTA | 0.731 ± 0.098 | 1.750 ± 0.813 |
| NoLA (ours, $L_{mom}$ only) | 1.041 ± 0.119 | 1.601 ± 0.523 |
|   $+$whiteness | 0.832 ± 0.082 | 1.497 ± 0.395 |
|   $+$anchor | 1.159 ± 0.082 | 1.933 ± 0.519 |
|   $+$whiteness $+$anchor | 0.722 ± 0.129 | 2.545 ± 0.856 |
|   whiteness$+$anchor, no moment | 0.762 ± 0.037 | 0.538 ± 0.039 |
| NoLA, oracle law | 0.828 ± 0.080 | 1.492 ± 0.392 |
| Supervised FT | 0.902 ± 0.123 | 0.220 ± 0.066 |
|   $-$orthogonality (old full) | 0.717 ± 0.132 | 3.229 ± 1.327 |

Notes. The three blocks use **different references** and must not be read across: PSNR$_{op}$/SSIM$_{op}$ isolate denoising error; the secondary block is comparable with published numbers but also charges every method for the B30-versus-ramp kernel mismatch; NPS power is a magnitude, not a quality score — the ideal output is the clean image, whose ratio is near 0, while 1.0 means as noisy as a full-dose scan, so no 'best' is marked for it.
GMSD, LPIPS and DISTS are distances (lower is better); VIF and the SSIM family are similarities (higher is better).
