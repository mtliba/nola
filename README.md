# NoLA — Noise-Law Adaptation

**Adapt a CT projection denoiser to a new scanner using only unlabelled target
sinograms.** No source data, no paired targets, no clean references.

A denoiser trained on one CT system degrades on another whose photon budget and
electronic-noise level differ. NoLA repairs it by estimating the target's
Poisson–Gaussian *variance law* directly from the noisy projections, then using
that law — rather than a target data distribution — as the adaptation signal.

On LoDoPaB → Mayo-geometry transfer at 25 % dose, NoLA improves the frozen
source model by **+2.97 dB** and recovers **65.5 %** of the supervised
fine-tuning gap, using no labels.

---

## Try it in one command

```bash
pip install -e ".[demo]"
python examples/demo.py
```

Runs on a laptop CPU in about six minutes and needs **no data download** and no
ASTRA. It builds a phantom, acquires it under two different noise laws, trains a
small source denoiser, estimates the target law from unlabelled sinograms,
adapts, and reports the gap closing. A typical run:

```
shift magnitude D = 2.18   -> adaptation should help

[3/6] estimating the target noise law from UNLABELLED sinograms
      estimated  I0=  34124.7  sigma_e=  8.73
      truth      I0=  37500.0  sigma_e= 10.00
      error           9.00%             12.68%   (no labels were read)

      no denoising               49.193 dB
      source model, zero-shot    49.530 dB    +0.34 dB vs noisy
      NoLA adapted (no labels)   51.855 dB    +2.66 dB vs noisy

      adaptation gain over the frozen source model: +2.32 dB
```

**The demo also tells you when the method should not be used.** It prints the
shift magnitude `D` — computable with no labels — before adapting. The defaults
mirror the paper's 25 % dose operating point (D = 2.18 here, 2.09 there). Run

```bash
python examples/demo.py --target-i0 8000 --target-sigma-e 12   # D = 0.50
```

and adaptation *loses* about half a decibel, exactly as the criterion predicts.
That failure is a feature of the demo, not an accident of it.

Steps 5 and 6 of the demo call the same `nola.lawfit` and `nola.adapt` used for
every number in the paper, so it exercises the shipped code rather than a
simplified restatement of it.

---

## The idea in three equations

For a post-log projection `p`, the Poisson–Gaussian acquisition model gives

```
Var(p) ≈ e^p / I₀  +  σ_e² e^{2p} / I₀²
```

Both systems store the same dimensionless line integral, but LoDoPaB divides
by `MU_MAX` and Mayo does not. Harmonising is therefore an **exact unit
conversion with no free parameter**, `p = c_M · y` with `c_M = MU_MAX =
81.35858`, and in common units both systems obey one two-parameter law:

```
v(y) = a·e^{c_M y} + b·e^{2 c_M y},     a = 1/(I₀ c_M²),   b = σ_e²/(I₀² c_M²)
```

The source acquisition is purely Poisson, so **b = 0 exactly** — it sits on the
boundary of the family. The target has electronic noise, so `b > 0`. That is
why the target law cannot be reached by rescaling the source law, and why
adaptation must *identify* `(a, b)` rather than shift a magnitude.

---

## What the code does

```python
from nola.units import harmonise
from nola.lawfit import fit_law
from nola.losses import NoLAWeights
from nola.adapt import adapt

law = fit_law([harmonise(y) for y in target_sinograms]).law   # no labels
print(law.i0, law.electronic_sigma)

adapt(model, loader, law, NoLAWeights(), steps=10_000, lr=1e-5)
```

The final objective is **one term** — the signal-conditional moment loss:
in each of K = 20 signal bands, the residual's mean and variance are matched
to the estimated law. Two further terms were tested and removed:

| term | what it does | verdict |
|---|---|---|
| `moment` | conditional residual mean and variance match the law | **kept — the whole objective** |
| `whiteness` | decorrelates the standardised residual, lags 1–5 | removed: −0.17 dB, buys decorrelation only |
| `anchor` | relative L2-SP toward the source weights | removed: +0.10 dB but not needed for stability |

`NoLAWeights()` still exposes all of them for the ablations. Residual–output
**orthogonality** is off for a stronger reason: the MMSE estimator violates it,
so driving it to zero moves the model away from the optimum.

---

## Leakage rule

`clean_sinogram`, `sigma_*` and `image_hu` are computed from the clean signal.
They may appear **only** in validation, the estimator-accuracy figure, the
oracle-law ablation, and the supervised baseline.

This is enforced, not just documented: `nola.data.TargetSinograms` refuses to
open them unless `allow_labels=True`, and nothing in the adaptation path sets
it. `NoiseLaw.mayo_truth()` exists for those four uses and nothing else.

---

## Reproducing the paper

The demo uses synthetic data. Reproducing the paper's numbers needs the real
datasets, which we cannot redistribute.

**1. Get the data**

- **LoDoPaB-CT** (source): https://zenodo.org/record/3384092
- **Mayo 2016 LDCT Grand Challenge** (target anatomy): apply at
  https://www.aapm.org/grandchallenge/lowdosect/

**2. Install with the tomography extras** (needs ODL + ASTRA)

```bash
conda install -c astra-toolbox astra-toolbox
pip install -e ".[full]"
```

**3. Run the pipeline**

```bash
python scripts/prepare_data.py   --data-root $DATA --out-dir data/
python scripts/estimate_law.py   --data-root $DATA --out-dir results/   # no labels
python scripts/adapt_nola.py     --data-root $DATA --out-dir results/ --dose 25
python scripts/evaluate.py       --data-root $DATA --out-dir results/ \
                                 --checkpoint results/nola_dose25.pt --dose 25
```

**Pretrained weights.** Both released checkpoints are the ones every number in
the paper came from:

```bash
python scripts/download_weights.py          # both, into checkpoints/
python scripts/download_weights.py --verify-only
```

| file | what it is |
|---|---|
| `theta_src.pt` | the source denoiser, trained on 8,000 LoDoPaB slices. The starting point for adaptation, and the "source-only" row. |
| `nola_dose25.pt` | `theta_src` after NoLA adaptation to the 25 % dose target, using unlabelled sinograms only. The "NoLA" row. |

Each is 67 MB and is checked against a recorded SHA-256 — a truncated torch
checkpoint usually loads without complaint and then produces quietly wrong
numbers. `scripts/train_source.py` trains θ_src from LoDoPaB if you would
rather start from scratch.

**A warning about the operator backend.** `astra_cpu` and `astra_cuda`
reconstruct the same sinogram differently enough to shift PSNR by **1.2 dB on
average and 2.9 dB at worst**, always in the same direction. Every evaluation
in a single table must use the same backend. `scripts/evaluate.py` records
`operator_impl` in every result and the table builders refuse to mix them.

---

## Results

Cross-system transfer, 25 % dose, 232 slices from four held-out patients.
`PSNR_op` is measured against `FBP(clean sinogram)`, which isolates denoising
error from reconstruction mismatch.

| Method | PSNR_op ↑ | ΔPSNR | SSIM_op ↑ | law err ↓ | max\|ρ\| ↓ |
|---|---|---|---|---|---|
| FBP (no denoising) | 33.44 ± 2.70 | −7.09 | 0.7448 | — | — |
| Source-only | 40.53 ± 0.58 | 0.00 | 0.9619 | 0.494 | 0.4067 |
| AdaBN | 41.60 ± 0.71 | +1.07 | 0.9684 | 0.354 | 0.2816 |
| Noise2Inverse (K=2) | 41.47 ± 2.59 | +0.94 | 0.9464 | — | — |
| Masked SSL-TTA | 40.45 ± 2.11 | −0.08 | 0.9232 | 0.607 | 0.2081 |
| Global-Variance TTA | 43.34 ± 1.93 | +2.82 | 0.9647 | 0.438 | 0.6344 |
| **NoLA** | **43.50 ± 1.83** | **+2.97** | 0.9665 | **0.150** | 0.0997 |
| NoLA, oracle law | 43.64 ± 1.81 | +3.11 | 0.9673 | 0.279 | 0.0969 |
| *Supervised fine-tune* | *45.06 ± 0.94* | *+4.53* | *0.9775* | *0.544* | *0.0707* |

Against Noise2Inverse — the like-for-like label-free baseline, trained on the
same target patients — NoLA wins on all four held-out patients individually
(+2.95, +2.15, +2.01, +0.99 dB) and on 96.1 % of slices. Against Masked
SSL-TTA from the same initialisation, which does not help at all (−0.08 dB vs
source), NoLA is +3.05 dB ahead on 100 % of slices.

### Matching the objective is not the same as improving the image

The residual audit on held-out patients shows the two do not move together, in
either direction. *Not necessary:* supervised fine-tuning has the **worst** law
match of any method (0.544) and the best image. *Not sufficient:* Global-Variance
TTA reaches almost NoLA's PSNR (43.34 vs 43.50) while leaving the law barely
matched (0.438 vs 0.150) and driving residual autocorrelation to **0.634 —
worse than the unadapted source model (0.407)**. It matches the average
variance and corrupts the structure. The signal-dependent law is what makes the
adapted residual actually obey the target physics; on PSNR alone the margin is
+0.15 dB.

*Estimation is not the bottleneck:* the same objective run with the **true**
target law instead of the estimate — the only thing changed — gives 43.64 dB,
+0.14 dB, on every slice. At 10 % dose the true law makes the adapted model
*worse* (36.30 vs 38.91 dB): that failure is the regime, not the estimator.

---

## Project page

Interactive versions of the figures, including material that did not fit in the
paper: **https://mtliba.github.io/nola/**

---

## Layout

```
nola/            the method. self-contained; numpy + torch only
  geometry.py      both acquisition models and the forward operators
  units.py         c_M, the NoiseLaw dataclass, harmonise/dehmonise
  lawfit.py        estimating (a, b) from unlabelled sinograms
  losses.py        the three objective terms
  adapt.py         the adaptation loop, with divergence guards
  audit.py         residual statistics on held-out data
examples/demo.py end-to-end, synthetic, no download
scripts/         the full reproduction pipeline
docs/            the project page
tests/           unit tests, including the leakage rule
```

## Citation

```bibtex
@inproceedings{nola2026,
  title     = {NoLA: Source-Free Noise-Law Adaptation for Cross-System
               CT Projection Denoising},
  booktitle = {Under Review},
  year      = {2026}
}
```

## License

MIT for the code. The datasets are governed by their own licences and are not
redistributed here.
