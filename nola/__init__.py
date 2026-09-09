"""NoLA — Noise-Law Adaptation for cross-system CT projection denoising.

A projection-domain denoiser trained on one scanner degrades on another whose
photon budget and electronic-noise level differ. NoLA repairs it using only
unlabelled target sinograms: no source data, no paired targets, no clean
references. It does so by estimating the target's Poisson–Gaussian variance
law from the noisy projections themselves and using that law — rather than a
target data distribution — as the adaptation signal.

    >>> from nola.units import NoiseLaw, harmonise
    >>> from nola.lawfit import fit_law
    >>> law = fit_law(harmonise(target_sinogram))     # no labels used
    >>> law.i0, law.electronic_sigma
    (37511.6, 9.98)

The package is self-contained: `nola.geometry` carries the forward operators
and both acquisition models, so nothing outside this repository is required.

LEAKAGE RULE, which the code enforces rather than merely documents
    `clean_sinogram`, `sigma_*` and `image_hu` are computed from the clean
    signal. They may appear only in validation, in the estimator-accuracy
    figure, in the oracle-law ablation and in the supervised baseline. No
    estimation or adaptation path may read them; `nola.data.TargetSinograms`
    refuses to open them unless `allow_labels=True`, and nothing in the
    adaptation path sets it.
"""
from __future__ import annotations

__version__ = "1.0.0"

__all__ = [
    "geometry", "units", "lawfit", "losses", "adapt", "unet", "splits",
    "nps", "audit", "data",
]
