"""Measuring whether a residual obeys a noise law.

Shared by ``scripts/residual_audit.py`` and ``scripts/system_grid.py`` so the
headline experiment and the controlled sweep score the claim the same way.

WHY THIS AND NOT PSNR
    On this pipeline the pixel ranking is inverted with respect to texture
    fidelity: at 25 percent dose CoreDiff leads PSNR by 3.1 dB while retaining
    29 percent of full-dose texture power, and the unadapted source model keeps
    10 percent while matching RED-CNN on PSNR and beating it on SSIM. PSNR
    rewards removing texture, so it cannot adjudicate a method whose claim is
    that it stops removing texture.

    The claim is checkable directly. If the model has removed exactly the
    noise, the residual's variance conditioned on the signal equals what the
    law predicts there. The discrepancy is measurable, has an absolute zero,
    and cannot be improved by smoothing: smoothing more makes the residual too
    large, smoothing less makes it too small, and only the right amount scores
    zero.

    It also fixes an artefact that shows up in the robustness sweep. Scored on
    PSNR, deliberately UNDER-stating the target noise by 25 percent beats using
    the correct law, because a model told the noise is small denoises less and
    the baseline was over-smoothing to begin with. Scored on law match, the
    correct law wins by construction, and the sweep measures what it was meant
    to measure.

ROBUST, AND ON A LOG SCALE
    The conditional variance is estimated with the median absolute deviation,
    not the sample variance, so a handful of structured outliers cannot inflate
    it. The discrepancy is a mean absolute log ratio, so being twice too large
    and twice too small cost the same 0.69, and the four orders of magnitude
    between an air ray and a ray through the spine do not let the dense bands
    dominate.
"""
from __future__ import annotations

import numpy as np

from .lawfit import MAD_TO_SIGMA
from .units import NoiseLaw

LAGS = (1, 2, 3, 4, 5)


def band_edges(y_hat: np.ndarray, n_bins: int = 20) -> np.ndarray:
    """Quantile bands of the signal estimate, with the ends opened out."""
    e = np.quantile(np.asarray(y_hat), np.linspace(0, 1, n_bins + 1))
    e[0] -= 1e-6
    e[-1] += 1e-6
    return e


def accumulate_bands(residual: np.ndarray, y_hat: np.ndarray,
                     edges: np.ndarray, per_band: list, band_signal: list,
                     stride: int = 11) -> None:
    """Append this slice's residuals and signal values to their bands.

    ``stride`` subsamples: 850k pixels per slice over 116 slices would put 100
    million values in a band and change no median.
    """
    n_bins = len(edges) - 1
    k = np.clip(np.digitize(y_hat.ravel(), edges) - 1, 0, n_bins - 1)
    rr, ss = residual.ravel(), y_hat.ravel()
    order = np.argsort(k, kind="stable")
    k, rr, ss = k[order], rr[order], ss[order]
    bounds = np.searchsorted(k, np.arange(n_bins + 1))
    for j in range(n_bins):
        a0, a1 = bounds[j], bounds[j + 1]
        if a1 > a0:
            per_band[j].append(rr[a0:a1][::stride].copy())
            band_signal[j].append(ss[a0:a1][::stride].copy())


def law_match(per_band: list, band_signal: list, law: NoiseLaw) -> dict:
    """Mean |log(measured / predicted)| over bands. Zero is perfect."""
    errs, centres, measured, predicted = [], [], [], []
    for pool, sig in zip(per_band, band_signal):
        if not pool:
            continue
        r = np.concatenate(pool)
        s = np.concatenate(sig)
        mad = np.median(np.abs(r - np.median(r)))
        v = (MAD_TO_SIGMA * mad) ** 2
        p = float(np.mean(law.variance(s)))
        if v <= 0 or p <= 0 or not np.isfinite(v):
            continue
        errs.append(abs(np.log(v / p)))
        centres.append(float(np.mean(s)))
        measured.append(float(v))
        predicted.append(p)
    return {"law_match": float(np.mean(errs)) if errs else float("nan"),
            "n_bands": len(errs),
            "bins": {"centre": centres, "measured": measured,
                     "predicted": predicted}}


def whiteness_and_orthogonality(residual: np.ndarray, y_hat: np.ndarray,
                                law: NoiseLaw) -> tuple[dict, float]:
    """Autocorrelation of the standardised residual, and cor(r, y_hat)^2."""
    sig = law.sigma(y_hat)
    z = residual / np.maximum(sig, 1e-12)
    z = z - z.mean()
    den = float((z * z).mean()) or 1.0
    rho = {}
    for lag in LAGS:
        rho[f"det{lag}"] = float((z[:, :-lag] * z[:, lag:]).mean()) / den
        rho[f"view{lag}"] = float((z[:-lag, :] * z[lag:, :]).mean()) / den
    rc = residual - residual.mean()
    sc = y_hat - y_hat.mean()
    denom = float((rc ** 2).mean() * (sc ** 2).mean()) or 1.0
    return rho, float((rc * sc).mean()) ** 2 / denom
