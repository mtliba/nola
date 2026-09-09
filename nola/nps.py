"""Radial noise power spectrum in uniform regions.

Protocol v2 section 9 asks for this as the anti-over-smoothing metric, and it
is the one number in the table that a model cannot improve by blurring.

WHY THE PIXEL METRICS NEED A COMPANION
    PSNR and SSIM both reward removing noise more than they punish removing
    texture, so a denoiser that flattens uniform tissue scores well while
    destroying exactly the low-contrast detectability that low-dose CT is
    judged on clinically. The noise power spectrum makes the trade visible:
    over-smoothing shows up as power missing at high spatial frequency, and no
    amount of it improves the score.

MAGNITUDE AND SHAPE ARE DIFFERENT QUESTIONS AND ARE REPORTED SEPARATELY
    Total noise power is not a quality measure in either direction, and
    treating it as one is a mistake worth stating plainly. The reference here
    is a full-dose RECONSTRUCTION, which is itself noisy; the ideal output is
    the clean image, whose high-frequency power in a flat region is near zero.
    So a power ratio of 1 means "as noisy as the full-dose scan", not "best".
    A method at 0.35 has removed more noise than one at 1.0 and may be better
    or worse - the pixel metrics decide that, not this one.

    What the CT literature actually objects to in over-smoothed reconstructions
    is a SHAPE change: power shifting to low frequency, which is what makes
    iterative and deep-denoised images look blotchy or waxy, and what destroys
    low-contrast detectability even when the noise magnitude looks acceptable.
    Shape is scale-free, so it is measured after normalising each spectrum to
    unit power over the band. That number has an unambiguous best value of
    zero and cannot be improved either by denoising harder or by denoising
    less - only by preserving the character of the noise.

WHAT IT IS COMPARED AGAINST
    The reference here is the full-dose reconstruction, which is not noiseless
    - it has its own texture, and matching that texture is the actual goal. So
    both images are high-pass filtered inside the same uniform patches and
    their radial spectra compared directly, without normalising away the total
    power. Normalising would discard the over-smoothing signal, which is
    precisely the thing being measured.

CHOOSING THE PATCHES
    Uniform means uniform in the REFERENCE, never in the method's output: a
    model that smooths aggressively would otherwise get to nominate its own
    flattest regions and be measured on those. Patches are ranked by the local
    standard deviation of a smoothed reference, so genuine anatomy is excluded
    while its noise is not.
"""
from __future__ import annotations

import numpy as np


def uniform_patches(reference_hu: np.ndarray, size: int = 64, n: int = 6,
                    stride: int = 16, body_threshold: float = -300.0,
                    smooth_sigma: float = 2.0) -> list[tuple[int, int]]:
    """Top-left corners of the ``n`` flattest in-body patches of the reference."""
    from scipy.ndimage import gaussian_filter

    ref = np.asarray(reference_hu, dtype=np.float32)
    smooth = gaussian_filter(ref, smooth_sigma)
    h, w = ref.shape
    cands = []
    for i in range(0, h - size + 1, stride):
        for j in range(0, w - size + 1, stride):
            block = smooth[i:i + size, j:j + size]
            if block.min() < body_threshold:      # touches air or lung edge
                continue
            cands.append((float(block.std()), i, j))
    cands.sort()
    chosen: list[tuple[int, int]] = []
    for _, i, j in cands:
        if all(abs(i - a) >= size or abs(j - b) >= size for a, b in chosen):
            chosen.append((i, j))
        if len(chosen) >= n:
            break
    return chosen


def radial_nps(patch: np.ndarray, smooth_sigma: float = 3.0
               ) -> tuple[np.ndarray, np.ndarray]:
    """Radially averaged power spectrum of the patch's high-frequency content.

    A 2-D Hann window is applied before the transform. Without it the patch
    edges act as a step and leak broadband power into every radial bin, which
    would swamp the difference the metric is trying to see.
    """
    from scipy.ndimage import gaussian_filter

    p = np.asarray(patch, dtype=np.float64)
    p = p - gaussian_filter(p, smooth_sigma)
    n = p.shape[0]
    win = np.hanning(n)
    p = p * win[:, None] * win[None, :]
    # Hann reduces the variance by a known factor; dividing it out keeps the
    # spectrum's absolute level comparable to an unwindowed one.
    p = p / np.sqrt((win ** 2).mean() ** 2)

    f = np.fft.fftshift(np.abs(np.fft.fft2(p)) ** 2) / (n * n)
    ky, kx = np.mgrid[:n, :n] - n // 2
    r = np.hypot(ky, kx).astype(int)
    nbins = n // 2
    prof = np.bincount(r.ravel(), f.ravel(), minlength=nbins + 1)[:nbins]
    cnt = np.bincount(r.ravel(), minlength=nbins + 1)[:nbins].clip(1)
    return np.arange(nbins) / n, prof / cnt


def nps_distance(method_hu: np.ndarray, reference_hu: np.ndarray,
                 size: int = 64, n_patches: int = 6,
                 f_lo: float = 0.05, f_hi: float = 0.45) -> dict:
    """Compare the two images' noise texture in the reference's flat regions.

    Returns the mean absolute log-ratio of the radial spectra over the usable
    band, and the total-power ratio, which is below one exactly when the method
    has removed texture the reference still has.
    """
    corners = uniform_patches(reference_hu, size=size, n=n_patches)
    if not corners:
        return {"nps_log_distance": float("nan"),
                "nps_power_ratio": float("nan"), "n_patches": 0}

    ratios, powers, shapes = [], [], []
    for i, j in corners:
        f, a = radial_nps(method_hu[i:i + size, j:j + size])
        _, b = radial_nps(reference_hu[i:i + size, j:j + size])
        band = (f >= f_lo) & (f <= f_hi)
        if not band.any():
            continue
        aa = np.maximum(a[band], 1e-12)
        bb = np.maximum(b[band], 1e-12)
        ratios.append(np.abs(np.log(aa / bb)).mean())
        powers.append(aa.sum() / bb.sum())
        # Shape: the same comparison after dividing out total power, so a
        # method cannot score well or badly merely by denoising more.
        shapes.append(np.abs(np.log((aa / aa.sum()) / (bb / bb.sum()))).mean())
    if not ratios:
        return {"nps_log_distance": float("nan"),
                "nps_power_ratio": float("nan"),
                "nps_shape_distance": float("nan"), "n_patches": 0}
    return {"nps_log_distance": float(np.mean(ratios)),
            "nps_power_ratio": float(np.mean(powers)),
            "nps_shape_distance": float(np.mean(shapes)),
            "n_patches": len(ratios)}
