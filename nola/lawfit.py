"""Estimate the target noise law from unlabelled sinograms.

Protocol v2 section 7. The whole claim of NoLA rests on this being possible
without a single clean target, so the estimator is built to be checkable rather
than merely plausible: the Mayo pipeline knows the true (I0, sigma_e), and
section 7's validation is an exact comparison, not a proxy.

WHAT MAKES THE TWO PARAMETERS SEPARABLE
    In harmonised units the law is  Var(y) = a exp(c_M y) + b exp(2 c_M y).
    The ratio of the second term to the first is (b/a) exp(c_M y), and
    b/a = sigma_e^2 / I0. At 25 percent dose that ratio is 2.7e-3, so through
    air the electronic term is a thousandth of the Poisson term and utterly
    invisible; through the densest rays, where c_M y reaches about 6, it is
    multiplied by exp(6) ~ 460 and becomes comparable. The dense tail is
    therefore the ONLY place b is identifiable, which has a direct
    consequence: an estimator that bins by quantile spends almost all its bins
    where b cannot be seen. Bins here are uniform in signal level, and the
    weighting is what compensates for the tail being sparse.

THE PSEUDO-RESIDUAL RUNS ALONG VIEWS, NOT ALONG THE DETECTOR
    Protocol section 7 specified a high-pass difference along the detector
    axis. Measured, that estimator is biased: it returns a 14 percent too
    large, which pushes I0 12 percent low and, because the fit has two
    parameters that trade off, drags sigma_e 22 percent low with it. The cause
    is signal leakage. A second difference removes a locally linear signal and
    leaves its curvature, and a sinogram is strongly curved ACROSS THE
    DETECTOR: fitting the measured bias back, the residual detector curvature
    is about 0.9 sigma, so it contributes a third of the apparent noise.

    Across VIEWS the same sinogram is almost flat. At 1152 views over 2 pi,
    adjacent views are 0.3125 degrees apart, and the second difference of a
    sinogram sinusoid over that step is about 3e-5 of its amplitude - roughly
    one percent of sigma, contributing 1e-4 of the variance. Four orders of
    magnitude less leakage, for free, by differencing the other way:

        t[v, i] = (y[v-1, i] - 2 y[v, i] + y[v+1, i]) / sqrt(6)

    ``axis="detector"`` keeps the original for comparison; the difference
    between the two is a one-line result worth reporting.

    Independence across views is a property of the acquisition: each view is a
    separate photon measurement, and the simulator draws every element of the
    sinogram independently. On real projections, detector afterglow and
    tube-current modulation correlate adjacent views slightly, which belongs
    in the limitations of the journal extension.

WHY THE SIGNAL ESTIMATE USES DETECTOR NEIGHBOURS, AND WHICH ONES
    Binning needs the signal level at (v, i), and taking it from the same
    measurement would correlate the bin assignment with the noise being
    measured. The residual now consumes bin i across three views, so the
    signal estimate is built from the OTHER detector bins of the same view:

        s[v, i] = (-y[v, i-2] + 4 y[v, i-1] + 4 y[v, i+1] - y[v, i+2]) / 6

    Disjoint detector bins, hence exactly independent. The coefficients are
    the cubic interpolant at i, not the plain average of the two neighbours:
    the average would inherit half the detector curvature that was just
    identified as the problem, biasing the regressor by about 0.5 sigma, while
    the cubic form annihilates any signal up to third order. It pays for that
    with a higher variance, 0.94 v against 0.5 v, which costs nothing here -
    regressor noise enters only through a Jensen term of order c_M^2 v / 2,
    which is 2e-4.

ROBUST SCALE, NOT SAMPLE VARIANCE
    Real sinograms are not locally linear everywhere. Edges, metal and the
    patient boundary put genuine curvature into d, and a sample variance would
    read that as noise and inflate a and b. The median absolute deviation
    ignores a minority of large values by construction, which is exactly the
    contamination model here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .units import C_M, NoiseLaw

MAD_TO_SIGMA = 1.4826


def pseudo_residuals(y: np.ndarray, axis: str = "view",
                     signal_views: int = 9
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(t, s)``: a high-pass residual and an independent signal
    estimate at the same locations.

    ``axis="view"`` differences across views and estimates the signal from
    detector neighbours. ``axis="detector"`` is the protocol's original,
    kept so the bias can be quantified rather than asserted.
    """
    from scipy.ndimage import uniform_filter1d

    y = np.asarray(y, dtype=np.float64)
    if axis == "view":
        # Residual: second difference across views, at detector bins 2..D-3.
        t = (y[:-2, 2:-2] - 2.0 * y[1:-1, 2:-2] + y[2:, 2:-2]) / np.sqrt(6.0)
        # Signal: cubic interpolant from detector bins i-2, i-1, i+1, i+2,
        # averaged over a window of views. The residual at (v, i) touches bin i
        # only, and this touches only bins i-2, i-1, i+1, i+2 at ANY view, so
        # widening the window across views keeps the two exactly independent
        # while cutting the regressor's own variance by the window length.
        # That matters more than it looks: the regressor enters through
        # exp(c_M s), whose expectation carries a Jensen factor
        # exp(c_M^2 Var(s) / 2), and at 5 percent dose in the dense tail
        # c_M sigma reaches 0.47, which inflates exp(2 c_M s) by half again.
        wide = (-y[:, :-4] + 4.0 * y[:, 1:-3]
                + 4.0 * y[:, 3:-1] - y[:, 4:]) / 6.0
        if signal_views > 1:
            wide = uniform_filter1d(wide, size=signal_views, axis=0,
                                    mode="nearest")
        return t, wide[1:-1]
    if axis == "detector":
        t = (y[1:-1, :-2] - 2.0 * y[1:-1, 1:-1] + y[1:-1, 2:]) / np.sqrt(6.0)
        sd = 0.5 * (y[:-2, 1:-1] + y[2:, 1:-1])
        return t, sd
    raise ValueError(f"axis must be 'view' or 'detector', not {axis!r}")


@dataclass
class LawFit:
    law: NoiseLaw
    bin_centres: np.ndarray
    bin_variance: np.ndarray
    bin_counts: np.ndarray
    fitted_variance: np.ndarray
    nonneg_clipped: str = ""
    n_slices: int = 0
    extra: dict = field(default_factory=dict)

    def relative_error(self, truth: NoiseLaw) -> dict:
        return {"a": abs(self.law.a - truth.a) / truth.a,
                "b": (abs(self.law.b - truth.b) / truth.b
                      if truth.b > 0 else float("nan")),
                "i0": abs(self.law.i0 - truth.i0) / truth.i0,
                "electronic_sigma": (
                    abs(self.law.electronic_sigma - truth.electronic_sigma)
                    / truth.electronic_sigma
                    if truth.electronic_sigma > 0 else float("nan"))}


def solve_ab(v: np.ndarray, n: np.ndarray, e1: np.ndarray, e2: np.ndarray,
             kappa: float, iters: int = 3, weight_cap: float = 20.0
             ) -> tuple[float, float, np.ndarray, str]:
    """Fit ``v ~= a e1 + b e2`` in log space. Returns ``(a, b, fitted, clipped)``.

    THE FIT IS IN LOG SPACE, AND THAT IS NOT A COSMETIC CHOICE
    A linear weighted least squares with the natural weight n / v^2 gives the
    air bins, where v is smallest and n largest, a weight larger than the
    dense-ray bins by about eight orders of magnitude. The consequence was
    visible in the numbers: a came out determined almost entirely by air, and
    b, which is identifiable ONLY in the dense tail, collapsed towards zero -
    measured 91 percent low at 5 percent dose before this changed. Variance
    spans four orders of magnitude here, so the meaningful error is relative,
    which is a residual in log space, and a robust scale estimate has
    Var(log v_hat) proportional to 1/n independent of v, making n the correct
    log-space weight.

    THE PARAMETERS ARE OPTIMISED AS LOGARITHMS
    a and b are of order 1e-9 and 1e-11 and are bounded below by zero, being
    variances. A bounded solve on the raw values fails in a way that is easy to
    miss: the linear starting point is dominated by the dense bins and returns
    a negative a, clipping puts the optimiser exactly on the boundary, and it
    then returns a point whose weighted cost is three orders of magnitude worse
    than simply plugging in the true parameters. Optimising log a and log b
    makes positivity structural, puts both parameters on a scale of order one,
    and lets the Jacobian be written down instead of differenced:

        r_k     = sqrt(w_k) (log(a e1_k + b e2_k) - log v_k)
        dr/dloga = sqrt(w_k) a e1_k / (a e1_k + b e2_k)
        dr/dlogb = sqrt(w_k) b e2_k / (a e1_k + b e2_k)

    THE WEIGHT IS CAPPED
    Weighting strictly by n lets the single lowest-signal bin, which holds most
    of the air and carries five times the count of any other, decide a on its
    own. That bin is also the one whose measurement is least trustworthy: it
    spans the widest spread of variances internally, and a robust scale
    estimate of a heterogeneous mixture sits nearer the mixture's median than
    its mean, which reads about eight percent low. Capping the weight at
    ``weight_cap`` times the median keeps the statistical logic and removes the
    single point of failure.

    JENSEN CORRECTION. The regressors are means of exp(c_M s) over each bin,
    but s is measured, not true. For s = s_true + e with Var(e) = kappa v,

        E[exp(c_M s)]   = exp(c_M s_true)   exp(c_M^2 kappa v / 2)
        E[exp(2 c_M s)] = exp(2 c_M s_true) exp(2 c_M^2 kappa v)

    so both regressors are inflated, the second four times harder in the
    exponent, exactly where b lives. Dividing it out needs v, which is what is
    being fitted, so the fit is iterated; two passes suffice because the
    correction is a few percent. ``kappa = 0`` when the signal is known
    exactly, which is the case for the empirical reference fit.
    """
    from scipy.optimize import least_squares

    w = np.minimum(n, weight_cap * np.median(n))
    sw = np.sqrt(w)
    logv = np.log(v)
    a = b = 0.0
    for _ in range(iters):
        c1 = np.exp(np.clip(C_M ** 2 * kappa * v / 2.0, None, 50.0))
        c2 = np.exp(np.clip(2.0 * C_M ** 2 * kappa * v, None, 50.0))
        E1, E2 = e1 / c1, e2 / c2

        def model(theta):
            return np.exp(theta[0]) * E1 + np.exp(theta[1]) * E2

        def residual(theta):
            return sw * (np.log(np.maximum(model(theta), 1e-300)) - logv)

        def jac(theta):
            m = np.maximum(model(theta), 1e-300)
            return np.stack([sw * np.exp(theta[0]) * E1 / m,
                             sw * np.exp(theta[1]) * E2 / m], axis=1)

        # Start from the two ends, where each term is on its own: the quietest
        # bin is nearly pure a, and whatever the tail has above that is b.
        a0 = max(v[0] / max(E1[0], 1e-300), 1e-30)
        b0 = max((v[-1] - a0 * E1[-1]) / max(E2[-1], 1e-300), a0 * 1e-8)
        sol = least_squares(residual, np.log([a0, b0]), jac=jac, method="lm",
                            xtol=1e-14, ftol=1e-14, gtol=1e-14)
        a, b = float(np.exp(sol.x[0])), float(np.exp(sol.x[1]))

    c1 = np.exp(np.clip(C_M ** 2 * kappa * v / 2.0, None, 50.0))
    c2 = np.exp(np.clip(2.0 * C_M ** 2 * kappa * v, None, 50.0))
    fitted = a * (e1 / c1) + b * (e2 / c2)
    # A vanishing component is reported rather than silently returned as a
    # tiny positive number: it means that part of the law is unresolved.
    clipped = "b" if b < a * 1e-6 else ""
    return a, b, fitted, clipped


def fit_law(sinograms, n_bins: int = 48, min_count: int = 200,
            axis: str = "view", signal_views: int = 9,
            lo_pct: float = 0.5, hi_pct: float = 99.9,
            max_samples_per_bin: int = 400_000) -> LawFit:
    """Fit (a, b) from a sequence of harmonised, unlabelled sinograms.

    ``sinograms`` is any iterable of 2-D arrays in harmonised units. Nothing
    else is read: no clean signal, no stored sigma, no dose label.
    """
    sinos = [np.asarray(y, dtype=np.float64) for y in sinograms]
    if not sinos:
        raise ValueError("no sinograms given")

    # Bin edges from a pooled sample of the signal estimate, uniform in signal
    # level so the dense tail keeps its own bins.
    probe = np.concatenate([pseudo_residuals(y, axis, signal_views)[1].ravel()[::37]
                            for y in sinos[:8]])
    lo, hi = np.percentile(probe, [lo_pct, hi_pct])
    edges = np.linspace(lo, hi, n_bins + 1)

    per_bin_d: list[list[np.ndarray]] = [[] for _ in range(n_bins)]
    per_bin_e1 = np.zeros(n_bins)
    per_bin_e2 = np.zeros(n_bins)
    per_bin_n = np.zeros(n_bins, dtype=np.int64)

    for y in sinos:
        d, s = pseudo_residuals(y, axis, signal_views)
        sr, dr = s.ravel(), d.ravel()
        k = np.digitize(sr, edges) - 1
        ok = (k >= 0) & (k < n_bins)
        k, sr, dr = k[ok], sr[ok], dr[ok]
        order = np.argsort(k, kind="stable")
        k, sr, dr = k[order], sr[order], dr[order]
        bounds = np.searchsorted(k, np.arange(n_bins + 1))
        e = np.clip(C_M * sr, None, 50.0)
        ex = np.exp(e)
        for j in range(n_bins):
            a0, a1 = bounds[j], bounds[j + 1]
            if a1 <= a0:
                continue
            chunk = dr[a0:a1]
            if per_bin_n[j] < max_samples_per_bin:
                per_bin_d[j].append(chunk)
            per_bin_n[j] += chunk.size
            per_bin_e1[j] += float(ex[a0:a1].sum())
            per_bin_e2[j] += float((ex[a0:a1] ** 2).sum())

    centres, variances, counts, e1, e2 = [], [], [], [], []
    for j in range(n_bins):
        if per_bin_n[j] < min_count or not per_bin_d[j]:
            continue
        pooled = np.concatenate(per_bin_d[j])
        mad = np.median(np.abs(pooled - np.median(pooled)))
        v = (MAD_TO_SIGMA * mad) ** 2
        if not np.isfinite(v) or v <= 0:
            continue
        centres.append(0.5 * (edges[j] + edges[j + 1]))
        variances.append(v)
        counts.append(int(per_bin_n[j]))
        e1.append(per_bin_e1[j] / per_bin_n[j])
        e2.append(per_bin_e2[j] / per_bin_n[j])

    centres = np.asarray(centres)
    v = np.asarray(variances)
    n = np.asarray(counts, dtype=np.float64)
    X = np.stack([np.asarray(e1), np.asarray(e2)], axis=1)

    if len(v) < 3:
        raise RuntimeError(f"only {len(v)} usable bins; check the input units")

    kappa = (0.9444 / max(1, signal_views)) if axis == "view" else 0.5
    a, b, fitted, clipped = solve_ab(v, n, X[:, 0], X[:, 1], kappa)

    law = NoiseLaw(a=a, b=b)
    return LawFit(law=law, bin_centres=centres, bin_variance=v,
                  bin_counts=n.astype(np.int64),
                  fitted_variance=fitted,
                  nonneg_clipped=clipped, n_slices=len(sinos),
                  extra={"edges": edges.tolist(), "n_bins_used": int(len(v)),
                     "axis": axis, "signal_views": signal_views,
                     "kappa": kappa})


def fit_empirical(clean, noisy, n_bins: int = 48, min_count: int = 200,
                  lo_pct: float = 0.5, hi_pct: float = 99.9) -> LawFit:
    """The conditional noise variance actually present in the data.

    VALIDATION ONLY. This reads the clean sinogram and is therefore forbidden
    in every estimation and adaptation path; it exists to answer one question
    that the delta-method law cannot answer about itself.

    ``NoiseLaw.mayo_truth`` is not, strictly, the truth. It is the first-order
    delta-method approximation to the variance of -log(N / I0), and the
    simulator used it only to WRITE the sigma arrays; the data itself was drawn
    from the exact Poisson-plus-Gaussian model. Those two differ at low counts,
    where the logarithm's curvature stops being negligible: the next term is of
    order 1/mu relative, and in the dense tail at 5 percent dose mu falls to
    about twenty photons.

    So when the estimator disagrees with mayo_truth, there are two candidate
    culprits and no way to tell them apart from the estimate alone. Binning the
    actual realised noise, noisy minus clean, by the exact clean signal gives
    the conditional variance with no approximation and no estimator in the way.
    If the estimate tracks THIS and mayo_truth does not, the estimator is right
    and the delta method is the thing that was approximate.
    """
    cleans = [np.asarray(c, dtype=np.float64) for c in clean]
    noisies = [np.asarray(x, dtype=np.float64) for x in noisy]

    probe = np.concatenate([c.ravel()[::37] for c in cleans[:8]])
    lo, hi = np.percentile(probe, [lo_pct, hi_pct])
    edges = np.linspace(lo, hi, n_bins + 1)

    per_bin: list[list[np.ndarray]] = [[] for _ in range(n_bins)]
    e1 = np.zeros(n_bins)
    e2 = np.zeros(n_bins)
    cnt = np.zeros(n_bins, dtype=np.int64)

    for c, x in zip(cleans, noisies):
        d = (x - c).ravel()
        sr = c.ravel()
        k = np.digitize(sr, edges) - 1
        ok = (k >= 0) & (k < n_bins)
        k, sr, d = k[ok], sr[ok], d[ok]
        order = np.argsort(k, kind="stable")
        k, sr, d = k[order], sr[order], d[order]
        bounds = np.searchsorted(k, np.arange(n_bins + 1))
        ex = np.exp(np.clip(C_M * sr, None, 50.0))
        for j in range(n_bins):
            a0, a1 = bounds[j], bounds[j + 1]
            if a1 <= a0:
                continue
            per_bin[j].append(d[a0:a1])
            cnt[j] += a1 - a0
            e1[j] += float(ex[a0:a1].sum())
            e2[j] += float((ex[a0:a1] ** 2).sum())

    centres, v, n, r1, r2, msq = [], [], [], [], [], []
    for j in range(n_bins):
        if cnt[j] < min_count or not per_bin[j]:
            continue
        pooled = np.concatenate(per_bin[j])
        mad = np.median(np.abs(pooled - np.median(pooled)))
        var = (MAD_TO_SIGMA * mad) ** 2
        if not np.isfinite(var) or var <= 0:
            continue
        centres.append(0.5 * (edges[j] + edges[j + 1]))
        v.append(var)
        msq.append(float(np.mean(pooled ** 2)))
        n.append(int(cnt[j]))
        r1.append(e1[j] / cnt[j])
        r2.append(e2[j] / cnt[j])

    v = np.asarray(v)
    n = np.asarray(n, dtype=np.float64)
    # kappa = 0: the signal is exact here, so there is no Jensen inflation.
    a, b, fitted, clipped = solve_ab(v, n, np.asarray(r1), np.asarray(r2), 0.0)
    return LawFit(law=NoiseLaw(a=a, b=b), bin_centres=np.asarray(centres),
                  bin_variance=v, bin_counts=n.astype(np.int64),
                  fitted_variance=fitted, nonneg_clipped=clipped,
                  n_slices=len(cleans),
                  extra={"n_bins_used": int(len(v)), "axis": "empirical",
                         "mean_square": msq})
