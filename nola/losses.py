"""The NoLA adaptation objective.

Protocol v2 section 8. Four terms, and each one exists because removing it
breaks something specific. The ablation table is the test of that claim, so
the implementation keeps them strictly separable.

Write the network's residual as r = y - f(y), and the estimated law as
Var(y) = a exp(c_M y) + b exp(2 c_M y).

1. SIGNAL-CONDITIONAL MOMENT MATCHING
   If f has removed exactly the noise, then within any narrow band of signal
   level the residual's variance must equal what the law predicts there. Not
   the global variance: a global match is satisfied by a model that
   over-smooths the dense rays and under-smooths the air, which is precisely
   the failure mode of naive rescaling. Conditioning on the signal is what
   makes the constraint bite across the whole four-order-of-magnitude range,
   and it is why the term needs the two-parameter law rather than a single
   scale.

   The bands are quantile bins of the network's own estimate f(y), so they
   adapt to whatever the sinogram contains instead of assuming a range.

2. WHITENESS
   Matching variance alone is satisfied by a model that leaves structured
   error behind as long as the error has the right magnitude, which is exactly
   what a denoiser transferred to the wrong system does: it smooths along the
   detector and leaves ripples. Standardising the residual by the predicted
   sigma and penalising its autocorrelation at short lags, along BOTH the
   detector and the view axis, removes that solution. Lags 1 to 5 cover the
   correlation length a convolutional denoiser can create.

3. ORTHOGONALITY - OFF BY DEFAULT, AND NOT AN IDENTIFICATION CONDITION
   This term was originally justified as excluding solutions that eat signal.
   That justification is wrong, and the counterexample is elementary. For
   x ~ N(0, sx^2), n ~ N(0, sn^2), y = x + n, the MMSE estimator is xhat = a y
   with 0 < a < 1, so the residual is r = (1 - a) y and

       Cov(r, xhat) = a (1 - a) Var(y) > 0.

   The ideal estimator does NOT have an orthogonal residual, so driving
   cov(r, f(y)) to zero pushes the model AWAY from MMSE. The measurements
   agree: removing this term gives the best law match at both doses tested
   (0.247 against 0.358 at 25 percent, 0.082 against 0.100 at 10 percent).

   It is kept, defaulted OFF, as an optional empirical regulariser against
   gross signal leakage - never as a guarantee.

4. L2-SP ANCHOR
   Adaptation runs on unlabelled data with no fidelity term, so nothing except
   the anchor stops the weights drifting to a degenerate solution that
   satisfies the statistics and has forgotten how to denoise. Anchoring to the
   source weights rather than to zero keeps the inductive bias that Stage 0
   paid for.

EVERY TERM READS ONLY y, f(y) AND THE ESTIMATED LAW. None reads a clean
sinogram or a stored sigma. That is the leakage rule in executable form.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .units import C_M


def predicted_variance(y_hat: torch.Tensor, a: float, b: float,
                       clamp_to: torch.Tensor | None = None) -> torch.Tensor:
    """Law variance at the network's own signal estimate.

    THE ARGUMENT IS DETACHED, AND THAT CLOSES A DEGENERATE SOLUTION
    Every loss below compares something about the residual against this
    prediction, and the prediction is a function of the model's own output.
    Left attached, the model has two ways to satisfy a moment constraint: fix
    the residual, which is the intended one, or raise y_hat until the
    PREDICTED variance moves to meet the residual it already has. The second
    costs nothing in the moment term and is available at every signal level,
    since the law is exponential in y_hat and a 1 percent change in a dense
    ray moves the prediction by half.

    Detaching makes the variance profile what it should be - a property of the
    signal, treated as data - so the only way to reduce these losses is to
    change the residual. The orthogonality guard and the anchor would
    eventually punish the degenerate route, but relying on them to clean up
    after a gradient that points the wrong way is not the same as not
    providing the gradient.

    THE PLUG-IN SIGNAL IS CLIPPED TO THE MEASUREMENT'S RANGE
    Detaching removes the gradient route but not a feedback route through the
    VALUE. The law is exponential in y_hat, so if the estimate drifts upward
    the demanded variance grows exponentially, which the model can only satisfy
    by making its residual larger, which it does by moving y_hat further from
    y - and upward drift is one of the ways to do that. The loop is slow at
    the Mayo operating points and violent where the electronic term dominates:
    the controlled sweep diverges at I0 = 7.5e3 with sigma_e = 20, where that
    term is 21x the Poisson term in the dense tail.

    Clipping the plug-in signal to the range of the measurement closes it. A
    signal estimate outside the range of the data it was estimated from is not
    physical, and refusing to extrapolate the law past the measured range costs
    nothing where the model is behaving.

    The exponent is clamped exactly as nola.units does. Under bf16 autocast
    exp(6) is fine but exp of a stray dense ray is not, and a single inf here
    poisons the whole batch gradient.
    """
    s = y_hat.detach().float()
    if clamp_to is not None:
        ref = clamp_to.detach().float()
        s = torch.clamp(s, float(ref.min()), float(ref.max()))
    e = torch.clamp(s * C_M, max=50.0)
    ex = torch.exp(e)
    return a * ex + b * ex * ex


def conditional_moment_loss(residual: torch.Tensor, y_hat: torch.Tensor,
                            a: float, b: float, n_bins: int = 20,
                            eps: float = 1e-12,
                            clamp_to: torch.Tensor | None = None,
                            min_count: int = 32) -> torch.Tensor:
    """Match the residual's first two conditional moments to the law.

    Per signal band k, with mean m_k and variance s2_k of the residual and
    predicted variance v_k:

        m_k^2 / v_k        the residual should be centred, and a band with a
                           non-zero mean means a systematic bias the variance
                           term alone cannot see;
        (s2_k - v_k)^2 / v_k^2    relative variance mismatch.

    Both are normalised by the predicted variance, which is what makes the
    four orders of magnitude between an air ray and a ray through the spine
    contribute comparably. The earlier form used a log-variance difference and
    omitted the mean entirely.

    WHAT THIS TERM CANNOT DO
    Matching these moments does not identify the ideal denoiser. Writing
    r = y - f(y) = n + (x - f(y)), the residual carries the estimation error
    as well as the noise, so even the MMSE estimator's residual is not
    distributed as the acquisition noise. The term is an adaptation signal
    that biases the model towards residual statistics compatible with the
    target acquisition, and it is the source anchor that decides which of the
    many compatible solutions is reached.
    """
    r2 = residual.float().reshape(-1) ** 2
    s = y_hat.detach().float().reshape(-1)
    v = predicted_variance(s, a, b, clamp_to).reshape(-1)

    # Quantile bands of the signal estimate. The bands define WHERE the
    # constraint is evaluated; letting the model move the boundaries is
    # another way to satisfy it without changing the residual.
    with torch.no_grad():
        q = torch.quantile(
            s, torch.linspace(0, 1, n_bins + 1, device=s.device)[1:-1])
        idx = torch.bucketize(s, q)

    raw = torch.bincount(idx, minlength=n_bins)
    counts = raw.clamp_min(1).to(r2.dtype)
    r1 = residual.float().reshape(-1)
    mean_k = torch.zeros(n_bins, device=s.device).index_add_(0, idx, r1) / counts
    m2_k = torch.zeros(n_bins, device=s.device).index_add_(0, idx, r2) / counts
    pred = torch.zeros(n_bins, device=s.device).index_add_(0, idx, v) / counts
    var_k = (m2_k - mean_k ** 2).clamp_min(0.0)

    # Bands with too few samples give unstable moments and are dropped rather
    # than allowed to dominate a mean over bands.
    used = raw >= min_count
    if not bool(used.any()):
        return s.new_zeros(())
    v_k = pred[used].clamp_min(eps)
    bias = (mean_k[used] ** 2) / v_k
    spread = ((var_k[used] - v_k) ** 2) / (v_k ** 2)
    return (bias + spread).mean()


def whiteness_loss(residual: torch.Tensor, y_hat: torch.Tensor,
                   a: float, b: float, lags: tuple[int, ...] = (1, 2, 3, 4, 5),
                   eps: float = 1e-12,
                   clamp_to: torch.Tensor | None = None,
                   axes: str = "both") -> torch.Tensor:
    """Penalise autocorrelation of the standardised residual on both axes.

    Standardising first matters. The raw residual is heteroscedastic by four
    orders of magnitude, so its unnormalised autocorrelation is dominated by
    whether neighbouring rays happen to be equally noisy, which is a property
    of the anatomy and not of the model.
    """
    z = residual.float() / torch.sqrt(
        predicted_variance(y_hat, a, b, clamp_to).clamp_min(eps))
    z = z - z.mean(dim=(-2, -1), keepdim=True)
    denom = (z * z).mean(dim=(-2, -1), keepdim=True).clamp_min(eps)

    # AXES IS A KNOB BECAUSE THE VIEW AXIS IS UNDER SUSPICION
    # At 10 percent dose the adapted model produces radial streaks that no
    # other method shows, and streaks are what view-correlated sinogram
    # structure becomes after back-projection. Penalising view-axis
    # autocorrelation asks the model to move view-correlated content OUT of
    # the residual, which puts it into the output. Being able to switch the
    # axes off separately turns that from a story into a test.
    use_det = axes in ("both", "detector")
    use_view = axes in ("both", "view")
    total = z.new_zeros(())
    n_terms = 0
    for lag in lags:
        if use_det and z.shape[-1] > lag:
            c = (z[..., :-lag] * z[..., lag:]).mean(dim=(-2, -1), keepdim=True)
            total = total + ((c / denom) ** 2).mean()
            n_terms += 1
        if use_view and z.shape[-2] > lag:
            c = (z[..., :-lag, :] * z[..., lag:, :]).mean(dim=(-2, -1),
                                                          keepdim=True)
            total = total + ((c / denom) ** 2).mean()
            n_terms += 1
    return total / max(n_terms, 1)


def orthogonality_loss(residual: torch.Tensor, y_hat: torch.Tensor,
                       eps: float = 1e-12) -> torch.Tensor:
    """Squared correlation between the residual and the estimate, per sample."""
    r = residual.float()
    s = y_hat.float()
    r = r - r.mean(dim=(-2, -1), keepdim=True)
    s = s - s.mean(dim=(-2, -1), keepdim=True)
    cov = (r * s).mean(dim=(-2, -1))
    norm = (r.pow(2).mean(dim=(-2, -1)) * s.pow(2).mean(dim=(-2, -1))).clamp_min(eps)
    return (cov.pow(2) / norm).mean()


class L2SPAnchor:
    """Squared distance to the source weights.

    Buffers and BatchNorm statistics are excluded: they are not parameters
    being regularised, and anchoring running statistics would fight the very
    adaptation AdaBN performs, making the ablation against AdaBN incoherent.
    """

    def __init__(self, model: nn.Module):
        self.reference = {k: v.detach().clone()
                          for k, v in model.named_parameters()}
        self._ref_sq = sum(float(v.pow(2).sum())
                           for v in self.reference.values()) or 1.0

    def __call__(self, model: nn.Module) -> torch.Tensor:
        """Relative L2-SP: ||theta - theta_src||^2 / ||theta_src||^2.

        Normalising by the source norm makes the weight independent of model
        size and of the scale the source weights happen to have, so the same
        lambda means the same thing across backbones.
        """
        total = None
        for k, v in model.named_parameters():
            d = (v - self.reference[k]).pow(2).sum()
            total = d if total is None else total + d
        return total / self._ref_sq


@dataclass
class NoLAWeights:
    """Default weights after review.

    ``orthogonality`` defaults to 0: the term is not an identification
    condition and the ideal estimator violates it (see orthogonality_loss).
    ``anchor`` is NOT tuned to 0. It was, briefly, by minimising law match -
    which is circular, since law match is the objective and the anchor is the
    regulariser constraining it. The anchor is the structural prior that picks
    which law-compatible solution is reached, and it stays on.
    """
    moment: float = 1.0
    whiteness: float = 1.0
    orthogonality: float = 0.0
    # The anchor is now the RELATIVE L2-SP, smaller than the raw sum by
    # ||theta_src||^2 ~ 1.8e4 for this backbone, so the weight is rescaled by
    # the same factor to keep the effective strength that was actually
    # measured. It is a default, not a tuned value; tune it on the adaptation
    # patients, never on test.
    anchor: float = 10.0
    sure: float = 0.0


def nola_loss(y: torch.Tensor, y_hat: torch.Tensor, a: float, b: float,
              model: nn.Module, anchor: L2SPAnchor, w: NoLAWeights,
              n_bins: int = 20, lags=(1, 2, 3, 4, 5),
              whiteness_axes: str = "both"):
    """The full objective. Returns ``(loss, parts)`` with parts detached."""
    residual = y.float() - y_hat.float()
    # The measurement bounds the plug-in signal: see predicted_variance.
    m = conditional_moment_loss(residual, y_hat, a, b, n_bins, clamp_to=y)
    wh = whiteness_loss(residual, y_hat, a, b, lags, clamp_to=y,
                        axes=whiteness_axes)
    o = orthogonality_loss(residual, y_hat)
    an = anchor(model)
    loss = w.moment * m + w.whiteness * wh + w.orthogonality * o + w.anchor * an
    parts = {"moment": float(m), "whiteness": float(wh),
             "orthogonality": float(o), "anchor": float(an)}
    if w.sure:
        # SURE is on a different scale from the other terms - it is a variance
        # in harmonised units, of order 1e-8 - so its weight carries that
        # scale and is not comparable with the others.
        su = sure_loss(y, y_hat, model, a, b, clamp_to=y)
        loss = loss + w.sure * su
        parts["sure"] = float(su)
    parts["total"] = float(loss)
    return loss, parts


def sure_loss(y: torch.Tensor, y_hat: torch.Tensor, model: nn.Module,
              a: float, b: float, eps_scale: float = 0.02,
              clamp_to: torch.Tensor | None = None) -> torch.Tensor:
    """Stein's unbiased risk estimate of the MSE to the CLEAN signal.

    WHY THIS TERM EXISTS
    Moment matching and whiteness constrain the residual's STATISTICS. They do
    not constrain its ALIGNMENT with the noise that is actually present, and
    that gap is measurable: at 10 percent dose the adapted model reaches a law
    match of 0.100 and a whiteness of 0.013 - it satisfies the objective almost
    perfectly - while producing an image 1.4 dB WORSE than no adaptation at
    all. Working back from the image noise power, its residual is 93.5 percent
    correlated with the true noise against the source model's 99.6 percent.
    Both residuals are the right size and both are white; one of them is the
    noise and the other merely looks like it.

    SURE closes exactly that gap. For y = s + n with independent zero-mean
    noise of known per-ray variance,

        E ||f(y) - s||^2  =  E ||y - f(y)||^2 - sum(var) + 2 sum(var * df/dy)

    every term of which is computable without ever seeing s. It is an unbiased
    estimate of the quantity that actually matters, and it needs precisely one
    thing NoLA already estimates from unlabelled data: the noise variance. The
    law estimate stops being only a target for moment matching and becomes the
    ingredient that makes label-free risk minimisation possible.

    THE DIVERGENCE IS ESTIMATED BY MONTE CARLO
    The exact divergence of a convolutional network is not available, so the
    standard single-probe estimator is used:

        sum(var * df/dy)  ~=  (1/eps) < var * b , f(y + eps b) - f(y) >,
        b ~ N(0, I)

    which costs one extra forward pass. ``eps`` is scaled by the data's own
    spread so the finite difference is neither swamped by float error nor so
    large that it stops approximating a derivative.

    Note the divergence term is what stops the trivial minimiser. Without it
    the expression is minimised by f(y) = y, which has zero data term; the
    divergence of the identity is maximal and penalises it exactly.
    """
    var = predicted_variance(y, a, b, clamp_to)
    data = ((y.float() - y_hat.float()) ** 2).mean()

    # THE PROBE PASS MUST BE FULL PRECISION AND THE STEP MUST BE VISIBLE
    # A finite difference needs the perturbation to survive the arithmetic.
    # Measured: with eps = 1e-3 * std the step is ~2e-5 against y ~ 2e-2, a
    # relative change of 1e-3, which is below bf16's ~4e-3 mantissa
    # resolution - so f(y + eps b) - f(y) was rounding noise and the estimated
    # divergence was meaningless. The resulting objective was minimised by the
    # identity, and adaptation returned an essentially undenoised image (28.5
    # dB against 39.0 for no adaptation at all).
    #
    # The step is now scaled to the NOISE, which is the scale SURE is about,
    # and the probe pass runs outside autocast in float32.
    #
    # eps must be small enough to still BE a derivative. At eps = 0.5 sigma the
    # finite difference stopped approximating one, the divergence term became
    # gameable, and the objective ran to -784 before exploding: SURE is bounded
    # below only through the true divergence, and a badly estimated divergence
    # can be driven arbitrarily negative. 0.02 sigma sits about three orders of
    # magnitude above float32 resolution here and well inside the linear
    # regime.
    probe = torch.randn_like(y)
    eps = (eps_scale * torch.sqrt(var.mean())).clamp_min(1e-9)
    with torch.autocast("cuda", enabled=False):
        perturbed = model((y.float() + eps * probe).float())
    div = ((var * probe) * (perturbed.float() - y_hat.float())).mean() / eps

    # Returned RELATIVE to the noise variance. Absolutely, this is of order
    # 1e-8 in harmonised units while every other term is of order one, so a
    # raw weight would have to be 1e8 and would mean nothing to a reader.
    # Dividing by the mean predicted variance makes it "risk in units of the
    # noise": 0 for a perfect denoiser, 1 for the identity, and directly
    # comparable across doses, which matters because the whole question is why
    # the objective behaves differently at 25 and 10 percent.
    return (data - var.mean() + 2.0 * div) / var.mean().clamp_min(1e-30)
