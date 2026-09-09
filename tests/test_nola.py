"""Tests for the claims the method rests on.

These are not smoke tests. Each one guards a property that was actually broken
at some point during development, and whose breakage did not look like a bug:

  * the unit conversion is exact, so it must round-trip to float precision
  * the source law sits at b = 0 exactly, which is why rescaling cannot reach
    the target -- if b drifts off zero the paper's central argument dissolves
  * the estimator must recover a known law from synthetic data it has never
    been told anything about
  * a zero-initialised head is an exact stationary point of the NoLA objective
    and looks exactly like convergence; adapt() must refuse it
  * the leakage rule must be enforced by the loader, not by good intentions
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from nola.geometry import MU_MAX
from nola.lawfit import fit_law
from nola.losses import NoLAWeights, predicted_variance
from nola.units import C_M, NoiseLaw, dehmonise, harmonise


# ------------------------------------------------------------------ units
def test_cm_is_mu_max_exactly():
    """c_M is a unit conversion, not a fitted constant."""
    assert C_M == pytest.approx(MU_MAX, rel=0, abs=0)
    assert C_M == pytest.approx(81.35858, abs=1e-4)


def test_harmonise_round_trips():
    y = np.linspace(0.0, 5.8, 512).astype(np.float32)
    back = dehmonise(harmonise(y))
    assert np.allclose(back, y, rtol=1e-5, atol=1e-6)


# ------------------------------------------------------------------- laws
def test_source_law_has_b_exactly_zero():
    """The whole argument for identifying (a, b) rests on this.

    The source acquisition is purely Poisson. If b were merely small rather
    than zero, a rescaling of the source law could approach the target and the
    paper's motivation would be quantitative rather than structural.
    """
    assert NoiseLaw.source().b == 0.0


def test_physical_round_trip():
    for i0, se in [(4096.0, 0.0), (37500.0, 10.0), (15000.0, 20.0)]:
        law = NoiseLaw.from_physical(i0, se)
        assert law.i0 == pytest.approx(i0, rel=1e-6)
        assert law.electronic_sigma == pytest.approx(se, rel=1e-6, abs=1e-9)


def test_variance_is_monotone_in_signal():
    law = NoiseLaw.from_physical(37500.0, 10.0)
    y = np.linspace(0.0, 5.8 / C_M, 200)
    v = law.variance(y)
    assert np.all(np.diff(v) > 0)


# -------------------------------------------------------------- estimator
def _synthetic_sinograms(i0, sigma_e, n=6, shape=(720, 128), seed=0):
    """Smooth line integrals plus a real Poisson-Gaussian acquisition.

    The signal is smooth along the view axis, which is the assumption the
    pseudo-residual makes; it spans p in [0, 5], as the real target does,
    because below about p = 2 the two exponential regressors are collinear and
    (a, b) genuinely is not identifiable.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        v = np.linspace(0, np.pi, shape[0])[:, None]
        d = np.linspace(-1, 1, shape[1])[None, :]
        p = 5.0 * np.exp(-4.0 * d ** 2) * (0.6 + 0.4 * np.cos(v + rng.random()))
        p = np.clip(p, 0, None)
        expected = i0 * np.exp(-p)
        counts = rng.poisson(expected).astype(np.float64)
        if sigma_e > 0:
            counts = counts + rng.normal(0.0, sigma_e, counts.shape)
        noisy = -np.log(np.maximum(counts, 1.0) / i0)
        out.append(harmonise(noisy.astype(np.float32)))
    return out


@pytest.mark.parametrize("i0,sigma_e", [(37500.0, 10.0), (15000.0, 10.0)])
def test_estimator_recovers_a_known_law(i0, sigma_e):
    """No clean signal, no stored sigma, no dose label is passed in."""
    fit = fit_law(_synthetic_sinograms(i0, sigma_e), n_bins=32, min_count=100)
    assert abs(fit.law.i0 - i0) / i0 < 0.25
    assert abs(fit.law.electronic_sigma - sigma_e) / sigma_e < 0.40


def test_estimator_finds_no_electronic_noise_when_there_is_none():
    """A purely Poisson acquisition must not invent a b > 0 term."""
    fit = fit_law(_synthetic_sinograms(4096.0, 0.0), n_bins=32, min_count=100)
    true = NoiseLaw.from_physical(4096.0, 0.0)
    assert fit.law.b < max(true.a, 1e-30) * 1e-2


# ----------------------------------------------------------------- losses
def test_predicted_variance_is_clipped_to_the_measurement():
    """Without the clip the model can lower the loss by inflating its output.

    exp(c_M * y_hat) feeds the predicted variance, so an unclipped plug-in
    signal gives the network a degenerate fixed point: push y_hat up, make the
    predicted variance enormous, and the relative variance term collapses.
    """
    a, b = 4.0e-9, 1.0e-11
    y = torch.linspace(0.0, 5.8 / C_M, 64)[None, None, :, None].repeat(1, 1, 1, 8)
    sane = predicted_variance(y, a, b, clamp_to=y)
    runaway = predicted_variance(y * 50.0, a, b, clamp_to=y)
    assert torch.isfinite(runaway).all()
    assert runaway.max() <= sane.max() * (1.0 + 1e-5)


def test_orthogonality_is_off_by_default():
    """The MMSE estimator violates it; it is an ablation, not a term."""
    assert NoLAWeights().orthogonality == 0.0
    assert NoLAWeights().moment > 0 and NoLAWeights().whiteness > 0


def test_anchor_is_on_by_default():
    assert NoLAWeights().anchor > 0


# ------------------------------------------------------------ degeneracy
def test_zero_head_is_rejected():
    """A zero-initialised head makes every gradient of the objective vanish.

    The loss is finite and flat, so the run looks converged from step one.
    adapt() asserts against it rather than letting it produce a plausible
    checkpoint.
    """
    from nola.adapt import _assert_nondegenerate

    class ZeroHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.c = torch.nn.Conv2d(1, 1, 1)
            torch.nn.init.zeros_(self.c.weight)
            torch.nn.init.zeros_(self.c.bias)

        def forward(self, x):
            return x - self.c(x) * 0.0        # output identical to input

    loader = [torch.randn(2, 1, 32, 32)]
    with pytest.raises((AssertionError, RuntimeError, ValueError)):
        _assert_nondegenerate(ZeroHead(), loader, "cpu")


# --------------------------------------------------------- leakage rule
def test_dataset_refuses_labels_by_default(tmp_path):
    """`clean_sinogram` and `sigma_*` must be unreachable without opting in."""
    from nola.data import TargetSinograms

    d = tmp_path / "train" / "L109"
    d.mkdir(parents=True)
    np.savez(d / "slice_0001.npz",
             sino_25=np.zeros((64, 32), np.float32),
             clean_sinogram=np.zeros((64, 32), np.float32),
             sigma_25=np.ones((64, 32), np.float32))

    ds = TargetSinograms(tmp_path, ["L109"], dose="25", split="train",
                         patch=16, patches_per_item=2, allow_labels=False)
    item = ds[0]
    assert isinstance(item, (torch.Tensor, np.ndarray))
    assert not getattr(ds, "allow_labels", False)
