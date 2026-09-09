"""Acquisition geometry and the forward model, for both systems.

This module is self-contained on purpose. The research code this was extracted
from imported the geometry from a larger internal toolbox; carrying that
dependency into a public repository would mean nobody could run anything. What
NoLA actually needs from it is small — two ray transforms, a Hounsfield
conversion and a low-dose simulator — so those are reproduced here in full.

Two systems appear throughout:

    SOURCE   LoDoPaB-CT [Leuschner et al., Sci. Data 2021]. Parallel beam,
             1000 x 513, N0 = 4096 photons per bin, no electronic noise.
             Projections are stored divided by MU_MAX (see units.py).

    TARGET   Mayo-2016 anatomy through a simulated Siemens-like fan-beam
             scanner, 1152 views x 736 bins, I0 = 1.5e5 at full dose and
             electronic noise sigma_e = 10 counts. Projections are stored as
             raw line integrals.

The one constant everything depends on is MU_MAX, which makes the two
conventions commensurable exactly and with no free parameter. See units.py.
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# Source system: LoDoPaB-CT
# --------------------------------------------------------------------------
# Linear attenuation in m^-1. MU_MAX is the attenuation of the densest tissue
# LoDoPaB represents (3071 HU), and is the divisor its stored projections use.
MU_WATER = 20.0
MU_AIR = 0.02
MU_MAX = 3071 * (MU_WATER - MU_AIR) / 1000 + MU_WATER      # 81.35858 m^-1

LODOPAB_PHOTONS = 4096          # N0 per detector bin
LODOPAB_ANGLES = 1000
LODOPAB_DET_BINS = 513
LODOPAB_IMAGE = (362, 362)
LODOPAB_MAX_PT = 0.13           # half field of view, metres

# --------------------------------------------------------------------------
# Target system: Mayo-2016 geometry
# --------------------------------------------------------------------------
MU_WATER_MM = 0.0192            # mm^-1 at ~70 keV effective energy
IMAGE_SIZE = 512
RECON_DIAMETER_MM = 340.0
SRC_TO_ISO_MM = 595.0
SRC_TO_DET_MM = 1085.6
ISO_TO_DET_MM = SRC_TO_DET_MM - SRC_TO_ISO_MM               # 490.6
NUM_DET_BINS = 736
NUM_ANGLES = 1152
DET_SPACING_MM = 1.2858         # gives the 500 mm data-collection diameter

I0_FULL = 1.5e5                 # full-dose incident flux, photons per bin
ELECTRONIC_SIGMA = 10.0         # detector electronic noise, in counts
DOSE_FRACTIONS = (0.25, 0.10, 0.05)


def hu_to_mu(hu):
    """Hounsfield units to linear attenuation in mm^-1."""
    return (np.asarray(hu, dtype=np.float32) / 1000.0 + 1.0) * MU_WATER_MM


def mu_to_hu(mu):
    """Linear attenuation in mm^-1 back to Hounsfield units."""
    return (np.asarray(mu, dtype=np.float32) / MU_WATER_MM - 1.0) * 1000.0


def build_fan_beam(impl: str | None = None):
    """The target scanner's fan-beam geometry, in millimetres.

    Returns ``(A, fbp)``: the ray transform and its filtered back-projection.

    ``impl`` is passed straight to ODL. Leaving it None tries ``astra_cuda``
    then falls back to ``astra_cpu``, which is a factor of ~25 slower; pass it
    explicitly if a silent fallback would turn a two-minute job into an hour.
    NOTE the backend is not cosmetic: astra_cpu and astra_cuda reconstruct the
    same sinogram differently enough to move PSNR by over a decibel, so a
    figure or table must not mix them.
    """
    import odl

    half = RECON_DIAMETER_MM / 2.0
    domain = odl.uniform_discr(
        [-half, -half], [half, half], (IMAGE_SIZE, IMAGE_SIZE), dtype="float32")
    angles = odl.uniform_partition(0, 2 * np.pi, NUM_ANGLES)
    det_half = NUM_DET_BINS * DET_SPACING_MM / 2.0
    detector = odl.uniform_partition(-det_half, det_half, NUM_DET_BINS)
    geometry = odl.tomo.FanBeamGeometry(
        angles, detector, src_radius=SRC_TO_ISO_MM, det_radius=ISO_TO_DET_MM)

    for cand in ([impl] if impl else ["astra_cuda", "astra_cpu"]):
        try:
            A = odl.tomo.RayTransform(domain, geometry, impl=cand)
            A(domain.zero())
            return A, odl.tomo.fbp_op(A)
        except Exception:                                    # noqa: BLE001
            continue
    raise RuntimeError("no working ASTRA backend for the fan-beam geometry")


def build_parallel_beam(impl: str | None = None):
    """The source system's parallel-beam geometry (LoDoPaB), in metres."""
    import odl

    m = LODOPAB_MAX_PT
    domain = odl.uniform_discr([-m, -m], [m, m], LODOPAB_IMAGE, dtype="float32")
    angles = odl.uniform_partition(0, np.pi, LODOPAB_ANGLES)
    detector = odl.uniform_partition(-m * np.sqrt(2), m * np.sqrt(2),
                                     LODOPAB_DET_BINS)
    geometry = odl.tomo.Parallel2dGeometry(angles, detector)

    for cand in ([impl] if impl else ["astra_cuda", "astra_cpu"]):
        try:
            A = odl.tomo.RayTransform(domain, geometry, impl=cand)
            A(domain.zero())
            return A, odl.tomo.fbp_op(A)
        except Exception:                                    # noqa: BLE001
            continue
    raise RuntimeError("no working ASTRA backend for the parallel-beam geometry")


def simulate_low_dose(clean_line_integral, dose_fraction, rng,
                      i0_full: float = I0_FULL,
                      electronic_sigma: float = ELECTRONIC_SIGMA):
    """Post-log Poisson + Gaussian measurement at a thinned dose.

    Returns ``(noisy_line_integral, sigma)``, where ``sigma`` is the per-ray
    standard deviation the delta method predicts. That prediction comes from
    the model, not from the realisation, which is what makes it usable as a
    reference: a sigma measured from the sample it is meant to describe would
    be circular.

    ``sigma`` is a *label*. It is derived from the clean signal and must never
    reach the adaptation or estimation code — see the note in README.
    """
    i0 = i0_full * dose_fraction
    p = np.asarray(clean_line_integral, dtype=np.float64)
    expected = i0 * np.exp(-p)
    counts = rng.poisson(expected).astype(np.float64)
    if electronic_sigma > 0:
        counts = counts + rng.normal(0.0, electronic_sigma, size=counts.shape)
    # A non-positive count has no logarithm. Flooring at one photon is the
    # standard handling and touches only rays through the densest anatomy.
    counts = np.maximum(counts, 1.0)
    noisy = -np.log(counts / i0)

    variance = (expected + electronic_sigma ** 2) / np.maximum(expected, 1.0) ** 2
    return noisy.astype(np.float32), np.sqrt(variance).astype(np.float32)


def simulate_source_dose(clean_line_integral, rng,
                         n0: float = LODOPAB_PHOTONS):
    """The source system's measurement: Poisson only, no electronic term.

    This is why the source sits on the boundary b = 0 of the law family, and
    therefore why no rescaling of it can reach a target with b > 0.
    """
    p = np.asarray(clean_line_integral, dtype=np.float64)
    expected = n0 * np.exp(-p)
    counts = np.maximum(rng.poisson(expected).astype(np.float64), 1.0)
    return (-np.log(counts / n0)).astype(np.float32)
