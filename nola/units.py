"""Unit harmonisation between the two systems, and the noise law they share.

THE HARMONISATION CONSTANT IS AN EXACT CONVERSION, NOT A RANGE MATCH
--------------------------------------------------------------------
Protocol v2 section 3 allowed for c_M to be "chosen so the value ranges
match". It does not have to be chosen at all, and saying so is worth a
sentence in the paper.

Both systems measure the same physical quantity: the dimensionless line
integral of linear attenuation along a ray. They differ only in what they
divide it by before storing it.

    LoDoPaB   stores  y = (integral of mu dl) / MU_MAX,  MU_MAX ~ 81.3564 m^-1
    Mayo      stores  p =  integral of mu dl             (mu in mm^-1, dl in mm)

So the map from Mayo units to LoDoPaB units is a division by MU_MAX and
nothing else. There is no free parameter and no fitting:

    c_M = MU_MAX

The numbers agree, which is the check that the reasoning is right rather than
merely tidy: the densest Mayo ray is p = 5.77, giving 0.0709 harmonised,
against a LoDoPaB maximum of 0.1305. Same scale, Mayo slightly thinner because
its 340 mm reconstruction field is imaged at higher flux. A range-matching
constant would have produced roughly 44 and silently absorbed a factor of two
of genuine physical difference into the "unit" conversion, where it would then
have been invisible to the law estimator.

THE TWO SYSTEMS SIT IN ONE TWO-PARAMETER FAMILY
-----------------------------------------------
Post-log measurements from a Poisson-plus-Gaussian detector have, by the delta
method, variance

    Var(p) = (I0 exp(-p) + sigma_e^2) / (I0 exp(-p))^2
           = exp(p) / I0  +  sigma_e^2 exp(2p) / I0^2

Write that in harmonised units, p = c_M * y:

    Var(y) = a exp(c_M y) + b exp(2 c_M y)
    a = 1 / (I0 c_M^2)                 b = sigma_e^2 / (I0^2 c_M^2)

This is the single most useful fact in the project. The LoDoPaB source system
is the SAME law with b = 0 exactly, since it simulates pure Poisson statistics
at N0 = 4096 with no electronic noise. Source and target are therefore not two
unrelated noise models; they are two points in one two-parameter family, and
the source sits on its boundary.

That is precisely why adaptation here must identify rather than rescale. A
method that assumes the target law has the source's shape and only fits a
scale is constrained to the b = 0 edge, and no scale on that edge reproduces a
target with b > 0. NoLA estimates (a, b) jointly and can leave the edge.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import geometry as _geo

# The conversion is exactly the constant LoDoPaB normalises by.
C_M: float = float(_geo.MU_MAX)

# Source system, for reference and for the L2-SP starting point.
SOURCE_PHOTONS: int = int(_geo.LODOPAB_PHOTONS)


def harmonise(mayo_sinogram: np.ndarray) -> np.ndarray:
    """Mayo line integrals (dimensionless) -> LoDoPaB stored units."""
    return (np.asarray(mayo_sinogram, dtype=np.float32) / C_M).astype(np.float32)



def dehmonise(harmonised: np.ndarray) -> np.ndarray:
    """LoDoPaB stored units -> Mayo line integrals."""
    return (np.asarray(harmonised, dtype=np.float32) * C_M).astype(np.float32)


@dataclass(frozen=True)
class NoiseLaw:
    """Var(y) = a exp(c_M y) + b exp(2 c_M y), y in harmonised units."""
    a: float
    b: float

    def variance(self, y: np.ndarray) -> np.ndarray:
        """Predicted variance at harmonised signal level ``y``.

        exp(c_M * y) overflows float32 for the densest rays, so the exponent is
        taken in float64 and clipped exactly as geometry.simulate_low_dose does. The
        clip bites only where the measurement carries almost no photons, which
        is also where the delta method has stopped being trustworthy.
        """
        e = np.clip(np.asarray(y, dtype=np.float64) * C_M, None, 50.0)
        ex = np.exp(e)
        return (self.a * ex + self.b * ex * ex).astype(np.float32)

    def sigma(self, y: np.ndarray) -> np.ndarray:
        return np.sqrt(np.maximum(self.variance(y), 0.0)).astype(np.float32)

    # --- physical readout, for the estimator-validation figure -------------
    @property
    def i0(self) -> float:
        return 1.0 / (self.a * C_M ** 2)

    @property
    def electronic_sigma(self) -> float:
        return float(np.sqrt(max(self.b, 0.0)) / (self.a * C_M))

    def as_dict(self) -> dict:
        return {"a": self.a, "b": self.b,
                "i0": self.i0, "electronic_sigma": self.electronic_sigma}

    # --- constructors ------------------------------------------------------
    @staticmethod
    def from_physical(i0: float, electronic_sigma: float) -> "NoiseLaw":
        a = 1.0 / (i0 * C_M ** 2)
        b = electronic_sigma ** 2 / (i0 ** 2 * C_M ** 2)
        return NoiseLaw(a=float(a), b=float(b))

    @staticmethod
    def source() -> "NoiseLaw":
        """The LoDoPaB law: pure Poisson at N0, so b = 0 exactly."""
        return NoiseLaw(a=1.0 / (SOURCE_PHOTONS * C_M ** 2), b=0.0)

    @staticmethod
    def mayo_truth(dose_fraction: float) -> "NoiseLaw":
        """The known target law. For validation and the oracle ablation ONLY.

        Never call this from an estimation or adaptation path: it encodes the
        answer the estimator is supposed to recover from unlabelled data.
        """
        return NoiseLaw.from_physical(_geo.I0_FULL * dose_fraction,
                                      _geo.ELECTRONIC_SIGMA)


DOSE_KEYS = {0.25: "25", 0.10: "10", 0.05: "05"}
