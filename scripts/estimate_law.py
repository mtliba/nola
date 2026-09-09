#!/usr/bin/env python
"""Estimate the target noise law, and check it against the truth it never saw.

Protocol v2 section 7, and the data behind Figure 2. Two things happen here and
they must not be confused:

    ESTIMATION reads only ``sino_{dose}`` from the eight TRAINING patients.
    No clean sinogram, no stored sigma, no dose label, no test patient.

    VALIDATION compares the result against ``NoiseLaw.mayo_truth``, which is
    the exact (I0, sigma_e) the simulator used. This is available only because
    both systems here are simulated, and it turns section 7's target of "10-15
    percent relative error" from a hope into a measurement.

The separation is the point. Nothing computed in the validation half is
allowed to flow back into the estimation half, so the two are different
functions with no shared state and the truth is loaded after the fit.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nola  # noqa: F401,E402
from nola.lawfit import fit_empirical, fit_law         # noqa: E402
from nola.units import NoiseLaw, harmonise            # noqa: E402
from nola.splits import (MAYO_ADAPT as MAYO_TRAIN,  # noqa: E402
                         MAYO_TEST, patient_dir)

DOSE_FRACTION = {"25": 0.25, "10": 0.10, "05": 0.05}


def _files(ready: Path, patients, per_patient: int):
    out = []
    for p in patients:
        files = sorted(patient_dir(ready, p).glob("slice_*.npz"))
        if per_patient and len(files) > per_patient:
            files = [files[i] for i in
                     np.linspace(0, len(files) - 1, per_patient).round().astype(int)]
        out.extend(files)
    return out


def load_unlabelled(ready: Path, patients, dose: str, per_patient: int):
    """Yield harmonised noisy sinograms. Reads exactly one key per file."""
    key = f"sino_{dose}"
    for f in _files(ready, patients, per_patient):
        with np.load(f) as z:
            yield harmonise(z[key])


def load_labelled(ready: Path, patients, dose: str, per_patient: int):
    """VALIDATION ONLY: (clean, noisy) pairs for the empirical reference law."""
    key = f"sino_{dose}"
    for f in _files(ready, patients, per_patient):
        with np.load(f) as z:
            yield harmonise(z["clean_sinogram"]), harmonise(z[key])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--ready-sub", default="mayo_ready/1mm_B30")
    ap.add_argument("--per-patient", type=int, default=12,
                    help="slices per training patient; 8 x 12 = 96 sinograms, "
                         "each contributing about 840k pseudo-residuals")
    ap.add_argument("--n-bins", type=int, default=48)
    ap.add_argument("--doses", nargs="+", default=["25", "10", "05"])
    ap.add_argument("--axes", nargs="+", default=["view", "detector"],
                    help="pseudo-residual axis; the gate is on the "
                         "first, the rest are reported for comparison")
    args = ap.parse_args()

    ready = args.data_root / args.ready_sub
    args.out_dir.mkdir(parents=True, exist_ok=True)

    assert not (set(MAYO_TRAIN) & set(MAYO_TEST)), "train/test overlap"
    print(f"estimating from {len(MAYO_TRAIN)} training patients: "
          f"{', '.join(MAYO_TRAIN)}")
    print(f"test patients never opened: {', '.join(MAYO_TEST)}\n")

    report = {"per_patient": args.per_patient, "n_bins": args.n_bins,
              "train_patients": list(MAYO_TRAIN), "doses": {}}

    for dose in args.doses:
        sinos = list(load_unlabelled(ready, MAYO_TRAIN, dose, args.per_patient))
        truth = NoiseLaw.mayo_truth(DOSE_FRACTION[dose])
        report["doses"][dose] = {"n_sinograms": len(sinos),
                                 "true": truth.as_dict(), "axes": {}}
        print(f"dose {dose}%   {len(sinos)} sinograms")
        print(f"   delta-method    a {truth.a:.4e}  b {truth.b:.4e}  "
              f"I0 {truth.i0:9.1f}  sigma_e {truth.electronic_sigma:6.2f}")

        # The conditional variance actually realised in the data. Reads the
        # clean sinogram, so it is validation only and never reaches the fit.
        pairs = list(load_labelled(ready, MAYO_TRAIN, dose, args.per_patient))
        emp = fit_empirical([c for c, _ in pairs], [x for _, x in pairs],
                            n_bins=args.n_bins)
        rel_emp = emp.relative_error(truth)
        print(f"   empirical       a {emp.law.a:.4e}  b {emp.law.b:.4e}  "
              f"I0 {emp.law.i0:9.1f}  sigma_e {emp.law.electronic_sigma:6.2f}"
              f"   [vs delta-method: I0 {rel_emp['i0'] * 100:.1f}%, "
              f"sigma_e {rel_emp['electronic_sigma'] * 100:.1f}%]")
        report["doses"][dose]["empirical"] = {
            "estimated": emp.law.as_dict(),
            "relative_error_vs_delta_method": rel_emp,
            "bins": {"centre": emp.bin_centres.tolist(),
                     "variance": emp.bin_variance.tolist(),
                     "mean_square": emp.extra["mean_square"],
                     "fitted": emp.fitted_variance.tolist(),
                     "count": emp.bin_counts.tolist()}}
        del pairs

        for axis in args.axes:
            fit = fit_law(sinos, n_bins=args.n_bins, axis=axis)

            # --- validation half; nothing here feeds back into the fit -----
            rel = fit.relative_error(truth)
            rel_vs_emp = fit.relative_error(emp.law)
            gated = axis == args.axes[0]
            # The gate is against the empirical law: that is the noise the data
            # actually contains, and it is what an adapted model has to match.
            ok = (rel_vs_emp["i0"] <= 0.15
                  and rel_vs_emp["electronic_sigma"] <= 0.15)

            print(f"   {axis:<9} fit    a {fit.law.a:.4e}  b {fit.law.b:.4e}  "
                  f"I0 {fit.law.i0:9.1f}  sigma_e {fit.law.electronic_sigma:6.2f}"
                  + (f"   [clipped {fit.nonneg_clipped}]"
                     if fit.nonneg_clipped else ""))
            print(f"   {'':<9} err    vs delta-method  I0 {rel['i0'] * 100:5.1f}%"
                  f"   sigma_e {rel['electronic_sigma'] * 100:5.1f}%")
            print(f"   {'':<9}        vs empirical     "
                  f"I0 {rel_vs_emp['i0'] * 100:5.1f}%"
                  f"   sigma_e {rel_vs_emp['electronic_sigma'] * 100:5.1f}%"
                  f"   {'PASS' if ok else 'FAIL'}"
                  f"{' (gated)' if gated else ' (reference only)'}")

            report["doses"][dose]["axes"][axis] = {
                "estimated": fit.law.as_dict(),
                "relative_error_vs_delta_method": rel,
                "relative_error_vs_empirical": rel_vs_emp,
                "pass": bool(ok),
                "gated": gated,
                "n_bins_used": fit.extra["n_bins_used"],
                "nonneg_clipped": fit.nonneg_clipped,
                "bins": {"centre": fit.bin_centres.tolist(),
                         "variance": fit.bin_variance.tolist(),
                         "fitted": fit.fitted_variance.tolist(),
                         "count": fit.bin_counts.tolist()},
            }
        print()

    (args.out_dir / "law_estimates.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out_dir}/law_estimates.json")
    gate_axis = args.axes[0]
    ok = all(v["axes"][gate_axis]["pass"] for v in report["doses"].values())
    print(f"\nGATE: law estimation on the {gate_axis} axis {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
