#!/usr/bin/env python
"""Does the adapted residual actually obey the target law, on held-out patients?

This is the direct test of NoLA's claim, and it exists because PSNR cannot be
one. On the 116-slice evaluation list the pixel ranking is inverted with respect
to texture fidelity: CoreDiff leads PSNR by 3.1 dB while keeping 29 percent of
full-dose texture power, and the source model keeps 10 percent while matching
RED-CNN on PSNR and beating it on SSIM. A metric that rewards removing texture
cannot adjudicate a method whose entire claim is that it stops removing texture.

So the three quantities NoLA optimises are measured again here, on the TEST
patients, which adaptation never saw:

    law match    mean |log (measured conditional variance / law prediction)|
                 over signal bands. Zero means the residual has exactly the
                 variance the law says it should, at every signal level.

    whiteness    the largest absolute autocorrelation of the standardised
                 residual over lags 1-5 along both sinogram axes. Zero means
                 nothing structured was left behind.

    orthogonality squared correlation between residual and estimate. Zero means
                 no signal was eaten.

Three things make this a test rather than a restatement of the training loss.
The patients are different. The measurement is robust (MAD) where the loss was
not, so a method cannot pass by exploiting the estimator. And the comparison is
run against BOTH the estimated law that NoLA was given and the true law it was
never told, which separates "matched what it was aimed at" from "matched
reality".

A method that scores well here and poorly on PSNR is reproducing the physics
and losing to smoother competitors on a metric that rewards smoothing. That is
a claim the table can support only if this file exists.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nola  # noqa: F401,E402
from nola.audit import (accumulate_bands, band_edges,  # noqa: E402
                        law_match, whiteness_and_orthogonality)
from nola.unet import ResidualUNet             # noqa: E402
from nola.units import NoiseLaw, harmonise     # noqa: E402
from nola.splits import (MAYO_ADAPT as MAYO_TRAIN,  # noqa: E402
                         MAYO_TEST, patient_dir)

DOSE_FRACTION = {"25": 0.25, "10": 0.10, "05": 0.05}
LAGS = (1, 2, 3, 4, 5)


def load_model(ckpt: Path, device: str):
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    base = blob.get("base") or blob.get("args", {}).get("base", 48)
    depth = blob.get("depth") or blob.get("args", {}).get("depth", 4)
    m = ResidualUNet(base=base, depth=depth)
    m.load_state_dict(blob["model"])
    return m.to(device).eval(), blob.get("manifest", {})


def audit_one(model, ready: Path, slices: dict, dose: str, laws: dict,
              device: str, n_bins: int = 20, split: str = "test") -> dict:
    """Accumulate residual statistics over the evaluation slices."""
    # Pooled per-band residuals, so the robust scale is estimated on the whole
    # test set rather than per slice where it would be noisy.
    edges = None
    per_band: list[list[np.ndarray]] = [[] for _ in range(n_bins)]
    band_signal: list[list[np.ndarray]] = [[] for _ in range(n_bins)]
    autocorr = {f"{ax}{lag}": [] for ax in ("det", "view") for lag in LAGS}
    orth = []

    patients = MAYO_TEST if split == "test" else MAYO_TRAIN
    for patient in patients:
        for idx in slices[patient]:
            f = patient_dir(ready, patient) / f"slice_{idx:04d}.npz"
            with np.load(f) as z:
                y = harmonise(z[f"sino_{dose}"])
            x = torch.from_numpy(y)[None, None].to(device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                yh = model(x).float()
            y_hat = yh.cpu().numpy()[0, 0]
            r = y - y_hat

            if edges is None:
                # Bands come from the NOISY INPUT, not from y_hat, and are
                # fixed on the first slice. Quantiles of y_hat would be
                # method-dependent, so each row would be scored on its own
                # bands and the law-match column would not be comparable
                # across the table. y is identical for every method.
                edges = band_edges(y, n_bins)
                per_band = [[] for _ in range(n_bins)]
                band_signal = [[] for _ in range(n_bins)]
            accumulate_bands(r, y_hat, edges, per_band, band_signal)

            rho, o = whiteness_and_orthogonality(r, y_hat, laws["used"])
            for k, v in rho.items():
                autocorr[k].append(v)
            orth.append(o)

    out: dict = {"n_slices": sum(len(v) for v in slices.values())}
    for label, law in laws.items():
        m = law_match(per_band, band_signal, law)
        out[f"law_match_{label}"] = m["law_match"]
        out[f"bins_{label}"] = m["bins"]

    rho = {k: float(np.mean(v)) for k, v in autocorr.items()}
    out["autocorr"] = rho
    out["whiteness_max_abs_rho"] = float(max(abs(v) for v in rho.values()))
    out["orthogonality"] = float(np.mean(orth))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--ready-sub", default="mayo_ready/1mm_B30")
    ap.add_argument("--checkpoints", type=Path, nargs="+", required=True,
                    help="checkpoint files, or directories to glob for *.pt")
    ap.add_argument("--dose", default="25", choices=["25", "10", "05"])
    ap.add_argument("--law-json", type=Path,
                    default=Path(__file__).resolve().parent.parent
                    / "results" / "law_estimates.json")
    ap.add_argument("--law-axis", default="view")
    ap.add_argument("--eval-slices", type=Path,
                    default=Path(__file__).resolve().parent.parent
                    / "results" / "eval_slices.json")
    ap.add_argument("--n-bins", type=int, default=20)
    ap.add_argument("--split", default="test", choices=["train", "test"],
                    help="ALWAYS use train for choosing anything. Law match "
                         "needs no labels, so hyperparameters can be selected "
                         "on the adaptation patients with no leakage; "
                         "splits.py forbids choosing on held-out data and this "
                         "flag is what makes obeying it free.")
    ap.add_argument("--train-slices-per-patient", type=int, default=20)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ready = args.data_root / args.ready_sub
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.split == "test":
        slices = json.loads(args.eval_slices.read_text())
    else:
        # Train patients have no fixed evaluation list; take an even spread.
        slices = {}
        for p in MAYO_TRAIN:
            ids = sorted(int(f.stem.split("_")[1])
                         for f in patient_dir(ready, p).glob("slice_*.npz"))
            step = max(1, len(ids) // args.train_slices_per_patient)
            slices[p] = ids[::step][:args.train_slices_per_patient]
    print(f"split: {args.split}  "
          f"({sum(len(v) for v in slices.values())} slices)")

    est = json.loads(args.law_json.read_text())["doses"][args.dose]
    laws = {
        "used": NoiseLaw(a=est["axes"][args.law_axis]["estimated"]["a"],
                         b=est["axes"][args.law_axis]["estimated"]["b"]),
        "true": NoiseLaw.mayo_truth(DOSE_FRACTION[args.dose]),
    }
    print(f"estimated law  I0 {laws['used'].i0:.0f}  "
          f"sigma_e {laws['used'].electronic_sigma:.2f}")
    print(f"true law       I0 {laws['true'].i0:.0f}  "
          f"sigma_e {laws['true'].electronic_sigma:.2f}\n")

    files: list[Path] = []
    for c in args.checkpoints:
        files.extend(sorted(c.glob("*.pt")) if c.is_dir() else [c])
    if not files:
        print("no checkpoints found", file=sys.stderr)
        return 2

    print(f"{'method':<22} {'law match':>10} {'vs true':>9} "
          f"{'whiteness':>10} {'orth':>10}")
    report = {}
    for ck in files:
        model, manifest = load_model(ck, device)
        name = ck.stem.replace(f"_dose{args.dose}", "")
        r = audit_one(model, ready, slices, args.dose, laws, device,
                      args.n_bins, args.split)
        r["manifest"] = manifest
        report[name] = r
        print(f"{name:<22} {r['law_match_used']:10.4f} "
              f"{r['law_match_true']:9.4f} "
              f"{r['whiteness_max_abs_rho']:10.4f} "
              f"{r['orthogonality']:10.2e}", flush=True)
        del model
        torch.cuda.empty_cache()

    dest = (args.out_dir /
            f"residual_audit_dose{args.dose}_{args.split}.json")
    dest.write_text(json.dumps(report, indent=2))
    print(f"\nlaw match is mean |log(measured/predicted)| over {args.n_bins} "
          f"signal bands; 0 is perfect, 0.69 is a factor of two.")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
