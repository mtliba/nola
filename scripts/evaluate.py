#!/usr/bin/env python
"""Score any method on the canonical evaluation slices, in image space.

One entry point for both blocks of the table, because the alternative is two
scoring paths that drift apart. Protocol v2 section 9.

    --checkpoint PATH   a sinogram-domain model from this project: the noisy
                        target sinogram is harmonised, denoised, returned to
                        Mayo units, reconstructed with the scanner's own
                        fan-beam operator and converted to Hounsfield units.

    --precomputed NAME  an existing image-domain result under
                        <data-root>/mayo_denoised/NAME, which is how the
                        published Block-B denoisers are rescored.

    --fbp-only          no model at all: FBP of the noisy sinogram. The floor.

WHY BLOCK B IS RESCORED RATHER THAN COPIED
    The existing table selected its slices as the first 80 files in sorted
    order, which is all 60 of one test patient and 20 of the other, and then
    skipped any slice a method had not produced while still reporting n = 80.
    CoreDiff was therefore averaged over 77 slices and the others over 80, and
    the two patients were unequally weighted throughout. Rescoring every method
    on the 116 slices all of them share costs one short job and removes the
    inconsistency rather than footnoting it.

METRIC CONVENTIONS ARE INHERITED, NOT REINVENTED
    FULL_RANGE for the pixel metrics and SOFT_WINDOW for the perceptual ones,
    identical to scripts/quality_metrics.py in SinoAudit, so the numbers stay
    comparable to everything already published from this pipeline.

THE PRIMARY IMAGE REFERENCE IS FBP(clean_sinogram)
    The target simulation knows its own clean sinogram, so the operator-
    consistent reconstruction x_op = FBP(clean_sinogram) is available and is
    the reference that isolates DENOISING error from forward-model and
    reconstruction mismatch. It is reported as psnr_op / ssim_op and is
    primary.

    ``image_hu`` remains as a secondary, clinically interpretable reference.
    Scored against it, a method is charged for forward-model discretisation,
    reconstruction operator mismatch and the B30-versus-ramp kernel difference
    on top of its own error. Both are reported; neither is selected by a
    threshold.

    The most direct measurement of all needs no reconstruction: sinogram MSE
    against clean_sinogram, reported as sino_nrmse for models that produce a
    sinogram.

TWO REFERENCES, AND ONLY FOR THE TEXTURE METRIC
    The pixel and perceptual metrics are scored against ``image_hu``, the
    full-dose reconstruction, because that is what every published number for
    this dataset uses and the Day-1 headroom check confirmed it is fair here.

    The noise power spectrum cannot use it alone. ``image_hu`` was
    reconstructed by the scanner with a Siemens B30 kernel; every method in
    Block A produces a sinogram that this script reconstructs with ODL's
    ramp-filtered FBP, and the two kernels have very different high-frequency
    response. Comparing their spectra measures the reconstruction filter at
    least as much as the denoising, which is why the raw power ratio sits near
    2.7 even for a strong denoiser rather than near 1.

    The fix is NOT ``FBP(clean_sinogram)``: that is noiseless, so it has no
    texture to match and every ratio against it is larger still - measured 4.1
    where image_hu gave 2.7. What the metric needs is a reference with the
    right kernel AND the right amount of noise, which is a FULL-DOSE
    acquisition put through this pipeline's own operator. The clean sinogram
    makes that available: simulate it at full dose with the pipeline's own
    noise model, reconstruct it with the same FBP, and the result is what a
    perfect denoiser of the low-dose data should look like. A power ratio of 1
    against it means the texture was reproduced; below 1 means over-smoothing.

    Both are reported. ``nps_*`` against image_hu stays comparable with the
    literature; ``fd_nps_*`` against the full-dose operator-consistent
    reference is the one the paper should read, and the seed is fixed so the
    reference is identical for every method.

PER-SLICE VALUES ARE WRITTEN OUT
    With two test patients the only honest interval is a slice-level bootstrap
    within patient, and that needs the individual values, not the mean.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nola  # noqa: F401,E402
from nola.nps import nps_distance                 # noqa: E402
from nola.unet import ResidualUNet                # noqa: E402
from nola.units import dehmonise, harmonise       # noqa: E402
from nola import geometry as mayo             # noqa: E402
from nola.splits import MAYO_TEST, patient_dir    # noqa: E402

FULL_RANGE = (-1024.0, 3072.0)
SOFT_WINDOW = (-160.0, 240.0)
PIXEL_METRICS = ("psnr", "ssim", "ms_ssim", "vif", "gmsd")
PERCEPTUAL_METRICS = ("lpips", "dists")
NEEDS_RGB = ("lpips", "dists", "vif", "gmsd")


def to01(hu, rng):
    lo, hi = rng
    return np.clip((hu - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def as_batch(x, device, rgb=False):
    t = torch.from_numpy(x)[None, None].to(device)
    return t.repeat(1, 3, 1, 1) if rgb else t


def load_model(ckpt: Path, device: str):
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    base = blob.get("base") or blob.get("args", {}).get("base", 48)
    depth = blob.get("depth") or blob.get("args", {}).get("depth", 4)
    model = ResidualUNet(base=base, depth=depth)
    model.load_state_dict(blob["model"])
    return model.to(device).eval(), blob.get("manifest", {})


def main() -> int:
    import pyiqa

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--ready-sub", default="mayo_ready/1mm_B30")
    ap.add_argument("--denoised-sub", default="mayo_denoised")
    ap.add_argument("--eval-slices", type=Path,
                    default=Path(__file__).resolve().parent.parent
                    / "results" / "eval_slices.json")
    ap.add_argument("--dose", nargs="+", default=["25"],
                    choices=["25", "10", "05"])
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--precomputed", nargs="*", default=None,
                    help="one or more names under mayo_denoised/")
    ap.add_argument("--fbp-only", action="store_true")
    ap.add_argument("--name", default=None)
    ap.add_argument("--skip-perceptual", action="store_true")
    args = ap.parse_args()

    modes = sum(x is not None and x is not False
                for x in [args.checkpoint, args.precomputed,
                          args.fbp_only or None])
    if modes != 1:
        print("give exactly one of --checkpoint, --precomputed, --fbp-only",
              file=sys.stderr)
        return 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ready = args.data_root / args.ready_sub
    args.out_dir.mkdir(parents=True, exist_ok=True)
    slices = json.loads(args.eval_slices.read_text())
    n_total = sum(len(v) for v in slices.values())
    print(f"{n_total} evaluation slices   device {device}")

    model, manifest = (load_model(args.checkpoint, device)
                       if args.checkpoint else (None, {}))
    # The operator is always needed now: the texture reference is a full-dose
    # acquisition reconstructed by this pipeline, which every method is
    # compared against.
    #
    # astra_cuda is asserted when a GPU is present, so a silent CPU fallback
    # cannot turn a two-minute job into a fifty-minute one unnoticed. With no
    # GPU the CPU backend is the deliberate choice: scoring precomputed
    # image-domain outputs needs no network, so these runs belong on the CPU
    # partition rather than queueing for a GPU they do not use.
    impl = "astra_cuda" if device == "cuda" else "astra_cpu"
    print(f"operator backend: {impl}")
    A, fbp = mayo.build_fan_beam(impl=impl)
    fd_cache: dict[tuple[str, int], np.ndarray] = {}
    op_cache: dict[tuple[str, int], np.ndarray] = {}

    def operator_reference(patient: str, idx: int, clean_sino) -> np.ndarray:
        """FBP(clean_sinogram) in HU: the primary, operator-consistent truth."""
        key = (patient, idx)
        if key not in op_cache:
            rec = np.asarray(fbp(A.range.element(clean_sino)), dtype=np.float32)
            op_cache[key] = mayo.mu_to_hu(rec)
        return op_cache[key]

    def fulldose_reference(patient: str, idx: int, clean_sino) -> np.ndarray:
        """FBP of a full-dose realisation of the same slice.

        The seed is derived from the slice identity, so every method in every
        run is compared against the identical reference image; drawing a fresh
        realisation per method would put an extra noise term into the metric
        and let a method win or lose on the reference's dice roll.
        """
        key = (patient, idx)
        if key not in fd_cache:
            # hashlib, not hash(): Python randomises str hashing per process
            # unless PYTHONHASHSEED is set, so the built-in would give a
            # different reference image on every run.
            seed = int(hashlib.sha256(f"{patient}_{idx}".encode())
                       .hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            fd_sino, _ = mayo.simulate_low_dose(clean_sino, 1.0, rng)
            rec = np.asarray(fbp(A.range.element(fd_sino)), dtype=np.float32)
            fd_cache[key] = mayo.mu_to_hu(rec)
        return fd_cache[key]

    wanted = PIXEL_METRICS + (() if args.skip_perceptual else PERCEPTUAL_METRICS)
    metrics = {}
    for m in wanted:
        try:
            metrics[m] = pyiqa.create_metric(m, device=device)
        except Exception as exc:                    # noqa: BLE001
            print(f"   {m}: unavailable ({type(exc).__name__})")
    print(f"metrics: {', '.join(metrics)}")

    jobs = ([(m, d) for m in args.precomputed for d in args.dose]
            if args.precomputed else [(None, d) for d in args.dose])
    all_out = {}
    incomplete: list[str] = []
    for precomputed, dose in jobs:
        label = args.name or precomputed or (
            args.checkpoint.stem if args.checkpoint else "fbp")
        rows, t0 = [], time.time()
        n_missing = 0
        for patient in MAYO_TEST:
            for idx in slices[patient]:
                f = patient_dir(ready, patient) / f"slice_{idx:04d}.npz"
                with np.load(f) as z:
                    ref_hu = z["image_hu"]
                    noisy = (z[f"sino_{dose}"]
                             if (args.checkpoint or args.fbp_only) else None)
                    clean_sino = z["clean_sinogram"]
                fd_ref_hu = fulldose_reference(patient, idx, clean_sino)
                op_ref_hu = operator_reference(patient, idx, clean_sino)

                sino_out = None
                if precomputed:
                    # Denoised outputs follow the same legacy layout.
                    sub = patient_dir(ready, patient).parent.name
                    g = (args.data_root / args.denoised_sub / precomputed
                         / f"dose{dose}" / sub / patient
                         / f"slice_{idx:04d}.npz")
                    if not g.exists():
                        print(f"   MISSING {g}")
                        n_missing += 1
                        continue
                    with np.load(g) as z:
                        hat_hu = z["denoised_hu"]
                else:
                    if args.fbp_only:
                        sino_out = noisy
                    else:
                        x = torch.from_numpy(harmonise(noisy))[None, None].to(device)
                        with torch.no_grad(), torch.autocast(
                                "cuda", dtype=torch.bfloat16):
                            out = model(x)
                        sino_out = dehmonise(out.float().cpu().numpy()[0, 0])
                    rec = np.asarray(fbp(A.range.element(sino_out)),
                                     dtype=np.float32)
                    hat_hu = mayo.mu_to_hu(rec)

                row = {"patient": patient, "slice": int(idx)}
                # Primary: against the operator-consistent reconstruction.
                for key in ("psnr", "ssim"):
                    if key not in metrics:
                        continue
                    rgb = key in NEEDS_RGB
                    a_ = as_batch(to01(hat_hu, FULL_RANGE), device, rgb)
                    b_ = as_batch(to01(op_ref_hu, FULL_RANGE), device, rgb)
                    with torch.no_grad():
                        row[f"{key}_op"] = float(metrics[key](a_, b_))
                # Sinogram domain, where no reconstruction intervenes at all.
                if sino_out is not None:
                    denom = float(np.sqrt(np.mean(clean_sino ** 2))) or 1.0
                    row["sino_nrmse"] = float(
                        np.sqrt(np.mean((sino_out - clean_sino) ** 2)) / denom)
                # Secondary: against the scanner reconstruction.
                for key, fn in metrics.items():
                    rng = SOFT_WINDOW if key in PERCEPTUAL_METRICS else FULL_RANGE
                    rgb = key in NEEDS_RGB
                    a = as_batch(to01(hat_hu, rng), device, rgb)
                    b = as_batch(to01(ref_hu, rng), device, rgb)
                    with torch.no_grad():
                        row[key] = float(fn(a, b))
                row.update(nps_distance(hat_hu, ref_hu))
                row.update({f"fd_{k}": v for k, v in
                            nps_distance(hat_hu, fd_ref_hu).items()})
                rows.append(row)

        if not rows:
            print(f"{label} dose{dose}: nothing scored")
            continue
        keys = [k for k in rows[0] if k not in ("patient", "slice")]
        summary = {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
        summary["n"] = len(rows)
        per_patient = {p: sum(1 for r in rows if r["patient"] == p)
                       for p in MAYO_TEST}
        print(f"\n{label:<12} dose {dose}%  " +
              "  ".join(f"{k} {summary[k]:.4f}" for k in
                        ("psnr_op", "ssim_op", "sino_nrmse", "psnr",
                         "fd_nps_shape_distance")
                        if k in summary))
        print(f"   n = {summary['n']} ({per_patient})  "
              f"{time.time() - t0:.0f}s", flush=True)

        # A cached prediction that is not on disk used to be skipped
        # with a printed line and nothing else: the JSON was written
        # short, the job exited 0, and the row could enter a table
        # scored on a different slice set from every other row. It is
        # recorded now, and the exit code says so.
        n_expected = sum(len(v) for v in slices.values())
        if n_missing:
            incomplete.append(f"{label}_dose{dose}: {n_missing} of "
                              f"{n_expected} predictions missing")
            print(f"   INCOMPLETE: {n_missing}/{n_expected} missing",
                  flush=True)
        out = {"name": label, "dose": dose, "summary": summary,
               "per_slice": rows, "manifest": manifest,
               "per_patient_n": per_patient,
               "n_expected": n_expected, "n_missing": n_missing,
               "complete": n_missing == 0,
               # The backend is part of the result, not of the run.
               # astra_cpu and astra_cuda reconstruct the SAME clean
               # sinogram differently enough to move psnr_op by 1.2 dB
               # on average (2.9 dB worst case) and NPS shape by 0.24,
               # always in the same direction. Those references are
               # rebuilt by the operator, so a CPU-evaluated row placed
               # beside a CUDA-evaluated one loses a decibel to the
               # partition it was scheduled on. Recorded so a table can
               # refuse the mixture instead of averaging it.
               "operator_impl": impl, "device": device}
        dest = args.out_dir / f"eval_{label}_dose{dose}.json"
        dest.write_text(json.dumps(out, indent=2))
        all_out[f"{label}_dose{dose}"] = summary

    (args.out_dir / "eval_summary.json").write_text(json.dumps(all_out, indent=2))
    print(f"\nwrote {len(all_out)} results to {args.out_dir}")
    if incomplete:
        print("\nINCOMPLETE EVALUATIONS (do not put these in a table):",
              file=sys.stderr)
        for line in incomplete:
            print(f"   {line}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
