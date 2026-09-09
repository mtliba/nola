#!/usr/bin/env python
"""Adapt the source denoiser to the target system without labels.

Protocol v2 section 8, plus the Block-A rows that share this code path.

    --method source   theta_src used zero-shot. No training. Row B0.
    --method adabn    BatchNorm statistics recomputed on target sinograms.
                      No gradient step is taken. Row AdaBN.
    --method nola     the full objective in nola.losses. Row NoLA.

The ablation rows are ``--method nola --ablate whiteness`` and so on, and the
oracle-law row is ``--oracle-law``, which substitutes the true (I0, sigma_e)
for the estimate and changes nothing else.

WHAT THIS SCRIPT IS NOT ALLOWED TO SEE
    Target training patients only, one key per file: ``sino_{dose}``. The
    dataset class refuses to open ``clean_sinogram`` or ``sigma_*`` unless
    ``allow_labels`` is set, and nothing here sets it. ``--oracle-law`` is the
    single deliberate exception, is off by default, and is recorded in the run
    manifest so an oracle row can never be mistaken for a real one.

WALL CLOCK IS A RESULT
    NoLA's claim includes being cheap. The elapsed time of the adaptation loop
    is measured and written into the manifest so the paper quotes a measured
    number rather than an estimate.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nola  # noqa: F401,E402
from nola.data import TargetSinograms                       # noqa: E402
from nola.adapt import adapt                             # noqa: E402
from nola.losses import NoLAWeights                      # noqa: E402
from nola.unet import ResidualUNet                          # noqa: E402
from nola.units import NoiseLaw                             # noqa: E402
from nola.splits import MAYO_ADAPT as MAYO_TRAIN, MAYO_TEST  # noqa: E402

DOSE_FRACTION = {"25": 0.25, "10": 0.10, "05": 0.05}


def load_source(ckpt: Path, device: str) -> tuple[ResidualUNet, dict]:
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    a = blob.get("args", {})
    model = ResidualUNet(base=a.get("base", 48), depth=a.get("depth", 4))
    model.load_state_dict(blob["model"])
    return model.to(device), blob


def resolve_law(args) -> tuple[NoiseLaw, str]:
    """Where (a, b) come from, and a label saying so."""
    if args.oracle_law:
        return NoiseLaw.mayo_truth(DOSE_FRACTION[args.dose]), "oracle"
    rep = json.loads(Path(args.law_json).read_text())
    entry = rep["doses"][args.dose]["axes"][args.law_axis]["estimated"]
    return NoiseLaw(a=entry["a"], b=entry["b"]), f"estimated/{args.law_axis}"


@torch.no_grad()
def run_adabn(model, loader, device, n_batches: int) -> int:
    """Recompute BatchNorm running statistics on target data.

    Momentum is set to None so each layer accumulates a cumulative average
    over everything it sees, which is what AdaBN means; leaving the default
    0.1 would make the result depend on the order of the last few batches.
    """
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None
    model.train()
    seen = 0
    for batch in loader:
        x = batch.to(device).flatten(0, 1) if batch.dim() == 5 else batch.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(x)
        seen += 1
        if seen >= n_batches:
            break
    model.eval()
    return seen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--ready-sub", default="mayo_ready/1mm_B30")
    ap.add_argument("--ckpt", type=Path,
                    default=Path(__file__).resolve().parent.parent
                    / "checkpoints" / "theta_src.pt")
    ap.add_argument("--method", choices=["nola", "adabn", "source"],
                    default="nola")
    ap.add_argument("--dose", default="25", choices=["25", "10", "05"])
    ap.add_argument("--law-json", type=Path,
                    default=Path(__file__).resolve().parent.parent
                    / "results" / "law_estimates.json")
    ap.add_argument("--law-axis", default="view")
    ap.add_argument("--oracle-law", action="store_true",
                    help="ABLATION ONLY: use the true (I0, sigma_e)")
    ap.add_argument("--ablate", nargs="*", default=[],
                    choices=["moment", "whiteness", "orthogonality", "anchor",
                             "sure"])
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--batch-items", type=int, default=8)
    ap.add_argument("--patches-per-item", type=int, default=8)
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--w-moment", type=float, default=NoLAWeights().moment)
    # Defaults come from NoLAWeights so the objective is defined in ONE place.
    # They were duplicated here, and when orthogonality was switched off after
    # review the dataclass changed and this did not: the run labelled "nola"
    # still carried the term it was supposed to have dropped.
    _dflt = NoLAWeights()
    ap.add_argument("--w-whiteness", type=float, default=_dflt.whiteness)
    ap.add_argument("--w-orthogonality", type=float, default=_dflt.orthogonality)
    ap.add_argument("--w-anchor", type=float, default=_dflt.anchor)
    ap.add_argument("--w-sure", type=float, default=0.0,
                    help="weight on the SURE risk term; 0 disables it")
    ap.add_argument("--n-bins", type=int, default=20)
    ap.add_argument("--whiteness-axes", default="both",
                    choices=["both", "detector", "view"],
                    help="which sinogram axes the whiteness term acts on")
    ap.add_argument("--adapt-params", default="all",
                    choices=["all", "decoder", "norm", "head"],
                    help="which parameters adaptation may change (review 19)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260906)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    device = "cuda"
    if not torch.cuda.is_available():
        print("FATAL: no GPU.", file=sys.stderr)
        return 2
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ready = args.data_root / args.ready_sub

    assert not (set(MAYO_TRAIN) & set(MAYO_TEST)), "train/test overlap"
    model, src_blob = load_source(args.ckpt, device)
    print(f"theta_src from {args.ckpt} (step {src_blob.get('step')}), "
          f"{model.n_parameters / 1e6:.1f}M parameters")
    print(f"adapting on {len(MAYO_TRAIN)} training patients at {args.dose}% dose")
    print(f"test patients untouched: {', '.join(MAYO_TEST)}")

    tag = args.tag or "_".join(
        [args.method, f"dose{args.dose}"]
        + (["oraclelaw"] if args.oracle_law else [])
        + ([f"no-{x}" for x in args.ablate]))

    manifest = {"tag": tag, "method": args.method, "dose": args.dose,
                "ablate": args.ablate, "oracle_law": bool(args.oracle_law),
                "args": {k: str(v) for k, v in vars(args).items()},
                "source_ckpt_step": src_blob.get("step")}

    if args.method == "source":
        print("\nmethod=source: zero-shot, no adaptation performed")
        model.eval()
        manifest["seconds"] = 0.0
    else:
        ds = TargetSinograms(ready, MAYO_TRAIN, dose=args.dose, split="train",
                             patch=args.patch,
                             patches_per_item=args.patches_per_item,
                             allow_labels=False, seed=args.seed)
        dl = DataLoader(ds, batch_size=args.batch_items, shuffle=True,
                        num_workers=args.workers, pin_memory=True,
                        drop_last=True, persistent_workers=args.workers > 0)
        print(f"target set: {len(ds)} sinograms, effective batch "
              f"{args.batch_items * args.patches_per_item} patches")

        t0 = time.time()
        if args.method == "adabn":
            n = run_adabn(model, dl, device, n_batches=200)
            print(f"AdaBN: statistics recomputed over {n} batches")
            manifest["adabn_batches"] = n
        else:
            law, law_source = resolve_law(args)
            print(f"law ({law_source}): a {law.a:.4e}  b {law.b:.4e}  "
                  f"-> I0 {law.i0:.1f}  sigma_e {law.electronic_sigma:.2f}")
            manifest["law"] = law.as_dict() | {"source": law_source}

            w = NoLAWeights(
                moment=0.0 if "moment" in args.ablate else args.w_moment,
                whiteness=0.0 if "whiteness" in args.ablate else args.w_whiteness,
                orthogonality=(0.0 if "orthogonality" in args.ablate
                               else args.w_orthogonality),
                anchor=0.0 if "anchor" in args.ablate else args.w_anchor,
                sure=0.0 if "sure" in args.ablate else args.w_sure)
            print(f"weights: {w}")

            out = adapt(model, dl, law, w, steps=args.steps, lr=args.lr,
                        warmup=args.warmup, n_bins=args.n_bins, device=device,
                        param_subset=args.adapt_params,
                        whiteness_axes=args.whiteness_axes)
            manifest["history"] = out["history"]
        manifest["seconds"] = round(time.time() - t0, 1)
        print(f"\nadaptation wall clock: {manifest['seconds'] / 60:.1f} min")

    ckpt_out = args.out_dir / f"{tag}.pt"
    torch.save({"model": model.state_dict(), "manifest": manifest,
                "base": src_blob.get("args", {}).get("base", 48),
                "depth": src_blob.get("args", {}).get("depth", 4)}, ckpt_out)
    (args.out_dir / f"{tag}.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {ckpt_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
