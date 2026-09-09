#!/usr/bin/env python
"""Stage 0: train the source denoiser on LoDoPaB, in the sinogram domain.

Protocol v2 section 5. This is the only model NoLA ever trains with labels,
and every later experiment starts from its weights, so it is worth training
once and properly rather than repeatedly and cheaply.

WHAT THE LOSS IS ON
    Plain MSE between the network output and the clean sinogram. Not a
    noise-weighted MSE: sigma spans a factor of roughly 200 across a LoDoPaB
    sinogram, and weighting by it would make the loss almost entirely about
    the few dense rays. Uniform MSE is also what the eventual comparison is
    against, so anything cleverer here would be an unstated advantage.

WHY VALIDATION USES WHOLE SINOGRAMS
    Training crops to 128x128; deployment sees 1152x736. A fully convolutional
    network handles that, but only if nothing in the pipeline has quietly
    learned the patch size, so validation is run at the full 1000x513 and the
    curve is the one that decides the checkpoint.

BF16, NOT FP16
    bf16 has the exponent range of fp32, and sinogram values span four orders
    of magnitude between an air ray and a ray through the spine. fp16 with a
    loss scaler works too, right up to the first time the scaler collapses on
    a dense-ray batch.
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
from nola.data import SourceSinograms  # noqa: E402
from nola.unet import ResidualUNet, count_norm_layers  # noqa: E402


def lr_at(step: int, total: int, base: float, warmup: int) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))


@torch.no_grad()
def validate(model, loader, device, limit: int) -> dict:
    model.eval()
    mse_model, mse_noisy, n = 0.0, 0.0, 0
    for obs, clean in loader:
        obs = obs.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(obs)
        mse_model += float(torch.mean((out.float() - clean) ** 2)) * obs.shape[0]
        mse_noisy += float(torch.mean((obs - clean) ** 2)) * obs.shape[0]
        n += obs.shape[0]
        if n >= limit:
            break
    model.train()
    return {"val_mse": mse_model / n, "val_mse_noisy": mse_noisy / n,
            "improvement_db": 10 * math.log10(mse_noisy / max(mse_model, 1e-30)),
            "n": n}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--source-sub", default="nola_source")
    ap.add_argument("--ckpt-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent / "checkpoints")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--base", type=int, default=48)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--val-slices", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260906)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"
    if not torch.cuda.is_available():
        print("FATAL: no GPU. This is the login node.", file=sys.stderr)
        return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    root = args.data_root / args.source_sub

    train = SourceSinograms(root, "train", patch=args.patch, seed=args.seed)
    val = SourceSinograms(root, "calib", patch=None, seed=args.seed + 1)
    print(f"train {len(train)} slices   val {len(val)} slices")

    tl = DataLoader(train, batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, pin_memory=True,
                    drop_last=True, persistent_workers=args.workers > 0)
    vl = DataLoader(val, batch_size=1, shuffle=False,
                    num_workers=max(2, args.workers // 4), pin_memory=True)

    model = ResidualUNet(base=args.base, depth=args.depth).to(device)
    model = model.to(memory_format=torch.channels_last)
    print(f"model: {model.n_parameters / 1e6:.1f}M parameters, "
          f"{count_norm_layers(model)} BatchNorm layers")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    history, best = [], math.inf
    step, t0 = 0, time.time()
    running = 0.0
    it = iter(tl)
    while step < args.steps:
        try:
            obs, clean = next(it)
        except StopIteration:
            it = iter(tl)
            obs, clean = next(it)

        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps, args.lr, args.warmup)

        obs = obs.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        clean = clean.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(obs)
            loss = torch.mean((out.float() - clean) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        running += float(loss)
        step += 1

        if step % 200 == 0:
            rate = step / (time.time() - t0)
            print(f"step {step:>6}/{args.steps}  loss {running / 200:.3e}  "
                  f"lr {opt.param_groups[0]['lr']:.2e}  {rate:.1f} it/s",
                  flush=True)
            running = 0.0

        if step % args.val_every == 0 or step == args.steps:
            v = validate(model, vl, device, args.val_slices)
            v["step"] = step
            history.append(v)
            print(f"   VAL step {step}  mse {v['val_mse']:.4e}  "
                  f"noisy {v['val_mse_noisy']:.4e}  "
                  f"+{v['improvement_db']:.2f} dB", flush=True)
            if v["val_mse"] < best:
                best = v["val_mse"]
                torch.save({"model": model.state_dict(),
                            "args": vars(args) | {"ckpt_dir": str(args.ckpt_dir),
                                                  "data_root": str(args.data_root),
                                                  "out_dir": str(args.out_dir)},
                            "step": step, "val": v},
                           args.ckpt_dir / "theta_src.pt")
                print(f"   saved {args.ckpt_dir / 'theta_src.pt'}", flush=True)
            (args.out_dir / "history.json").write_text(json.dumps(history, indent=2))

    print(f"\ndone in {(time.time() - t0) / 60:.1f} min, best val mse {best:.4e}")
    (args.out_dir / "history.json").write_text(json.dumps(history, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
