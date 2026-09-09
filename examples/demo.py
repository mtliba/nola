#!/usr/bin/env python
"""NoLA end to end, on a laptop, in a few minutes, with nothing to download.

This runs the whole argument of the paper on synthetic data:

    1. build a phantom and its clean sinogram
    2. acquire it twice, under two different physical noise laws
         SOURCE  Poisson only          (b = 0 exactly)
         TARGET  Poisson + electronic  (b > 0)
    3. train a small denoiser on the SOURCE system, with labels
    4. show that it degrades on the TARGET system
    5. estimate the target law from unlabelled target sinograms
    6. adapt the source model to that law, with no labels at all
    7. show the gap close

Steps 5 and 6 call the same `nola.lawfit` and `nola.adapt` used for every
number in the paper, so this is a test of the shipped code rather than a
restatement of it.

Deliberately no ODL and no ASTRA: the Radon transform here is twenty lines of
scipy. The tomography is therefore crude, but NoLA operates in the projection
domain and never sees the reconstruction, so nothing about the method is
being faked. To reproduce the paper's numbers on real anatomy you need the
real geometry -- see README, "Reproducing the paper".

    python examples/demo.py            # CPU, ~3 minutes
    python examples/demo.py --steps 4000 --device cuda
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nola.adapt import adapt                                  # noqa: E402
from nola.geometry import MU_MAX                              # noqa: E402
from nola.lawfit import fit_law                               # noqa: E402
from nola.losses import NoLAWeights                           # noqa: E402
from nola.units import NoiseLaw, harmonise                    # noqa: E402


# ----------------------------------------------------------------- phantom
def phantom(n: int = 128, rng=None) -> np.ndarray:
    """A Shepp-Logan-like ellipse phantom, in attenuation units."""
    rng = rng or np.random.default_rng(0)
    y, x = np.mgrid[-1:1:n * 1j, -1:1:n * 1j]
    img = np.zeros((n, n), dtype=np.float64)
    # (cx, cy, major, minor, angle, value)
    ellipses = [(0, 0, 0.92, 0.69, 0, 1.0), (0, -0.02, 0.87, 0.66, 0, -0.8),
                (0.22, 0, 0.11, 0.31, -18, -0.2), (-0.22, 0, 0.16, 0.41, 18, -0.2),
                (0, 0.35, 0.21, 0.25, 0, 0.1), (0, 0.1, 0.046, 0.046, 0, 0.1),
                (-0.08, -0.605, 0.046, 0.023, 0, 0.1),
                (0.06, -0.605, 0.023, 0.023, 0, 0.1)]
    for cx, cy, a, b, ang, v in ellipses:
        t = np.deg2rad(ang)
        xr = (x - cx) * np.cos(t) + (y - cy) * np.sin(t)
        yr = -(x - cx) * np.sin(t) + (y - cy) * np.cos(t)
        img[(xr / a) ** 2 + (yr / b) ** 2 <= 1] += v
    return np.clip(img, 0, None)


def radon(img: np.ndarray, n_views: int = 180) -> np.ndarray:
    """Parallel-beam forward projection, by rotating and summing."""
    from scipy.ndimage import rotate
    out = np.empty((n_views, img.shape[0]), dtype=np.float64)
    for i, a in enumerate(np.linspace(0.0, 180.0, n_views, endpoint=False)):
        out[i] = rotate(img, a, reshape=False, order=1, mode="constant").sum(0)
    return out


def iradon(sino: np.ndarray, size: int) -> np.ndarray:
    """Filtered back-projection with a ramp filter. For display only."""
    from scipy.ndimage import rotate
    n_views, n_det = sino.shape
    freq = np.fft.rfftfreq(n_det)
    filt = 2.0 * freq                                     # ramp
    fil = np.fft.irfft(np.fft.rfft(sino, axis=1) * filt, n=n_det, axis=1)
    rec = np.zeros((size, size), dtype=np.float64)
    for i, a in enumerate(np.linspace(0.0, 180.0, n_views, endpoint=False)):
        band = np.tile(fil[i], (size, 1))
        rec += rotate(band, -a, reshape=False, order=1, mode="constant")
    return rec * np.pi / (2.0 * n_views)


# ------------------------------------------------------------- acquisition
def acquire(clean_p: np.ndarray, i0: float, sigma_e: float, rng) -> np.ndarray:
    """Post-log Poisson(+Gaussian) measurement of a line-integral field."""
    expected = i0 * np.exp(-clean_p)
    counts = rng.poisson(expected).astype(np.float64)
    if sigma_e > 0:
        counts = counts + rng.normal(0.0, sigma_e, counts.shape)
    return (-np.log(np.maximum(counts, 1.0) / i0)).astype(np.float32)


# ------------------------------------------------------------------ model
class SmallUNet(torch.nn.Module):
    """A deliberately small residual denoiser, so the demo runs on a CPU."""

    def __init__(self, width: int = 32):
        super().__init__()
        c = width
        blk = lambda i, o: torch.nn.Sequential(                # noqa: E731
            torch.nn.Conv2d(i, o, 3, padding=1), torch.nn.BatchNorm2d(o),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(o, o, 3, padding=1), torch.nn.BatchNorm2d(o),
            torch.nn.ReLU(inplace=True))
        self.e1, self.e2 = blk(1, c), blk(c, 2 * c)
        self.d1 = blk(3 * c, c)
        self.out = torch.nn.Conv2d(c, 1, 1)
        self.pool = torch.nn.MaxPool2d(2)
        torch.nn.init.zeros_(self.out.bias)
        # NOT zero-initialised: a zero head makes the NoLA objective's
        # gradient vanish identically -- an exact stationary point that looks
        # like convergence. nola.adapt asserts against it.
        torch.nn.init.normal_(self.out.weight, std=1e-3)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        u = torch.nn.functional.interpolate(e2, size=e1.shape[-2:],
                                            mode="nearest")
        return x - self.out(self.d1(torch.cat([e1, u], 1)))


def psnr(a, b):
    lo, hi = float(min(a.min(), b.min())), float(max(a.max(), b.max()))
    mse = float(np.mean(((a - b) / (hi - lo)) ** 2))
    return 10.0 * np.log10(1.0 / max(mse, 1e-30))


class Patches(torch.utils.data.Dataset):
    """Random patches of unlabelled target sinograms, for adaptation."""

    def __init__(self, sinos, patch=64, per_item=8, seed=0):
        self.s, self.p, self.k = sinos, patch, per_item
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        y = self.s[i]
        out = []
        for _ in range(self.k):
            r = self.rng.integers(0, y.shape[0] - self.p)
            c = self.rng.integers(0, y.shape[1] - self.p)
            out.append(y[r:r + self.p, c:c + self.p])
        return torch.from_numpy(np.stack(out)[:, None].astype(np.float32))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=800, help="adaptation steps")
    ap.add_argument("--src-steps", type=int, default=1200)
    ap.add_argument("--n-train", type=int, default=16)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--attenuation", type=float, default=0.12,
                    help="scales the line integrals. The default puts "
                         "p_max near 5, as in the real target system; "
                         "much lower and (a, b) stop being separable")
    ap.add_argument("--views", type=int, default=720,
                    help="angular sampling. Below ~360 the view-axis "
                         "pseudo-residual starts measuring signal "
                         "curvature instead of noise and I0 comes out low")
    ap.add_argument("--out", type=Path, default=Path("demo_out"))
    ap.add_argument("--seed", type=int, default=0)
    # Defaults mirror the paper's 25% dose operating point (D = 2.09).
    # Try --target-i0 8000 --target-sigma-e 12 to land at D = 0.50 and
    # watch adaptation fail exactly as the criterion predicts.
    ap.add_argument("--target-i0", type=float, default=37500.0)
    ap.add_argument("--target-sigma-e", type=float, default=10.0)
    args = ap.parse_args()

    dev = args.device
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    # Two systems, two laws. The source has NO electronic noise, so b = 0
    # exactly and it sits on the boundary of the two-parameter family; that is
    # why no rescaling of the source law can reach the target.
    SRC_I0, SRC_SE = 4096.0, 0.0
    TGT_I0, TGT_SE = args.target_i0, args.target_sigma_e

    print(__doc__.split("\n\n")[0])
    print(f"\nsource system : I0={SRC_I0:>8.0f}  sigma_e={SRC_SE:>5.1f}   (b = 0)")
    print(f"target system : I0={TGT_I0:>8.0f}  sigma_e={TGT_SE:>5.1f}   (b > 0)")
    print(f"c_M = {MU_MAX:.5f}")

    # The shift magnitude decides whether adaptation is worth doing AT ALL,
    # and it is computable before adapting, with no labels. The paper measures
    # rho = 0.953 between D and the gain over a 4x4 grid of acquisitions, and
    # every configuration that got worse had D < 1. So print it, and say what
    # it predicts -- including when it predicts failure.
    src_law = NoiseLaw.from_physical(SRC_I0, SRC_SE)
    tgt_law = NoiseLaw.from_physical(TGT_I0, TGT_SE)
    pp = np.linspace(0.0, 5.0, 200)
    wgt = np.exp(-((pp - 1.9) / 1.5) ** 2); wgt /= wgt.sum()
    D = float(np.sum(wgt * np.abs(np.log(tgt_law.variance(pp / MU_MAX)
                                         / src_law.variance(pp / MU_MAX)))))
    verdict = ("adaptation should help" if D > 1.3 else
               "LOW MISMATCH: the paper's criterion predicts adaptation will "
               "not help here, and may hurt" if D < 0.9 else
               "borderline; the 10% dose case that fails in the paper is D=1.05")
    print(f"shift magnitude D = {D:.2f}   -> {verdict}")
    print(f"(the paper's 25% dose operating point is D = 2.09)\n")

    # ---- data ------------------------------------------------------------
    print("[1/6] building phantoms and clean sinograms")
    cleans = []
    for k in range(args.n_train + 4):
        img = phantom(args.size, rng)
        img = img * (0.8 + 0.4 * rng.random())
        cleans.append(radon(img, args.views) * args.attenuation)
    clean_tr, clean_te = cleans[:args.n_train], cleans[args.n_train:]

    src_tr = [acquire(c, SRC_I0, SRC_SE, rng) for c in clean_tr]
    tgt_tr = [acquire(c, TGT_I0, TGT_SE, rng) for c in clean_tr]
    tgt_te = [acquire(c, TGT_I0, TGT_SE, rng) for c in clean_te]

    # ---- train the source model (this is the only place labels are used) --
    print(f"[2/6] training the source denoiser ({args.src_steps} steps, labels OK here)")
    model = SmallUNet().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    # Train on PATCHES, not whole sinograms. A 720 x 128 sinogram at batch 4
    # is minutes per hundred steps on a CPU, which would make this demo
    # something nobody runs. The law estimator still sees the full sinograms,
    # because it needs the view-axis structure that patches would break up.
    Xf = harmonise(np.stack(src_tr)[:, None].astype(np.float32))
    Yf = harmonise(np.stack(clean_tr)[:, None].astype(np.float32))
    X = torch.from_numpy(Xf).to(dev)
    Y = torch.from_numpy(Yf).to(dev)
    P = 96
    model.train()
    t0 = time.time()
    g = torch.Generator().manual_seed(args.seed)
    for step in range(args.src_steps):
        i = torch.randint(0, X.shape[0], (8,), generator=g)
        r = torch.randint(0, X.shape[2] - P, (1,), generator=g).item()
        c = torch.randint(0, X.shape[3] - P, (1,), generator=g).item()
        xb, yb = X[i, :, r:r + P, c:c + P], Y[i, :, r:r + P, c:c + P]
        loss = torch.nn.functional.mse_loss(model(xb), yb)
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 400 == 0:
            print(f"      step {step+1:>5}/{args.src_steps}  mse {loss.item():.3e}")
    model.eval()
    print(f"      done in {time.time()-t0:.0f}s")

    def evaluate(m):
        out = []
        with torch.no_grad():
            for y, c in zip(tgt_te, clean_te):
                x = torch.from_numpy(harmonise(y))[None, None].to(dev)
                out.append(psnr(m(x).cpu().numpy()[0, 0], harmonise(c)))
        return float(np.mean(out))

    src_psnr = evaluate(model)
    noisy_psnr = float(np.mean([psnr(harmonise(y), harmonise(c))
                                for y, c in zip(tgt_te, clean_te)]))

    # ---- estimate the target law, from noisy sinograms only --------------
    print("[3/6] estimating the target noise law from UNLABELLED sinograms")
    fit = fit_law([harmonise(y) for y in tgt_tr], n_bins=32, min_count=100)
    est = fit.law
    true = NoiseLaw.from_physical(TGT_I0, TGT_SE)
    e_i0 = abs(est.i0 - true.i0) / true.i0 * 100
    e_se = abs(est.electronic_sigma - true.electronic_sigma) / true.electronic_sigma * 100
    print(f"      estimated  I0={est.i0:9.1f}  sigma_e={est.electronic_sigma:6.2f}")
    print(f"      truth      I0={true.i0:9.1f}  sigma_e={true.electronic_sigma:6.2f}")
    print(f"      error         {e_i0:6.2f}%            {e_se:6.2f}%   "
          "(no labels were read)")

    # ---- adapt ------------------------------------------------------------
    print(f"[4/6] adapting to the ESTIMATED law ({args.steps} steps, still no labels)")
    loader = torch.utils.data.DataLoader(
        Patches([harmonise(y) for y in tgt_tr], seed=args.seed),
        batch_size=4, shuffle=True, drop_last=True)
    adapt(model, loader, est, NoLAWeights(), steps=args.steps, lr=2e-4,
          warmup=100, n_bins=12, device=dev, log_every=max(1, args.steps // 5))
    nola_psnr = evaluate(model)

    # ---- report -----------------------------------------------------------
    print("\n[5/6] result on held-out target slices (PSNR, projection domain)\n")
    gap = src_psnr - noisy_psnr
    rows = [("no denoising", noisy_psnr), ("source model, zero-shot", src_psnr),
            ("NoLA adapted (no labels)", nola_psnr)]
    w = max(len(r[0]) for r in rows)
    for name, v in rows:
        d = "" if name.startswith("no denoising") else f"   {v - noisy_psnr:+6.2f} dB vs noisy"
        print(f"      {name:<{w}}  {v:7.3f} dB{d}")
    print(f"\n      adaptation gain over the frozen source model: "
          f"{nola_psnr - src_psnr:+.2f} dB")
    if gap > 0:
        print(f"      that is {100*(nola_psnr-src_psnr)/max(gap,1e-9):.0f}% of "
              "what the source model itself was worth")

    # ---- figure -----------------------------------------------------------
    print("\n[6/6] writing figure")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        y = tgt_te[0]
        with torch.no_grad():
            src_out = model  # after adaptation; re-run source separately below
        fig, ax = plt.subplots(1, 3, figsize=(9, 3.2))
        ax[0].imshow(iradon(clean_te[0], args.size), cmap="gray")
        ax[0].set_title("clean")
        ax[1].imshow(iradon(y, args.size), cmap="gray")
        ax[1].set_title(f"target, no denoising\n{noisy_psnr:.2f} dB")
        with torch.no_grad():
            rec = model(torch.from_numpy(harmonise(y))[None, None].to(dev))
        ax[2].imshow(iradon(rec.cpu().numpy()[0, 0] * float(MU_MAX), args.size),
                     cmap="gray")
        ax[2].set_title(f"NoLA adapted\n{nola_psnr:.2f} dB")
        for a in ax:
            a.set_xticks([]); a.set_yticks([])
        fig.tight_layout()
        fig.savefig(args.out / "demo.png", dpi=140)
        print(f"      wrote {args.out / 'demo.png'}")
    except ImportError:
        print("      matplotlib not installed, skipping figure")

    print("\nDone. The law was estimated and the model adapted without ever "
          "reading a clean\nsinogram, a stored sigma, or a dose label.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
