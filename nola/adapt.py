"""The adaptation loop, shared by the Mayo runs and the controlled grid.

Kept out of the scripts so that the single-cell experiment and the sixteen-cell
sweep run identical code. A grid that quietly differed from the headline
experiment - a different schedule, a different clip, a different batch - would
make the sweep unusable as evidence about the headline number, which is the
whole reason the sweep exists.
"""
from __future__ import annotations

import math
import time

import torch

from .losses import L2SPAnchor, NoLAWeights, nola_loss
from .units import NoiseLaw


def cosine_with_warmup(step: int, total: int, base: float, warmup: int) -> float:
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))


def select_parameters(model, subset: str = "all") -> list:
    """Freeze everything outside ``subset`` and return what stays trainable.

    WHY RESTRICTING THE SUBSET IS THE RIGHT KNOB HERE
    The adaptation objective is under-determined: matching the residual's
    conditional variance and whiteness pins its statistics, not its alignment
    with the noise actually present. Measured at 10 percent dose, the full
    network reaches a law match of 0.106 and a whiteness of 0.012 - the best of
    any method - while its image is 3.9 dB WORSE than no adaptation. It is
    solving the stated problem and choosing a bad solution among the many that
    solve it.

    Shrinking the hypothesis space is the standard remedy for that, and it is
    more honest than adding another penalty: fewer degrees of freedom means
    fewer ways to satisfy the statistics while drifting from a good image, and
    the source model supplies the rest. It also makes the method cheaper to
    deploy, which is the practical story anyway.

        all      every parameter
        decoder  upsampling path, decoder blocks and head
        norm     BatchNorm affine parameters only (~0.1 percent of weights)
        head     the final 1x1 residual projection only
    """
    import torch.nn as nn

    for p in model.parameters():
        p.requires_grad_(subset == "all")
    if subset == "all":
        return [p for p in model.parameters()]

    chosen = []
    if subset == "decoder":
        mods = [model.ups, model.decoders, model.head]
    elif subset == "head":
        mods = [model.head]
    elif subset == "norm":
        mods = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    else:
        raise ValueError(f"unknown parameter subset {subset!r}")
    for m in mods:
        for p in m.parameters():
            p.requires_grad_(True)
            chosen.append(p)
    return chosen


def _assert_nondegenerate(model, loader, device) -> None:
    """Refuse to adapt a model whose residual is identically zero.

    Every term in the objective is a statement about r = y - f(y). The backbone
    initialises its head to zero so that an untrained model is exactly the
    identity, which makes r exactly zero, which makes all three data terms and
    all of their gradients exactly zero. The identity is therefore a genuine
    stationary point of the NoLA objective, and adaptation started there runs
    to completion, reports falling losses that never move, and returns the
    weights it was given.

    That cannot happen in a real run, because adaptation starts from a trained
    theta_src. It happens immediately if a checkpoint fails to load, or if
    someone adapts a freshly constructed model, and both are silent failures
    worth one forward pass to catch.
    """
    batch = next(iter(loader))
    y = batch.to(device)
    if y.dim() == 5:
        y = y.flatten(0, 1)
    with torch.no_grad():
        r = y - model(y)
    rms = float(r.float().pow(2).mean().sqrt())
    scale = float(y.float().std())
    if rms <= 1e-9 * max(scale, 1e-12):
        raise RuntimeError(
            "the model's residual is identically zero, so every NoLA loss and "
            "every gradient is zero and adaptation would be a no-op. This is "
            "what an untrained backbone looks like: check that theta_src "
            f"loaded (residual rms {rms:.3e}, signal std {scale:.3e}).")


def adapt(model, loader, law: NoiseLaw, weights: NoLAWeights, *,
          steps: int = 10000, lr: float = 1e-5, warmup: int = 500,
          n_bins: int = 20, device: str = "cuda", log_every: int = 250,
          verbose: bool = True, param_subset: str = "all",
          whiteness_axes: str = "both") -> dict:
    """Run NoLA adaptation in place. Returns history and wall clock.

    The anchor is captured from the model AS PASSED IN, so the reference is
    whatever weights adaptation starts from. That is the source model in every
    real run; making it implicit rather than a parameter removes the chance of
    anchoring a grid cell to the wrong starting point.
    """
    _assert_nondegenerate(model, loader, device)
    anchor = L2SPAnchor(model)
    trainable = select_parameters(model, param_subset)
    n_train = sum(p.numel() for p in trainable)
    n_all = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"adapting {param_subset}: {n_train / 1e6:.3f}M of "
              f"{n_all / 1e6:.1f}M parameters "
              f"({100 * n_train / n_all:.2f}%)", flush=True)
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    model.train()

    # DIVERGENCE IS RECORDED, NOT SURVIVED
    # The controlled sweep found a cell where adaptation destroys the model: at
    # I0 = 7.5e3 with sigma_e = 20 the electronic term is 21x the Poisson term
    # in the dense tail. Measured there, the source model at 28.66 dB came out
    # at 9.55 dB.
    #
    # AND THE LOSS STAYED FINITE THE WHOLE TIME, which is why the check below
    # tests the model and not only the number. The failure is not arithmetic:
    # the law is exponential in the plug-in signal, so a model that drifts
    # upward is DEMANDED to produce a larger residual, which it supplies by
    # drifting further. That fixed point satisfies the moment loss with a
    # wrecked image, so a loss-magnitude test sees nothing wrong. Watching the
    # output leave the range of the data catches it; clipping the plug-in
    # signal in nola.losses closes the loop that creates it.
    history: list[dict] = []
    diverged, diverge_step, first_loss = False, None, None
    step, it, t0 = 0, iter(loader), time.time()
    while step < steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        y = batch.to(device, non_blocking=True)
        if y.dim() == 5:                 # (items, patches, 1, H, W)
            y = y.flatten(0, 1)

        for g in opt.param_groups:
            g["lr"] = cosine_with_warmup(step, steps, lr, warmup)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            y_hat = model(y)
        loss, parts = nola_loss(y, y_hat.float(), law.a, law.b,
                                model, anchor, weights, n_bins=n_bins,
                                whiteness_axes=whiteness_axes)

        value = float(loss)
        if first_loss is None:
            first_loss = value
        # The output must stay in the neighbourhood of the data it denoises.
        # A denoiser whose output spreads several times wider than its input
        # has stopped denoising and started inventing, whatever the loss says.
        if step % 50 == 0:
            spread = float(y_hat.detach().float().std())
            data_spread = float(y.float().std())
            if spread > 5.0 * max(data_spread, 1e-12):
                diverged, diverge_step = True, step
                if verbose:
                    print(f"DIVERGED at step {step}: output std {spread:.4e} "
                          f"against input std {data_spread:.4e}", flush=True)
                break
        # Two failure shapes: a non-finite loss, and one that has run away
        # while still being a number. The second is the one that produces a
        # confident wrong row, so it is checked too.
        if not math.isfinite(value) or value > 1e4 * max(abs(first_loss), 1e-12):
            diverged, diverge_step = True, step
            if verbose:
                print(f"DIVERGED at step {step}: loss {value:.4e} "
                      f"(started {first_loss:.4e})", flush=True)
            break

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        step += 1

        if step % log_every == 0 or step == 1:
            parts["step"] = step
            parts["lr"] = opt.param_groups[0]["lr"]
            history.append(parts)
            if verbose:
                print(f"step {step:>6}/{steps}  total {parts['total']:.4e}  "
                      f"mom {parts['moment']:.3e}  "
                      f"white {parts['whiteness']:.3e}  "
                      f"orth {parts['orthogonality']:.3e}  "
                      f"anch {parts['anchor']:.3e}", flush=True)

    model.eval()
    return {"history": history, "seconds": round(time.time() - t0, 1),
            "param_subset": param_subset, "n_trainable": n_train,
            "whiteness_axes": whiteness_axes,
            "steps": steps, "completed_steps": step,
            "diverged": diverged, "diverge_step": diverge_step,
            "first_loss": first_loss}
