"""The denoising backbone, shared by the source model and every adapted copy.

RESIDUAL HEAD. The network predicts the noise r and returns y - r. Two reasons,
both about NoLA rather than about denoising quality. The adaptation losses are
all statements about the residual - its conditional variance, its whiteness,
its orthogonality to the estimate - so predicting r directly means the losses
read a network output instead of a difference of two large, nearly equal
numbers. And at initialisation a near-zero residual head is the identity, which
is the right prior for a model being moved to a system it has never seen.

BATCH NORM, DELIBERATELY. GroupNorm would be the reflexive choice for a small
model, and the auditor in SinoAudit uses it. Here BatchNorm is required by the
experimental design: AdaBN is a Block-A baseline, and AdaBN is precisely the
act of recomputing BatchNorm's running statistics on target data. A backbone
without BatchNorm cannot host that baseline, and dropping the baseline would
remove the cheapest competitor NoLA has to beat. Batches are 32 patches or
more, so the usual small-batch objection does not apply.

FULLY CONVOLUTIONAL. Trained on 128x128 patches of 1000x513 source sinograms
and applied to whole 1152x736 target sinograms, with no resizing anywhere.
Neither spatial size is a multiple of the downsampling factor, so the input is
reflection-padded up and cropped back; interpolating instead would shift the
sinogram by a fraction of a detector bin, which is a geometry error dressed up
as a resampling convenience.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(kind: str, channels: int) -> nn.Module:
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    if kind == "group":
        return nn.GroupNorm(min(8, channels), channels)
    raise ValueError(f"unknown norm {kind!r}")


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, norm: str = "batch"):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            _norm(norm, cout),
            nn.SiLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            _norm(norm, cout),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.body(x)


class ResidualUNet(nn.Module):
    """y -> y - r. ``base=48, depth=4`` gives about 22M parameters."""

    def __init__(self, base: int = 48, depth: int = 4, norm: str = "batch",
                 in_channels: int = 1):
        super().__init__()
        self.depth = depth
        widths = [base * 2 ** i for i in range(depth)]

        self.encoders = nn.ModuleList()
        cin = in_channels
        for w in widths:
            self.encoders.append(ConvBlock(cin, w, norm))
            cin = w
        self.bottleneck = ConvBlock(widths[-1], widths[-1] * 2, norm)

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        cin = widths[-1] * 2
        for w in reversed(widths):
            self.ups.append(nn.ConvTranspose2d(cin, w, 2, stride=2))
            self.decoders.append(ConvBlock(w * 2, w, norm))
            cin = w
        self.head = nn.Conv2d(widths[0], 1, 1)
        self.pool = nn.MaxPool2d(2)

        # Start as the identity: zero head means zero residual means y -> y.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        factor = 2 ** self.depth
        ph = (factor - h % factor) % factor
        pw = (factor - w % factor) % factor
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")

        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            x = dec(torch.cat([x, skip], dim=1))
        return self.head(x)[..., :h, :w]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.residual(x)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def count_norm_layers(model: nn.Module) -> int:
    return sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d))
