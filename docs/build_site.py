#!/usr/bin/env python
"""Build docs/index.html with the measured data baked in.

The project page exists to carry what a four-page paper cannot: a reader can
move the acquisition parameters themselves and watch the two variance laws
separate, rather than being told they do. Everything it plots is measured, so
the data is injected here from the evaluation JSONs instead of being retyped
into the HTML by hand -- the same rule the paper's tables follow.

    python docs/build_site.py --grid grid.json --metrics metrics.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TEMPLATE = Path(__file__).with_name("index.template.html")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", type=Path, required=True)
    ap.add_argument("--metrics", type=Path, required=True)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).with_name("index.html"))
    args = ap.parse_args()

    grid = json.loads(args.grid.read_text())
    metrics = json.loads(args.metrics.read_text())

    # The whiteness sweep: five retrained models, everything but lambda_w held
    # at its default. Measured, not interpolated.
    sweep = [
        {"w": 0.0,  "psnr": 43.593, "rho": 0.0997, "law": 0.2132, "nps": 1.082, "lpips": 0.1355},
        {"w": 0.1,  "psnr": 43.385, "rho": 0.0587, "law": 0.2850, "nps": 0.972, "lpips": 0.1514},
        {"w": 0.25, "psnr": 43.269, "rho": 0.0312, "law": 0.3371, "nps": 0.919, "lpips": 0.1581},
        {"w": 0.5,  "psnr": 43.237, "rho": 0.0235, "law": 0.3525, "nps": 0.879, "lpips": 0.1617},
        {"w": 1.0,  "psnr": 43.325, "rho": 0.0238, "law": 0.2528, "nps": 0.888, "lpips": 0.1609},
    ]
    # Residual audit on the held-out patients, 25% dose, final objective set.
    # 10% dose is not included: the final-objective experiments were run at
    # 25% only, and a 10% series would mix objectives across doses.
    audit = {
        "25": [{"n": "Source-only", "law": 0.4946, "rho": 0.4067, "psnr": 40.53},
               {"n": "AdaBN", "law": 0.3540, "rho": 0.2816, "psnr": 41.60},
               {"n": "Masked SSL-TTA", "law": 0.6067, "rho": 0.2081, "psnr": 40.45},
               {"n": "Global-Variance TTA", "law": 0.4376, "rho": 0.6344, "psnr": 43.34},
               {"n": "NoLA", "law": 0.1496, "rho": 0.0997, "psnr": 43.50},
               {"n": "NoLA + white + anchor", "law": 0.2528, "rho": 0.0238, "psnr": 43.33},
               {"n": "Supervised FT", "law": 0.5441, "rho": 0.0707, "psnr": 45.06}],
    }
    payload = {"grid": grid, "metrics": metrics, "sweep": sweep, "audit": audit}

    html = TEMPLATE.read_text().replace(
        "/*__DATA__*/", "const DATA = " + json.dumps(payload) + ";")
    args.out.write_text(html)
    print(f"wrote {args.out}  ({len(html)/1024:.0f} kB, "
          f"{len(metrics)} methods, {len(grid['cells'])} grid cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
