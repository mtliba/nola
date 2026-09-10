#!/usr/bin/env python
"""Fetch the released checkpoints.

Two are published, and they are the ones every number in the paper was
produced from:

    theta_src.pt      the SOURCE denoiser, trained on 8,000 LoDoPaB slices.
                      This is the starting point for adaptation, and the model
                      the "source-only" row evaluates.

    nola_dose25.pt    theta_src after NoLA adaptation to the 25% dose target,
                      using unlabelled target sinograms only. The "NoLA" row.

Both are 67 MB and are checked against a SHA-256 recorded here, because a
truncated download of a torch checkpoint frequently loads without complaint
and then produces quietly wrong numbers.

    python scripts/download_weights.py                 # both, into checkpoints/
    python scripts/download_weights.py --only theta_src
    python scripts/download_weights.py --verify-only   # check what is on disk
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

RELEASE = "https://github.com/mtliba/nola/releases/download/v1.0.0"

WEIGHTS = {
    "theta_src": {
        "file": "theta_src.pt",
        "sha256": "9001a38742cc14f70d418b0f617eb766e30a41fded48d065e9c22382835f7d28",
        "what": "source denoiser, trained on LoDoPaB",
    },
    "nola_dose25": {
        "file": "nola_dose25.pt",
        "sha256": "b030ab1a9607f522bf88a532cf258320d98be97dd62d680cd6ab78b94ef9e2b6",
        "what": "adapted to the 25% dose target, no labels",
    },
}


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        with open(tmp, "wb") as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
                done += len(b)
                if total:
                    pct = 100 * done / total
                    print(f"\r      {done/1e6:6.1f} / {total/1e6:.1f} MB "
                          f"({pct:5.1f}%)", end="", file=sys.stderr)
        print("", file=sys.stderr)
    # Only becomes the real file once complete, so an interrupted download
    # cannot leave a plausible-looking truncated checkpoint behind.
    tmp.replace(dest)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent / "checkpoints")
    ap.add_argument("--only", choices=sorted(WEIGHTS))
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    names = [args.only] if args.only else list(WEIGHTS)
    bad = 0
    for n in names:
        spec = WEIGHTS[n]
        dest = args.out_dir / spec["file"]
        print(f"{spec['file']}  ({spec['what']})")

        if dest.is_file() and not args.force:
            got = sha256(dest)
            if got == spec["sha256"]:
                print("      already present, checksum OK")
                continue
            print(f"      checksum MISMATCH\n        expected {spec['sha256']}"
                  f"\n        got      {got}", file=sys.stderr)
            if args.verify_only:
                bad += 1
                continue
            print("      re-downloading")
        elif args.verify_only:
            print("      not present", file=sys.stderr)
            bad += 1
            continue

        url = f"{RELEASE}/{spec['file']}"
        try:
            fetch(url, dest)
        except urllib.error.HTTPError as e:
            print(f"      download failed: HTTP {e.code} for {url}\n"
                  f"      if the release is not published yet, train the source "
                  f"model with scripts/train_source.py", file=sys.stderr)
            bad += 1
            continue
        got = sha256(dest)
        if got != spec["sha256"]:
            print(f"      checksum MISMATCH after download; file kept at {dest}",
                  file=sys.stderr)
            bad += 1
        else:
            print(f"      OK -> {dest}")

    if bad:
        print(f"\n{bad} file(s) missing or corrupt", file=sys.stderr)
        return 1
    print("\nAll checkpoints present and verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
