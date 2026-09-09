"""Datasets for the source training, and for unlabelled target adaptation.

TWO LOADERS, DELIBERATELY DIFFERENT IN WHAT THEY MAY READ
    ``SourceSinograms`` yields (noisy, clean) pairs. It is used only for
    Stage 0 on LoDoPaB, where a clean target is legitimate.

    ``TargetSinograms`` yields noisy patches and NOTHING ELSE. It refuses to
    open the ``clean_sinogram`` and ``sigma_*`` arrays even though they sit in
    the same npz, because those are functions of the clean signal and using
    them anywhere in the adaptation path is label leakage. The refusal is
    enforced in code rather than by discipline: the class never names those
    keys, and ``allow_labels=True`` exists solely for the oracle fine-tune
    baseline and the estimator-validation figure, which are supposed to see
    them.

PATCHES, NOT WHOLE SINOGRAMS, FOR TRAINING
    A 1000x513 sinogram is one sample with enormously correlated content.
    128x128 crops give a batch of 32 genuinely different noise realisations at
    a fraction of the memory, and the network is fully convolutional so
    nothing about the crop size is baked in. Validation uses whole sinograms,
    because that is the shape the model is deployed at and patch-edge effects
    would otherwise go unmeasured.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .splits import patient_dir
from .units import harmonise


def _worker_rng(seed: int):
    """A generator that differs per DataLoader worker.

    Workers are forked, so a generator built in ``__init__`` is copied intact
    and every worker draws the identical crop sequence. With eight workers that
    quietly reduces the effective patch diversity eightfold, and nothing in the
    loss curve looks wrong. Seeding from the worker id fixes it and stays
    reproducible.
    """
    info = torch.utils.data.get_worker_info()
    return np.random.default_rng(seed + 100003 * (info.id if info else 0))


class SourceSinograms(Dataset):
    """(noisy, clean) pairs from the consolidated LoDoPaB shards."""

    def __init__(self, root: Path, part: str = "train",
                 patch: int | None = 128, seed: int = 0):
        import h5py

        self.files = sorted(Path(root, part).glob(f"{_official(part)}_*.h5"))
        if not self.files:
            raise FileNotFoundError(f"no shards under {Path(root, part)}")
        self.patch = patch
        self._h5 = h5py
        self._handles: dict[int, object] = {}
        self.counts = []
        for f in self.files:
            with h5py.File(f, "r") as h:
                self.counts.append(h["observation"].shape[0])
        self.offsets = np.cumsum([0] + self.counts)
        self.seed = seed
        self._rng = None

    @property
    def rng(self):
        if self._rng is None:
            self._rng = _worker_rng(self.seed)
        return self._rng

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def _open(self, i: int):
        # Opened lazily and per worker: an h5py handle inherited across a fork
        # returns silently corrupt data, which is the classic way this fails.
        if i not in self._handles:
            self._handles[i] = self._h5.File(self.files[i], "r")
        return self._handles[i]

    def __getitem__(self, idx: int):
        shard = int(np.searchsorted(self.offsets, idx, side="right") - 1)
        off = idx - int(self.offsets[shard])
        h = self._open(shard)
        obs = h["observation"][off]
        clean = h["clean"][off]
        if self.patch:
            obs, clean = _random_crop_pair(obs, clean, self.patch, self.rng)
        return (torch.from_numpy(np.ascontiguousarray(obs))[None],
                torch.from_numpy(np.ascontiguousarray(clean))[None])


def _official(part: str) -> str:
    return {"train": "train", "calib": "validation", "test": "test"}[part]


def _random_crop_pair(a: np.ndarray, b: np.ndarray, size: int, rng):
    h, w = a.shape
    if h < size or w < size:
        return a, b
    i = int(rng.integers(0, h - size + 1))
    j = int(rng.integers(0, w - size + 1))
    return a[i:i + size, j:j + size], b[i:i + size, j:j + size]


class TargetSinograms(Dataset):
    """Unlabelled harmonised Mayo sinograms at one dose.

    ``allow_labels`` additionally returns the clean sinogram. It is False by
    default and every NoLA code path must leave it False; only the oracle
    fine-tune and the validation figures set it.
    """

    NOISY_KEY = "sino_{dose}"
    CLEAN_KEY = "clean_sinogram"

    def __init__(self, ready: Path, patients, dose: str = "25",
                 split: str = "train", patch: int | None = 128,
                 patches_per_item: int = 8,
                 slices=None, allow_labels: bool = False, seed: int = 0):
        self.files = []
        for p in patients:
            # split is accepted for interface stability; the directory is
            # resolved per patient because the on-disk layout predates the
            # logical split (see nola.splits.patient_dir).
            fs = sorted(patient_dir(ready, p).glob("slice_*.npz"))
            if slices is not None:
                keep = set(slices[p]) if isinstance(slices, dict) else set(slices)
                fs = [f for f in fs if int(f.stem.split("_")[1]) in keep]
            self.files.extend(fs)
        if not self.files:
            raise FileNotFoundError(f"no slices for {patients} under {ready}")
        self.dose = dose
        self.patch = patch
        # One npz read costs 3.4 MB even when only the noisy key is touched, so
        # drawing a single 128x128 patch from it wastes 99 percent of the read.
        # Several patches per opened file turn an I/O-bound loop into a
        # compute-bound one; they are correlated, which is why the batch is
        # assembled from several DIFFERENT slices as well.
        self.patches_per_item = max(1, patches_per_item) if patch else 1
        self.allow_labels = allow_labels
        self.seed = seed
        self._rng = None

    @property
    def rng(self):
        if self._rng is None:
            self._rng = _worker_rng(self.seed)
        return self._rng

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        with np.load(self.files[idx]) as z:
            noisy = harmonise(z[self.NOISY_KEY.format(dose=self.dose)])
            clean = harmonise(z[self.CLEAN_KEY]) if self.allow_labels else None

        if not self.patch:
            t = torch.from_numpy(np.ascontiguousarray(noisy))[None]
            if clean is None:
                return t
            return t, torch.from_numpy(np.ascontiguousarray(clean))[None]

        ns, cs = [], []
        for _ in range(self.patches_per_item):
            if clean is None:
                a = _random_crop_single(noisy, self.patch, self.rng)
                ns.append(np.ascontiguousarray(a))
            else:
                a, b = _random_crop_pair(noisy, clean, self.patch, self.rng)
                ns.append(np.ascontiguousarray(a))
                cs.append(np.ascontiguousarray(b))
        # (patches_per_item, 1, patch, patch); the training loop flattens the
        # leading two dimensions so the effective batch is items x patches.
        t = torch.from_numpy(np.stack(ns))[:, None]
        if clean is None:
            return t
        return t, torch.from_numpy(np.stack(cs))[:, None]


def _random_crop_single(a: np.ndarray, size: int, rng):
    h, w = a.shape
    if h < size or w < size:
        return a
    i = int(rng.integers(0, h - size + 1))
    j = int(rng.integers(0, w - size + 1))
    return a[i:i + size, j:j + size]
