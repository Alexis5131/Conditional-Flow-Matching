"""Wrapper around ``clean-fid`` to compute FID against CIFAR-10 train stats.

We compute FID in a sample-folder mode: the generator dumps PNG images to a
temp folder, then ``clean_fid.fid.compute_fid(folder, dataset_name="cifar10",
dataset_split="train", mode="clean")`` does the rest, using the cached
reference statistics shipped by the library.

Samples must be in [0, 1] float tensors, shape (N, 3, H, W). They are clamped
and converted to uint8 PNGs (no anti-aliasing) before scoring.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor


def _save_batch_as_png(batch: Tensor, start_idx: int, folder: Path) -> None:
    """Save a (B, 3, H, W) float tensor in [0, 1] as PNG files."""
    batch = batch.clamp(0.0, 1.0).mul(255).round().to(torch.uint8).cpu()
    batch = batch.permute(0, 2, 3, 1).numpy()
    for i, arr in enumerate(batch):
        Image.fromarray(arr).save(folder / f"{start_idx + i:07d}.png")


def dump_samples(samples: Iterator[Tensor], folder: Path) -> int:
    """Iterate over batches of samples, dump them, return total count."""
    folder.mkdir(parents=True, exist_ok=True)
    n = 0
    for batch in samples:
        _save_batch_as_png(batch, n, folder)
        n += batch.size(0)
    return n


def compute_fid_cifar10(
    sample_iter: Iterator[Tensor],
    *,
    n_samples: int,
    workdir: Path | None = None,
    keep_samples: bool = False,
) -> float:
    """Compute FID between ``sample_iter`` and the CIFAR-10 train set.

    ``sample_iter`` must yield enough batches to reach ``n_samples`` images
    (extras are ignored). Uses ``clean-fid`` so the score is comparable to
    other papers using the same library.
    """
    from cleanfid import fid

    tmp_ctx = tempfile.TemporaryDirectory() if workdir is None else None
    folder = Path(workdir if workdir is not None else tmp_ctx.name)
    folder.mkdir(parents=True, exist_ok=True)
    try:
        produced = 0

        def _capped() -> Iterator[Tensor]:
            nonlocal produced
            for batch in sample_iter:
                remaining = n_samples - produced
                if remaining <= 0:
                    return
                if batch.size(0) > remaining:
                    batch = batch[:remaining]
                produced += batch.size(0)
                yield batch

        total = dump_samples(_capped(), folder)
        if total < n_samples:
            raise RuntimeError(f"Only {total} samples produced, expected {n_samples}.")
        score = fid.compute_fid(
            str(folder),
            dataset_name="cifar10",
            dataset_split="train",
            mode="clean",
            dataset_res=32,
        )
        return float(score)
    finally:
        if tmp_ctx is not None and not keep_samples:
            tmp_ctx.cleanup()
        elif workdir is not None and not keep_samples:
            for p in folder.glob("*.png"):
                os.remove(p)
