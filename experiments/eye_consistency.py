"""Left/right eye colour agreement of generated samples ("heterochromia rate").

Eye colour agreement is a long-range property no per-pixel loss can see: the
two irises are ~25 px apart.  This measures it directly.  The dataset-mean
eye mask (channel 1 of the semantic masks) is split at the vertical midline
into a left and a right region; for each image the mean chroma ``(Cb, Cr)``
of each region is taken and the two are compared.  Reported per checkpoint:

* ``mean dist``: mean Euclidean distance between the two eyes' mean chroma;
* ``hetero %``: fraction of images whose distance exceeds the real data's
  95th percentile -- i.e. the rate of odd-eyed faces beyond what the data
  itself contains.

Real data (two disjoint halves) gives the reference and its noise.

Usage::

    uv run python experiments/eye_consistency.py name=path.eqx [name=path ...]
        [--n 256] [--seed 0]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax  # noqa: E402
import numpy as np  # noqa: E402
import typer  # noqa: E402

from data.animefaces import load_masks, preprocess_all  # noqa: E402
from models.imagefm import EYE_CHANNEL, ImageFM  # noqa: E402
from models.unet import UNet  # noqa: E402


def chroma(x: np.ndarray) -> np.ndarray:
    """BT.601 ``(Cb, Cr)`` of an RGB array in [-1, 1], shape ``(..., 2)``."""
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return np.stack([0.5 * (b - y) / (1 - 0.114), 0.5 * (r - y) / (1 - 0.299)], -1)


def eye_distance(x: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Per-image distance between mean chroma of the left and right eye regions."""
    c = chroma(x)  # (N, H, W, 2)
    ml = (c * left[..., None]).sum((1, 2)) / left.sum()
    mr = (c * right[..., None]).sum((1, 2)) / right.sum()
    return np.linalg.norm(ml - mr, axis=-1)


def main(arms: list[str], n: int = 256, seed: int = 0):
    """Print one row per real half and per ``name=path`` checkpoint."""
    arr = preprocess_all("./data/anime-faces")
    m = load_masks()[..., EYE_CHANNEL].astype(np.float32).mean(0) / 255.0
    left, right = m.copy(), m.copy()
    left[:, m.shape[1] // 2 :] = 0
    right[:, : m.shape[1] // 2] = 0

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(arr), 2 * n, replace=False)
    d_real = eye_distance(arr[idx[:n]], left, right)
    d_real2 = eye_distance(arr[idx[n:]], left, right)
    thresh = float(np.quantile(d_real, 0.95))
    rows = [("real", d_real), ("real (2nd half)", d_real2)]

    noise = jax.random.normal(jax.random.key(seed), (n, 64, 64, 3))
    for arm in arms:
        name, path = arm.split("=", 1)
        model = ImageFM.load(path, lambda key, **hp: UNet(**hp, key=key))
        gen = np.clip(np.asarray(model.generate(noise)), -1, 1)
        rows.append((name, eye_distance(gen, left, right)))

    print(f"threshold (real p95) = {thresh:.4f}\n")
    print(f"{'':>20}{'mean dist':>11}{'median':>9}{'hetero %':>10}")
    for name, d in rows:
        print(
            f"{name:>20}{d.mean():>11.4f}{np.median(d):>9.4f}"
            f"{100 * (d > thresh).mean():>10.1f}"
        )


if __name__ == "__main__":
    typer.run(main)
