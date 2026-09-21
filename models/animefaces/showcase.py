"""Showcase figures from a checkpoint: a sample grid and layout-to-image pairs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import numpy as np
from PIL import Image

from data.animefaces import to_uint8
from data.layouts import LayoutPrior

if TYPE_CHECKING:
    from pathlib import Path

    from models.animefaces.flow import ImageFM


def mask_rgb(masks: np.ndarray) -> np.ndarray:
    """Face/eyes/mouth masks ``(N, H, W, 3)`` as images in ``[-1, 1]``."""
    face, eyes, mouth = masks[..., :1], masks[..., 1:2], masks[..., 2:3]
    rgb = face * 0.45 + eyes * [0.0, 0.5, 0.6] + mouth * [0.6, -0.2, -0.2]
    return np.clip(rgb, 0, 1) * 2 - 1


def tile(rows: list[np.ndarray], scale: int = 2) -> Image.Image:
    """Stack rows of ``(N, H, W, 3)`` images into one, nearest-neighbour upscaled."""
    grid = np.concatenate([np.concatenate(list(r), axis=1) for r in rows], axis=0)
    img = Image.fromarray(to_uint8(grid))
    return img.resize((img.width * scale, img.height * scale), Image.Resampling.NEAREST)


def showcase(model: ImageFM, outdir: Path, n_jumps: int = 0, seed: int = 0) -> None:
    """Write ``samples.png`` (a 6x8 grid) and ``layout_to_image.png``.

    ``n_jumps > 0`` samples with the distilled flow map instead of the ODE solver.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    prior = LayoutPrior.load()
    k1, k2 = jax.random.split(jax.random.key(seed))

    def sample(key, n, seed_):
        masks = prior.sample_masks(n, seed=seed_)
        noise = jax.random.normal(key, (n, 64, 64, 3))
        m = jax.numpy.asarray(masks)
        out = model.jump(noise, m, n_jumps) if n_jumps else model.generate(noise, m)
        return masks, np.asarray(out)

    _, grid = sample(k1, 48, seed)
    tile([grid[i : i + 8] for i in range(0, 48, 8)]).save(outdir / "samples.png")
    masks, imgs = sample(k2, 8, seed + 1)
    tile([mask_rgb(masks), imgs]).save(outdir / "layout_to_image.png")
