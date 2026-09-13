"""Dump dataset images next to the auxiliary-loss feature maps computed on them.

Writes to ``./.inspect/`` (git-ignored), for eyeballing what the Sobel edge
loss and the dark-line loss in ``models/imagefm.py`` actually respond to.
For each of ``n`` random dataset images the script saves

* ``<i>_image.png``   the image,
* ``<i>_sobel.png``   ``sqrt(gx^2 + gy^2)`` averaged over RGB (``ImageFM._sobel``),
* ``<i>_ink.png``     the soft-closing ink map (``ImageFM._ink``),
* ``<i>_ink_box.png`` the old 3x3 box-mean top-hat, for comparison,

plus ``grid.png`` with one row per image and those four columns.  Maps are
normalised to their 99.5th percentile over the whole batch so that panels are
comparable across images.

Usage:  ``uv run python inspect_losses.py [--n 16] [--seed 0] [--beta 10]``
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import PIL.Image as Pilimage
import typer

from data.animefaces import preprocess_all, to_uint8
from models.imagefm import ImageFM

OUTDIR = Path("./.inspect")


def _box_ink(x: jnp.ndarray) -> jnp.ndarray:
    """The pre-soft-closing ink map ``relu(box3(L) - L)``, kept for comparison."""
    luma = x @ jnp.array([0.299, 0.587, 0.114], dtype=jnp.float32)
    padded = jnp.pad(luma, ((0, 0), (1, 1), (1, 1)), mode="edge")
    h, w = luma.shape[1:]
    window = jnp.stack(
        [padded[:, i : i + h, j : j + w] for i in range(3) for j in range(3)]
    )
    return jax.nn.relu(window.mean(axis=0) - luma)


def _to_gray_png(m: np.ndarray, scale: float) -> np.ndarray:
    """Map a non-negative array to uint8 with ``scale`` -> 255, clipped."""
    return np.clip(m / max(scale, 1e-6) * 255, 0, 255).astype(np.uint8)


def main(n: int = 16, seed: int = 0, beta: float = 10.0, scale: int = 3):
    """Write the images, maps, and a summary grid to ``./.inspect/``."""
    arr = preprocess_all("./data/anime-faces")
    idx = np.sort(np.random.default_rng(seed).choice(len(arr), n, replace=False))
    x = jnp.asarray(arr[idx])  # (n, H, W, C) in [-1, 1]

    gx, gy = ImageFM._sobel(x)
    maps = {
        "sobel": np.asarray(jnp.sqrt(gx**2 + gy**2).mean(axis=-1)),
        "ink": np.asarray(ImageFM._ink(x, beta)),
        "ink_box": np.asarray(_box_ink(x)),
    }
    scales = {k: float(np.quantile(v, 0.995)) for k, v in maps.items()}

    OUTDIR.mkdir(exist_ok=True)
    rows = []
    for row, image in enumerate(np.asarray(x)):
        panels = [to_uint8(image)]
        Pilimage.fromarray(panels[0]).save(OUTDIR / f"{row:02d}_image.png")
        for name, m in maps.items():
            gray = _to_gray_png(m[row], scales[name])
            Pilimage.fromarray(gray).save(OUTDIR / f"{row:02d}_{name}.png")
            panels.append(np.repeat(gray[..., None], 3, axis=-1))
        rows.append(np.concatenate(panels, axis=1))
    grid = np.concatenate(rows, axis=0)
    big = Pilimage.fromarray(grid).resize(
        (grid.shape[1] * scale, grid.shape[0] * scale), Pilimage.Resampling.NEAREST
    )
    big.save(OUTDIR / "grid.png")

    for name, m in maps.items():
        print(
            f"{name:>8}: zeros {(m <= 1e-6).mean():5.1%}  mean {m.mean():.3f}  "
            f"p99.5 {scales[name]:.3f}"
        )
    print(
        f"wrote {n} images x {len(maps)} maps to {OUTDIR}/ ; grid.png columns: "
        f"image | {' | '.join(maps)}"
    )


if __name__ == "__main__":
    typer.run(main)
