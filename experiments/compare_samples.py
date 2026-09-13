"""Sample grid across checkpoints from identical noise, one labelled row each.

Usage::

    uv run python experiments/compare_samples.py OUT.png name=path.eqx [name=path ...]
        [--n 8] [--seed 0] [--scale 2]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax  # noqa: E402
import numpy as np  # noqa: E402
import PIL.Image as Pilimage  # noqa: E402
import typer  # noqa: E402
from PIL import ImageDraw, ImageFont  # noqa: E402

from data.animefaces import to_uint8  # noqa: E402
from models.imagefm import ImageFM  # noqa: E402
from models.unet import UNet  # noqa: E402


def main(out: str, arms: list[str], n: int = 8, seed: int = 0, scale: int = 2):
    """Write ``out``: one row per ``name=path`` arm, ``n`` samples from shared noise."""
    noise = jax.random.normal(jax.random.key(seed), (n, 64, 64, 3))
    rows, names = [], []
    for arm in arms:
        name, path = arm.split("=", 1)
        model = ImageFM.load(path, lambda key, **hp: UNet(**hp, key=key))
        imgs = to_uint8(model.generate(noise))
        rows.append(np.concatenate(list(imgs), axis=1))
        names.append(name)
    grid = np.concatenate(rows, axis=0)
    body = Pilimage.fromarray(grid).resize(
        (grid.shape[1] * scale, grid.shape[0] * scale), Pilimage.Resampling.NEAREST
    )
    label_w = 110
    img = Pilimage.new("RGB", (body.width + label_w, body.height), (255, 255, 255))
    img.paste(body, (label_w, 0))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=12)
    for i, name in enumerate(names):
        draw.text((4, i * 64 * scale + 4), name, fill=(0, 0, 0), font=font)
    img.save(out)
    print(f"wrote {out}: rows {names}")


if __name__ == "__main__":
    typer.run(main)
