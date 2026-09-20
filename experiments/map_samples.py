"""Sample grid from a flow-map checkpoint: 1 / 2 / 4 jumps against the ODE sampler.

Same noise and the same prior layouts in every row, so the rows differ only
in the sampler.  Runs on the CPU by default (``JAX_PLATFORMS=cpu``) so it can
be used while the GPU is training.

Usage::

    uv run python experiments/map_samples.py runs/mf_rp_100/model.eqx [--n 8]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import PIL.Image as PilImage  # noqa: E402
import typer  # noqa: E402
from PIL import ImageDraw  # noqa: E402

from data.animefaces import to_uint8  # noqa: E402
from data.layouts import LayoutPrior  # noqa: E402
from models.imagefm import ImageFM  # noqa: E402
from models.unet import UNet  # noqa: E402


def main(checkpoint: str, n: int = 8, seed: int = 7, out: str = "", jumps: str = "1,2"):
    """Write ``<run dir>/map_samples.png``: one row per jump count, then Dopri5-16.

    ``--jumps 1,2,4`` etc.  A distilled map (``experiments/distill_map.py``) is
    trained only on the jumps 0 -> 1, 0 -> 1/2, 1/2 -> 1, so anything else --
    including the ODE row, which uses the untrained diagonal -- is meaningless
    for it; pass ``--jumps 1,2`` and ignore the last row.
    """
    model = ImageFM.load(checkpoint, UNet.from_hparams)
    masks = None
    if model.cond_channels:
        masks = jnp.asarray(
            LayoutPrior.load().sample_masks(n, seed=seed)[..., : model.cond_channels]
        )
    noise = jax.random.normal(jax.random.key(seed), (n, 64, 64, 3))
    rows = [
        (f"{k} jumps ({k} NFE)", model.generate_map(noise, masks, k))
        for k in (int(j) for j in jumps.split(","))
    ]
    model.n_steps = 16
    rows.append(("dopri5-16 (~96 NFE)", model.generate(noise, masks)))

    w, pad = 64, 14
    im = PilImage.new("RGB", (n * w, len(rows) * (w + pad)), "white")
    draw = ImageDraw.Draw(im)
    for r, (name, arr) in enumerate(rows):
        y0 = r * (w + pad)
        draw.text((2, y0), name, fill=(0, 0, 0))
        for c in range(n):
            tile = to_uint8(np.clip(np.asarray(arr[c]), -1, 1))
            im.paste(PilImage.fromarray(tile), (c * w, y0 + pad))
    im = im.resize((im.width * 2, im.height * 2), PilImage.Resampling.NEAREST)
    out = out or str(Path(checkpoint).parent / "map_samples.png")
    im.save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    typer.run(main)
