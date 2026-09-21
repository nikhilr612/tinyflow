"""Run a training job: checkpoint every epoch, log the loss, save a sample.

``flow.train`` is only the optimisation loop; this module consumes
it and owns the bookkeeping.  Evaluation (FID, the iris-agreement metric) is
deliberately not here; it is run on the checkpoints this writes.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import jax
import numpy as np
from PIL import Image
from tqdm import tqdm

from data.animefaces import to_uint8
from data.layouts import LayoutPrior
from models.animefaces.flow import ImageFM, TrainConfig, train

if TYPE_CHECKING:
    from pathlib import Path


def save_grid(images: np.ndarray, path: Path) -> None:
    """Write ``(N, H, W, 3)`` images in ``[-1, 1]`` as one row of tiles."""
    Image.fromarray(to_uint8(np.concatenate(list(images), axis=1))).save(path)


def run(
    key: jax.Array,
    model: ImageFM,
    dataset,
    batches_per_epoch: int,
    cfg: TrainConfig,
    outdir: Path,
) -> ImageFM:
    """Train; after every epoch write ``model.eqx``, ``losses.json`` and a sample row.

    The sample row is eight fixed (noise, layout) pairs, so the series
    ``sample_epoch_*.png`` shows the generator's output for the same latents
    evolving over the run.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    key, k_noise = jax.random.split(key)
    noise = jax.random.normal(k_noise, (8, 64, 64, 3))
    masks = jax.numpy.asarray(LayoutPrior.load().sample_masks(8, seed=0))
    history = []
    for epoch, loss in tqdm(
        train(key, model, dataset, batches_per_epoch, cfg),
        total=cfg.n_epochs,
        desc="train",
    ):
        history.append({"epoch": epoch, "loss": loss})
        model.save(str(outdir / "model.eqx"))
        (outdir / "losses.json").write_text(json.dumps(history, indent=1))
        save_grid(
            np.asarray(model.generate(noise, masks)),
            outdir / f"sample_epoch_{epoch:04d}.png",
        )
    return model
