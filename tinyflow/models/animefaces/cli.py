"""The anime-face experiment's commands: ``train``, ``distil``, ``showcase``.

Mounted by ``main.py`` as ``main.py anime ...``.  Each command only wires the
pieces together: the data (``data/``), the network (``unet.py``), the maths
(``flow.py``) and the shared loops (``trainer.py``, ``distil.py``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import jax
import numpy as np
import typer
from tqdm import tqdm

import tinyflow.data.animefaces as faces
from tinyflow import distil, trainer
from tinyflow.data.layouts import LayoutPrior
from tinyflow.models.animefaces import showcase as sc
from tinyflow.models.animefaces.flow import (
    T_END,
    FaceMapTrainer,
    FlowTrainer,
    ImageFM,
    jump_step,
    solve,
)
from tinyflow.models.animefaces.unet import UNet

app = typer.Typer(add_completion=False, help="Layout-conditioned 64x64 anime faces.")

SHAPE = (64, 64, 3)
SCHEDULE_EPOCHS = 200  # the cosine's length; the run stops early (FlowTrainer)


@app.command()
def train(
    outdir: Annotated[
        str, typer.Argument(help="Run directory: model.eqx, losses.json, samples")
    ] = "./runs/anime",
    n_epochs: int = 125,
    base_channels: int = 64,
    batch_size: int = 128,
    seed: int = 49,
):
    """Train the layout-conditioned anime-face model (the recipe in flow.py)."""
    images, masks = faces.load_images(), faces.load_masks()
    keep = faces.curated_indices()
    dataset, batches = faces.make_dataset(images[keep], masks[keep], batch_size, seed)
    hparams = {"base_channels": base_channels}
    key, k_net = jax.random.split(jax.random.key(seed))
    model = ImageFM(UNet(key=k_net, **hparams), hparams)

    # Eight fixed (noise, layout) pairs: the same latents rendered every epoch.
    key, k_noise = jax.random.split(key)
    noise = jax.random.normal(k_noise, (8, *SHAPE))
    layouts = jax.numpy.asarray(LayoutPrior.load().sample_masks(8, seed=0))

    fits = FlowTrainer(schedule_steps=SCHEDULE_EPOCHS * batches).fit(
        model.net, dataset, n_epochs, batches, key
    )
    trainer.run(fits, model, Path(outdir), lambda m: m.sample(noise, layouts), n_epochs)


@app.command(name="distil")
def distil_(
    checkpoint: Annotated[
        str, typer.Argument(help="A trained model.eqx")
    ] = "./runs/anime/model.eqx",
    outdir: Annotated[
        str, typer.Argument(help="Where the flow map and its ONNX export go")
    ] = "./runs/distil",
    n_pairs: int = 100_000,
    n_epochs: int = 40,
    seed: int = 0,
):
    """Distil the sampler into a one/two-jump flow map and export it to ONNX."""
    model = ImageFM.load(checkpoint, UNet)

    def teacher(x_0, masks):
        states = solve(model.net, x_0, masks, (T_END / 2, T_END))
        return states[:, 0], states[:, 1]

    # The source is Gaussian noise; one prior layout per pair.
    masks = LayoutPrior.load().sample_masks(n_pairs, seed=seed)
    teacher_batch = 250

    def inputs():
        key = jax.random.key(seed)
        for i in range(0, n_pairs, teacher_batch):
            key, sk = jax.random.split(key)
            m = jax.numpy.asarray(masks[i : i + teacher_batch])
            yield jax.random.normal(sk, (len(m), *SHAPE)), m

    n_teacher = -(-n_pairs // teacher_batch)
    pairs = distil.teacher_pairs(
        teacher, tqdm(inputs(), total=n_teacher, desc="teacher"), cond=masks
    )

    # Each tile of the sample row: one jump above, two jumps below.
    noise = jax.random.normal(jax.random.key(seed), (8, *SHAPE))
    layouts = jax.numpy.asarray(masks[:8])

    def sample_fn(m: ImageFM):
        rows = [np.asarray(m.sample(noise, layouts, k)) for k in (1, 2)]
        return np.concatenate(rows, axis=1)

    jumps, batches = distil.jump_batches(pairs, batch_size=128, seed=seed)
    fits = FaceMapTrainer(schedule_steps=n_epochs * batches).fit(
        model.net, jumps, n_epochs, batches, jax.random.key(seed)
    )
    trainer.run(fits, model, Path(outdir), sample_fn, n_epochs)

    spec = jax.ShapeDtypeStruct(SHAPE, jax.numpy.float32)
    distil.export_onnx(
        lambda x, m: distil.jump(jump_step, model.net, x, m, 2),
        [spec, spec],
        Path(outdir) / "sampler_2jump.onnx",
        ["noise", "masks"],
        ["image"],
    )


@app.command()
def showcase(
    checkpoint: Annotated[
        str, typer.Argument(help="A model.eqx (trained, or distilled)")
    ] = "./runs/anime/model.eqx",
    outdir: Annotated[str, typer.Argument(help="Where the figures go")] = "",
    n_jumps: Annotated[
        int, typer.Option(help="0: the ODE sampler; 1 or 2: the distilled map")
    ] = 0,
    seed: int = 0,
):
    """Sample grid and layout-to-image pairs from a checkpoint."""
    model = ImageFM.load(checkpoint, UNet)
    sc.showcase(
        model, Path(outdir or str(Path(checkpoint).parent / "figures")), n_jumps, seed
    )
