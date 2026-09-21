"""The tinyflow CLI: ``toy``, ``anime``, ``distil``, ``showcase``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import jax
import typer

import data.animefaces as faces
import data.toycardioid
from models import ToyFM, toyfm
from models.animefaces.flow import ImageFM, TrainConfig
from models.animefaces.unet import UNet
from viz import create_animation

app = typer.Typer(add_completion=False)


@app.command()
def toy(
    fpath: Annotated[str, typer.Argument(help="Synthetic data; see data.toycardioid")],
    outpath: Annotated[
        str, typer.Argument(help="Where to save the model")
    ] = "./runs/toymodel.eqx",
    n_epochs: int = 12,
    seed: int = 49,
):
    """Train the 2-D toy flow model and animate its samples."""
    dataset = data.toycardioid.cardioid_dataset(fpath, seed=seed - 7)
    key, k1, k2 = jax.random.split(jax.random.key(seed), 3)
    model = toyfm.train_on(key, ToyFM.from_key(k1), dataset, n_epochs=n_epochs)
    points = model.sample(16, k2, jax.numpy.linspace(0, 1, 20))
    create_animation(fpath, points, outpath + ".svg")
    model.save(outpath)


@app.command()
def anime(
    outdir: Annotated[
        str, typer.Argument(help="Run directory: model.eqx, losses.json, samples")
    ] = "./runs/anime",
    n_epochs: int = 125,
    base_channels: int = 64,
    batch_size: int = 128,
    seed: int = 49,
):
    """Train the layout-conditioned anime-face model (the recipe in animefaces.flow)."""
    from models.animefaces import train as training

    images, masks = faces.load_images(), faces.load_masks()
    keep = faces.curated_indices()
    dataset, batches = faces.make_dataset(images[keep], masks[keep], batch_size, seed)
    hparams = {"base_channels": base_channels}
    key, k_net = jax.random.split(jax.random.key(seed))
    model = ImageFM(UNet(key=k_net, **hparams), hparams)
    training.run(
        key, model, dataset, batches, TrainConfig(n_epochs=n_epochs), Path(outdir)
    )


@app.command()
def distil(
    checkpoint: Annotated[str, typer.Argument(help="A trained model.eqx")],
    outdir: Annotated[
        str, typer.Argument(help="Where the flow map and its ONNX export go")
    ] = "./runs/distil",
    n_pairs: int = 100_000,
    n_epochs: int = 40,
    seed: int = 0,
):
    """Distil the sampler into a one/two-jump flow map and export it to ONNX."""
    from models.animefaces import distill

    model = ImageFM.load(checkpoint, UNet)
    pairs = distill.teacher_pairs(model, n_pairs, seed)
    distill.train_map(model, pairs, Path(outdir), n_epochs, seed=seed)
    distill.export_onnx(model, Path(outdir) / "sampler_2jump.onnx")


@app.command()
def showcase(
    checkpoint: Annotated[
        str, typer.Argument(help="A model.eqx (trained, or distilled)")
    ],
    outdir: Annotated[str, typer.Argument(help="Where the figures go")] = "",
    n_jumps: Annotated[
        int, typer.Option(help="0: the ODE sampler; 1 or 2: the distilled map")
    ] = 0,
    seed: int = 0,
):
    """Sample grid and layout-to-image pairs from a checkpoint."""
    from models.animefaces import showcase as sc

    model = ImageFM.load(checkpoint, UNet)
    sc.showcase(
        model, Path(outdir or str(Path(checkpoint).parent / "figures")), n_jumps, seed
    )


if __name__ == "__main__":
    app()
