"""Main entry point for the tinyflow CLI."""

from pathlib import Path
from typing import Annotated

import jax
import PIL.Image as Pilimage
import typer

import data.animefaces
import data.toycardioid
import models.toyfm as toyfm
from data.animefaces import preprocess_all, to_uint8
from metrics import compute_real_stats
from models import ToyFM
from models.imagefm import ImageFM, train_on_image
from models.unet import UNet
from viz import create_animation

app = typer.Typer()


@app.command()
def toy(
    fpath: Annotated[
        str,
        typer.Argument(help="The path to synthetic data. Check data.toycardioid"),
    ],
    outpath: Annotated[
        str,
        typer.Argument(help="Output path to save the model, and test samples."),
    ] = "./runs/toymodel.eqx",
    n_epochs: Annotated[int, typer.Option(help="Number of epochs to train for")] = 12,
    seed: Annotated[int, typer.Option(help="Seed to use")] = 49,
):
    """Train a toy flow matching model on synthetic data."""
    dataset = data.toycardioid.cardioid_dataset(fpath, seed=seed - 7)
    key = jax.random.key(seed)
    key, sk1, sk2 = jax.random.split(key, num=3)
    model = toyfm.train_on(key, ToyFM.from_key(sk1), dataset, n_epochs=n_epochs)

    points = model.sample(16, sk2, jax.numpy.linspace(0, 1, 20))
    create_animation(fpath, points, outpath + ".svg")
    model.save(outpath)


@app.command()
def anime(
    outpath: str = "./runs/imagemodel.eqx",
    n_epochs: int = 100,
    batch_size: int = 128,
    seed: int = 49,
    base_channels: int = 32,
    time_embedding_dim: int = 128,
    n_blocks: int = 4,
    eval_every: int = 5,
    early_stop_patience: int = 3,
    n_steps: int = 64,
    edge_weight: float = 0.1,
    line_weight: float = 0.0,
):
    """Train a flow matching model on the anime faces dataset."""
    assert base_channels % 8 == 0, (
        f"base_channels={base_channels} must be divisible by 8"
    )
    assert time_embedding_dim % 2 == 0, (
        f"time_embedding_dim={time_embedding_dim} must be even"
    )

    arr = preprocess_all("./data/anime-faces")
    real_stats = compute_real_stats(arr, batch_size=batch_size)

    dataset, batches_per_epoch = data.animefaces.wrap_dataset(
        arr, batch_size=batch_size, seed=seed
    )
    key = jax.random.key(seed)
    key, sk1, sk2, sk3 = jax.random.split(key, num=4)
    unet = UNet(base_channels, time_embedding_dim, sk1, n_blocks, in_channels=3)
    model = ImageFM(
        unet,
        hparams={
            "base_channels": base_channels,
            "time_embedding_dim": time_embedding_dim,
            "n_blocks": n_blocks,
            "in_channels": 3,
        },
        n_steps=n_steps,
    )
    model = train_on_image(
        key,
        model,
        dataset,
        batches_per_epoch,
        n_epochs=n_epochs,
        outpath=outpath,
        eval_every=eval_every,
        real_stats=real_stats,
        early_stop_patience=early_stop_patience,
        edge_weight=edge_weight,
        line_weight=line_weight,
    )
    model.save(outpath)

    outdir = Path(outpath).parent
    x_0 = jax.random.normal(sk3, (1, 64, 64, 3))
    final_img = to_uint8(model.generate(x_0)[0])
    Pilimage.fromarray(final_img).save(str(outdir / "sample_final.png"))
    print(f"Saved model to {outpath}")
    print(f"Saved final sample to {outdir / 'sample_final.png'}")


if __name__ == "__main__":
    app()
