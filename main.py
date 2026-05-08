"""Main entry point for the tinyflow CLI."""

from typing import Annotated

import jax
import typer

import data.toycardioid
import models.toyfm as toyfm
from models import ToyFM
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


if __name__ == "__main__":
    app()
