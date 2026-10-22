"""The tinyflow CLI: ``toy``, and one sub-command group per experiment (``anime``)."""

from __future__ import annotations

from typing import Annotated

import jax
import typer

import data.toycardioid
from models import ToyFM, toyfm
from models.animefaces import cli as animefaces

app = typer.Typer(add_completion=False)
app.add_typer(animefaces.app, name="anime")


@app.command()
def toy(
    fpath: Annotated[str, typer.Argument(help="Synthetic data; see data.toycardioid")],
    outpath: Annotated[
        str, typer.Argument(help="Where to save the model")
    ] = "./runs/toymodel.eqx",
    n_epochs: int = 12,
    seed: int = 49,
):
    """Train the 2-D toy flow model."""
    dataset = data.toycardioid.cardioid_dataset(fpath, seed=seed - 7)
    key, k_model = jax.random.split(jax.random.key(seed))
    model = toyfm.train_on(key, ToyFM.from_key(k_model), dataset, n_epochs=n_epochs)
    model.save(outpath)


if __name__ == "__main__":
    app()
