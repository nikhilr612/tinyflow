"""The tinyflow CLI: ``toy``, and one sub-command group per experiment.

``anime`` (layout-conditioned anime faces) and ``senbridge`` (radar to optical).
"""

from __future__ import annotations

from typing import Annotated

import jax
import typer

from tinyflow.data import toycardioid
from tinyflow.models import ToyFM, toyfm
from tinyflow.models.animefaces import cli as animefaces
from tinyflow.models.senbridge import cli as senbridge

app = typer.Typer(add_completion=False)
app.add_typer(animefaces.app, name="anime")
app.add_typer(senbridge.app, name="senbridge")


@app.command()
def toy(
    fpath: Annotated[
        str, typer.Argument(help="Synthetic data; see tinyflow.data.toycardioid")
    ],
    outpath: Annotated[
        str, typer.Argument(help="Where to save the model")
    ] = "./runs/toymodel.eqx",
    n_epochs: int = 12,
    seed: int = 49,
):
    """Train the 2-D toy flow model."""
    dataset = toycardioid.cardioid_dataset(fpath, seed=seed - 7)
    key, k_model = jax.random.split(jax.random.key(seed))
    model = toyfm.train_on(key, ToyFM.from_key(k_model), dataset, n_epochs=n_epochs)
    model.save(outpath)


if __name__ == "__main__":
    app()
