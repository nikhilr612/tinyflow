# tinyflow

A short and sweet implementation of flow matching models in the JAX ecosystem,
with an emphasis on readability, simplicity, cleanliness and reproducibility.

Two tracks share the flow-matching recipe:

- **Toy** (`data/toycardioid.py`, `models/toyfm.py`): a standalone,
  instructive implementation of the plain flow matching recipe on a 2-D
  dataset, seconds to train, no dependency on the image track.
- **Anime faces** (`data/animefaces.py`, `data/layouts.py`,
  `models/animefaces/`): 64x64 unconditional generation with layout-conditioned
  region pooling and a distilled 2-evaluation sampler. See `data/README.md`
  for the dataset and its provenance.

## Get started

This project uses [`uv`](https://astral.sh/uv) and [`just`](https://github.com/casey/just):

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
cargo install just   # or: brew install just / apt install just
```

Then, from the repo root:

```sh
just setup   # locked env + the anime-faces dataset
just         # list every recipe
```

[OPTIONAL] install a platform-specific JAX runtime (e.g. TPU) after `just setup`;
`jax[cuda13]` is the default and JAX auto-selects among whichever are installed:

```sh
uv add jax[tpu]
```

## Recipes

```sh
just toy                                  # 2-D toy model -> runs/toymodel.eqx + an animated SVG
just anime                                # train the anime-face model -> runs/anime/model.eqx (~1h on one 4090)
just distil                               # distill it into a 2-evaluation sampler + ONNX -> runs/distil/
just showcase                             # sample figures from a checkpoint (ODE sampler by default)
just showcase ./runs/distil/model.eqx 2   # ... or the distilled map, two jumps
just check                                # ruff + ty, the CI gates
```

`uv run main.py --help` (and `--help` on any subcommand) documents the
hyperparameters directly.
