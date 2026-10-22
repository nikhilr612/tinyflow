# tinyflow
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22876650.svg)](https://doi.org/10.5281/zenodo.22876650)

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
just toy                                              # 2-D toy model -> runs/toymodel.eqx
just anime train                                      # train the anime-face model -> runs/anime/ (~1h on one 4090)
just anime distil                                     # distil it into a 2-evaluation sampler + ONNX -> runs/distil/
just anime showcase                                   # sample figures from a checkpoint (ODE sampler by default)
just anime showcase ./runs/distil/model.eqx --n-jumps 2   # ... or the distilled map, two jumps
just check                                            # ruff + ty, the CI gates
```

Each experiment keeps its recipes in its own directory (`models/animefaces/justfile`):
from there the same recipes run without the prefix (`just train`), and paths stay
relative to the repo root.  Extra arguments go straight to the command, e.g.
`just anime train --n-epochs 50`; `uv run main.py anime --help` (and `--help` on
any subcommand) documents them.

The training loop is shared: `trainer.py` (optimiser, schedule, weight average,
checkpoints) and `distil.py` (flow-map distillation) are generic, and an
experiment supplies only its loss and sampler.
