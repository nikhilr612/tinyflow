# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`tinyflow` is a from-scratch implementation of generative flow matching models in the JAX
ecosystem (JAX + Equinox + Optax + diffrax + Grain), for unconditional image generation.
The README states the priorities explicitly: readability, simplicity, cleanliness, and
reproducibility. Prefer a clear, well-documented implementation over a clever one, and keep
the mathematical derivation in module/class docstrings the way `models/toyfm.py` and
`models/imagefm.py` do.

## Commands

Everything runs through `uv` (Python >= 3.13; `jax[cuda13]` is the default backend).

```sh
uv sync                                    # install (use --locked --all-extras --dev to match CI)
uv run main.py --help                      # CLI entry point (typer)
uv run ruff check                          # lint  (CI gate)
uv run ty check                            # type check (CI gate)
uv run python data/toycardioid.py          # regenerate toycardioid.npy in the cwd
uv run python generate_figures.py          # paper figures from ./runs into ./runs/figures
```

Training:

```sh
uv run main.py toy ./data/toycardioid.npy  # 2D toy model -> ./runs/toymodel.eqx + animated SVG
uv run main.py anime                       # image model; see --help for hparams
```

CI (`.github/workflows/main.yml`) runs only `ruff check` and `ty check` on push/PR to `master`.
**There is no test suite**; verify changes by running the toy pipeline, which is fast, before
touching anything image-side.

Ruff is configured in `pyproject.toml` with a broad rule set (google-convention `D`
docstrings on everything, `FA` future annotations, `PTH` pathlib over `os.path`, `TC`
TYPE_CHECKING blocks, `I` import sorting). `F722` is ignored because jaxtyping annotations
are string literals that Pyflakes misreads.

## Architecture

Two independent tracks share the flow-matching recipe but almost no code:

**Toy track** — `data/toycardioid.py` -> `models/toyfm.py` -> `viz.py`.
`toyfm.py` is deliberately standalone and deliberately carries an explicit batch dimension
through every function instead of using `vmap`, for pedagogical clarity. Its docstring says
so. Do not "fix" this by unifying it with `imagefm.py` or by vmapping it.

**Image track** — `data/animefaces.py` -> `models/unet.py` -> `models/imagefm.py`
-> `metrics.py`, driven by the `anime` command in `main.py`.

### The x_hat parameterization (the central non-obvious contract)

`UNet.__call__(x, t)` returns a **denoised image prediction `x_hat`**, not a velocity. The
velocity is derived wherever it is needed as `(x_hat - x_t) / max(1 - t, 1e-5)`. This
appears in three places that must stay consistent: `ImageFM.train_step`, `ImageFM.sample`,
and the standalone ODE term in `metrics.evaluate_fid`. `models/unet.py` documents why the
final bounded activation is omitted; the parameterization is what makes the auxiliary Sobel
edge loss in `ImageFM._edge_loss` possible, since it operates on `x_hat` against `x_1`
directly.

`ImageFM` is generic over the velocity network: anything implementing
`__call__(x: (H, W, C), t: scalar) -> (H, W, C)` works as `net_theta`. Keep it that way —
do not import `UNet` into `imagefm.py`.

### Array layout

All public boundaries — datasets, `ImageFM`, `metrics.py`, saved images — use `(H, W, C)`
with values in `[-1, 1]` (anime faces are 64x64x3). `models/unet.py` is the sole exception:
Equinox `Conv2d` is channels-first, so `UNet.__call__` transposes to `(C, H, W)` on entry and
back on exit, and every internal module in that file operates channels-first. Denormalize for
display with `(x + 1) * 127.5`.

### Serialization

Both `ToyFM` and `ImageFM` split weights from architecture: `eqx.tree_serialise_leaves` writes
the pytree to `path`, and a sibling `path + ".hparams"` JSON holds the constructor arguments.
Deserialization needs a skeleton of the right shape first — `ToyFM.load` builds one with
`eqx.filter_eval_shape`, while `ImageFM.load` takes a `skeleton_fn(key, **hparams)` callable
(see `_load_model` in `generate_figures.py`). `ImageFM.load` coerces every hparam value with
`int()`, so only integer hyperparameters can live in that dict.

### Runtime shape checking

Forward passes and loss functions are decorated `@jaxtyped(typechecker=beartype)` with
`jaxtyping` shape annotations. Where it is combined with `@staticmethod`, `jaxtyped` goes
outermost (see `ImageFM.train_step`). Keep the annotations accurate — they are the primary
documentation of tensor shapes in this codebase. `.env` sets `JAX_CHECK_TRACER_LEAKS=on`.

### Data pipelines and caching

Both datasets are Grain pipelines. `data/animefaces.py` splits work into offline
preprocessing (`preprocess_all` loads PNGs from `./data/anime-faces/images`, rescales to
`[-1, 1]`, caches to `./.preprocessed/anime_faces.npy`) and online augmentation
(`RandomHorizontalFlip`, `ColorJitter` as `grain.transforms.RandomMap` steps running on
worker threads). Inception statistics for FID are cached separately at
`./.preprocessed/real_stats.npz` by `metrics.compute_real_stats`. Both caches are keyed only
by path — delete them by hand after changing preprocessing, or stale data will be reused
silently.

### Training loop

The loop is split in two. `models/imagefm.py:train_on_image` is only the optimisation loop:
it takes a `TrainConfig` (epochs, LR, loss weights, `t` distribution), updates
`model.net_theta` in place, and *yields* `(epoch, mean_loss)` once per epoch. It imports
nothing from `metrics` or `data`. `training.py:run` consumes that generator and owns every
bookkeeping concern via a `RunConfig`: checkpoint to `outpath` every epoch, sample PNGs
from fixed noise (`sample.png` + `sample_epoch_NNNN.png`), FID with `fidax` every
`eval_every` epochs, `best_model.eqx`, `losses.json` (what `generate_figures.py` reads),
and early stopping — consecutive FID degradations beyond a 1% tolerance against
`early_stop_patience` — which is just `break`ing out of the generator. Keep new
bookkeeping in `training.py`; keep `imagefm.py` to the maths.

## Known deviations

`UBlock` in `models/unet.py` uses non-standard skip connections: they run from pre-encoder to
post-upsample (circumventing one encoder block) rather than the conventional post-encoder to
pre-decoder. This is documented in the class docstring and carries a `TODO(n)`. It is a known
open issue, not an accident — do not silently change it while working on something else.
