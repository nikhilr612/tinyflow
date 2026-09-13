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

Everything runs through `uv` (Python >= 3.13; `jax[cuda13]` is the default backend);
the `justfile` at the root wraps the common loops (`just` lists them).

```sh
uv sync                                    # install (use --locked --all-extras --dev to match CI)
just check                                 # ruff check + ty check  (the CI gates)
uv run main.py --help                      # CLI entry point (typer)
uv run python data/toycardioid.py          # regenerate toycardioid.npy in the cwd
just paper NAME 300 50                     # train the current best recipe, evaluate, paper figures
just figures runs/NAME                     # paper figures + showcase page for a finished run
```

Training:

```sh
uv run main.py toy ./data/toycardioid.npy  # 2D toy model -> ./runs/toymodel.eqx + animated SVG
uv run main.py anime                       # image model; see --help for hparams
```

CI (`.github/workflows/main.yml`) runs only `ruff check` and `ty check` on push/PR to `master`.
**There is no test suite**; verify changes by running the toy pipeline, which is fast, before
touching anything image-side, and `just paper smoke 2 1 --base-channels 8 --time-embedding-dim 32`
for the full image pipeline (~2 min).

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

**Image track** — `data/animefaces.py` + `data/layouts.py` -> `models/unet.py` -> `models/imagefm.py`
-> `training.py` -> `metrics.py`, driven by the `anime` command in `main.py`.

### The x_hat parameterization (the central non-obvious contract)

`UNet.__call__(x, t)` returns a **denoised image prediction `x_hat`**, not a velocity. The
velocity is derived wherever it is needed as `(x_hat - x_t) / max(1 - t, denom_floor)`. This
appears in `ImageFM.train_step`, `ImageFM.velocity` (used by `_solve` for sampling) and the
guided sampler in `experiments/guidance.py`; keep them consistent. `models/unet.py` documents
why the final bounded activation is omitted.

The training loss is the plain flow-matching loss and **deliberately has no auxiliary
terms**. A campaign of edge / line / palette losses on `x_hat`, mask-weighted variants and
per-pixel re-weightings was measured to hurt or do nothing, for a reason derived in
`experiments/METHODS.md` (sections 1-2): the minimiser is `E[x_1 | x_t]`, such losses are
irreducible there, and `x_1`-dependent weights tilt the learned field. Do not re-add them;
priors about samples belong at sampling time (guidance) or in the architecture.

`ImageFM` is generic over the network: anything implementing
`__call__(x: (H, W, C_in), t: scalar) -> (H, W, C_out)` works as `net_theta`. Keep it that way —
do not import `UNet` into `imagefm.py`. Checkpoints are loaded with
`ImageFM.load(path, UNet.from_hparams)`; `from_hparams` ignores hparam keys from the archived
experiment branch.

### Layout conditioning and region pooling (the current best model)

The best model is a **pure conditional** one: `cond_channels=3` concatenates the semantic
masks (face, eyes, mouth hulls from `hysts/anime-face-detector` landmarks) plus an indicator
channel to `x_t` (`imagefm.cond_token`), and `region_pool=1` adds `RegionPool` layers after
the 16x16 and 32x32 up blocks: features inside each region (face, eyes, mouth, hair/background)
are mean-pooled, projected by a zero-initialised 1x1 conv and broadcast back into that region,
so both irises render from one shared feature. This took the left/right iris-mismatch rate
from ~34% to the real data's ~6% at slightly better FID (`experiments/METHODS.md` 7.3).

At sampling time layouts come from `data/layouts.py:LayoutPrior` (a Gaussian mixture over a
point-distribution model of the landmarks, fitted by `experiments/landmark_prior.py`, stored in
`.preprocessed/landmark_prior.npz`), so no real image enters generation; `training.run`
evaluates a conditioned model with a bank of prior-sampled masks. `cond_dropout > 0` trains a
null token as well (unconditional mode + classifier-free guidance); it cost ~7 FID at short
horizons and is off in the recipe.

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
(`RandomHorizontalFlip` — which flips the masks with the image — and `ColorJitter`, as
`grain.transforms.RandomMap` steps on worker threads). With masks, batches are
`(images, masks)` tuples. Inception statistics for FID are cached at
`./.preprocessed/real_stats.npz` by `metrics.compute_real_stats`. Both caches are keyed only
by path — delete them by hand after changing preprocessing, or stale data will be reused
silently.

Detector outputs live beside them and are expensive to rebuild (they need the detector's own
environment; `experiments/extract_landmarks.py`): `anime_faces_masks.npy` (N, 64, 64, 3),
`anime_faces_landmarks.npy` (N, 28, 2) in `[-1, 1]`, `anime_faces_landmark_scores.npy`
(N, 28). `--min-landmark-score 0.3` (default) drops the 1.5% of images the detector rejects
(non-faces) from training only; FID references always use the full set.

### Training loop

The loop is split in two. `models/imagefm.py:train_on_image` is only the optimisation loop:
it takes a `TrainConfig` (epochs, LR, EMA, conditioning), updates `model.net_theta` in
place, and *yields* `(epoch, mean_loss)` once per epoch. It imports nothing from `metrics`
or `data`. `training.py:run` consumes that generator and owns every bookkeeping concern via a
`RunConfig`: checkpoint to `outpath` every epoch, sample PNGs from fixed noise, FID every
`eval_every` epochs (16 sampler steps by default — within 0.5 FID of 64 at a quarter of the
cost — with prior-sampled masks for conditioned models), `best_model.eqx`, `losses.json`
(what `generate_figures.py` reads), and early stopping. Keep new bookkeeping in
`training.py`; keep `imagefm.py` to the maths.

### Experiments and evaluation

`experiments/METHODS.md` is the record of every measured result and the reasoning behind the
current design; read it before proposing an inductive bias. The scripts beside it are the
evaluation tools: `cond_eval.py` (FID with prior / real layouts + iris-mismatch rate),
`eye_consistency.py`, `bottleneck.py` (error maps, precision/recall, FID floor, sampler
steps), `guidance.py` (sampling-time edge guidance, -3 to -8 FID), `compare_samples.py`,
`landmark_prior.py`. Screening protocol from the campaign: FID noise at 30 epochs is ±4, so
judge architecture changes by training loss (stable to ±0.0003) plus the iris metric, and
confirm anything within 5 FID with a second seed or a longer horizon. The full working tree
of the campaign, including everything that was dropped, is on the branch
`exp/aux-loss-campaign`.

## Known deviations

`UBlock` in `models/unet.py` uses non-standard skip connections: they run from pre-encoder to
post-upsample (circumventing one encoder block) rather than the conventional post-encoder to
pre-decoder. Traced precisely, it is the standard pattern shifted down one level. This is
documented in the class docstring and carries a `TODO(n)`. It was tested: the standard
layout is neutral at 9M parameters and needs a lower learning rate at 37M (METHODS.md 6.1),
so it stays — do not silently change it while working on something else.
