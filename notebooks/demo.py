# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "tinyflow[datasets] @ git+https://github.com/nikhilr612/tinyflow@v0.1.0",
#     "jax[cuda13]>=0.10.0,<0.11; sys_platform == 'linux'",
# ]
# ///
"""tinyflow as a library: train and sample both of its models (a marimo notebook)."""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    # The header above is read by uv and marimo's sandbox.  A hosted notebook
    # (molab) instead installs a missing import by its name, and "tinyflow" on
    # PyPI is another project: so where this one or JAX's CUDA plugin is
    # missing, install both from the release before any cell imports them.
    import importlib.util
    import shutil
    import subprocess
    import sys
    from importlib.metadata import distributions
    from pathlib import Path

    _spec = importlib.util.find_spec("tinyflow")  # finds, does not import
    _dirs = _spec.submodule_search_locations or [] if _spec else []
    _ours = any((Path(d) / "trainer.py").exists() for d in _dirs)
    _cuda = any(d.metadata["Name"] == "jax-cuda13-plugin" for d in distributions())
    if not (_ours and (_cuda or sys.platform != "linux")):
        _uv = shutil.which("uv")
        _pip = [_uv, "pip"] if _uv else [sys.executable, "-m", "pip"]
        subprocess.run(
            [
                *_pip,
                "install",
                *(["--python", sys.executable] if _uv else []),
                "tinyflow[datasets] @ git+https://github.com/nikhilr612/tinyflow@v0.1.0",
                "jax[cuda13]>=0.10.0,<0.11; sys_platform == 'linux'",
            ],
            check=True,
        )
        importlib.invalidate_caches()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # tinyflow: two flow models, trained from scratch

    [tinyflow](https://github.com/nikhilr612/tinyflow) is a short and sweet
    implementation of flow matching in JAX, with an emphasis on simplicity
    and readability.

    The following components comprise each implementation:
    * a **denoiser**: a plain Equinox module (`UNet`, `DiT`) that predicts the
      clean image from a noisy one;
    * a **stochastic interpolant** between source and target distributions.
    * CFM loss and sampler (`flow.py`,
      `bridge.py`), generic over the denoiser;
    * a **`Trainer`**: Adam on a warm-up and a cosine, gradient clipping and an
      average of the weights. A recipe subclasses it with one method,
      `loss(net, batch, key)`, and `fit` yields the averaged denoiser after
      every epoch.

    This notebook trains both of the package's models on one GPU: anime faces
    drawn to a layout, and satellite radar translated to optical images.
    Each part trains its model in one cell (the epoch count is at the top of
    it); everything after that is sampling and follows the sliders.
    """)
    return


@app.cell
def _():
    import base64
    import io

    import equinox as eqx
    import jax
    import marimo as mo
    import numpy as np
    from PIL import Image

    from tinyflow.trainer import to_uint8

    def tiles(*rows):
        """Rows of ``(N, H, W, 3)`` images in ``[-1, 1]``, as one picture."""
        return to_uint8(np.concatenate([np.concatenate(list(r), 1) for r in rows]))

    def gif(states, ms=100):
        """Trajectories ``(N, T, H, W, 3)`` in ``[-1, 1]``, as one animated row."""
        states = np.asarray(states)
        first, *rest = (Image.fromarray(tiles(f)) for f in states.swapaxes(0, 1))
        buf = io.BytesIO()
        hold = [ms] * len(rest) + [2000]  # rest on the last frame
        first.save(buf, "GIF", save_all=True, append_images=rest, duration=hold, loop=0)
        data = base64.b64encode(buf.getvalue()).decode()
        return mo.image(f"data:image/gif;base64,{data}", width="100%")

    def architecture(name, net):
        """``net``'s module tree, every layer and weight shape, folded away."""
        n = sum(x.size for x in jax.tree.leaves(eqx.filter(net, eqx.is_array)))
        tree = mo.plain_text(eqx.tree_pformat(net))
        return mo.accordion({f"{name}: {n / 1e6:.1f}M parameters": tree})

    jax.devices()
    return architecture, gif, jax, mo, np, tiles


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · Anime faces, drawn to a layout

    Flow matching carries noise $x_0$ to an image $x_1$ along the straight line
    $x_t = t\,x_1 + (1-t)\,x_0$. The U-Net sees $x_t$, the time $t$ and three
    masks saying where the face, the eyes and the mouth go, and predicts the
    clean image $\hat x$; the velocity $(\hat x - x_t)/(1-t)$ is integrated
    from $t=0$ to $1$ in 8 midpoint steps.

    The first run downloads the 21,551 faces (64×64) and the landmarks the
    masks are drawn from, and caches both in the working directory.
    """)
    return


@app.cell
def _(mo, tiles):
    import tinyflow.data.animefaces as faces

    images, masks = faces.load_images(), faces.load_masks()
    keep = faces.curated_indices()  # drops the 1.5 % that are not faces
    face_data, face_steps = faces.make_dataset(images[keep], masks[keep])

    mo.vstack(
        [
            mo.md(f"{len(keep):,} training faces; their masks in red, green, blue:"),
            mo.image(tiles(images[:8], 2 * masks[:8] - 1)),
        ]
    )
    return face_data, face_steps


@app.cell
def _(architecture, jax):
    from tinyflow.models.animefaces.unet import UNet

    face_net = UNet(key=jax.random.key(1))
    architecture("U-Net", face_net)
    return (face_net,)


@app.cell
def _(face_data, face_net, face_steps, jax, mo):
    from tinyflow.models.animefaces.flow import FlowTrainer, ImageFM

    _epochs = 40  # 25 s each on an RTX 4090
    # FlowTrainer is Trainer plus the draw of t and the noise for ImageFM.loss.
    _recipe = FlowTrainer(schedule_steps=_epochs * face_steps)
    _fits = _recipe.fit(face_net, face_data, _epochs, face_steps, jax.random.key(0))
    with mo.status.progress_bar(total=_epochs, title="U-Net") as _bar:
        for _epoch, _loss, _net in _fits:  # the averaged weights, once an epoch
            _bar.update(subtitle=f"loss {_loss:.4f}")
    face_model = ImageFM(_net, hparams={})
    face_model.save("faces.eqx")  # ImageFM.load("faces.eqx", UNet) reads it back
    return (face_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The layouts below are not taken from real images: they are drawn from a
    Gaussian mixture fitted to the landmarks (`LayoutPrior`). Each layout is
    rendered twice, from two draws of the noise: the layout fixes the pose,
    the noise picks the character.  Below, the first row as it is integrated,
    from the noise at $t = 0$ to the faces at $t = 1$.
    """)
    return


@app.cell
def _(mo):
    face_seed = mo.ui.slider(0, 99, label="seed", show_value=True)
    face_seed
    return (face_seed,)


@app.cell
def _(face_model, face_seed, gif, jax, mo, np, tiles):
    from tinyflow.data.layouts import LayoutPrior
    from tinyflow.models.animefaces.flow import T_END, solve

    _layouts = jax.numpy.asarray(LayoutPrior.load().sample_masks(8, face_seed.value))
    _noise = jax.random.normal(jax.random.key(face_seed.value), (2, 8, 64, 64, 3))
    _ts = tuple(np.linspace(0, T_END, 33).tolist())  # between the 8 steps: interpolated
    mo.vstack(
        [
            mo.image(
                tiles(
                    2 * _layouts - 1, *(face_model.sample(n, _layouts) for n in _noise)
                ),
                width="100%",
            ),
            gif(solve(face_model.net, _noise[0], _layouts, _ts)),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Radar to optical

    Sentinel-1's radar sees through cloud and at night; Sentinel-2's camera
    does not. The bridge model starts at the radar image $s$ instead of noise,
    and ends at the optical image $x_1$ taken within two days of it:

    $$x_t = \alpha(t)\,s + \beta(t)\,x_1 + \gamma(t)\,z,\qquad
      \beta = \sin^2(\pi t/2),\ \alpha = 1-\beta,\ \gamma = \sqrt{4t(1-t)},$$

    a stochastic interpolant whose noise $z$ is loudest at $t = 1/2$ and
    vanishes at both ends, so one radar image has many optical renderings.
    The transformer (`DiT`) predicts $\hat x$ from $x_t$, the radar image and
    seven numbers: the season, the location, the time gap and the orbit.

    The first run downloads the [SEN-Bridge](https://huggingface.co/datasets/nikhilr01/sen-bridge)
    arrays: 14 GB for 125,000 training and 15,000 validation pairs (128×128).
    """)
    return


@app.cell
def _(mo, np, tiles):
    from sen_bridge import data

    import tinyflow.data.senbridge as sb
    from tinyflow.models.senbridge.bridge import to_unit

    train_pairs, val_pairs = data.load_dataset("train"), data.load_dataset("validation")
    bridge_data, bridge_steps = sb.make_dataset(train_pairs)

    _i = np.arange(8) * 1000
    mo.vstack(
        [
            mo.md("Training pairs: radar (false colour) over optical."),
            mo.image(
                tiles(to_unit(train_pairs.sar[_i]), to_unit(train_pairs.optical[_i]))
            ),
        ]
    )
    return bridge_data, bridge_steps, to_unit, val_pairs


@app.cell
def _(architecture, jax):
    from tinyflow.models.senbridge.dit import DiT

    bridge_net = DiT(jax.random.key(1))
    architecture("DiT", bridge_net)
    return (bridge_net,)


@app.cell
def _(bridge_data, bridge_net, bridge_steps, jax, mo):
    from tinyflow.models.senbridge.bridge import Bridge, BridgeTrainer

    _epochs = 20  # 2 min each on an RTX 4090
    # BridgeTrainer is Trainer plus the draw of t and z for Bridge.loss.  Its
    # peak of 1e-4 is for 120 epochs; a short run needs a hotter one (at 10
    # epochs, 3e-4: FID 135 against 164; 6e-4 was unstable).
    _recipe = BridgeTrainer(schedule_steps=_epochs * bridge_steps, peak_lr=3e-4)
    _key = jax.random.key(0)
    _fits = _recipe.fit(bridge_net, bridge_data, _epochs, bridge_steps, _key)
    with mo.status.progress_bar(total=_epochs, title="DiT") as _bar:
        for _epoch, _loss, _net in _fits:  # the averaged weights, once an epoch
            _bar.update(subtitle=f"loss {_loss:.4f}")
    bridge_model = Bridge(_net, hparams={})
    bridge_model.save("bridge.eqx")  # Bridge.load("bridge.eqx", DiT) reads it back
    return (bridge_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Held-out validation pairs: the radar image, two samples from it (two draws
    of $z$), and the optical image that was actually taken.  Below, the first
    sample's path: out of the radar image, through the noise at $t = 1/2$, into
    the optical image.
    """)
    return


@app.cell
def _(mo):
    bridge_seed = mo.ui.slider(0, 99, label="seed", show_value=True)
    bridge_seed
    return (bridge_seed,)


@app.cell
def _(bridge_model, bridge_seed, gif, jax, mo, np, tiles, to_unit, val_pairs):
    _rng = np.random.default_rng(bridge_seed.value)
    _rows = np.sort(_rng.choice(len(val_pairs), 8, replace=False))
    _sar, _meta = to_unit(val_pairs.sar[_rows]), val_pairs.meta[_rows]
    _z = jax.random.normal(jax.random.key(bridge_seed.value), (2, *_sar.shape))
    _samples = [bridge_model.sample(_sar, _meta, z) for z in _z]
    mo.vstack(
        [
            mo.image(
                tiles(_sar, *_samples, to_unit(val_pairs.optical[_rows])), width="100%"
            ),
            gif(bridge_model.sample(_sar, _meta, _z[0], trajectory=True)),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
