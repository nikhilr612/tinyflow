"""Paper figures for one training run: curves, sample grids, and a showcase.

Reads a run directory as written by ``training.run`` (``losses.json``,
``model.eqx`` or ``best_model.eqx``) and writes to ``<run>/figures/``.

Two commands, one per dataset (masks come from different places):

* ``anime`` -- loss/FID curves, 8x8 samples (layouts from the layout prior
  for a conditioned model), layout-to-image, real-vs-generated, noise
  interpolation, iris pairs, and the ``showcase.png`` page;
* ``celeba`` -- a single ``celeba_grid.png``: row 1 the held-out masks,
  row 2 the generations rendered from them, row 3 the paired real faces.
  Layouts cycle from the held-out validation bank, not the GMM prior.

Usage::

    uv run python generate_figures.py anime RUN_DIR [--checkpoint best_model.eqx]
    uv run python generate_figures.py celeba RUN_DIR [--n-pairs 8]
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib as mpl
import numpy as np
import PIL.Image as Pilimage
import typer

from data.animefaces import preprocess_all, to_uint8
from data.layouts import LayoutPrior
from models.imagefm import ImageFM
from models.unet import UNet

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import rcParams  # noqa: E402

rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
    }
)

EYE_ROWS, EYE_COLS = slice(16, 38), slice(10, 54)  # eye band of the aligned faces

# Overlay palette per mask channel: face red, eyes green, mouth blue, nose yellow.
OVERLAY_COLORS = np.array(
    [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]], dtype=np.float32
)


def grid(images: np.ndarray, cols: int) -> np.ndarray:
    """Tile uint8 images ``(N, H, W, 3)`` into ``ceil(N / cols)`` rows."""
    n = len(images)
    rows = [
        np.concatenate(list(images[r : r + cols]), axis=1) for r in range(0, n, cols)
    ]
    return np.concatenate(rows, axis=0)


def mask_overlay(masks: np.ndarray) -> np.ndarray:
    """Render layout masks ``(N, H, W, K)`` in channel colors on grey, uint8.

    The first three channels use the legacy scale (face red, eyes green,
    mouth blue), kept bit-identical to the original formula; extra channels
    tint in ``OVERLAY_COLORS`` order (nose yellow).
    """
    base = np.full(masks.shape[:-1] + (3,), 128, np.uint8)
    tint = np.zeros(masks.shape[:-1] + (3,), np.float32)
    n3 = min(masks.shape[-1], 3)
    tint[..., :n3] = masks[..., :n3] * 127
    for i in range(3, masks.shape[-1]):
        tint += masks[..., i : i + 1] * (OVERLAY_COLORS[i % len(OVERLAY_COLORS)] / 2)
    return np.clip(base * 0.5 + tint, 0, 255).astype(np.uint8)


def curves(history: list[dict], outdir: Path) -> tuple[Path, Path | None]:
    """Loss and FID curves from ``losses.json`` records."""
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(epochs, [h["loss"] for h in history], linewidth=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Flow-matching loss")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    loss_path = outdir / "loss_curve.png"
    fig.savefig(loss_path)
    plt.close(fig)

    fid = [(h["epoch"], h["fid"]) for h in history if "fid" in h]
    if not fid:
        return loss_path, None
    e, f = zip(*fid)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(e, f, marker="o", linewidth=1.2, markersize=4)
    best = int(np.argmin(f))
    ax.annotate(
        f"best {f[best]:.1f}",
        (e[best], f[best]),
        xytext=(5, 5),
        textcoords="offset points",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("FID (5000 samples)")
    ax.grid(True, alpha=0.3)
    fid_path = outdir / "fid_curve.png"
    fig.savefig(fid_path)
    plt.close(fig)
    return loss_path, fid_path


def slerp(a: jnp.ndarray, b: jnp.ndarray, n: int) -> jnp.ndarray:
    """``n`` points on the great circle from ``a`` to ``b`` (flattened arrays)."""
    a_n, b_n = a / jnp.linalg.norm(a), b / jnp.linalg.norm(b)
    omega = jnp.arccos(jnp.clip(jnp.dot(a_n, b_n), -1, 1))
    ts = jnp.linspace(0, 1, n)
    return jnp.stack(
        [
            (jnp.sin((1 - t) * omega) * a + jnp.sin(t * omega) * b) / jnp.sin(omega)
            for t in ts
        ]
    )


app = typer.Typer()


def _load_run_model(run: Path, checkpoint: str) -> ImageFM:
    """Load ``checkpoint`` from ``run``, falling back to ``model.eqx``."""
    ckpt = run / checkpoint
    if not ckpt.exists():
        ckpt = run / "model.eqx"
    return ImageFM.load(str(ckpt), UNet.from_hparams)


@app.command()
def anime(
    run_dir: str,
    checkpoint: str = "model.eqx",
    n_steps: int = 64,
    seed: int = 0,
):
    """Write every figure for ``run_dir`` into ``run_dir/figures``."""
    main(run_dir, checkpoint, n_steps, seed)


@app.command()
def celeba(
    run_dir: str,
    checkpoint: str = "best_model.eqx",
    n_pairs: int = 8,
    n_steps: int = 64,
    seed: int = 0,
):
    """Write ``celeba_grid.png`` for ``run_dir`` into ``run_dir/figures``.

    One 3-row grid over ``n_pairs`` seeded-random pairs from the held-out
    validation bank: row 1 the masks, row 2 the generations rendered from
    them, row 3 the paired real faces.  Nothing else is written.
    """
    run = Path(run_dir)
    out = run / "figures"
    out.mkdir(parents=True, exist_ok=True)
    model = _load_run_model(run, checkpoint)
    model.n_steps = n_steps
    size = int(model.hparams.get("image_size", 64))
    sfx = "" if size == 64 else f"_{size}"
    bank = np.load(f"./.preprocessed/celebamask_eval_masks{sfx}.npy")
    faces = np.load(f"./.preprocessed/celebamask_faces{sfx}.npy")
    # The bank is the tail of the dataset arrays, so faces pair by offset.
    offset = len(faces) - len(bank)
    idx = np.random.default_rng(seed).choice(len(bank), n_pairs, replace=False)
    refs = to_uint8(faces[offset + idx])
    masks = None
    if model.cond_channels:
        masks = jnp.asarray(
            bank[idx, ..., : model.cond_channels].astype(np.float32) / 255.0
        )
    noise = jax.random.normal(jax.random.key(seed), (n_pairs, size, size, 3))
    gen = to_uint8(np.clip(np.asarray(model.generate(noise, masks)), -1, 1))
    rows = [grid(gen, n_pairs), grid(refs, n_pairs)]
    if masks is not None:
        rows.insert(0, grid(mask_overlay(np.asarray(masks)), n_pairs))
    panel = np.concatenate(rows, axis=0)
    Pilimage.fromarray(panel).save(out / "celeba_grid.png")
    print(f"wrote celeba_grid.png ({n_pairs} pairs) to {out}/")


def main(
    run_dir: str,
    checkpoint: str = "model.eqx",
    n_steps: int = 64,
    seed: int = 0,
):
    """Write every figure for ``run_dir`` into ``run_dir/figures``."""
    run = Path(run_dir)
    out = run / "figures"
    out.mkdir(parents=True, exist_ok=True)
    ckpt = run / checkpoint
    if not ckpt.exists():
        ckpt = run / "model.eqx"
    model = ImageFM.load(str(ckpt), UNet.from_hparams)
    model.n_steps = n_steps
    key = jax.random.key(seed)
    panels: dict[str, np.ndarray] = {}

    with (run / "losses.json").open() as f:
        history = json.load(f)
    loss_path, fid_path = curves(history, out)

    # ---- samples (64) with layouts from the prior when conditioned
    k_noise, k_interp, key = jax.random.split(key, 3)
    noise = jax.random.normal(k_noise, (64, 64, 64, 3))
    masks = None
    if model.cond_channels:
        masks = jnp.asarray(
            LayoutPrior.load().sample_masks(64, seed)[..., : model.cond_channels]
        )
    samples = to_uint8(np.clip(np.asarray(model.generate(noise, masks)), -1, 1))
    panels["sample_grid"] = grid(samples, 8)
    Pilimage.fromarray(panels["sample_grid"]).save(out / "sample_grid.png")

    if masks is not None:
        over = mask_overlay(np.asarray(masks[:8]))
        panels["layout_to_image"] = np.concatenate(
            [grid(over, 8), grid(samples[:8], 8)], 0
        )
        Pilimage.fromarray(panels["layout_to_image"]).save(out / "layout_to_image.png")

    # ---- real vs generated
    arr = preprocess_all("./data/anime-faces")
    real = to_uint8(arr[np.random.default_rng(seed).choice(len(arr), 8, replace=False)])
    panels["real_vs_generated"] = np.concatenate(
        [grid(real, 8), grid(samples[8:16], 8)], 0
    )
    Pilimage.fromarray(panels["real_vs_generated"]).save(out / "real_vs_generated.png")

    # ---- interpolation in noise, fixed layout
    a, b = jax.random.normal(k_interp, (2, 64 * 64 * 3))
    z = slerp(a, b, 8).reshape(8, 64, 64, 3)
    m_fixed = None if masks is None else jnp.repeat(masks[:1], 8, axis=0)
    interp = to_uint8(np.clip(np.asarray(model.generate(z, m_fixed)), -1, 1))
    panels["interpolation"] = grid(interp, 8)
    Pilimage.fromarray(panels["interpolation"]).save(out / "interpolation.png")

    # ---- iris pairs: eye-band crops of 16 samples, 3x upscaled
    crops = samples[:16, EYE_ROWS, EYE_COLS]
    crops = np.repeat(np.repeat(crops, 3, axis=1), 3, axis=2)
    panels["iris_pairs"] = grid(crops, 4)
    Pilimage.fromarray(panels["iris_pairs"]).save(out / "iris_pairs.png")

    # ---- showcase page
    n_fig = 2 + len(panels)
    fig = plt.figure(figsize=(12, 3.0 * ((n_fig + 1) // 2)))
    gs = fig.add_gridspec((n_fig + 1) // 2, 2)
    items = [("Training loss", plt.imread(loss_path))]
    if fid_path is not None:
        items.append(("FID over training", plt.imread(fid_path)))
    titles = {
        "sample_grid": "Samples (layouts from the prior)"
        if masks is not None
        else "Samples",
        "layout_to_image": "Layout mask (top) and the sample rendered from it",
        "real_vs_generated": "Real (top) vs generated (bottom)",
        "interpolation": "Noise interpolation at a fixed layout",
        "iris_pairs": "Eye regions: left/right iris agreement",
    }
    items += [(titles[k], v) for k, v in panels.items()]
    for i, (title, img) in enumerate(items):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    hp = model.hparams
    fid_final = next((h["fid"] for h in reversed(history) if "fid" in h), None)
    fig.suptitle(
        f"{run.name}: {len(history)} epochs, base_channels={hp['base_channels']}, "
        f"cond_channels={hp.get('cond_channels', 0)}, "
        f"region_pool={hp.get('region_pool', 0)}"
        + (f", final FID {fid_final:.1f}" if fid_final is not None else ""),
        y=1.0,
    )
    fig.tight_layout()
    fig.savefig(out / "showcase.png")
    plt.close(fig)
    print(f"wrote {sorted(p.name for p in out.iterdir())} to {out}/")


if __name__ == "__main__":
    app()
