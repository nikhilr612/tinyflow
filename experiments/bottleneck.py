"""Where and why a checkpoint falls short: spatial error maps and global diagnostics.

Two questions, no training:

**Where in the image is the model struggling?**

* ``error_maps.png`` -- for ``t`` in {0.3, 0.5, 0.7, 0.9}, the per-pixel
  squared error of ``x_hat(x_t, t)`` against ``x_1`` averaged over real
  pairs (top row), and the same divided by the dataset's per-pixel variance
  (bottom row): error relative to how unpredictable that pixel is a priori.
  Regions that stay bright in the *bottom* row are the ones the model is
  genuinely bad at, not merely the ones with high intrinsic variance.
* ``region_table`` -- the same errors pooled over semantic regions (face,
  eyes, mouth, and hair/background = everything outside the face hull)
  using the dataset-mean masks.
* ``texture_maps.png`` -- generated vs real on an 8x8 grid of cells: relative
  difference in edge mass and in luma std (local detail), and in chroma std
  (local colour variety).  Red = generated has less than real.
* ``worst_samples.png`` / ``best_samples.png`` -- samples ranked by the
  distance of their Inception feature to the nearest real features (the
  per-sample quantity behind precision): what the failures look like.

**What kind of gap is it?**

* FID floor: 5000 real images against the real statistics -- the best any
  model can score at this sample count.
* Precision / recall (Kynkaanniemi et al. 2019, k = 3): fidelity vs
  diversity, for the plain sampler and for edge guidance.
* FID vs sampler steps (16 / 32 / 64 / 128 Dopri5 steps).

Usage::

    uv run python experiments/bottleneck.py [--checkpoint ...] [--n-fid 5000]
        [--skip-sampler] [--outdir runs/ablation/bottleneck]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib as mpl  # noqa: E402
import numpy as np  # noqa: E402
import PIL.Image as Pilimage  # noqa: E402
import typer  # noqa: E402
from fidax.fid import FrechetInceptionDistance, _extract_activations  # noqa: E402

from data.animefaces import load_masks, preprocess_all, to_uint8  # noqa: E402
from data.layouts import LayoutPrior  # noqa: E402
from experiments.guidance import GuidedSampler, edge_mass, sobel_magnitude  # noqa: E402
from metrics import _to_inception_input, compute_real_stats, evaluate_fid  # noqa: E402
from models.imagefm import ImageFM, cond_token  # noqa: E402
from models.unet import UNet  # noqa: E402

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LUMA = np.array([0.299, 0.587, 0.114], np.float32)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def features(images: np.ndarray, fid: FrechetInceptionDistance, bs: int = 128):
    """Inception pool features ``(N, 2048)`` of images in ``[-1, 1]``."""
    out = []
    for i in range(0, len(images), bs):
        x = _to_inception_input(images[i : i + bs]).astype(jnp.float32) * 2 - 1
        out.append(np.asarray(_extract_activations(fid.model, x)))
    return np.concatenate(out)


def knn_radii(x: np.ndarray, k: int) -> np.ndarray:
    """Distance from every row of ``x`` to its ``k``-th nearest other row."""
    d = np.sqrt(np.maximum(sq_dists(x, x), 0))
    np.fill_diagonal(d, np.inf)
    return np.partition(d, k - 1, axis=1)[:, k - 1]


def sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distances ``(len(a), len(b))``."""
    return (a**2).sum(1)[:, None] + (b**2).sum(1)[None] - 2 * a @ b.T


def precision_recall(real: np.ndarray, fake: np.ndarray, k: int = 3):
    """Kynkaanniemi et al. 2019: fraction of fake (real) inside the real (fake) set."""
    r_real, r_fake = knn_radii(real, k), knn_radii(fake, k)
    d_fr = np.sqrt(np.maximum(sq_dists(fake, real), 0))
    precision = (d_fr <= r_real[None]).any(1).mean()
    recall = (r_fake[None] >= d_fr.T).any(1).mean()
    return float(precision), float(recall), d_fr.min(1)


def generate(model, n: int, key, bs: int = 256, sampler=None, bank=None) -> np.ndarray:
    """``n`` samples in ``[-1, 1]`` from ``model`` (or a wrapper with ``.generate``).

    ``bank``: prior-sampled masks for a layout-conditioned ``model`` (cycled).
    """
    out = []
    for i in range(0, n, bs):
        key, sk = jax.random.split(key)
        m = min(bs, n - i)
        noise = jax.random.normal(sk, (m, 64, 64, 3))
        if sampler is not None:
            x = sampler.generate(noise)
        elif bank is not None:
            idx = (np.arange(m) + i) % len(bank)
            x = model.generate(noise, jnp.asarray(bank[idx]))
        else:
            x = model.generate(noise)
        out.append(np.clip(np.asarray(x), -1, 1))
    return np.concatenate(out)


def cell_stats(x: np.ndarray, cells: int = 8) -> dict[str, np.ndarray]:
    """Per-cell edge mass, luma std and chroma std, ``(cells, cells)`` each."""
    edge = np.asarray(sobel_magnitude(jnp.asarray(x)))
    luma = x @ LUMA
    r, b = x[..., 0], x[..., 2]
    cb, cr = 0.5 * (b - luma) / 0.886, 0.5 * (r - luma) / 0.701
    s = 64 // cells

    def pool(m, fn):
        return fn(m.reshape(len(m), cells, s, cells, s), axis=(2, 4)).mean(0)

    return {
        "edge mass": pool(edge, np.mean),
        "luma std": pool(luma, np.std),
        "chroma std": pool(np.sqrt(cb**2 + cr**2), np.std),
    }


def save_grid(imgs: np.ndarray, path: Path, cols: int = 16, scale: int = 2):
    """Save uint8 images as a grid."""
    rows = [
        np.concatenate(list(imgs[r * cols : (r + 1) * cols]), 1)
        for r in range(len(imgs) // cols)
    ]
    g = np.concatenate(rows, 0)
    Pilimage.fromarray(g).resize(
        (g.shape[1] * scale, g.shape[0] * scale), Pilimage.Resampling.NEAREST
    ).save(path)


class _Banked:
    """``generate(noise)`` for a conditioned model with a cycled mask bank."""

    def __init__(self, model, bank):
        self.model, self.bank, self.pos = model, jnp.asarray(bank), 0

    def generate(self, x_0):
        """Sample with the next masks from the bank."""
        idx = (self.pos + jnp.arange(len(x_0))) % len(self.bank)
        self.pos = (self.pos + len(x_0)) % len(self.bank)
        return self.model.generate(x_0, self.bank[idx])


# --------------------------------------------------------------------------


def main(
    checkpoint: str = "runs/exp_long/wide_noaux/model.eqx",
    n_fid: int = 5000,
    n_maps: int = 512,
    skip_sampler: bool = False,
    outdir: str = "runs/ablation/bottleneck",
    seed: int = 0,
):
    """Run every diagnostic; write PNGs, ``REPORT.md`` and ``results.json``."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    arr = preprocess_all("./data/anime-faces")
    real_stats = compute_real_stats(arr)
    masks = load_masks().astype(np.float32).mean(0) / 255  # dataset-mean (H, W, 3)
    model = ImageFM.load(checkpoint, UNet.from_hparams)
    bank = None
    if model.cond_channels:
        bank = LayoutPrior.load().sample_masks(5000, seed)[..., : model.cond_channels]
    rng = np.random.default_rng(seed)
    results: dict = {"checkpoint": checkpoint}
    lines = [f"# Bottleneck analysis: `{checkpoint}`\n"]

    # ---------------- 1. spatial error maps at fixed t (training-time view)
    idx = rng.choice(len(arr), n_maps, replace=False)
    x1 = jnp.asarray(arr[idx])
    x0 = jax.random.normal(jax.random.key(seed), x1.shape)
    forward = eqx.filter_jit(lambda net, xt, t: jax.vmap(lambda a, b: net(a, b))(xt, t))
    ts = [0.3, 0.5, 0.7, 0.9]
    var_map = arr[rng.choice(len(arr), 4000, replace=False)].var(0).mean(-1)  # (H, W)
    err_maps, rows = {}, []
    regions = {
        "face": masks[..., 0],
        "eyes": masks[..., 1],
        "mouth": masks[..., 2],
        "hair/background": 1 - masks[..., 0],
    }
    cond_maps = None
    if model.cond_channels:
        m_real = load_masks()[idx].astype(np.float32) / 255
        cond_maps = cond_token(
            jnp.asarray(m_real[..., : model.cond_channels]),
            (n_maps, 64, 64, model.cond_channels),
        )
    for t in ts:
        xt = t * x1 + (1 - t) * x0
        x_in = xt if cond_maps is None else jnp.concatenate([xt, cond_maps], -1)
        xh = forward(model.net_theta, x_in, jnp.full((n_maps,), t))
        e = np.asarray(((xh - x1) ** 2).mean(0).mean(-1))  # (H, W)
        err_maps[t] = e
        rows.append(
            [f"t={t}"]
            + [float((e * m).sum() / m.sum()) for m in regions.values()]
            + [
                float((e / (var_map + 1e-4) * m).sum() / m.sum())
                for m in regions.values()
            ]
        )
    fig, axes = plt.subplots(2, len(ts), figsize=(3.2 * len(ts), 6.4))
    vmax = max(m.max() for m in err_maps.values())
    for j, t in enumerate(ts):
        axes[0, j].imshow(err_maps[t], vmin=0, vmax=vmax, cmap="magma")
        axes[0, j].set_title(f"sq. error, t={t}")
        axes[1, j].imshow(
            err_maps[t] / (var_map + 1e-4), vmin=0, vmax=1.0, cmap="magma"
        )
        axes[1, j].set_title("error / data variance")
        for ax in axes[:, j]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "error_maps.png", dpi=110)
    plt.close(fig)
    lines += [
        "## Where: prediction error by region (training-time view)\n",
        "Squared error of x_hat vs x_1, mean over 512 real pairs; right half divided "
        "by the dataset's per-pixel variance (1.0 = as bad as the dataset mean).\n",
        "| t | "
        + " | ".join(regions)
        + " | "
        + " | ".join(f"{r} /var" for r in regions)
        + " |",
        "|---|" + "---|" * (2 * len(regions)),
    ]
    for r in rows:
        lines.append(f"| {r[0]} | " + " | ".join(f"{v:.4f}" for v in r[1:]) + " |")
    results["region_error"] = {r[0]: r[1:] for r in rows}
    print("\n".join(lines[-len(rows) - 2 :]))

    # ---------------- 2. generated vs real local texture (sampling-time view)
    n_tex = 1024
    gen = generate(model, n_tex, jax.random.key(seed + 1), bank=bank)
    real = arr[rng.choice(len(arr), n_tex, replace=False)]
    cs_g, cs_r = cell_stats(gen), cell_stats(real)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    tex_rows = []
    for ax, k in zip(axes, cs_g):
        rel = (cs_g[k] - cs_r[k]) / (cs_r[k] + 1e-6)
        im = ax.imshow(rel, vmin=-0.5, vmax=0.5, cmap="RdBu")
        ax.set_title(f"{k}: (gen - real) / real")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
        tex_rows.append(
            [k]
            + [
                float((rel * m[::8, ::8]).sum() / m[::8, ::8].sum())
                for m in regions.values()
            ]
        )
    fig.tight_layout()
    fig.savefig(out / "texture_maps.png", dpi=110)
    plt.close(fig)
    lines += [
        "\n## Where: generated vs real local statistics (sampling-time view)\n",
        "Relative difference per region (negative = generated has less).\n",
        "| statistic | " + " | ".join(regions) + " |",
        "|---|" + "---|" * len(regions),
    ]
    for r in tex_rows:
        lines.append(f"| {r[0]} | " + " | ".join(f"{v:+.3f}" for v in r[1:]) + " |")
    print("\n".join(lines[-len(tex_rows) - 2 :]))

    # ---------------- 3. FID floor, precision/recall, worst/best samples
    fid = FrechetInceptionDistance()
    n_pr = min(n_fid, 5000)
    real_a = arr[rng.choice(len(arr), n_pr, replace=False)]
    for i in range(0, n_pr, 256):
        fid.update(_to_inception_input(real_a[i : i + 256]), real=False)
    mu_f, sig_f = fid.get_fake_stats()
    floor = float(
        FrechetInceptionDistance._fid_from_stats(
            mu_f, sig_f, jnp.asarray(real_stats["mu"]), jnp.asarray(real_stats["sigma"])
        )
    )
    f_real = features(real_a, fid)
    pr_rows = []
    gen_pr = generate(model, n_pr, jax.random.key(seed + 2), bank=bank)
    f_gen = features(gen_pr, fid)
    p, r, d_near = precision_recall(f_real, f_gen)
    pr_rows.append(["plain sampler", p, r])
    m_edge_real = float(jax.vmap(edge_mass)(jnp.asarray(real_a[:512])).mean())
    guided = GuidedSampler(
        model, lambda x: jax.nn.relu(m_edge_real - edge_mass(x)) ** 2, 0.02, masks=bank
    )
    gen_g = generate(model, n_pr, jax.random.key(seed + 2), sampler=guided)
    p_g, r_g, _ = precision_recall(f_real, features(gen_g, fid))
    pr_rows.append(["edge guidance lam=0.02", p_g, r_g])
    p_rr, r_rr, _ = precision_recall(
        f_real, features(arr[rng.choice(len(arr), n_pr, replace=False)], fid)
    )
    pr_rows.append(["real vs real (ceiling)", p_rr, r_rr])
    order = np.argsort(d_near)
    save_grid(to_uint8(gen_pr[order[-32:]]), out / "worst_samples.png")
    save_grid(to_uint8(gen_pr[order[:32]]), out / "best_samples.png")
    lines += [
        f"\n## What kind of gap\n\nFID floor (5000 real vs real stats): "
        f"**{floor:.1f}**\n",
        "| sampler | precision | recall |",
        "|---|---|---|",
    ]
    for r in pr_rows:
        lines.append(f"| {r[0]} | {r[1]:.3f} | {r[2]:.3f} |")
    lines.append(
        "\n`worst_samples.png` / `best_samples.png`: samples farthest from / "
        "closest to the real Inception manifold."
    )
    results.update({"fid_floor": floor, "precision_recall": pr_rows})
    print(f"\nFID floor {floor:.1f}")
    print("\n".join(lines[-len(pr_rows) - 3 : -1]))

    # ---------------- 4. FID vs sampler steps
    if not skip_sampler:
        lines += [
            "\n## FID vs sampler steps (Dopri5)\n",
            "| steps | FID |",
            "|---|---|",
        ]
        steps_rows = []
        for n_steps in (16, 32, 64, 128):
            model.n_steps = n_steps
            src = model if bank is None else _Banked(model, bank)
            f = evaluate_fid(src, real_stats, jax.random.key(seed + 3), n_fid)
            steps_rows.append([n_steps, f])
            lines.append(f"| {n_steps} | {f:.1f} |")
            print(f"steps {n_steps}: FID {f:.1f}")
        results["fid_vs_steps"] = steps_rows

    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    with (out / "results.json").open("w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out}/REPORT.md")


if __name__ == "__main__":
    typer.run(main)
