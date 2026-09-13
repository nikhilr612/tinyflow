"""Colour statistics of generated samples vs. the dataset -- no training.

Answers "does the model already get colours right?" before anyone writes a
palette loss.  Compares ``n`` generated samples against ``n`` real images on:

* per-channel mean / std in RGB and YCbCr (global colour cast, contrast);
* saturation (HSV ``S``) mean and histogram;
* a 32x32 (Cb, Cr) chroma histogram: L1 and Hellinger distance to the real
  one, and, as a yardstick, the same distances between two disjoint halves of
  the real data (the sampling-noise floor);
* palette size: median number of 16^3 RGB bins covering 90% of a picture's
  pixels (a per-image "how many colours" measure);
* chroma in the face region (eye/face masks are aligned): mean Cb, Cr.

Writes ``color_stats.png`` with the two chroma histograms and the saturation
histograms side by side.

Usage::

    uv run python experiments/color_stats.py [--checkpoint ...] [--n 1000]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax  # noqa: E402
import matplotlib as mpl  # noqa: E402
import numpy as np  # noqa: E402
import typer  # noqa: E402

from data.animefaces import load_masks, preprocess_all  # noqa: E402
from models.imagefm import ImageFM  # noqa: E402
from models.unet import UNet  # noqa: E402

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def ycbcr(x: np.ndarray) -> np.ndarray:
    """BT.601 (Y, Cb, Cr) of an RGB array in [-1, 1]; chroma in [-0.5, 0.5]-ish."""
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return np.stack([y, 0.5 * (b - y) / (1 - 0.114), 0.5 * (r - y) / (1 - 0.299)], -1)


def saturation(x: np.ndarray) -> np.ndarray:
    """HSV saturation of an RGB array in [-1, 1]."""
    v = (x + 1) / 2
    mx, mn = v.max(-1), v.min(-1)
    return np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)


def chroma_hist(x: np.ndarray, bins: int = 32) -> np.ndarray:
    """Normalised 2-D (Cb, Cr) histogram over all pixels of ``x``."""
    c = ycbcr(x)[..., 1:].reshape(-1, 2)
    h, _, _ = np.histogram2d(c[:, 0], c[:, 1], bins=bins, range=[[-0.5, 0.5]] * 2)
    return h / h.sum()


def hellinger(p: np.ndarray, q: np.ndarray) -> float:
    """Hellinger distance between two normalised histograms."""
    return float(np.sqrt(0.5 * ((np.sqrt(p) - np.sqrt(q)) ** 2).sum()))


def palette_n90(x: np.ndarray, levels: int = 16) -> float:
    """Median over images of the number of RGB bins covering 90% of pixels."""
    q = np.clip(((x + 1) / 2 * (levels - 1)).round(), 0, levels - 1).astype(int)
    ids = q[..., 0] * levels**2 + q[..., 1] * levels + q[..., 2]
    out = []
    for im in ids.reshape(len(x), -1):
        counts = np.sort(np.bincount(im, minlength=levels**3))[::-1]
        out.append(int(np.searchsorted(np.cumsum(counts), 0.9 * im.size)) + 1)
    return float(np.median(out))


def main(
    checkpoint: str = "runs/exp_edge/edge0/model.eqx",
    n: int = 1000,
    seed: int = 0,
    out: str = "runs/ablation/color_stats.png",
):
    """Print the comparison table and write the histogram figure."""
    arr = preprocess_all("./data/anime-faces")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(arr), 2 * n, replace=False)
    real, real2 = arr[idx[:n]], arr[idx[n:]]
    masks = load_masks()
    face = masks[..., 0].astype(np.float32).mean(0) / 255  # dataset-mean face map
    face = face / face.sum()

    model = ImageFM.load(checkpoint, lambda key, **hp: UNet(**hp, key=key))
    gen = []
    for i in range(0, n, 100):
        noise = jax.random.normal(jax.random.key(seed * 1000 + i), (100, 64, 64, 3))
        gen.append(np.clip(np.asarray(model.generate(noise)), -1, 1))
    gen = np.concatenate(gen)[:n]

    def stats(x):
        yc = ycbcr(x)
        s = saturation(x)
        return {
            "R mean": x[..., 0].mean(),
            "G mean": x[..., 1].mean(),
            "B mean": x[..., 2].mean(),
            "Y mean": yc[..., 0].mean(),
            "Y std": yc[..., 0].std(),
            "Cb mean": yc[..., 1].mean(),
            "Cr mean": yc[..., 2].mean(),
            "Cb std": yc[..., 1].std(),
            "Cr std": yc[..., 2].std(),
            "saturation mean": s.mean(),
            "saturation p90": np.quantile(s, 0.9),
            "face Cb": (yc[..., 1] * face).sum(-1).sum(-1).mean(),
            "face Cr": (yc[..., 2] * face).sum(-1).sum(-1).mean(),
            "palette n90": palette_n90(x),
        }

    sr, sr2, sg = stats(real), stats(real2), stats(gen)
    print(f"{'':>18}{'real':>10}{'real (2nd half)':>17}{'generated':>12}")
    for k in sr:
        print(f"{k:>18}{sr[k]:>10.3f}{sr2[k]:>17.3f}{sg[k]:>12.3f}")
    hr, hr2, hg = chroma_hist(real), chroma_hist(real2), chroma_hist(gen)
    print("\nchroma histogram distance   real-vs-real2   real-vs-generated")
    print(f"{'L1':>26}{np.abs(hr - hr2).sum():>16.4f}{np.abs(hr - hg).sum():>20.4f}")
    print(f"{'Hellinger':>26}{hellinger(hr, hr2):>16.4f}{hellinger(hr, hg):>20.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, h, title in zip(axes[:2], [hr, hg], ["real", "generated"]):
        ax.imshow(np.log1p(h.T * 1e4), origin="lower", extent=[-0.5, 0.5, -0.5, 0.5])
        ax.set_xlabel("Cb")
        ax.set_ylabel("Cr")
        ax.set_title(f"chroma histogram, {title} (log)")
    ax = axes[2]
    ax.hist(
        saturation(real).ravel(),
        bins=50,
        range=(0, 1),
        density=True,
        alpha=0.5,
        label="real",
    )
    ax.hist(
        saturation(gen).ravel(),
        bins=50,
        range=(0, 1),
        density=True,
        alpha=0.5,
        label="generated",
    )
    ax.set_xlabel("saturation")
    ax.set_title("saturation distribution")
    ax.legend()
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    typer.run(main)
