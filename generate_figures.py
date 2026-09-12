"""Generate paper-quality figures from training results."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

from data.animefaces import preprocess_all, to_uint8
from models.imagefm import ImageFM
from models.unet import UNet

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


def _load_model(model_path: str) -> ImageFM:
    with open(model_path + ".hparams") as f:
        hparams = json.load(f)

    def skeleton(key, **hp):
        return UNet(**hp, key=key)

    return ImageFM.load(model_path, skeleton)


def plot_loss_curve(losses_path: str, output_path: str):
    with open(losses_path) as f:
        history = json.load(f)
    epochs = [h["epoch"] for h in history]
    losses = [h["loss"] for h in history]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, losses, linewidth=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CFM Loss")
    ax.set_title("Training Loss Curve")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved loss curve to {output_path}")


def plot_fid_curve(losses_path: str, output_path: str):
    with open(losses_path) as f:
        history = json.load(f)
    fid_entries = [(h["epoch"], h["fid"]) for h in history if "fid" in h]
    if not fid_entries:
        print("No FID entries found in losses.json")
        return
    epochs, fids = zip(*fid_entries)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, fids, marker="o", linewidth=1.2, markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("FID")
    ax.set_title("FID Score Over Training")
    ax.axhline(
        y=min(fids),
        color="green",
        linestyle="--",
        alpha=0.5,
        label=f"Best: {min(fids):.1f}",
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved FID curve to {output_path}")


def generate_sample_grid(
    model_path: str, output_path: str, n_row: int = 4, n_col: int = 4
):
    model = _load_model(model_path)
    key = jax.random.key(42)
    key, sk = jax.random.split(key)
    noise = jax.random.normal(sk, (n_row * n_col, 64, 64, 3))
    samples = to_uint8(model.generate(noise))
    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col * 2, n_row * 2))
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i])
        ax.axis("off")
    fig.suptitle("Generated Anime Faces (Flow Matching)")
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved sample grid to {output_path}")


def generate_interpolation(model_path: str, output_path: str, n_steps: int = 10):
    model = _load_model(model_path)
    key = jax.random.key(42)
    sk1, sk2 = jax.random.split(key)
    z1 = jax.random.normal(sk1, (1, 64, 64, 3))
    z2 = jax.random.normal(sk2, (1, 64, 64, 3))
    alphas = jnp.linspace(0, 1, n_steps)
    samples = []
    for alpha in alphas:
        z = (1 - alpha) * z1 + alpha * z2
        samples.append(to_uint8(model.generate(z)[0]))
    fig, axes = plt.subplots(1, n_steps, figsize=(n_steps * 1.5, 1.5))
    for i, (ax, img) in enumerate(zip(axes, samples)):
        ax.imshow(img)
        ax.set_title(f"\u03b1={float(alphas[i]):.1f}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Latent Interpolation")
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved interpolation to {output_path}")


def compare_real_vs_generated(model_path: str, output_path: str, n: int = 8):
    arr = preprocess_all("./data/anime-faces")
    idx = np.random.RandomState(0).choice(len(arr), n, replace=False)
    real_imgs = to_uint8(arr[idx])

    model = _load_model(model_path)
    key = jax.random.key(42)
    key, sk = jax.random.split(key)
    noise = jax.random.normal(sk, (n, 64, 64, 3))
    gen_imgs = to_uint8(model.generate(noise))

    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3))
    for i in range(n):
        axes[0, i].imshow(real_imgs[i])
        axes[0, i].axis("off")
        axes[1, i].imshow(gen_imgs[i])
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Real", fontsize=10)
    axes[1, 0].set_ylabel("Generated", fontsize=10)
    fig.suptitle("Real vs. Generated Anime Faces")
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved real vs generated comparison to {output_path}")


if __name__ == "__main__":
    from datetime import datetime

    outdir = Path("./runs/figures")
    outdir.mkdir(parents=True, exist_ok=True)

    losses_path = "./runs/losses.json"
    model_path = "./runs/best_model.eqx"

    print(f"[{datetime.now():%H:%M:%S}] Plotting loss curve...")
    plot_loss_curve(losses_path, str(outdir / "loss_curve.png"))

    print(f"[{datetime.now():%H:%M:%S}] Plotting FID curve...")
    plot_fid_curve(losses_path, str(outdir / "fid_curve.png"))

    print(f"[{datetime.now():%H:%M:%S}] Generating sample grid...")
    generate_sample_grid(model_path, str(outdir / "sample_grid.png"))

    print(f"[{datetime.now():%H:%M:%S}] Generating interpolation...")
    generate_interpolation(model_path, str(outdir / "interpolation.png"))

    print(f"[{datetime.now():%H:%M:%S}] Generating real vs generated comparison...")
    compare_real_vs_generated(model_path, str(outdir / "real_vs_generated.png"))

    print(f"[{datetime.now():%H:%M:%S}] All figures saved to {outdir}/")
