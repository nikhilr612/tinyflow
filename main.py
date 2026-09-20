"""Main entry point for the tinyflow CLI."""

from pathlib import Path
from typing import Annotated

import equinox as eqx
import jax
import numpy as np
import typer

import data.animefaces
import data.toycardioid
import models.toyfm as toyfm
import training
from data.animefaces import (
    curated_indices,
    load_landmark_scores,
    load_masks,
    preprocess_all,
)
from data.layouts import LayoutPrior, rasterize
from metrics import compute_real_stats
from models import ToyFM
from models.imagefm import ImageFM, TrainConfig
from models.unet import UNet
from training import RunConfig
from viz import create_animation

app = typer.Typer()


@app.command()
def toy(
    fpath: Annotated[
        str,
        typer.Argument(help="The path to synthetic data. Check data.toycardioid"),
    ],
    outpath: Annotated[
        str,
        typer.Argument(help="Output path to save the model, and test samples."),
    ] = "./runs/toymodel.eqx",
    n_epochs: Annotated[int, typer.Option(help="Number of epochs to train for")] = 12,
    seed: Annotated[int, typer.Option(help="Seed to use")] = 49,
):
    """Train a toy flow matching model on synthetic data."""
    dataset = data.toycardioid.cardioid_dataset(fpath, seed=seed - 7)
    key = jax.random.key(seed)
    key, sk1, sk2 = jax.random.split(key, num=3)
    model = toyfm.train_on(key, ToyFM.from_key(sk1), dataset, n_epochs=n_epochs)

    points = model.sample(16, sk2, jax.numpy.linspace(0, 1, 20))
    create_animation(fpath, points, outpath + ".svg")
    model.save(outpath)


@app.command()
def anime(
    outpath: str = "./runs/imagemodel.eqx",
    n_epochs: int = 100,
    batch_size: int = 128,
    seed: int = 49,
    base_channels: int = 32,
    time_embedding_dim: int = 128,
    n_blocks: int = 4,
    eval_every: int = 5,
    early_stop_patience: int = 3,
    n_steps: int = 64,
    init_lr: float = 1e-3,
    warmup_steps: int = 500,
    lr_end_frac: float = 0.01,
    ema_decay: float = 0.999,
    init_from: str = "",
    min_landmark_score: float = 0.3,
    cond_channels: int = 0,
    cond_dropout: float = 0.0,
    region_pool: int = 0,
    mask_path: str = "",
    flow_map: int = 0,
    flow_map_frac: float = 0.5,
    mask_input: int = 1,
    color_jitter: int = 1,
    fid_batch_size: int = 256,
    fid_n_steps: int = 16,
    dataset_name: str = "anime",
    image_size: int = 64,
):
    """Train a flow matching model on the anime faces (or CelebAMask-HQ) dataset.

    ``--dataset-name celeba`` trains on the arrays ``data/celebamask.py``
    writes: images and four-channel masks (face, eyes, mouth, nose), the
    curation in ``celebamask_keep.npy`` instead of ``--min-landmark-score``,
    and the held-out real label maps as the evaluation layout bank (no
    layout prior).  ``--mask-path`` defaults per dataset.  ``--image-size 128``
    uses the ``_128`` arrays: for CelebA those ``data/celebamask.py --size 128``
    writes, for anime the Real-ESRGAN-upscaled sources from
    ``experiments/upscale_anime.py build`` with masks rasterised from the
    landmarks; the U-Net places its region-pooling layers by resolution.

    ``--cond-channels 3`` conditions the model on the cached layout masks at
    ``--mask-path`` (face, eyes, mouth); ``--region-pool 1`` adds the
    mask-guided region-pooling layers that make both irises render from one
    shared feature.  A conditioned model is evaluated (sample PNGs, FID) with
    layouts drawn from the prior at ``.preprocessed/landmark_prior.npz``, so no
    real image enters generation.  ``--min-landmark-score`` drops
    detector-rejected non-faces from training (the FID reference stays the
    full set).  ``--fid-n-steps`` is the sampler length for training-time FID
    only; 16 Dopri5 steps score within 0.5 FID of 64 on this data.  The
    learning rate warms up linearly and decays with a cosine
    (``--warmup-steps``, ``--lr-end-frac``).  ``--init-from CKPT`` starts
    from a saved model's weights (same architecture) with a fresh optimiser.

    ``--flow-map 1`` builds the two-time (MeanFlow) network and trains
    ``--flow-map-frac`` of every batch on the average-velocity identity, so
    the model can also sample in one or two jumps (``ImageFM.generate_map``;
    their FID is logged beside the ODE sampler's).  ``--init-from`` a velocity
    checkpoint of the same architecture starts the fine-tune from it.
    """
    assert base_channels % 8 == 0, (
        f"base_channels={base_channels} must be divisible by 8"
    )
    assert time_embedding_dim % 2 == 0, (
        f"time_embedding_dim={time_embedding_dim} must be even"
    )

    if dataset_name not in ("anime", "celeba"):
        raise ValueError(f"unknown --dataset-name {dataset_name!r}")
    celeba = dataset_name == "celeba"
    if image_size not in (64, 128):
        raise ValueError("--image-size must be 64 or 128")
    sfx = "" if image_size == 64 else f"_{image_size}"
    if celeba:
        arr = np.load(f"./.preprocessed/celebamask_faces{sfx}.npy")
        real_stats = compute_real_stats(
            arr,
            batch_size=batch_size,
            cache_path=f"./.preprocessed/celebamask_stats{sfx}.npz",
        )
        mask_path = mask_path or f"./.preprocessed/celebamask_masks{sfx}.npy"
    elif image_size == 64:
        arr = preprocess_all("./data/anime-faces")
        real_stats = compute_real_stats(arr, batch_size=batch_size)
        mask_path = mask_path or "./.preprocessed/anime_faces_masks.npy"
    else:
        # 128 px anime: the 64 px sources upscaled by Real-ESRGAN anime-6B
        # (``experiments/upscale_anime.py build``); FID is therefore measured
        # against the upscaler's rendering of the data.  Masks are rasterised
        # from the stored landmarks at this size (``layouts.rasterize``
        # reproduces the 64 px training masks to within 0.2 % of pixels).
        arr = np.load("./.preprocessed/anime_faces_128_anime6b.npy")
        arr = arr.astype(np.float32) / 127.5 - 1.0
        real_stats = compute_real_stats(
            arr, batch_size=batch_size, cache_path="./.preprocessed/anime_stats_128.npz"
        )
        mask_path = mask_path or "./.preprocessed/anime_faces_masks_128.npy"
        if not Path(mask_path).exists():
            lm = np.load("./.preprocessed/anime_faces_landmarks.npy")
            np.save(
                mask_path,
                (rasterize(lm, size=image_size) * 255).round().astype(np.uint8),
            )

    masks = load_masks(mask_path)[..., :cond_channels] if cond_channels > 0 else None
    if masks is not None and masks.shape[-1] < cond_channels:
        raise ValueError(
            f"{mask_path} has {masks.shape[-1]} channels, need {cond_channels}"
        )
    # Curation applies to *training* only; the FID reference stays the full
    # set so scores remain comparable across runs.
    if celeba:
        keep = np.flatnonzero(np.load(f"./.preprocessed/celebamask_keep{sfx}.npy"))
    elif min_landmark_score > 0:
        keep = curated_indices(load_landmark_scores(), min_landmark_score)
    else:
        keep = np.arange(len(arr))
    print(f"curation: keeping {len(keep)} of {len(arr)} images")
    arr = arr[keep]
    masks = None if masks is None else masks[keep]
    aug = data.animefaces.AugmentationConfig()
    if not color_jitter:  # flips only; jitter widens the photometric marginals
        aug = data.animefaces.AugmentationConfig(
            brightness_max=0.0, contrast_range=(1.0, 1.0), saturation_range=(1.0, 1.0)
        )
    dataset, batches_per_epoch = data.animefaces.wrap_dataset(
        arr, masks, batch_size=batch_size, seed=seed, aug_config=aug
    )
    eval_masks = None
    if cond_channels > 0 and celeba:
        bank = np.load(f"./.preprocessed/celebamask_eval_masks{sfx}.npy")
        idx = np.random.default_rng(seed).choice(len(bank), 5000)
        eval_masks = bank[idx, ..., :cond_channels].astype(np.float32) / 255.0
    elif cond_channels > 0:
        eval_masks = LayoutPrior.load().sample_masks(5000, seed=seed, size=image_size)[
            ..., :cond_channels
        ]

    hparams = {
        "base_channels": base_channels,
        "time_embedding_dim": time_embedding_dim,
        "n_blocks": n_blocks,
        "in_channels": 3 + (cond_channels + 1 if cond_channels else 0),
        "out_channels": 3,
        "cond_channels": cond_channels,
        "region_pool": region_pool,
        "image_size": image_size,
        "flow_map": flow_map,
        "mask_input": mask_input,
    }
    key = jax.random.key(seed)
    key, sk1 = jax.random.split(key)
    unet = UNet(key=sk1, **hparams)
    model = ImageFM(unet, hparams=hparams, n_steps=n_steps)
    if init_from:
        loaded = ImageFM.load(init_from, UNet.from_hparams)
        # Older checkpoints lack keys that were added with their default
        # value later (image_size, flow_map); compare with defaults filled in.
        defaults = {"image_size": 64, "flow_map": 0, "mask_input": 1}
        old = {**defaults, **loaded.hparams}
        if {k: v for k, v in old.items() if k != "flow_map"} != {
            k: v for k, v in hparams.items() if k != "flow_map"
        }:
            raise ValueError(f"--init-from architecture {loaded.hparams} != {hparams}")
        if loaded.hparams.get("flow_map", 0) == flow_map:
            model.net_theta = loaded.net_theta
        else:
            # A velocity checkpoint into a flow-map network: everything but
            # the (zero-initialised) span projection is transplanted, so the
            # fine-tune starts as the loaded model on the diagonal s = t.
            model.net_theta = eqx.tree_at(
                lambda m: (m.in_conv, m.out_conv, m.time_mlp, m.blocks, m.region_pools),
                unet,
                (
                    loaded.net_theta.in_conv,
                    loaded.net_theta.out_conv,
                    loaded.net_theta.time_mlp,
                    loaded.net_theta.blocks,
                    loaded.net_theta.region_pools,
                ),
            )
        print(f"initialised from {init_from}")
    training.run(
        key,
        model,
        dataset,
        batches_per_epoch,
        TrainConfig(
            n_epochs=n_epochs,
            init_lr=init_lr,
            warmup_steps=warmup_steps,
            lr_end_frac=lr_end_frac,
            ema_decay=ema_decay,
            cond_channels=cond_channels,
            cond_dropout=cond_dropout,
            flow_map_frac=flow_map_frac if flow_map else 0.0,
        ),
        RunConfig(
            outpath=outpath,
            eval_every=eval_every,
            early_stop_patience=early_stop_patience,
            fid_batch_size=fid_batch_size,
            fid_n_steps=fid_n_steps,
        ),
        real_stats=real_stats,
        eval_masks=eval_masks,
    )
    print(f"Saved model to {outpath}")


if __name__ == "__main__":
    app()
