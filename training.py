"""Bookkeeping around the image-model training loop.

``models.imagefm.train_on_image`` is a bare optimisation loop that yields once
per epoch.  This module consumes it and layers on everything a *run* needs
that the model does not: checkpoints, sample PNGs from fixed noise, FID on a
schedule, best-checkpoint retention, early stopping, and ``losses.json``.
Keeping these here means ``imagefm.py`` depends only on the maths.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import jax
import PIL.Image as Pilimage
from jaxtyping import PRNGKeyArray
from tqdm import tqdm

from data.animefaces import to_uint8
from metrics import evaluate_fid
from models.imagefm import ImageFM, TrainConfig, train_on_image


@dataclass(frozen=True)
class RunConfig:
    """Where to write, how often to evaluate, and when to stop.

    Attributes:
        outpath: Checkpoint path, overwritten every epoch.  Samples,
            ``best_model.eqx`` and ``losses.json`` go in the same directory.
        eval_every: Evaluate FID every N epochs (0 = never).  FID is measured
            on the epoch clock, so each score lands on the record holding the
            loss it belongs with.
        early_stop_patience: Stop after this many consecutive FID evaluations
            that degrade beyond a 1% tolerance (0 = disabled).
        fid_n_samples: Generated images per FID evaluation.  FID is biased in
            ``n``, so the value is recorded next to each score.
        fid_batch_size: Images per sampling / Inception batch during FID.
            Sampling dominates the evaluation cost (a 64-step Dopri5 solve is
            ~380 network evaluations per batch), so use the largest batch
            that fits; 256 is fine for the models in this repo on a 24 GB card
            when nothing else shares it.
    """

    outpath: str
    eval_every: int = 1
    early_stop_patience: int = 0
    fid_n_samples: int = 5000
    fid_batch_size: int = 256


def save_sample(model: ImageFM, outdir: Path, noise: jax.Array, epoch: int) -> None:
    """Save a sample from fixed noise, both as the latest and as a per-epoch file.

    Because the noise is fixed, the ``sample_epoch_*.png`` series shows how the
    generator's output for one latent evolves over the run.
    """
    img = Pilimage.fromarray(to_uint8(model.generate(noise)[0]))
    img.save(str(outdir / "sample.png"))
    img.save(str(outdir / f"sample_epoch_{epoch:04d}.png"))


def run(
    key: PRNGKeyArray,
    model: ImageFM,
    dataset,
    batches_per_epoch: int,
    train_cfg: TrainConfig,
    run_cfg: RunConfig,
    real_stats: dict | None = None,
) -> ImageFM:
    """Train ``model`` with checkpointing, sampling, FID and early stopping.

    ``losses.json`` is a list of per-epoch records ``{"epoch", "loss"}`` with
    ``"fid"`` and ``"fid_n_samples"`` added on evaluated epochs; it is what
    ``generate_figures.py`` reads.

    Args:
        key: JAX PRNG key; split between training, the fixed sample noise and
            FID generation.
        model: The ``ImageFM`` instance to train (updated in place).
        dataset: See ``train_on_image``.
        batches_per_epoch: See ``train_on_image``.
        train_cfg: Optimisation and loss hyperparameters.
        run_cfg: Bookkeeping settings.
        real_stats: Real-image Inception statistics for FID.  ``None`` skips
            FID, and with it best-checkpoint tracking and early stopping.

    Returns:
        The trained ``ImageFM`` model.
    """
    key, sample_key, train_key = jax.random.split(key, num=3)
    outdir = Path(run_cfg.outpath).parent
    outdir.mkdir(parents=True, exist_ok=True)

    first = next(iter(dataset))
    image_shape = (first[0] if isinstance(first, tuple) else first).shape[1:]
    sample_noise = jax.random.normal(sample_key, (1, *image_shape))

    history: list[dict] = []
    best_fid = float("inf")
    fid_patience = 0

    epochs = train_on_image(train_key, model, dataset, batches_per_epoch, train_cfg)
    for epoch, loss in (pbar := tqdm(epochs, desc="run", total=train_cfg.n_epochs)):
        record: dict = {"epoch": epoch, "loss": loss}
        history.append(record)
        pbar.set_postfix({"loss": f"{loss:.4f}"})

        model.save(run_cfg.outpath)
        save_sample(model, outdir, sample_noise, epoch)

        evaluate = real_stats is not None and run_cfg.eval_every
        if evaluate and (epoch + 1) % run_cfg.eval_every == 0:
            key, eval_key = jax.random.split(key)
            fid = evaluate_fid(
                model,
                real_stats,
                eval_key,
                n_samples=run_cfg.fid_n_samples,
                batch_size=run_cfg.fid_batch_size,
            )
            print(f"\nEpoch {epoch}: FID = {fid:.2f}", flush=True)
            record["fid"] = round(fid, 2)
            record["fid_n_samples"] = run_cfg.fid_n_samples

            if fid < best_fid * 1.01:
                if fid < best_fid:
                    best_fid = fid
                    model.save(str(outdir / "best_model.eqx"))
                fid_patience = 0
            else:
                fid_patience += 1
                print(f"  FID degradation {fid_patience}/{run_cfg.early_stop_patience}")

        with (outdir / "losses.json").open("w") as f:
            json.dump(history, f, indent=2)

        if 0 < run_cfg.early_stop_patience <= fid_patience:
            print(f"Early stopping at epoch {epoch}: best FID {best_fid:.2f}")
            break

    return model
