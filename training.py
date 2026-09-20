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
import jax.numpy as jnp
import numpy as np
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
        fid_n_steps: Sampler steps for training-time FID; the checkpoint's
            own ``n_steps`` is restored afterwards.  16 Dopri5 steps score
            within 0.5 FID of 64 on this data at a quarter of the cost.
    """

    outpath: str
    eval_every: int = 1
    early_stop_patience: int = 0
    fid_n_samples: int = 5000
    fid_batch_size: int = 256
    fid_n_steps: int = 16


class MaskedSampler:
    """``generate(noise)`` for a layout-conditioned model, masks drawn from a bank.

    ``metrics.evaluate_fid`` only knows ``generate(noise)``; a pure conditional
    model needs a layout for every sample.  The bank is cycled in order, so
    a fixed bank gives a reproducible evaluation.
    """

    def __init__(self, model: ImageFM, masks: np.ndarray):
        """``masks``: ``(N, H, W, K)`` in ``[0, 1]`` sampled from the layout prior."""
        self.model, self.masks, self._pos = model, jnp.asarray(masks), 0

    def generate(self, x_0):
        """Sample ``len(x_0)`` images with the next masks from the bank."""
        idx = (self._pos + jnp.arange(len(x_0))) % len(self.masks)
        self._pos = (self._pos + len(x_0)) % len(self.masks)
        return self.model.generate(x_0, self.masks[idx])


class MapSampler:
    """``generate(noise)`` through ``ImageFM.generate_map`` (flow-map jumps)."""

    def __init__(self, model: ImageFM, masks: np.ndarray | None, n_steps: int):
        """``masks``: bank as in ``MaskedSampler`` or ``None``; ``n_steps`` jumps."""
        self.model, self.n_steps, self._pos = model, n_steps, 0
        self.masks = None if masks is None else jnp.asarray(masks)

    def generate(self, x_0):
        """Sample ``len(x_0)`` images in ``n_steps`` jumps."""
        m = None
        if self.masks is not None:
            idx = (self._pos + jnp.arange(len(x_0))) % len(self.masks)
            self._pos = (self._pos + len(x_0)) % len(self.masks)
            m = self.masks[idx]
        return self.model.generate_map(x_0, m, self.n_steps)


def save_sample(sampler, outdir: Path, noise: jax.Array, epoch: int) -> None:
    """Save a sample from fixed noise, both as the latest and as a per-epoch file.

    Because the noise is fixed, the ``sample_epoch_*.png`` series shows how the
    generator's output for one latent evolves over the run.
    """
    img = Pilimage.fromarray(to_uint8(sampler.generate(noise)[0]))
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
    eval_masks: np.ndarray | None = None,
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
        eval_masks: For a layout-conditioned model, a bank of prior-sampled
            masks ``(N, H, W, K)`` used for the sample PNGs and FID; ``None``
            samples with ``model.generate(noise)`` (unconditional / null token).

    Returns:
        The trained ``ImageFM`` model.
    """
    key, sample_key, train_key = jax.random.split(key, num=3)
    outdir = Path(run_cfg.outpath).parent
    outdir.mkdir(parents=True, exist_ok=True)

    first = next(iter(dataset))
    image_shape = (first[0] if isinstance(first, tuple) else first).shape[1:]
    sample_noise = jax.random.normal(sample_key, (1, *image_shape))
    # The sample PNG always uses the bank's first mask; FID cycles the bank.
    png_sampler = (
        MaskedSampler(model, eval_masks[:1]) if eval_masks is not None else model
    )

    history: list[dict] = []
    best_fid = float("inf")
    fid_patience = 0

    epochs = train_on_image(train_key, model, dataset, batches_per_epoch, train_cfg)
    for epoch, loss in (pbar := tqdm(epochs, desc="run", total=train_cfg.n_epochs)):
        record: dict = {"epoch": epoch, "loss": loss}
        history.append(record)
        pbar.set_postfix({"loss": f"{loss:.4f}"})
        if not np.isfinite(loss):
            print(f"\nEpoch {epoch}: loss is {loss}; stopping.", flush=True)
            with (outdir / "losses.json").open("w") as f:
                json.dump(history, f, indent=2)
            break

        model.save(run_cfg.outpath)
        save_sample(png_sampler, outdir, sample_noise, epoch)

        evaluate = real_stats is not None and run_cfg.eval_every
        if evaluate and (epoch + 1) % run_cfg.eval_every == 0:
            key, eval_key = jax.random.split(key)
            fid_sampler = (
                MaskedSampler(model, eval_masks) if eval_masks is not None else model
            )
            n_steps, model.n_steps = model.n_steps, run_cfg.fid_n_steps
            fid = evaluate_fid(
                fid_sampler,
                real_stats,
                eval_key,
                n_samples=run_cfg.fid_n_samples,
                batch_size=run_cfg.fid_batch_size,
                image_size=image_shape[0],
            )
            model.n_steps = n_steps
            print(f"\nEpoch {epoch}: FID = {fid:.2f}", flush=True)
            record["fid"] = round(fid, 2)
            record["fid_n_samples"] = run_cfg.fid_n_samples
            if train_cfg.flow_map_frac > 0:
                # The flow map's own samplers: one and two jumps (1 / 2 NFE).
                for n_jumps in (1, 2):
                    jump = MapSampler(model, eval_masks, n_jumps)
                    f = evaluate_fid(
                        jump,
                        real_stats,
                        eval_key,
                        n_samples=run_cfg.fid_n_samples,
                        batch_size=run_cfg.fid_batch_size,
                        image_size=image_shape[0],
                    )
                    print(f"  {n_jumps}-jump FID = {f:.2f}", flush=True)
                    record[f"fid_map{n_jumps}"] = round(f, 2)

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
