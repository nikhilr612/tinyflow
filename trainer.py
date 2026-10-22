"""The optimisation loop and its bookkeeping, shared by every model trained here.

``Trainer`` is the recipe: Adam on a linear warm-up and a cosine decay,
gradients clipped by global norm, and an exponential moving average of the
weights, warmed up as ``(1 + n) / (10 + n)`` so early averages are not
dominated by the initialisation.  The optimiser steps the raw weights; the
averaged ones are what ``fit`` hands back.  A subclass supplies ``loss`` -- the
part that is about the model -- and inherits the rest.  Every feature has a
neutral value that turns it off:

    warmup_steps=0     no warm-up          end_lr_frac=1.0    a constant rate
    clip_norm=None     no clipping         ema_decay=None     no average

``ema_decay=1`` is rejected rather than read as "off": under the warm-up the
decay would then be ``(1 + n) / (10 + n)`` forever, a running mean of every
iterate since the initialisation.  Anything structural -- another optimiser or
schedule -- is a method override.

The jit contract.  A trainer is a frozen dataclass of Python scalars, so it is
hashable and ``eqx.filter_jit`` treats it as static: one compilation per set of
hyperparameters, and subclasses must keep to scalar fields.  ``loss`` is traced
inside the jitted step, so it is pure JAX and draws all of its randomness from
``key``.  The network is any pytree; its inexact-array leaves are the trained
parameters and everything else is held fixed.

``run`` consumes ``fit`` and owns the bookkeeping: a checkpoint, the loss
history and a row of samples after every epoch.
"""

from __future__ import annotations

import abc
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from PIL import Image
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator
    from pathlib import Path

    from jaxtyping import Array, Float, PRNGKeyArray


@dataclass(frozen=True)
class Trainer(abc.ABC):
    """Adam, warm-up + cosine decay, clipping and a weight average; ``loss`` abstract.

    ``schedule_steps`` is the length of the cosine, which may exceed the number
    of steps actually run.  The warm-up is capped at a tenth of it, so a short
    run is not all warm-up.
    """

    schedule_steps: int
    peak_lr: float = 1e-3
    warmup_steps: int = 500
    end_lr_frac: float = 0.01
    clip_norm: float | None = 1.0
    ema_decay: float | None = 0.999

    def __post_init__(self) -> None:
        """Reject an ``ema_decay`` outside ``[0, 1)``; ``None`` is off."""
        if self.ema_decay is not None and not 0 <= self.ema_decay < 1:
            raise ValueError(f"ema_decay must be in [0, 1) or None: {self.ema_decay}")

    @abc.abstractmethod
    def loss(self, net: Any, batch: Any, key: PRNGKeyArray) -> Float[Array, ""]:
        """The objective for one batch; what a subclass is about."""

    def schedule(self) -> optax.Schedule:
        """Linear warm-up to ``peak_lr``, cosine decay to ``end_lr_frac`` of it."""
        warmup = min(self.warmup_steps, self.schedule_steps // 10)
        return optax.warmup_cosine_decay_schedule(
            0.0,
            self.peak_lr,
            warmup,
            self.schedule_steps,
            self.peak_lr * self.end_lr_frac,
        )

    def optimizer(self) -> optax.GradientTransformation:
        """Adam on ``schedule``, after clipping by global norm if set."""
        adam = optax.adam(self.schedule())
        if self.clip_norm is None:
            return adam
        return optax.chain(optax.clip_by_global_norm(self.clip_norm), adam)

    @eqx.filter_jit
    def step(self, params, static, ema, opt_state, n, batch, key: PRNGKeyArray):
        """One optimiser step and the average's update; ``n`` is the step count."""
        loss, grads = jax.value_and_grad(
            lambda p: self.loss(eqx.combine(p, static), batch, key)
        )(params)
        updates, opt_state = self.optimizer().update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        if self.ema_decay is not None:
            decay = jnp.minimum(self.ema_decay, (1 + n) / (10 + n))
            ema = optax.incremental_update(params, ema, 1 - decay)
        return params, ema, opt_state, loss

    def fit[N](
        self,
        net: N,
        batches: Iterable,
        n_epochs: int,
        batches_per_epoch: int,
        key: PRNGKeyArray,
    ) -> Iterator[tuple[int, float, N]]:
        """Train ``net``; yields ``(epoch, mean loss, averaged net)`` once per epoch.

        ``batches`` is one stream for the whole run (``n_epochs *
        batches_per_epoch`` batches are drawn from it), so the caller's
        pipeline decides how passes over the data are shuffled.
        """
        params, static = eqx.partition(net, eqx.is_inexact_array)
        opt_state = self.optimizer().init(params)
        ema = params if self.ema_decay is not None else None
        stream = iter(batches)
        for epoch in range(n_epochs):
            total = jnp.zeros(())
            for b in range(batches_per_epoch):
                key, sk = jax.random.split(key)
                n = jnp.asarray(epoch * batches_per_epoch + b + 1)  # an int recompiles
                params, ema, opt_state, loss = self.step(
                    params, static, ema, opt_state, n, next(stream), sk
                )
                total += loss
            averaged = params if ema is None else ema
            yield epoch, float(total) / batches_per_epoch, eqx.combine(averaged, static)


def to_uint8(x) -> np.ndarray:
    """``[-1, 1]`` floats to ``uint8`` pixels, clipped: the model may overshoot."""
    return np.clip((np.asarray(x) + 1) * 127.5, 0, 255).astype(np.uint8)


def save_grid(images: np.ndarray, path: Path) -> None:
    """Write ``(N, H, W, 3)`` images in ``[-1, 1]`` as one row of tiles."""
    Image.fromarray(to_uint8(np.concatenate(list(images), axis=1))).save(path)


class Checkpoint(Protocol):
    """What ``run`` needs of a model: its network, and a way to save it."""

    net: Any

    def save(self, path: str) -> None:
        """Write the model to ``path``."""
        ...


def run[M: Checkpoint](
    fits: Iterator[tuple[int, float, Any]],
    model: M,
    outdir: Path,
    sample_fn: Callable[[M], Any],
    n_epochs: int,
) -> M:
    """Consume ``fits``; after each epoch save ``model.eqx``, ``losses.json``, samples.

    ``sample_fn(model)`` returns ``(N, H, W, 3)`` images in ``[-1, 1]``, written
    as the row ``sample_epoch_*.png``.  Drawn from fixed inputs, the series
    shows the model's output for the same latents evolving over the run.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch, loss, net in tqdm(fits, total=n_epochs, desc="train"):
        model.net = net
        history.append({"epoch": epoch, "loss": loss})
        model.save(str(outdir / "model.eqx"))
        (outdir / "losses.json").write_text(json.dumps(history, indent=1))
        save_grid(
            np.asarray(sample_fn(model)), outdir / f"sample_epoch_{epoch:04d}.png"
        )
    return model
