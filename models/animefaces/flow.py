"""Flow matching on images with x-prediction, generic over the denoising network.

The recipe (Lipman et al. 2023; the x-prediction form of Li & He 2026):

    x_t = t x_1 + (1 - t) x_0,      x_0 ~ N(0, I),  x_1 ~ data,  t in [0, 1]

The network predicts the clean image, ``x_hat(x_t, m, t)``, conditioned on the
layout masks ``m``; the velocity field is derived from it,

    v = (x_hat - x_t) / max(1 - t, floor),

and trained with the plain conditional flow-matching loss in velocity space,

    L = E || v - (x_1 - x_0) ||^2  =  E || x_hat - x_1 ||^2 / max(1 - t, floor)^2,

so fine detail decided late in ``t`` is up-weighted, with the ``1 / (1 - t)^2``
weight bounded by the floor (0.05, as in JiT).  The minimiser is
``E[x_1 | x_t, m]``; no auxiliary term is added to it, because any loss on
``x_hat`` against the sharp ``x_1`` is irreducible at that optimum and only
biases the network toward hedged predictions (derived and measured: edge,
line, palette and mask-weighted variants all hurt or did nothing).

Sampling integrates ``dx/dt = v(x, t)`` from noise at ``t = 0`` to ``t = 1``
with the explicit midpoint scheme in 8 fixed steps (16 network evaluations;
measured to match Dopri5 with 16 steps at a sixth of the cost).  The same
network, given a second time ``s``, is read as a *flow map* ``x_s = x_t +
(s - t) u(x_t, m, t, s)`` for the distilled one- and two-jump samplers.

``net`` is any callable ``(x: (H, W, 3), masks: (H, W, 3), t: scalar[, s]) ->
(H, W, 3)``; the class never imports the U-Net.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import diffrax
import equinox as eqx
import jax
import optax
from beartype import beartype
from einops import rearrange
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped

if TYPE_CHECKING:
    from collections.abc import Iterator

FLOOR = 0.05
T_END = 1.0 - 1e-3
N_STEPS = 8


def velocity(net: Any, x, masks, t):
    """``(x_hat - x) / max(1 - t, FLOOR)`` for one sample."""
    t = jax.numpy.asarray(t, x.dtype)
    return (net(x, masks, t) - x) / jax.numpy.maximum(1 - t, FLOOR)


def mean_velocity(net: Any, x, masks, t, s):
    """The flow map's velocity averaged over ``[t, s]``: ``x_s = x + (s - t) u``.

    ``u(x_t, t, s) = (1 / (s - t)) * int_t^s v(x_tau, tau) dtau`` along the
    trajectory through ``x_t``, so one jump is exact by construction; as
    ``s -> t`` it is the instantaneous ``velocity``.  Read through the same
    ``x_hat`` parametrisation, with the network's second time input.
    """
    t, s = jax.numpy.asarray(t, x.dtype), jax.numpy.asarray(s, x.dtype)
    return (net(x, masks, t, s) - x) / jax.numpy.maximum(1 - t, FLOOR)


class ImageFM:
    """A denoising network plus everything that turns it into a generator."""

    def __init__(self, net: Any, hparams: dict) -> None:
        """``hparams``: the network's constructor arguments, saved with the weights."""
        self.net = net
        self.hparams = hparams

    def save(self, path: str) -> None:
        """Write the weights to ``path`` and the constructor arguments beside them."""
        eqx.tree_serialise_leaves(path, self.net)
        Path(path + ".hparams").write_text(json.dumps(self.hparams))

    @classmethod
    def load(cls, path: str, skeleton_fn) -> ImageFM:
        """Rebuild with ``skeleton_fn(key, **hparams)`` and read the weights into it."""
        hparams = json.loads(Path(path + ".hparams").read_text())
        skeleton = skeleton_fn(key=jax.random.key(0), **hparams)
        return cls(eqx.tree_deserialise_leaves(path, skeleton), hparams)

    def generate(self, x_0: jax.Array, masks: jax.Array) -> jax.Array:
        """Images from noise ``(B, H, W, 3)`` by solving the ODE, one layout per sample.

        The solver is the midpoint scheme with 8 fixed steps (module docstring).
        """
        return _generate(self.net, x_0, masks)

    def jump(self, x_0: jax.Array, masks: jax.Array, n_jumps: int = 2) -> jax.Array:
        """Images from noise with the distilled flow map in ``n_jumps`` evaluations."""
        return _jump(self.net, x_0, masks, n_jumps)

    @staticmethod
    @jaxtyped(typechecker=beartype)
    def loss(
        net: Any,
        t: Float[Array, " B"],
        x_0: Float[Array, " B H W 3"],
        x_1: Float[Array, " B H W 3"],
        masks: Float[Array, " B H W 3"],
    ) -> Float[Array, ""]:
        """The conditional flow-matching loss in velocity space (module docstring)."""
        t_b = rearrange(t, "b -> b 1 1 1")
        x_t = t_b * x_1 + (1 - t_b) * x_0
        v = jax.vmap(lambda x, m, ti: velocity(net, x, m, ti))(x_t, masks, t)
        return optax.l2_loss(v, x_1 - x_0).mean()


@eqx.filter_jit
def _generate(net, x_0, masks):
    term = diffrax.ODETerm(lambda t, x, m: velocity(net, x, m, t))

    def solve(x, m):
        return diffrax.diffeqsolve(
            term, diffrax.Midpoint(), 0.0, T_END, T_END / N_STEPS, x, args=m
        ).ys[0]

    return jax.vmap(solve)(x_0, masks)


@eqx.filter_jit
def _jump(net, x_0, masks, n_jumps: int):
    x = x_0
    for k in range(n_jumps):
        t, s = k / n_jumps, (k + 1) / n_jumps
        x = x + (s - t) * jax.vmap(
            lambda xi, mi, t=t, s=s: mean_velocity(net, xi, mi, t, s)
        )(x, masks)
    return x


@dataclass(frozen=True)
class TrainConfig:
    """The optimisation recipe.

    Adam at a peak of 1e-3 with a 500-step linear warm-up and a cosine decay to
    1 % over ``schedule_epochs`` -- of which only ``n_epochs`` are run: FID peaks
    when the integrated learning rate reaches about 100 peak-epochs (epoch 125
    of a 200-epoch cosine) and drifts up after, so the run ends there.  A
    constant 1e-3 diverged in one epoch on four occasions; the schedule never
    did.  Gradients are clipped at global norm 1.  The published weights are an
    exponential moving average (decay 0.999, warmed up as ``(1 + n) / (10 + n)``
    so early averages are not dominated by the initialisation); the optimiser
    steps the raw weights.  ``t`` is logit-normal, ``sigmoid(N(-0.8, 1))``:
    more mass at high noise, and a thin tail at ``t -> 1`` balancing the
    ``1 / (1 - t)^2`` loss weight there.
    """

    n_epochs: int = 125
    schedule_epochs: int = 200
    peak_lr: float = 1e-3
    warmup_steps: int = 500
    end_lr_frac: float = 0.01
    ema_decay: float = 0.999
    t_mu: float = -0.8
    t_sigma: float = 1.0


def train(
    key: PRNGKeyArray, model: ImageFM, dataset, batches_per_epoch: int, cfg: TrainConfig
) -> Iterator[tuple[int, float]]:
    """Train ``model.net`` in place; yields ``(epoch, mean loss)`` once per epoch.

    ``model.net`` holds the EMA weights at every yield, so the caller can
    checkpoint or sample from it between epochs.
    """
    total = cfg.schedule_epochs * batches_per_epoch
    warmup = min(cfg.warmup_steps, total // 10)  # short runs: at most 10 %
    schedule = optax.warmup_cosine_decay_schedule(
        0.0, cfg.peak_lr, warmup, total, cfg.peak_lr * cfg.end_lr_frac
    )
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))
    params, static = eqx.partition(model.net, eqx.is_inexact_array)
    opt_state = opt.init(params)
    ema = params

    @eqx.filter_jit
    def step(key, params, ema, opt_state, n, images, masks):
        k_t, k_noise = jax.random.split(key)
        t = jax.nn.sigmoid(
            cfg.t_mu + cfg.t_sigma * jax.random.normal(k_t, (len(images),))
        )
        noise = jax.random.normal(k_noise, images.shape)
        loss, grads = jax.value_and_grad(
            lambda p: ImageFM.loss(eqx.combine(p, static), t, noise, images, masks)
        )(params)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        decay = jax.numpy.minimum(cfg.ema_decay, (1 + n) / (10 + n))
        ema = optax.incremental_update(params, ema, 1 - decay)
        return params, ema, opt_state, loss

    stream = iter(dataset)
    for epoch in range(cfg.n_epochs):
        total_loss = jax.numpy.zeros(())
        for b in range(batches_per_epoch):
            key, sk = jax.random.split(key)
            images, masks = next(stream)
            n = jax.numpy.asarray(
                epoch * batches_per_epoch + b + 1
            )  # an int would recompile
            params, ema, opt_state, loss = step(
                sk, params, ema, opt_state, n, images, masks
            )
            total_loss += loss
        model.net = eqx.combine(ema, static)
        yield epoch, float(total_loss) / batches_per_epoch
