"""Flow matching on images with x-prediction, generic over the learned denoiser.

The recipe (Lipman et al. 2023; the x-prediction form of Li & He 2026):

    x_t = t x_1 + (1 - t) x_0,      x_0 ~ N(0, I),  x_1 ~ data,  t in [0, 1]

The learned denoiser predicts the clean image, ``x_hat(x_t, m, t)``,
conditioned on the layout masks ``m``; the velocity field is derived from it,

    v = (x_hat - x_t) / max(1 - t, floor),

and trained with the plain conditional flow-matching loss in velocity space,

    L = E || v - (x_1 - x_0) ||^2  =  E || x_hat - x_1 ||^2 / max(1 - t, floor)^2,

so fine detail decided late in ``t`` is up-weighted, with the ``1 / (1 - t)^2``
weight bounded by the floor (0.05, as in JiT).  The minimiser is
``E[x_1 | x_t, m]``; no auxiliary term is added to it, because any loss on
``x_hat`` against the sharp ``x_1`` is irreducible at that optimum and only
biases the denoiser toward hedged predictions (derived and measured: edge,
line, palette and mask-weighted variants all hurt or did nothing).

Sampling integrates ``dx/dt = v(x, t)`` from noise at ``t = 0`` to ``t = 1``
with the explicit midpoint scheme in 8 fixed steps (16 denoiser evaluations;
measured to match Dopri5 with 16 steps at a sixth of the cost).  The same
denoiser, given a second time ``s``, is read as a *flow map* ``x_s = x_t +
(s - t) u(x_t, m, t, s)`` for the distilled one- and two-jump samplers
(``distil.py``).

``net`` is any ``Denoiser``; the module never imports the U-Net.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import diffrax
import equinox as eqx
import jax
import optax
from beartype import beartype
from einops import rearrange
from jaxtyping import Array, Float, jaxtyped

from tinyflow import distil, trainer

if TYPE_CHECKING:
    from jaxtyping import PRNGKeyArray

FLOOR = 0.05
T_END = 1.0 - 1e-3
N_STEPS = 8


@runtime_checkable
class Denoiser(Protocol):
    """The denoiser's signature: one sample in, its denoised prediction out.

    ``s``, the flow map's second time, is only given by the distilled samplers.
    """

    def __call__(
        self,
        x: Float[Array, " H W 3"],
        masks: Float[Array, " H W 3"],
        t: Float[Array, ""],
        s: Float[Array, ""] | None = None,
    ) -> Float[Array, " H W 3"]:
        """The clean image predicted from ``x`` at ``t``, given ``masks``."""
        ...


def velocity(net: Denoiser, x, masks, t):
    """``(x_hat - x) / max(1 - t, FLOOR)`` for one sample."""
    t = jax.numpy.asarray(t, x.dtype)
    return (net(x, masks, t) - x) / jax.numpy.maximum(1 - t, FLOOR)


def mean_velocity(net: Denoiser, x, masks, t, s):
    """The flow map's velocity averaged over ``[t, s]``: ``x_s = x + (s - t) u``.

    ``u(x_t, t, s) = (1 / (s - t)) * int_t^s v(x_tau, tau) dtau`` along the
    trajectory through ``x_t``, so one jump is exact by construction; as
    ``s -> t`` it is the instantaneous ``velocity``.

    The denoiser still predicts an image, ``x_hat(x_t, t, s)``: the endpoint at
    ``t = 1`` of the straight line through ``x_t`` with slope ``u``, so
    ``u = (x_hat - x_t) / (1 - t)`` -- divided by the time left to ``t = 1``,
    not by ``s - t``.  Equivalently, a jump mixes ``x_t`` and ``x_hat`` by the
    ratio of the remaining times,

        x_s = (1 - s) / (1 - t) * x_t + (s - t) / (1 - t) * x_hat,

    which is Euler's step with ``x_hat`` in place of ``E[x_1 | x_t]``.  It is
    a reparametrisation, not an approximation -- every ``u`` has an ``x_hat``
    -- and it buys three things: the output keeps the velocity model's meaning
    (``x_hat -> E[x_1 | x_t]`` as ``s -> t``), so the student starts as the
    teacher; it stays on the data's scale; and a jump that ends at ``s = 1``
    returns ``x_hat`` itself.  The same form is CTM's (Kim et al. 2024:
    ``G = (s/t) x_t + (1 - s/t) g`` with ``g -> E[x | x_t]``; other weightings
    trained unstably) and X-Pred MeanFlow's (Xie et al. 2026:
    ``u = (z_t - x_hat) / t``), both with the data at ``t = 0``.  The floor
    never binds here: the trained jumps start at ``t <= 1/2``.
    """
    t, s = jax.numpy.asarray(t, x.dtype), jax.numpy.asarray(s, x.dtype)
    return (net(x, masks, t, s) - x) / jax.numpy.maximum(1 - t, FLOOR)


def jump_step(net: Denoiser, x, masks, t, s):
    """One flow-map jump of one sample, ``x_t -> x_s``."""
    return x + (s - t) * mean_velocity(net, x, masks, t, s)


@eqx.filter_jit
def solve(
    net: Denoiser, x_0: jax.Array, masks: jax.Array, save_at: tuple[float, ...]
) -> jax.Array:
    """The ODE from noise ``(B, H, W, 3)``, states at ``save_at``: ``(B, S, H, W, 3)``.

    The midpoint scheme with 8 fixed steps (module docstring); the times in
    ``save_at`` should be step boundaries (``k * T_END / 8``) to be exact.
    """
    term = diffrax.ODETerm(lambda t, x, m: velocity(net, x, m, t))
    saveat = diffrax.SaveAt(ts=list(save_at))

    def one(x, m):
        return diffrax.diffeqsolve(
            term,
            diffrax.Midpoint(),
            0.0,
            T_END,
            T_END / N_STEPS,
            x,
            args=m,
            saveat=saveat,
        ).ys

    return jax.vmap(one)(x_0, masks)


@eqx.filter_jit
def _jump(net: Denoiser, x_0: jax.Array, masks: jax.Array, n_jumps: int):
    return jax.vmap(lambda x, m: distil.jump(jump_step, net, x, m, n_jumps))(x_0, masks)


class ImageFM(trainer.Checkpoint):
    """A ``Denoiser`` plus everything that turns it into a generator.

    ``trainer.Checkpoint`` holds the denoiser and saves and loads it.
    """

    net: Denoiser

    def sample(self, x_0: jax.Array, masks: jax.Array, n_jumps: int = 0) -> jax.Array:
        """Images from noise ``(B, H, W, 3)``, one layout per sample.

        ``n_jumps = 0`` solves the ODE (midpoint, 8 steps); ``n_jumps > 0`` uses
        the distilled flow map in that many evaluations.
        """
        if n_jumps:
            return _jump(self.net, x_0, masks, n_jumps)
        return solve(self.net, x_0, masks, (T_END,))[:, -1]

    @staticmethod
    @jaxtyped(typechecker=beartype)
    def loss(
        net: Denoiser,
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


@dataclass(frozen=True)
class FlowTrainer(trainer.Trainer):
    """The anime-face training recipe: ``Trainer``'s defaults plus the draw of ``t``.

    Adam at a peak of 1e-3 with a 500-step linear warm-up and a cosine decay to
    1 % over ``schedule_steps`` (200 epochs), of which only 125 epochs are run:
    FID peaks when the integrated learning rate reaches about 100 peak-epochs
    and drifts up after, so the run ends there.  A constant 1e-3 diverged in
    one epoch on four occasions; the schedule never did.  Gradients are clipped
    at global norm 1; the published weights are the EMA (decay 0.999).  ``t``
    is logit-normal, ``sigmoid(N(-0.8, 1))``: more mass at high noise, and a
    thin tail at ``t -> 1`` balancing the ``1 / (1 - t)^2`` loss weight there.
    """

    t_mu: float = -0.8
    t_sigma: float = 1.0

    def loss(self, net: Any, batch: Any, key: PRNGKeyArray) -> Float[Array, ""]:
        """``ImageFM.loss`` on ``(images, masks)``, drawing ``t`` and the noise."""
        images, masks = batch
        k_t, k_noise = jax.random.split(key)
        t = jax.nn.sigmoid(
            self.t_mu + self.t_sigma * jax.random.normal(k_t, (len(images),))
        )
        noise = jax.random.normal(k_noise, images.shape)
        return ImageFM.loss(net, t, noise, images, masks)


@dataclass(frozen=True)
class FaceMapTrainer(distil.MapTrainer):
    """Distillation of the anime-face sampler: jumps read through x-prediction."""

    def jump_step(self, net: Any, x, cond, t, s) -> jax.Array:
        """``flow.jump_step``."""
        return jump_step(net, x, cond, t, s)
