"""Radar-to-optical translation: a stochastic interpolant from radar to optical.

The interpolant (Albergo, Boffi & Vanden-Eijnden 2023) bridges every radar
image ``sar`` to its paired optical image ``x_1`` (a data-dependent coupling,
Albergo et al. 2024), with a latent ``z`` between them:

    x_t = alpha(t) sar + beta(t) x_1 + gamma(t) z,        z ~ N(0, I),

    beta = sin^2(pi t / 2),   alpha = 1 - beta,   gamma = sqrt(a t (1 - t)),

with ``a = 4``: ``gamma`` peaks at 1 at ``t = 1/2``, the latent as loud as
the images there.  Measured against ``a = 1`` it gave -7 FID and -0.004
LPIPS, and the paper's linear ``beta = t`` +3 FID and +0.013 LPIPS.

The pairs are drawn jointly, so the velocity ``E[dx_t/dt | x_t, sar]`` is the
transport of ``p(optical | sar)``: the pairing enters the field.  The only
Gaussian is ``z``, which is what makes the samples of one radar image differ.
The flat ``beta`` is flat at ``t = 0``, so given the radar image ``x_t`` at a
small ``t0`` is ``N(alpha sar, gamma^2)`` up to ``beta(t0) ~ (pi t0)^2 / 4``:
the sampler starts there, not at the point mass ``t = 0``, which no ODE
spreads.

The denoiser predicts the optical image, ``x_hat(x_t, sar, meta, t)``.  With
``sar`` an input, ``x_hat`` fixes everything else:

    z_hat = (x_t - alpha sar - beta x_hat) / gamma,
    b = alpha' sar + beta' x_hat + gamma' z_hat.

Trained in velocity space, the error is the image error times one weight,

    b_hat - b = w(t) (x_hat - x_1),      w = beta' - beta gamma' / gamma,

where ``gamma' / gamma = (1 - 2t) / (2t (1 - t))``: ``a`` cancels.  ``w ~ 3
pi^2 t / 8`` near ``0``, where ``x_t`` is still the radar image and holds
nothing new, and ``~ 1 / (2 (1 - t))`` near ``1``, bounded there by the floor
on ``1 - t`` as in ``flow.py``.  As there, the minimiser is ``E[x_1 | x_t,
sar]`` and no auxiliary term is added.

Sampling integrates ``dx/dt = b`` from ``T0`` to ``t = 1``.  With ``x_hat``
held fixed the equation is linear in ``x``, and solved exactly over a step,

    x_s = alpha_s sar + beta_s x_hat + (gamma_s / gamma_t)(x_t - alpha_t sar
          - beta_t x_hat),

an exponential-integrator step: DDIM's form, the first-order DPM-Solver (Lu et
al. 2022), written for the bridge.  Two of them make each interval second
order -- a step to the midpoint, then the whole step with the prediction made
there -- for the evaluations of the explicit midpoint scheme.  It is exact
when ``x_hat`` is, and reaches ``gamma = 0`` at ``t = 1`` with no floor: the
last step returns ``x_hat``.  The equation is semilinear, ``r = gamma' /
gamma`` its linear part, singular at both ends; integrating ``r`` exactly
(``int r = log gamma``) leaves only the drift of ``x_hat`` within a step to
discretise.  An explicit solver on ``b`` must instead approximate ``gamma_s /
gamma_t`` by a polynomial in ``h r``, and ``|h r|`` is not small on the end
intervals: on a synthetic denoiser, diffrax's fixed-step solvers at the same
32 evaluations were 20-80 times less accurate, and only adaptive Dopri5 did
as well, at a cost that varies per input.  The steps are on ``grid``, ``t =
sin^2(pi u / 2)`` for uniform ``u``: dense at both ends.

The SDE with diffusion ``kappa gamma^2`` -- which vanishes at both ends, where
the score ``-z_hat / gamma`` diverges -- is solved the same way.  With
``x_hat`` fixed the residual ``x - alpha sar - beta x_hat`` is an
Ornstein-Uhlenbeck process, so the exact step decays it by a further
``exp(-kappa (s - t))`` and adds ``gamma_s sqrt(1 - exp(-2 kappa (s - t)))``
of fresh noise, keeping its variance at ``gamma_s^2``.  ``kappa = 0`` is the
ODE; ``kappa -> inf`` re-noises fully at every step.  The midpoint prediction
is made deterministically, so for ``kappa > 0`` the scheme is first order in
the noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from beartype import beartype
from einops import rearrange
from jaxtyping import Array, Float, jaxtyped

from tinyflow import trainer

if TYPE_CHECKING:
    from jaxtyping import PRNGKeyArray, UInt8

A = 4.0  # gamma's scale: peak noise sqrt(A) / 2 = 1 at t = 1/2
FLOOR = 0.05  # on 1 - t in the loss weight's gamma' / gamma
T_MIN = 1e-6  # on t there; only training draws come near it
T0 = 1e-3
N_STEPS = 16  # the sampler's default: 2 N_STEPS denoiser evaluations
COMPUTE_DTYPE = "bfloat16"  # the denoiser's forward passes, on float32 weights


def to_unit(images: UInt8[Any, "*b H W 3"]) -> Float[Array, "*b H W 3"]:
    """8-bit images to ``[-1, 1]``, on whichever device holds them.

    Batches stay ``uint8`` up to the device, a quarter of float32's bytes.
    """
    return images.astype("float32") / 127.5 - 1


def grid(n_steps: int) -> jax.Array:
    """``n_steps + 1`` times, ``T0`` to 1 as ``sin^2(pi u / 2)``: dense at both ends."""
    u = jnp.arange(n_steps + 1) / n_steps
    return T0 + (1 - T0) * jnp.sin(jnp.pi * u / 2) ** 2


@runtime_checkable
class Denoiser(Protocol):
    """The denoiser's signature: one sample in, its predicted optical image out."""

    def __call__(
        self,
        x: Float[Array, " H W 3"],
        sar: Float[Array, " H W 3"],
        meta: Float[Array, " M"],
        t: Float[Array, ""],
    ) -> Float[Array, " H W 3"]:
        """The optical image predicted from ``x`` at ``t``, given ``sar``, ``meta``."""
        ...


def coefficients(t):
    """``(alpha, beta, gamma)`` at ``t``."""
    beta = jnp.sin(jnp.pi * t / 2) ** 2
    return 1 - beta, beta, jnp.sqrt(A * t * (1 - t))


def weight(t):
    """``w = beta' - beta gamma' / gamma``: velocity error per unit image error.

    ``t`` and ``1 - t`` are floored in ``gamma' / gamma``, which bounds ``w``
    near ``1`` (module docstring).
    """
    d_beta = jnp.pi / 2 * jnp.sin(jnp.pi * t)
    d_log_gamma = (1 - 2 * t) / (2 * jnp.maximum(t, T_MIN) * jnp.maximum(1 - t, FLOOR))
    return d_beta - coefficients(t)[1] * d_log_gamma


def step(x, x_hat, sar, t, s, kappa: float = 0.0, xi=None):
    """``x_t -> x_s`` with ``x_hat`` held fixed: the exact solution over the step.

    The ODE's for ``kappa = 0``; for ``kappa > 0`` the SDE's with diffusion
    ``kappa gamma^2``, ``xi`` its standard-normal draw (module docstring).
    """
    alpha_t, beta_t, gamma_t = coefficients(t)
    alpha_s, beta_s, gamma_s = coefficients(s)
    residual = x - alpha_t * sar - beta_t * x_hat  # gamma_t z_hat
    decay = jnp.exp(-kappa * (s - t))
    x_s = alpha_s * sar + beta_s * x_hat + gamma_s / gamma_t * decay * residual
    if xi is None:
        return x_s
    return x_s + gamma_s * jnp.sqrt(1 - decay**2) * xi


def cast(net: Any, dtype: str) -> Any:
    """``net`` with its floating-point leaves in ``dtype``: the mixed-precision policy.

    The weights stay float32 everywhere else (the optimiser, the average, the
    checkpoint); the denoiser casts its inputs to its weights' dtype and its
    output back, so this one cast is all a forward pass in ``dtype`` needs.
    """
    return jax.tree.map(
        lambda a: a.astype(dtype) if eqx.is_inexact_array(a) else a, net
    )


@eqx.filter_jit
def solve(
    net: Denoiser,
    sar: jax.Array,
    meta: jax.Array,
    z: jax.Array,
    kappa: float = 0.0,
    key: PRNGKeyArray | None = None,
    n_steps: int = N_STEPS,
    trajectory: bool = False,
) -> jax.Array:
    """Optical images ``(B, H, W, 3)`` from radar images, their ``meta``, noise ``z``.

    Starts at ``T0`` from ``alpha sar + gamma z`` and steps to ``t = 1`` on
    ``grid(n_steps)``, two exact steps per interval (module docstring): ``2
    n_steps`` denoiser evaluations.  ``kappa > 0`` samples the SDE instead,
    drawing its noise from ``key``.  ``trajectory`` returns every state on the
    grid instead, ``(B, n_steps + 1, H, W, 3)``.
    """
    if kappa and key is None:
        raise ValueError("kappa > 0 samples an SDE: pass a key for its noise")
    ts = grid(n_steps)
    alpha0, _, gamma0 = coefficients(ts[0])

    def one(s, m, n, k):
        def interval(x, tuk):
            t, u, ki = tuk
            mid = step(x, net(x, s, m, t), s, t, (t + u) / 2)
            x_hat = net(mid, s, m, (t + u) / 2)
            xi = jax.random.normal(ki, x.shape) if kappa else None
            x = step(x, x_hat, s, t, u, kappa, xi)
            return x, x if trajectory else None

        steps = (ts[:-1], ts[1:], jax.random.split(k, n_steps))
        x_0 = alpha0 * s + gamma0 * n
        x_1, xs = jax.lax.scan(interval, x_0, steps)
        return jnp.concatenate([x_0[None], xs]) if trajectory else x_1

    keys = jax.random.split(jax.random.key(0) if key is None else key, len(sar))
    return jax.vmap(one)(sar, meta, z, keys)


class Bridge(trainer.Checkpoint):
    """A ``Denoiser`` plus everything that turns it into a radar-to-optical sampler.

    ``trainer.Checkpoint`` holds the denoiser and saves and loads it.
    """

    net: Denoiser

    def sample(
        self,
        sar: jax.Array,
        meta: jax.Array,
        z: jax.Array,
        kappa: float = 0.0,
        key: PRNGKeyArray | None = None,
        n_steps: int = N_STEPS,
        trajectory: bool = False,
    ) -> jax.Array:
        """Optical images in ``[-1, 1]`` for radar images ``(B, H, W, 3)``, also in it.

        ``z`` ``(B, H, W, 3)`` is the noise, one draw per sample; ``kappa > 0``
        samples the SDE, its noise drawn from ``key``; ``2 n_steps`` denoiser
        evaluations.  The denoiser runs in ``COMPUTE_DTYPE``; the state is
        integrated in float32.  ``trajectory``: every state, as in ``solve``.
        """
        net = cast(self.net, COMPUTE_DTYPE)
        return solve(net, sar, meta, z, kappa, key, n_steps, trajectory)

    @staticmethod
    @jaxtyped(typechecker=beartype)
    def loss(
        net: Denoiser,
        t: Float[Array, " B"],
        z: Float[Array, " B H W 3"],
        sar: Float[Array, " B H W 3"],
        meta: Float[Array, " B M"],
        x_1: Float[Array, " B H W 3"],
    ) -> Float[Array, ""]:
        """The velocity-space loss: image error times ``weight`` (module docstring)."""
        coeffs = coefficients(t)
        alpha, beta, gamma = (rearrange(c, "b -> b 1 1 1") for c in coeffs)
        x_t = alpha * sar + beta * x_1 + gamma * z
        x_hat = jax.vmap(net)(x_t, sar, meta, t)
        w = rearrange(weight(t), "b -> b 1 1 1")
        return optax.l2_loss(w * x_hat, w * x_1).mean()


@dataclass(frozen=True)
class BridgeTrainer(trainer.Trainer):
    """The SEN-Bridge recipe: ``Trainer``'s, transformer defaults, drawing ``t``, ``z``.

    Adam at a peak of 1e-4 (DiT and SiT train at 1e-4) after a 1000-step
    warm-up (SD3), with ``Trainer``'s cosine and clipping; the weight average
    at 0.9999 (DiT, SiT, JiT).  The denoiser runs in ``COMPUTE_DTYPE`` on
    float32 weights (``cast``).

    ``t`` is uniform (Albergo et al. 2024).  ``gamma`` decays like ``sqrt(1 -
    t)``, so the low-noise regime where texture is resolved (``gamma < 0.1``)
    lies in ``t > 0.99``: SD3's logit-normal ``sigmoid(N(0, 1))`` puts ~2e-6 of
    its mass there, and its denoiser was worse than the identity at ``t =
    0.999`` (more sampler steps gave worse samples); the sampler's own grid,
    ``sin^2(pi u / 2)``, puts 6 % there but only 0.64 of uniform's density at
    ``t = 1/2``, where the layout is decided.  Uniform, with 1 % above 0.99,
    beat both.
    """

    peak_lr: float = 1e-4
    warmup_steps: int = 1000
    ema_decay: float | None = 0.9999

    def loss(self, net: Any, batch: Any, key: PRNGKeyArray) -> Float[Array, ""]:
        """``Bridge.loss`` on ``(optical, (sar, meta))``, drawing ``t`` and ``z``."""
        optical, (sar, meta) = batch
        x_1, sar = to_unit(optical), to_unit(sar)
        k_t, k_z = jax.random.split(key)
        t = jax.random.uniform(k_t, (len(x_1),))
        z = jax.random.normal(k_z, x_1.shape)
        return Bridge.loss(cast(net, COMPUTE_DTYPE), t, z, sar, meta, x_1)
