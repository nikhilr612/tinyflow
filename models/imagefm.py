"""An implementation of flow matching on images, generic over the velocity model.

This module implements the Flow Matching Recipe for image-shaped data.
It is agnostic to the choice of velocity model: any eqx.Module implementing
``__call__(x, t) -> velocity`` with ``x`` of shape ``(H, W, C)`` and scalar
``t`` can be used as ``net_theta``.
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import beartype
import diffrax
import equinox as eqx
import jax
import optax
from einops import rearrange
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped


def cond_token(
    masks: Float[Array, " B H W K"] | None, shape: tuple, keep=None
) -> Float[Array, " B H W K1"]:
    """Build the conditioning channels ``[masks, indicator]`` for a batch.

    ``keep`` is a per-sample ``(B,)`` 0/1 array: 1 passes the mask through
    with indicator 1, 0 replaces it by the *null token* -- zero mask channels
    and indicator 0.  The indicator disambiguates "unconditioned" from "a
    face with no parts" (all-zero masks), the channel-space analogue of the
    learned null embedding in classifier-free guidance.  ``masks=None`` gives
    the null token for the whole batch; ``shape`` is ``(B, H, W, K)``.
    """
    b, h, w, k = shape
    if masks is None:
        return jax.numpy.zeros((b, h, w, k + 1))
    if keep is None:
        keep = jax.numpy.ones((b,))
    keep = rearrange(keep, "b -> b 1 1 1")
    ind = jax.numpy.broadcast_to(keep, (b, h, w, 1))
    return jax.numpy.concatenate([masks * keep, ind], axis=-1)


@eqx.filter_jit
def _solve(
    net_theta: Any,
    x_0,
    ts,
    t1: float,
    n_steps: int,
    denom_floor: float,
    cond=None,
):
    """Solve the probability-flow ODE on ``[0, t1]`` with ``n_steps`` Dopri5 steps.

    The single solver behind every sampling path, so training-time samples, FID
    samples and figure samples are all drawn from the same sampler.

    Args:
        net_theta: The denoising model.
        x_0: Initial noise, shape ``(B, H, W, C)``.
        ts: Times at which to record the trajectory, all ``<= t1``.
        t1: Terminal time (static).
        n_steps: Number of fixed steps, setting ``dt0 = t1 / n_steps`` (static).
        denom_floor: Floor on ``1 - t`` in the velocity (static).
        cond: Optional conditioning channels ``(B, H, W, K1)`` concatenated to
            the state before every network call (see ``cond_token``).

    Returns:
        Trajectories of shape ``(B, len(ts), H, W, C)``.
    """
    ode_term = diffrax.ODETerm(
        lambda t, x, c: ImageFM.velocity(net_theta, x, t, denom_floor, c),
    )

    @jax.vmap
    def solve_one(x_i, c_i):
        sol = diffrax.diffeqsolve(
            ode_term,
            diffrax.Dopri5(),
            t0=0.0,
            t1=t1,
            y0=x_i,
            args=c_i,
            dt0=t1 / n_steps,
            saveat=diffrax.SaveAt(ts=ts),
        )
        return sol.ys

    return solve_one(x_0, cond)


class ImageFM:
    """Flow matching on images, generic over the velocity field architecture."""

    def __init__(
        self,
        net_theta: Any,
        hparams: dict,
        n_steps: int = 64,
        denom_floor: float = 0.05,
        t_eps: float = 1e-3,
    ) -> None:
        """Initialize ImageFM with a denoising model and its hyperparameters.

        Args:
            net_theta: The denoising model implementing ``__call__(x, t)``
                where ``x`` is ``(H, W, C)`` and ``t`` is scalar.
                Output is interpreted as denoised prediction ``x_hat``,
                not velocity (velocity is derived during training/sampling).
            hparams: Architecture hyperparameters needed for serialization.
                Saved as JSON alongside model weights.
            n_steps: Number of fixed Dopri5 steps used by every sampling path.
                This is the single knob controlling sampler cost/accuracy; it is
                a property of the sampler, not the architecture, so it is not
                serialized into ``hparams``.
            denom_floor: Floor on the ``1 - t`` denominator in the velocity
                ``(x_hat - x_t) / (1 - t)``, applied identically in the training
                loss and the sampler.  It bounds the per-sample loss weight
                ``1 / (1 - t)^2`` at ``1 / denom_floor^2`` and keeps the field
                finite through ``t = 1``.  Follows JiT (Li & He, CVPR 2026,
                Alg. 1-2), which clips at 0.05.
            t_eps: Terminal time offset; sampling integrates to ``1 - t_eps``.
                Not needed for stability (``denom_floor`` handles that), so it is
                kept small; raise it to stop short and inspect partially evolved
                images.
        """
        self.net_theta = net_theta
        self.hparams = hparams
        self.n_steps = n_steps
        self.denom_floor = denom_floor
        self.t_eps = t_eps
        # Layout conditioning: ``hparams["cond_channels"]`` mask channels plus
        # one indicator channel are concatenated to the image on every call.
        self.cond_channels = int(hparams.get("cond_channels", 0))

    def save(self, path: str) -> None:
        """Serialize the model weights and hyperparameters to disk.

        Weights go to ``path`` (binary eqx format).
        Hyperparameters go to ``path + ".hparams"`` (JSON).
        """
        with Path(path + ".hparams").open("w") as f:
            json.dump(self.hparams, f)
        with Path(path).open("wb") as f:
            eqx.tree_serialise_leaves(f, self.net_theta)

    @staticmethod
    def load(path: str, skeleton_fn) -> "ImageFM":
        """Deserialize a model from disk.

        Args:
            path: Base path (without .hparams suffix) to load from.
            skeleton_fn: Callable ``(key: PRNGKeyArray, **hparams) -> eqx.Module``
                that constructs a skeleton model of the correct architecture.

        Returns:
            The deserialized ``ImageFM`` instance.
        """
        with Path(path + ".hparams").open("r") as f:
            hparams = json.load(f)
        key = jax.random.key(42)
        like = skeleton_fn(key, **{k: int(v) for k, v in hparams.items()})
        with Path(path).open("rb") as f:
            net_theta = eqx.tree_deserialise_leaves(f, like)
        return ImageFM(net_theta, hparams)

    @staticmethod
    def velocity(net_theta: Any, x, t, denom_floor: float, cond=None):
        """Velocity field of the reparameterized flow, ``(x_hat - x) / (1 - t)``.

        Args:
            net_theta: The denoising model.
            x: Current state, shape ``(H, W, C)``.
            t: Scalar time in ``[0, 1]``.
            denom_floor: Floor on ``1 - t``; see ``ImageFM.__init__``.
            cond: Optional conditioning channels ``(H, W, K1)`` concatenated
                to ``x`` before the network call.

        Returns:
            Velocity of the same shape as ``x``.
        """
        t = jax.numpy.array(t)
        x_in = x if cond is None else jax.numpy.concatenate([x, cond], axis=-1)
        return (net_theta(x_in, t) - x) / jax.numpy.maximum(1 - t, denom_floor)

    @jaxtyped(typechecker=beartype.beartype)
    def sample(
        self,
        x_0: Float[Array, " B ..."],
        ts: Float[Array, " L"],
        masks: Float[Array, " B H W K"] | None = None,
    ) -> Float[Array, " B L ..."]:
        """Sample by solving the ODE from ``t=0`` to ``t=1 - t_eps``.

        ``ts`` is given on the nominal ``[0, 1]`` schedule and clamped to
        ``1 - t_eps``.

        Args:
            x_0: Initial noise of shape ``(batch_size, ...)``, drawn from a prior.
            ts: Timestamps at which to record the trajectory.
            masks: Layout masks for a conditioned model; see ``generate``.

        Returns:
            Trajectories of shape ``(batch_size, len(ts), ...)``.
        """
        t1 = 1.0 - self.t_eps
        return _solve(
            self.net_theta,
            x_0,
            jax.numpy.minimum(ts, t1),
            t1,
            self.n_steps,
            self.denom_floor,
            self._cond(x_0, masks),
        )

    def _cond(self, x_0, masks):
        """Conditioning channels for sampling, or ``None`` for unconditioned models.

        With ``cond_channels > 0`` and ``masks=None`` this is the null token,
        i.e. the model runs in its unconditional mode.
        """
        if self.cond_channels == 0:
            if masks is not None:
                raise ValueError("this model was not trained with conditioning")
            return None
        shape = (*x_0.shape[:3], self.cond_channels)
        return cond_token(masks, shape)

    @jaxtyped(typechecker=beartype.beartype)
    def generate(
        self,
        x_0: Float[Array, " B ..."],
        masks: Float[Array, " B H W K"] | None = None,
    ) -> Float[Array, " B ..."]:
        """Generate final images only, discarding the intermediate trajectory.

        Args:
            x_0: Initial noise of shape ``(batch_size, ...)``, drawn from a prior.
            masks: Layout masks ``(B, H, W, K)`` in ``[0, 1]`` for a model trained
                with ``cond_channels = K``; ``None`` samples unconditionally
                (null token).  Must be ``None`` for unconditioned models.

        Returns:
            Generated images of shape ``(batch_size, ...)``.
        """
        t1 = 1.0 - self.t_eps
        ts = jax.numpy.array([t1])
        # (B, 1, H, W, C) -> (B, H, W, C): drops the length-1 saved-time axis.
        return _solve(
            self.net_theta,
            x_0,
            ts,
            t1,
            self.n_steps,
            self.denom_floor,
            self._cond(x_0, masks),
        )[:, 0]

    @jaxtyped(typechecker=beartype.beartype)
    @staticmethod
    def train_step(
        net_theta: Any,
        t: Float[Array, " B"],
        x_0: Float[Array, " B H W C"],
        x_1: Float[Array, " B H W C"],
        denom_floor: float = 0.05,
        masks: Float[Array, " B H W K"] | None = None,
        cond_channels: int = 0,
        cond_dropout: float = 0.0,
        cond_key: PRNGKeyArray | None = None,
    ) -> Float[Array, ""]:
        """Single training step: x-prediction with a velocity-space loss.

        The network predicts the clean image ``x_hat`` from ``(x_t, t)``
        (x-prediction: its output stays on the image manifold rather than
        carrying off-manifold noise).  The loss is taken in velocity space,
        ``|(x_hat - x_t)/(1 - t) - (x_1 - x_0)|^2 = |x_hat - x_1|^2 / (1 - t)^2``,
        which up-weights late ``t`` where fine detail is decided.  The
        denominator is floored at ``denom_floor`` so that weight is bounded;
        without the floor a single ``t`` near 1 owns the whole batch gradient.
        This is the JiT recipe (Li & He, CVPR 2026, Tab. 1 (3)(a), Alg. 1).

        There are deliberately no auxiliary terms.  The population minimiser
        of this loss is the conditional mean ``E[x_1 | x_t]``; any extra loss
        on ``x_hat`` against the sharp ``x_1`` (edges, lines, palettes) is
        almost entirely *irreducible* at that optimum and only biases the
        network toward hedged predictions, and any per-pixel re-weighting that
        depends on ``x_1`` tilts the learned field toward a re-weighted data
        distribution.  Both were measured to hurt or do nothing
        (experiments/METHODS.md, sections 1-2).  Priors about *samples* belong
        at sampling time (guidance) or in the architecture (``RegionPool``).

        Layout conditioning (``cond_channels > 0``) concatenates
        ``cond_token(masks)`` to ``x_t`` so the network learns
        ``E[x_1 | x_t, m]``: the layout is given and the network renders.  It
        leaves the loss untouched.

        Args:
            net_theta: The denoising model.
            t: Timesteps in ``[0, 1]``, shape ``(B,)``.
            x_0: Noise samples, shape ``(B, H, W, C)``.
            x_1: Data samples, shape ``(B, H, W, C)``.
            denom_floor: Floor on ``1 - t``; see ``ImageFM.__init__``.
            masks: Soft semantic masks of ``x_1`` in ``[0, 1]``, shape
                ``(B, H, W, K)``; required when ``cond_channels > 0``.
            cond_channels: ``K > 0`` conditions the network on the first ``K``
                mask channels; the network's input width must be ``C + K + 1``.
                ``0`` disables it.
            cond_dropout: Probability of replacing a sample's mask by the null
                token during training so the same network also works
                unconditionally (Ho & Salimans 2022).  Needs ``cond_key``.
                Measured to cost ~7 FID at 30 epochs; ``0`` gives a pure
                conditional model (experiments/METHODS.md, section 7.2).
            cond_key: PRNG key for the dropout draw.

        Returns:
            Scalar mean loss.
        """
        if masks is None and cond_channels:
            raise ValueError("cond_channels > 0 needs masks")
        t_b = rearrange(t, "b -> b 1 1 1")
        x_t = t_b * x_1 + (1 - t_b) * x_0
        x_in = x_t
        if cond_channels and masks is not None:
            keep = None
            if cond_dropout > 0:
                if cond_key is None:
                    raise ValueError("cond_dropout needs cond_key")
                keep = jax.random.uniform(cond_key, (x_t.shape[0],)) >= cond_dropout
                keep = keep.astype(x_t.dtype)
            cond = cond_token(
                masks[..., :cond_channels], masks.shape[:3] + (cond_channels,), keep
            )
            x_in = jax.numpy.concatenate([x_t, cond], axis=-1)
        x_hat = jax.vmap(lambda x, ti: net_theta(x, ti))(x_in, t)
        u = (x_hat - x_t) / jax.numpy.maximum(1 - t_b, denom_floor)
        return optax.l2_loss(u, x_1 - x_0).mean()


@dataclass(frozen=True)
class TrainConfig:
    """Optimisation and loss hyperparameters: everything the gradient step needs.

    Attributes:
        n_epochs: Number of full passes over the dataset.
        init_lr: Learning rate for Adam.
        t_mu: Mean of the logit-normal timestep distribution
            ``t = sigmoid(N(t_mu, t_sigma^2))``.  Negative values sample lower
            ``t`` (higher noise) more often; JiT uses -0.8.
        t_sigma: Standard deviation of the logit-normal timestep distribution.
        cond_channels: Number of mask channels the network is conditioned on
            (``0`` = unconditioned); see ``train_step``.
        cond_dropout: Null-token probability for classifier-free conditioning.
        ema_decay: Decay of the exponential moving average of the weights that
            is published as ``model.net_theta`` (checkpointed, sampled and
            evaluated) while the optimiser keeps stepping the raw weights.
            Standard in diffusion/flow training since DDPM (Ho et al. 2020,
            0.9999) -- a single Adam iterate at a constant learning rate is a
            noisy point; the average sits nearer the basin and makes FID
            reproducible between evaluations.  Shorter than the literature's
            0.9999 because our runs are ~17k steps, not 500k.  ``0`` disables
            it.  The decay
            is warmed up as ``min(ema_decay, (1 + step) / (10 + step))`` so
            early averages are not dominated by the random initialisation
            (the ``num_updates`` scheme of TensorFlow's
            ``ExponentialMovingAverage``, used by the DDPM reference code).
    """

    n_epochs: int = 100
    init_lr: float = 1e-3
    cond_channels: int = 0
    cond_dropout: float = 0.0
    t_mu: float = -0.8
    t_sigma: float = 1.0
    ema_decay: float = 0.999


def train_on_image(
    key: PRNGKeyArray,
    model: ImageFM,
    dataset,
    batches_per_epoch: int,
    cfg: TrainConfig = TrainConfig(),
) -> Iterator[tuple[int, float]]:
    """Train ``model`` in place, yielding ``(epoch, mean_loss)`` per epoch.

    This is only the optimisation loop.  It knows nothing about checkpoints,
    samples, evaluation or stopping criteria; ``training.run`` layers those on
    by consuming this generator and breaking out of it when it wants to stop.
    ``model.net_theta`` is updated in place, so it is current at every yield:
    it holds the EMA weights when ``cfg.ema_decay > 0`` and the raw weights
    otherwise.  The optimiser always steps the raw weights, which live only
    inside this generator.

    Args:
        key: JAX PRNG key.
        model: The ``ImageFM`` instance to train.
        dataset: An endless dataset yielding batches of shape ``(B, H, W, C)``,
            or ``(images, masks)`` pairs of such batches when ``cond_channels > 0``.
        batches_per_epoch: Number of batches making up one pass over the data;
            the dataset never stops on its own, so this is what bounds training.
        cfg: Optimisation and loss hyperparameters.

    Yields:
        ``(epoch, mean_loss)`` at the end of each epoch, ``epoch`` from 0.
    """
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(cfg.init_lr),
    )
    optimizer_state = optimizer.init(eqx.filter(model.net_theta, eqx.is_inexact_array))

    def ema_update(ema_theta, net_theta, step):
        """One step of the exponential moving average of the weights.

        With decay ``d`` the update is

            theta_ema  <-  d * theta_ema + (1 - d) * theta
                        =  theta_ema + (1 - d) * (theta - theta_ema),

        the second form being what ``optax.incremental_update`` computes.
        Unrolled, ``theta_ema`` after ``n`` steps is ``(1 - d) * sum_k d^k
        theta_{n-k}``: a geometric weighting of past iterates with half-life
        ``ln 2 / (1 - d)`` steps (about 700 steps at ``d = 0.999``).  Early
        on that sum is dominated by ``theta_0``, so ``d`` is warmed up as
        ``min(d, (1 + step) / (10 + step))``: the effective decay is ~0.2 at
        step 1, 0.9 at step ~80, 0.99 at step ~900, and the ramp only meets
        the cap of 0.999 at step ~9000, and the initialisation is forgotten
        within the first few hundred steps.
        """
        decay = jax.numpy.minimum(cfg.ema_decay, (1 + step) / (10 + step))
        ema_params, static = eqx.partition(ema_theta, eqx.is_inexact_array)
        net_params = eqx.filter(net_theta, eqx.is_inexact_array)
        ema_params = optax.incremental_update(net_params, ema_params, 1 - decay)
        return eqx.combine(ema_params, static)

    @eqx.filter_jit
    def make_update(key, net_theta, ema_theta, step, batch, optimizer_state, masks):
        newkey, sk1, sk2, sk3 = jax.random.split(key, num=4)
        batch_size = batch.shape[0]
        # Logit-normal t (JiT, Tab. 3): mu < 0 shifts mass toward low t, i.e.
        # high noise, where the modelling is hard; the thin tail at t -> 1
        # also balances the 1/(1-t)^2 loss weight there.
        t = jax.nn.sigmoid(
            cfg.t_mu + cfg.t_sigma * jax.random.normal(sk1, (batch_size,))
        )
        rand_input = jax.random.normal(sk2, shape=batch.shape)
        loss, grad = eqx.filter_value_and_grad(ImageFM.train_step)(
            net_theta,
            t,
            rand_input,
            batch,
            model.denom_floor,
            masks,
            cfg.cond_channels,
            cfg.cond_dropout,
            sk3,
        )
        updates, optimizer_state = optimizer.update(grad, optimizer_state)
        net_theta = eqx.apply_updates(net_theta, updates)
        if cfg.ema_decay > 0:
            ema_theta = ema_update(ema_theta, net_theta, step)
        else:
            ema_theta = net_theta
        return newkey, net_theta, ema_theta, optimizer_state, loss

    # The dataset is endless, so the step budget is what ends the run and epoch
    # boundaries fall out of the step counter.
    total_steps = cfg.n_epochs * batches_per_epoch
    net_theta = ema_theta = model.net_theta
    net_loss = 0.0
    for step, batch in zip(range(1, total_steps + 1), dataset):
        masks = None
        if isinstance(batch, tuple):
            batch, masks = batch
        key, net_theta, ema_theta, optimizer_state, loss = make_update(
            key,
            net_theta,
            ema_theta,
            jax.numpy.asarray(step),
            batch,
            optimizer_state,
            masks,
        )
        net_loss += float(loss)
        if step % batches_per_epoch == 0:
            model.net_theta = ema_theta
            yield step // batches_per_epoch - 1, net_loss / batches_per_epoch
            net_loss = 0.0
