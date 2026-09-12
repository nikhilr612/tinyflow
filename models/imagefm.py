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
from einops import pack, rearrange, reduce, unpack
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped


@eqx.filter_jit
def _solve(net_theta: Any, x_0, ts, t1: float, n_steps: int, denom_floor: float):
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

    Returns:
        Trajectories of shape ``(B, len(ts), H, W, C)``.
    """
    ode_term = diffrax.ODETerm(
        lambda t, x, _args: ImageFM.velocity(net_theta, x, t, denom_floor),
    )

    @jax.vmap
    def solve_one(x_i):
        sol = diffrax.diffeqsolve(
            ode_term,
            diffrax.Dopri5(),
            t0=0.0,
            t1=t1,
            y0=x_i,
            dt0=t1 / n_steps,
            saveat=diffrax.SaveAt(ts=ts),
        )
        return sol.ys

    return solve_one(x_0)


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
    def velocity(net_theta: Any, x, t, denom_floor: float):
        """Velocity field of the reparameterized flow, ``(x_hat - x) / (1 - t)``.

        Args:
            net_theta: The denoising model.
            x: Current state, shape ``(H, W, C)``.
            t: Scalar time in ``[0, 1]``.
            denom_floor: Floor on ``1 - t``; see ``ImageFM.__init__``.

        Returns:
            Velocity of the same shape as ``x``.
        """
        t = jax.numpy.array(t)
        return (net_theta(x, t) - x) / jax.numpy.maximum(1 - t, denom_floor)

    @jaxtyped(typechecker=beartype.beartype)
    def sample(
        self,
        x_0: Float[Array, " B ..."],
        ts: Float[Array, " L"],
    ) -> Float[Array, " B L ..."]:
        """Sample by solving the ODE from ``t=0`` to ``t=1 - t_eps``.

        ``ts`` is given on the nominal ``[0, 1]`` schedule and clamped to
        ``1 - t_eps``.

        Args:
            x_0: Initial noise of shape ``(batch_size, ...)``, drawn from a prior.
            ts: Timestamps at which to record the trajectory.

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
        )

    @jaxtyped(typechecker=beartype.beartype)
    def generate(self, x_0: Float[Array, " B ..."]) -> Float[Array, " B ..."]:
        """Generate final images only, discarding the intermediate trajectory.

        Args:
            x_0: Initial noise of shape ``(batch_size, ...)``, drawn from a prior.

        Returns:
            Generated images of shape ``(batch_size, ...)``.
        """
        t1 = 1.0 - self.t_eps
        ts = jax.numpy.array([t1])
        # (B, 1, H, W, C) -> (B, H, W, C): drops the length-1 saved-time axis.
        return _solve(self.net_theta, x_0, ts, t1, self.n_steps, self.denom_floor)[:, 0]

    @jaxtyped(typechecker=beartype.beartype)
    @staticmethod
    def train_step(
        net_theta: Any,
        t: Float[Array, " B"],
        x_0: Float[Array, " B H W C"],
        x_1: Float[Array, " B H W C"],
        edge_weight: float = 0.1,
        line_weight: float = 0.0,
        denom_floor: float = 0.05,
        masks: Float[Array, " B H W K"] | None = None,
        aux_weight: float = 0.0,
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
        Auxiliary losses on ``x_hat`` vs ``x_1`` provide perceptual signals; they
        are only possible because the network predicts ``x_hat`` directly.

        A further, representation-level auxiliary is available when ``masks``
        are given: ``net_theta.forward_aux(x_t, t)`` must then return
        ``(x_hat, logits)`` with ``logits`` a coarse ``(h, w, K)`` map, and the
        loss adds ``aux_weight`` times the binary cross-entropy against the
        area-averaged masks.  Unlike the edge/line terms this does not touch
        ``x_hat``; it asks a hidden layer to know *where* the semantic parts of
        the clean image are while looking at the noisy one (cf. REPA, Yu et
        al. 2025).  Because it is a separate head with its own target it does
        not re-weight the flow-matching term and so leaves its minimizer, the
        conditional expectation, intact.

        Args:
            net_theta: The denoising model.
            t: Timesteps in ``[0, 1]``, shape ``(B,)``.
            x_0: Noise samples, shape ``(B, H, W, C)``.
            x_1: Data samples, shape ``(B, H, W, C)``.
            edge_weight: Weight for the auxiliary Sobel edge loss.
            line_weight: Weight for the auxiliary dark-line loss.
            denom_floor: Floor on ``1 - t``; see ``ImageFM.__init__``.
            masks: Soft semantic masks of ``x_1`` in ``[0, 1]``, shape
                ``(B, H, W, K)``; ``None`` disables the auxiliary head.
            aux_weight: Weight for the auxiliary mask loss.

        Returns:
            Scalar mean loss.
        """
        t_b = rearrange(t, "b -> b 1 1 1")
        x_t = t_b * x_1 + (1 - t_b) * x_0
        if masks is None:
            x_hat = jax.vmap(lambda x, ti: net_theta(x, ti))(x_t, t)
            aux_loss = 0.0
        else:
            x_hat, logits = jax.vmap(lambda x, ti: net_theta.forward_aux(x, ti))(x_t, t)
            aux_loss = ImageFM._aux_loss(logits, masks)
        u = (x_hat - x_t) / jax.numpy.maximum(1 - t_b, denom_floor)
        vel_loss = optax.l2_loss(u, x_1 - x_0).mean()
        edge_loss = ImageFM._edge_loss(x_hat, x_1)
        line_loss = ImageFM._line_loss(x_hat, x_1)
        return (
            vel_loss
            + edge_weight * edge_loss
            + line_weight * line_loss
            + aux_weight * aux_loss
        )

    @jaxtyped(typechecker=beartype.beartype)
    @staticmethod
    def _aux_loss(
        logits: Float[Array, " B h w K"], masks: Float[Array, " B H W K"]
    ) -> Float[Array, ""]:
        """Binary cross-entropy between coarse mask logits and area-pooled masks.

        The masks are pooled to the logit resolution with a mean, so the target
        is the fraction of each cell covered by the part -- a soft label, which
        sigmoid cross-entropy handles as is.
        """
        stride = masks.shape[1] // logits.shape[1]
        target = reduce(
            masks, "b (h s1) (w s2) k -> b h w k", "mean", s1=stride, s2=stride
        )
        return optax.sigmoid_binary_cross_entropy(logits, target).mean()

    @staticmethod
    def _sobel(
        x: Float[Array, " B H W C"],
    ) -> tuple[Float[Array, " B H W C"], Float[Array, " B H W C"]]:
        """Depthwise 3x3 Sobel responses ``(gx, gy)`` of every channel."""
        n = x.shape[-1]
        # One input channel, one output channel                        HWIO
        sobel_x = rearrange(
            jax.numpy.array(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=jax.numpy.float32
            ),
            "h w -> h w 1 1",
        )
        sobel_y = rearrange(sobel_x, "h w i o -> w h i o")

        # Edge-replicate rather than zero-pad: in [-1, 1] zero is mid-grey, so
        # zero padding would put a phantom step on every border pixel and turn
        # the edge loss into an intensity penalty along the frame.
        xp = jax.numpy.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="edge")

        # NHWC input, HWIO kernel, NHWC output.  feature_group_count = C makes
        # every channel its own group, so the same Sobel is applied depthwise.
        def conv(kernel):
            return jax.lax.conv_general_dilated(
                xp,
                jax.numpy.broadcast_to(kernel, (3, 3, 1, n)),
                window_strides=(1, 1),
                padding="VALID",
                feature_group_count=n,
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
            )

        return conv(sobel_x), conv(sobel_y)

    @staticmethod
    def _edge_loss(
        pred: Float[Array, " B H W C"],
        target: Float[Array, " B H W C"],
    ) -> Float[Array, ""]:
        """Sobel gradient-difference loss: L1 on the signed per-axis gradients.

        The Sobel responses ``gx`` and ``gy`` are matched *separately* rather
        than through their magnitude ``sqrt(gx^2 + gy^2)``.  Magnitude discards
        orientation, so a stroke reproduced at the wrong angle costs nothing;
        the signed form also penalises the stroke's direction and polarity
        (dark-on-light vs light-on-dark).  This is the Gradient Difference Loss
        of Mathieu, Couprie & LeCun (ICLR 2016, arXiv:1511.05440, Sec. 3.2)
        with Sobel filters in place of forward differences, and L1 (their
        ``alpha = 1``) so that its scale matches the previous magnitude loss.
        """
        combined, ps = pack([pred, target], "b h w *")  # (B, H, W, 2C)
        gx, gy = ImageFM._sobel(combined)
        pred_gx, target_gx = unpack(gx, ps, "b h w *")  # each (B, H, W, C)
        pred_gy, target_gy = unpack(gy, ps, "b h w *")
        return 0.5 * (
            jax.numpy.abs(pred_gx - target_gx).mean()
            + jax.numpy.abs(pred_gy - target_gy).mean()
        )

    @staticmethod
    def _ink(
        x: Float[Array, " B H W C"],
        beta: float = 10.0,
    ) -> Float[Array, " B H W"]:
        """Ink map: soft black top-hat of luma with a 3x3 window.

        Anime line art is thin, dark strokes on lighter fill.  The black
        top-hat ``closing(L) - L`` (Serra, *Image Analysis and Mathematical
        Morphology*, 1982; cf. the line channel of XDoG, Winnemoeller et al.
        2012) is positive exactly where luma sits in a valley narrower than
        the structuring element -- a one-pixel stroke at 64x64 -- and zero on
        flat fill, smooth shading, and any dark region wider than the window.
        The closing ``min3(max3(L))`` is made differentiable everywhere by
        replacing max/min with log-mean-exp at inverse temperature ``beta``:

            smax(L) = log(mean_3x3(exp(beta L))) / beta,  smin(L) = -smax(-L),
            ink     = relu(smin(smax(L)) - L),  L = BT.601 luma.

        ``beta -> inf`` recovers the hard closing; ``beta -> 0`` degrades to a
        5x5 box blur, i.e. a Laplacian that also responds to mild curvature in
        shading.  On the anime-faces data ``beta = 10`` is 0.98-correlated
        with the hard closing while still routing gradient to all nine window
        pixels; unlike the hard version it also picks up strokes inside dark
        hair, where the box mean under-responds.  The ``relu`` only absorbs
        the small finite-``beta`` slack (the hard closing is ``>= L``).
        """
        # BT.601 luma; the weights sum to one so the affine [-1, 1] range needs
        # no correction.  Border pixels are edge-replicated so a window never
        # sees synthetic dark/bright values.
        luma_w = jax.numpy.array([0.299, 0.587, 0.114], dtype=jax.numpy.float32)
        luma = x @ luma_w  # (B, H, W)

        def smax(z):
            # log-mean-exp over each 3x3 window, via the nine shifted copies.
            zp = jax.numpy.pad(z, ((0, 0), (1, 1), (1, 1)), mode="edge")
            h, w = z.shape[1:]
            shifts = [zp[:, i : i + h, j : j + w] for i in range(3) for j in range(3)]
            stacked = jax.numpy.stack(shifts, axis=0)  # (9, B, H, W)
            lse = jax.scipy.special.logsumexp(beta * stacked, axis=0)
            return (lse - jax.numpy.log(9.0)) / beta

        closing = -smax(-smax(luma))
        return jax.nn.relu(closing - luma)

    @staticmethod
    def _line_loss(
        pred: Float[Array, " B H W C"],
        target: Float[Array, " B H W C"],
        beta: float = 10.0,
    ) -> Float[Array, ""]:
        """Dark-line loss: L1 between the ink maps of ``pred`` and ``target``.

        See ``ImageFM._ink``.  It is one-sided (ink, not highlights) and fires
        once per stroke rather than on both flanks as a Sobel loss does.  On
        the 64x64 anime-faces data the map is ~45% exact zeros and its L1 mass
        sits in roughly [0.1, 0.6], so no thresholding is needed.
        """
        return jax.numpy.abs(
            ImageFM._ink(pred, beta) - ImageFM._ink(target, beta)
        ).mean()


@dataclass(frozen=True)
class TrainConfig:
    """Optimisation and loss hyperparameters: everything the gradient step needs.

    Attributes:
        n_epochs: Number of full passes over the dataset.
        init_lr: Learning rate for Adam.
        edge_weight: Weight for the auxiliary Sobel edge loss.
        line_weight: Weight for the auxiliary dark-line loss.
        aux_weight: Weight for the auxiliary mask loss; needs a dataset that
            yields masks and a ``net_theta`` with ``forward_aux``.
        t_mu: Mean of the logit-normal timestep distribution
            ``t = sigmoid(N(t_mu, t_sigma^2))``.  Negative values sample lower
            ``t`` (higher noise) more often; JiT uses -0.8.
        t_sigma: Standard deviation of the logit-normal timestep distribution.
    """

    n_epochs: int = 100
    init_lr: float = 1e-3
    edge_weight: float = 0.1
    line_weight: float = 0.0
    aux_weight: float = 0.0
    t_mu: float = -0.8
    t_sigma: float = 1.0


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
    ``model.net_theta`` is updated in place, so it is current at every yield.

    Args:
        key: JAX PRNG key.
        model: The ``ImageFM`` instance to train.
        dataset: An endless dataset yielding batches of shape ``(B, H, W, C)``,
            or ``(images, masks)`` pairs of such batches when ``aux_weight > 0``.
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

    @eqx.filter_jit
    def make_update(key, net_theta, batch, optimizer_state, masks=None):
        newkey, sk1, sk2 = jax.random.split(key, num=3)
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
            cfg.edge_weight,
            cfg.line_weight,
            model.denom_floor,
            masks,
            cfg.aux_weight,
        )
        updates, optimizer_state = optimizer.update(grad, optimizer_state)
        net_theta = eqx.apply_updates(net_theta, updates)
        return newkey, net_theta, optimizer_state, loss

    # The dataset is endless, so the step budget is what ends the run and epoch
    # boundaries fall out of the step counter.
    total_steps = cfg.n_epochs * batches_per_epoch
    net_loss = 0.0
    for step, batch in zip(range(1, total_steps + 1), dataset):
        masks = None
        if isinstance(batch, tuple):
            batch, masks = batch
        key, model.net_theta, optimizer_state, loss = make_update(
            key, model.net_theta, batch, optimizer_state, masks
        )
        net_loss += float(loss)
        if step % batches_per_epoch == 0:
            yield step // batches_per_epoch - 1, net_loss / batches_per_epoch
            net_loss = 0.0
