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

# Channels of the semantic masks produced for the anime-faces data (see
# ``data.animefaces.load_masks``): 0 face hull, 1 union of both eye hulls,
# 2 mouth hull.
EYE_CHANNEL = 1


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
        edge_weight: float = 0.0,
        line_weight: float = 0.0,
        denom_floor: float = 0.05,
        masks: Float[Array, " B H W K"] | None = None,
        aux_weight: float = 0.0,
        edge_norm: str = "l1",
        fm_edge_weight: float = 0.0,
        fm_ink_weight: float = 0.0,
        fm_eye_weight: float = 0.0,
        fm_self_edge_weight: float = 0.0,
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

        Spatial re-weighting of the flow-matching term (``fm_*_weight``)
        multiplies the per-pixel squared error by ``1 + sum_k w_k m_k`` with
        ``m_k`` a non-negative map (see ``_fm_weight_map``).  Whether that
        moves the minimizer depends only on what the map is a function of:
        for a weight ``w`` and prediction ``c`` at fixed ``x_t``,
        ``argmin E[w |c - x_1|^2 | x_t] = E[w x_1 | x_t] / E[w | x_t]``, which
        equals the conditional mean ``E[x_1 | x_t]`` iff ``w`` is independent
        of ``x_1`` given ``x_t``.  Maps of the *clean* image (Sobel magnitude,
        ink, eye mask of ``x_1``) therefore tilt the minimizer toward posterior
        samples the map favours -- a sharpening prior, strongest at low ``t``
        (weak for the eye mask, whose position is nearly fixed on aligned
        faces).  ``fm_self_edge_weight`` uses the Sobel magnitude of
        ``stop_gradient(x_hat)`` instead: a function of ``(x_t, t)`` only, so
        the minimizer is unchanged while gradient still concentrates on the
        pixels the model currently draws edges at.  ``edge_norm="l2"`` is the
        analogous fix for the Sobel term: L2 on a *linear* map has minimizer
        ``sobel(E[x_1 | x_t])``, consistent with the flow-matching term,
        whereas L1 selects a median and rewards hedged, flatter predictions
        wherever the target's detail is not knowable from ``x_t``.

        Args:
            net_theta: The denoising model.
            t: Timesteps in ``[0, 1]``, shape ``(B,)``.
            x_0: Noise samples, shape ``(B, H, W, C)``.
            x_1: Data samples, shape ``(B, H, W, C)``.
            edge_weight: Weight for the auxiliary Sobel edge loss.
            line_weight: Weight for the auxiliary dark-line loss.
            denom_floor: Floor on ``1 - t``; see ``ImageFM.__init__``.
            masks: Soft semantic masks of ``x_1`` in ``[0, 1]``, shape
                ``(B, H, W, K)``; required when ``aux_weight > 0`` (auxiliary
                head) or ``fm_eye_weight > 0`` (eye channel as a weight map).
            aux_weight: Weight for the auxiliary mask loss.
            edge_norm: ``"l1"`` (gradient-difference loss, median-type) or
                ``"l2"`` (mean-type, bias-free) for the Sobel term.
            fm_edge_weight: Flow-matching re-weighting by Sobel magnitude.
            fm_ink_weight: Flow-matching re-weighting by the ink map.
            fm_eye_weight: Flow-matching re-weighting by the eye mask.
            fm_self_edge_weight: Flow-matching re-weighting by the Sobel
                magnitude of the model's own (stop-gradient) ``x_hat``.
            cond_channels: ``K > 0`` conditions the network on ``masks`` by
                concatenating ``cond_token(masks)`` (``K + 1`` channels) to
                ``x_t``; the network's input width must be ``C + K + 1``.
                Conditioning changes the target to ``E[x_1 | x_t, m]`` --
                layout is given, the network renders -- without touching the
                loss.  ``0`` disables it.
            cond_dropout: Probability of replacing a sample's mask by the null
                token during training so the same network also works
                unconditionally (Ho & Salimans 2022).  Needs ``cond_key``.
            cond_key: PRNG key for the dropout draw.

        Returns:
            Scalar mean loss.
        """
        if masks is None and (aux_weight > 0 or fm_eye_weight > 0 or cond_channels):
            raise ValueError("aux_weight, fm_eye_weight and cond_channels need masks")
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
        if aux_weight > 0 and masks is not None:
            x_hat, logits = jax.vmap(lambda x, ti: net_theta.forward_aux(x, ti))(
                x_in, t
            )
            aux_loss = ImageFM._aux_loss(logits, masks)
        else:
            x_hat = jax.vmap(lambda x, ti: net_theta(x, ti))(x_in, t)
            aux_loss = 0.0
        u = (x_hat - x_t) / jax.numpy.maximum(1 - t_b, denom_floor)
        w_map = ImageFM._fm_weight_map(
            x_1, masks, fm_edge_weight, fm_ink_weight, fm_eye_weight
        )
        if fm_self_edge_weight > 0:
            gx, gy = ImageFM._sobel(jax.lax.stop_gradient(x_hat))
            m = jax.numpy.sqrt(gx**2 + gy**2).mean(-1)
            w_map = w_map + fm_self_edge_weight * m / (m.mean() + 1e-8)
        vel_loss = (w_map[..., None] * optax.l2_loss(u, x_1 - x_0)).mean()
        edge_loss = ImageFM._edge_loss(x_hat, x_1, edge_norm)
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

    @jaxtyped(typechecker=beartype.beartype)
    @staticmethod
    def _fm_weight_map(
        x_1: Float[Array, " B H W C"],
        masks: Float[Array, " B H W K"] | None,
        edge_weight: float,
        ink_weight: float,
        eye_weight: float,
    ) -> Float[Array, " B H W"]:
        """Per-pixel weight ``1 + sum_k w_k m_k`` for the flow-matching term.

        Each map is normalised by its batch mean, so ``w_k`` is the *average*
        extra weight it adds and the maps are comparable regardless of their
        native scale (Sobel magnitudes are O(1), ink is mostly zero, the eye
        mask covers ~5% of pixels).  Weights of zero skip the map entirely.
        """
        w = jax.numpy.ones(x_1.shape[:3], dtype=x_1.dtype)

        def add(w, m, coef):
            return w + coef * m / (m.mean() + 1e-8)

        if edge_weight > 0:
            gx, gy = ImageFM._sobel(x_1)
            w = add(w, jax.numpy.sqrt(gx**2 + gy**2).mean(-1), edge_weight)
        if ink_weight > 0:
            w = add(w, ImageFM._ink(x_1), ink_weight)
        if eye_weight > 0 and masks is not None:
            w = add(w, masks[..., EYE_CHANNEL], eye_weight)
        return w

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
        norm: str = "l1",
    ) -> Float[Array, ""]:
        """Sobel gradient-difference loss on the signed per-axis gradients.

        The Sobel responses ``gx`` and ``gy`` are matched *separately* rather
        than through their magnitude ``sqrt(gx^2 + gy^2)``.  Magnitude discards
        orientation, so a stroke reproduced at the wrong angle costs nothing;
        the signed form also penalises the stroke's direction and polarity
        (dark-on-light vs light-on-dark).  This is the Gradient Difference Loss
        of Mathieu, Couprie & LeCun (ICLR 2016, arXiv:1511.05440, Sec. 3.2)
        with Sobel filters in place of forward differences, and L1 (their
        ``alpha = 1``) so that its scale matches the previous magnitude loss.

        ``norm="l2"`` uses ``alpha = 2``.  The choice matters more than it
        looks: Sobel is linear, so the L2 minimizer is ``sobel(E[x_1 | x_t])``
        -- consistent with the flow-matching term -- while the L1 minimizer
        is a per-pixel median of the gradient, which under uncertainty is
        closer to zero than the mean and so favours flatter predictions.
        """
        if norm not in ("l1", "l2"):
            raise ValueError(f"norm must be 'l1' or 'l2', got {norm!r}")
        combined, ps = pack([pred, target], "b h w *")  # (B, H, W, 2C)
        gx, gy = ImageFM._sobel(combined)
        pred_gx, target_gx = unpack(gx, ps, "b h w *")  # each (B, H, W, C)
        pred_gy, target_gy = unpack(gy, ps, "b h w *")
        dist = jax.numpy.abs if norm == "l1" else jax.numpy.square
        return 0.5 * (
            dist(pred_gx - target_gx).mean() + dist(pred_gy - target_gy).mean()
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
        edge_weight: Weight for the auxiliary Sobel edge loss.  Default 0:
            at its former default of 0.1 the L1 term was found to *reduce*
            edge content and FID (experiments/METHODS.md, sections 1-2).
        line_weight: Weight for the auxiliary dark-line loss.
        aux_weight: Weight for the auxiliary mask loss; needs a dataset that
            yields masks and a ``net_theta`` with ``forward_aux``.
        t_mu: Mean of the logit-normal timestep distribution
            ``t = sigmoid(N(t_mu, t_sigma^2))``.  Negative values sample lower
            ``t`` (higher noise) more often; JiT uses -0.8.
        t_sigma: Standard deviation of the logit-normal timestep distribution.
        edge_norm: ``"l1"`` or ``"l2"`` for the Sobel term; see ``train_step``.
        fm_edge_weight: Flow-matching re-weighting by the Sobel magnitude of
            ``x_1`` (average extra weight per pixel); see ``train_step``.
        fm_ink_weight: Flow-matching re-weighting by the ink map of ``x_1``.
        fm_eye_weight: Flow-matching re-weighting by the eye mask; needs a
            dataset that yields masks.
        fm_self_edge_weight: Flow-matching re-weighting by the Sobel magnitude
            of the model's own ``x_hat`` (stop-gradient); see ``train_step``.
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
    edge_weight: float = 0.0
    line_weight: float = 0.0
    aux_weight: float = 0.0
    edge_norm: str = "l1"
    fm_edge_weight: float = 0.0
    fm_ink_weight: float = 0.0
    fm_eye_weight: float = 0.0
    fm_self_edge_weight: float = 0.0
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
            cfg.edge_weight,
            cfg.line_weight,
            model.denom_floor,
            masks,
            cfg.aux_weight,
            cfg.edge_norm,
            cfg.fm_edge_weight,
            cfg.fm_ink_weight,
            cfg.fm_eye_weight,
            cfg.fm_self_edge_weight,
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
