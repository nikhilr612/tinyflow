"""An implementation of flow matching on images, generic over the velocity model.

This module implements the Flow Matching Recipe for image-shaped data.
It is agnostic to the choice of velocity model: any eqx.Module implementing
``__call__(x, t) -> velocity`` with ``x`` of shape ``(H, W, C)`` and scalar
``t`` can be used as ``net_theta``.
"""

import json
from pathlib import Path
from typing import Any

import beartype
import diffrax
import equinox as eqx
import jax
import optax
import PIL.Image as Pilimage
from einops import pack, rearrange, unpack
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped
from tqdm import tqdm

from data.animefaces import to_uint8
from metrics import evaluate_fid


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

        Args:
            net_theta: The denoising model.
            t: Timesteps in ``[0, 1]``, shape ``(B,)``.
            x_0: Noise samples, shape ``(B, H, W, C)``.
            x_1: Data samples, shape ``(B, H, W, C)``.
            edge_weight: Weight for the auxiliary Sobel edge loss.
            line_weight: Weight for the auxiliary dark-line loss.
            denom_floor: Floor on ``1 - t``; see ``ImageFM.__init__``.

        Returns:
            Scalar mean loss.
        """
        t_b = rearrange(t, "b -> b 1 1 1")
        x_t = t_b * x_1 + (1 - t_b) * x_0
        x_hat = jax.vmap(lambda x, ti: net_theta(x, ti))(x_t, t)
        u = (x_hat - x_t) / jax.numpy.maximum(1 - t_b, denom_floor)
        vel_loss = optax.l2_loss(u, x_1 - x_0).mean()
        edge_loss = ImageFM._edge_loss(x_hat, x_1)
        line_loss = ImageFM._line_loss(x_hat, x_1)
        return vel_loss + edge_weight * edge_loss + line_weight * line_loss

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


def _save_sample(model, outdir, sample_noise, epoch):
    """Save a sample from fixed noise, both as the latest and as a per-epoch file.

    Because the noise is fixed, the ``sample_epoch_*.png`` series shows how the
    generator's output for one latent evolves over the run.
    """
    img = Pilimage.fromarray(to_uint8(model.generate(sample_noise)[0]))
    img.save(str(outdir / "sample.png"))
    img.save(str(outdir / f"sample_epoch_{epoch:04d}.png"))


def train_on_image(
    key: PRNGKeyArray,
    model: ImageFM,
    dataset,
    batches_per_epoch: int,
    n_epochs: int = 100,
    init_lr: float = 1e-3,
    edge_weight: float = 0.1,
    line_weight: float = 0.0,
    outpath: str | None = None,
    eval_every: int = 1,
    real_stats: dict | None = None,
    early_stop_patience: int = 0,
    fid_n_samples: int = 5000,
    t_mu: float = -0.8,
    t_sigma: float = 1.0,
) -> ImageFM:
    """Train the velocity model via flow matching on an image dataset.

    Args:
        key: JAX PRNG key.
        model: The ``ImageFM`` instance to train.
        dataset: An endless dataset yielding batches of shape ``(B, H, W, C)``.
        batches_per_epoch: Number of batches making up one pass over the data;
            the dataset never stops on its own, so this is what bounds training.
        n_epochs: Number of full passes over the dataset.
        init_lr: Initial learning rate for Adam.
        edge_weight: Weight for the auxiliary Sobel edge loss.
        line_weight: Weight for the auxiliary dark-line loss.
        outpath: Path for periodic checkpoint saves (overwritten each time).
            A checkpoint and a sample PNG are written at the end of every epoch.
        eval_every: Evaluate FID every N epochs (0 = disabled).  FID is measured
            on the epoch clock, so each score lands on the record holding the
            loss it belongs with.
        real_stats: Real-image Inception statistics for FID (``None`` skips FID
            evaluation, and with it best-checkpoint tracking and early stopping).
        early_stop_patience: Stop after this many consecutive FID evaluations
            that degrade beyond a 1% tolerance (0 = disabled).
        fid_n_samples: Number of generated images per FID evaluation.  FID is
            biased in ``n``, so the value is recorded next to each score.
        t_mu: Mean of the logit-normal timestep distribution
            ``t = sigmoid(N(t_mu, t_sigma^2))``.  Negative values sample lower
            ``t`` (higher noise) more often; JiT uses -0.8.
        t_sigma: Standard deviation of the logit-normal timestep distribution.

    Returns:
        The trained ``ImageFM`` model.
    """
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(init_lr),
    )
    optimizer_state = optimizer.init(eqx.filter(model.net_theta, eqx.is_inexact_array))

    key, sample_key = jax.random.split(key)
    sample_noise = jax.random.normal(sample_key, (1, 64, 64, 3))

    global_step = 0

    @eqx.filter_jit
    def make_update(key, net_theta, batch, optimizer_state):
        newkey, sk1, sk2 = jax.random.split(key, num=3)
        batch_size = batch.shape[0]
        # Logit-normal t (JiT, Tab. 3): mu < 0 shifts mass toward low t, i.e.
        # high noise, where the modelling is hard; the thin tail at t -> 1
        # also balances the 1/(1-t)^2 loss weight there.
        t = jax.nn.sigmoid(t_mu + t_sigma * jax.random.normal(sk1, (batch_size,)))
        rand_input = jax.random.normal(sk2, shape=batch.shape)
        loss, grad = eqx.filter_value_and_grad(ImageFM.train_step)(
            net_theta,
            t,
            rand_input,
            batch,
            edge_weight,
            line_weight,
            model.denom_floor,
        )
        updates, optimizer_state = optimizer.update(grad, optimizer_state)
        net_theta = eqx.apply_updates(net_theta, updates)
        return newkey, net_theta, optimizer_state, loss

    history = []
    best_fid = float("inf")
    fid_patience = 0
    net_loss = 0
    count = 0

    # The dataset is endless, so the step budget is what ends the run and epoch
    # boundaries fall out of the step counter.
    total_steps = n_epochs * batches_per_epoch
    batches = zip(range(total_steps), dataset)

    for _, batch in (pbar := tqdm(desc="run", iterable=batches, total=total_steps)):
        key, model.net_theta, optimizer_state, loss = make_update(
            key,
            model.net_theta,
            batch,
            optimizer_state,
        )
        global_step += 1
        net_loss += loss
        count += 1

        if global_step % batches_per_epoch:
            continue

        # End of an epoch: everything periodic happens here, on one clock.
        epoch = global_step // batches_per_epoch - 1
        record = {"epoch": epoch, "loss": float(net_loss / count)}
        net_loss = 0
        count = 0
        history.append(record)
        pbar.set_postfix({"loss": f"{record['loss']:.4f}", "step": global_step})

        if outpath is None:
            continue
        outdir = Path(outpath).parent
        model.save(outpath)
        _save_sample(model, outdir, sample_noise, epoch)

        if real_stats is not None and eval_every and (epoch + 1) % eval_every == 0:
            key, eval_key = jax.random.split(key)
            fid = evaluate_fid(model, real_stats, eval_key, n_samples=fid_n_samples)
            print(f"\nEpoch {epoch}: FID = {fid:.2f}", flush=True)
            record["fid"] = round(fid, 2)
            record["fid_n_samples"] = fid_n_samples

            if fid < best_fid * 1.01:
                if fid < best_fid:
                    best_fid = fid
                    model.save(str(outdir / "best_model.eqx"))
                fid_patience = 0
            else:
                fid_patience += 1
                print(f"  FID degradation {fid_patience}/{early_stop_patience}")

        with (outdir / "losses.json").open("w") as f:
            json.dump(history, f, indent=2)

        if early_stop_patience > 0 and fid_patience >= early_stop_patience:
            print(f"Early stopping at epoch {epoch}: best FID {best_fid:.2f}")
            return model

    return model
