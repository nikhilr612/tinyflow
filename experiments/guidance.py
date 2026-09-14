"""Sampling-time edge guidance -- a sample-level prior with no retraining.

Training-time losses on ``x_hat`` against ``x_1`` are almost entirely
irreducible at the flow-matching optimum (``experiments/METHODS.md`` section 1);
a prior about *finished samples* belongs at sampling time.  At every ODE step
the prediction is nudged down the gradient of a unary energy,

    E(x_hat) = relu(m_real - mean|Sobel(x_hat)|)^2        (edge-mass deficit)
    g = dE/dx_t,   x_hat' = x_hat - lam * gate(t) * g / rms(g),
    v = (x_hat' - x_t) / max(1 - t, floor),

with ``m_real`` the data's mean Sobel magnitude (measured here), the gradient
taken through the network and RMS-normalised per sample so ``lam`` is in pixel
units, and ``gate(t) = 1[t_lo < t < t_hi]`` (guidance in a limited interval,
Kynkaanniemi et al. 2024).  One-sided: it switches off once the sample has
the data's edge mass, which is why small ``lam`` lands near ``edge/real = 1``.

``--scale sigma2`` replaces the constant step by ``lam * ((1 - t) / t)^2``
(clipped at 4, i.e. its value at ``t = 1/3``; equal to ``lam`` at ``t = 1/2``).
This is the posterior-variance factor of energy/classifier guidance (Dhariwal &
Nichol 2021; Chung et al. 2023): a shift of the posterior mean ``x_hat`` by
``-sigma_t^2 dE/dx`` is what a tilt ``p(x_1 | x_t) exp(-E)`` induces, and
``sigma_t^2 = ((1 - t) / t)^2`` in ``x_1`` units for the linear path.  The
constant schedule under-guides early and over-guides late by comparison.

Measured (METHODS.md section 4): -8 FID on a 9M model, -3 on the 37M
200-epoch model at ``lam = 0.02``; the best 9M setting was ``lam = 0.05`` on
``t in (0.5, 1)``.  Cost: one backward pass per network evaluation.

Usage::

    uv run python experiments/guidance.py [--checkpoint ...] [--n-fid 5000]
        [--lams 0,0.02,0.05] [--intervals 0-1,0.5-1] [--outdir ...]

For a layout-conditioned checkpoint, masks are drawn from the layout prior.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import diffrax  # noqa: E402
import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import PIL.Image as Pilimage  # noqa: E402
import typer  # noqa: E402

from data.animefaces import preprocess_all, to_uint8  # noqa: E402
from data.layouts import LayoutPrior  # noqa: E402
from metrics import compute_real_stats, evaluate_fid  # noqa: E402
from models.imagefm import ImageFM, cond_token  # noqa: E402
from models.unet import UNet  # noqa: E402


def sobel_magnitude(x: jnp.ndarray) -> jnp.ndarray:
    """Mean-over-channels Sobel magnitude of ``(B, H, W, C)`` images, ``(B, H, W)``."""
    kx = jnp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], jnp.float32)
    n = x.shape[-1]
    xp = jnp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="edge")

    def conv(k):
        return jax.lax.conv_general_dilated(
            xp,
            jnp.broadcast_to(k[..., None, None], (3, 3, 1, n)),
            (1, 1),
            "VALID",
            feature_group_count=n,
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )

    gx, gy = conv(kx), conv(kx.T)
    return jnp.sqrt(gx**2 + gy**2).mean(-1)


def edge_mass(x):
    """Mean Sobel magnitude of a single ``(H, W, C)`` image."""
    return sobel_magnitude(x[None]).mean()


class GuidedSampler:
    """``ImageFM``-compatible sampler whose velocity is guided by an energy.

    Compiled once; ``lam``, ``t_lo`` and ``t_hi`` are traced arguments so a
    sweep reuses the executable.  ``masks`` (a bank, cycled) are required for
    a layout-conditioned model and must be ``None`` otherwise.
    """

    def __init__(
        self,
        model: ImageFM,
        energy,
        lam: float = 0.0,
        t_lo: float = 0.0,
        t_hi: float = 1.0,
        masks: np.ndarray | None = None,
        scale: str = "const",
    ):
        """Wrap ``model``; ``energy(x_hat) -> scalar`` on one ``(H, W, C)`` image.

        ``scale``: ``"const"`` (step ``lam``) or ``"sigma2"`` (step
        ``lam * min(((1 - t) / t)^2, 4)``).
        """
        if scale not in ("const", "sigma2"):
            raise ValueError(f"unknown scale {scale!r}")
        self.model, self.lam, self.t_lo, self.t_hi = model, lam, t_lo, t_hi
        self.sigma2 = scale == "sigma2"
        self.masks = None if masks is None else jnp.asarray(masks)
        self._pos = 0
        net, floor = model.net_theta, model.denom_floor
        t1 = 1.0 - model.t_eps

        def x_hat_and_grad(x, t, c):
            def e(x):
                xh = net(x if c is None else jnp.concatenate([x, c], -1), t)
                return energy(xh), xh

            g, xh = jax.grad(e, has_aux=True)(x)
            return xh, g

        def velocity(t, x, args):
            (lam, t_lo, t_hi), c = args
            xh, g = x_hat_and_grad(x, t, c)
            rms = jnp.sqrt((g**2).mean()) + 1e-12
            gate = jnp.where((t > t_lo) & (t < t_hi), 1.0, 0.0)
            if self.sigma2:
                gate = gate * jnp.minimum(((1 - t) / jnp.maximum(t, 1e-3)) ** 2, 4.0)
            xh = xh - lam * gate * g / rms
            return (xh - x) / jnp.maximum(1 - t, floor)

        term = diffrax.ODETerm(velocity)

        def one(x_i, c_i, params):
            sol = diffrax.diffeqsolve(
                term,
                diffrax.Dopri5(),
                t0=0.0,
                t1=t1,
                y0=x_i,
                args=(params, c_i),
                dt0=t1 / model.n_steps,
                saveat=diffrax.SaveAt(t1=True),
            )
            return sol.ys[0]

        self._solve = eqx.filter_jit(jax.vmap(one, in_axes=(0, 0, None)))

    def generate(self, x_0):
        """Solve the guided ODE from noise ``x_0`` of shape ``(B, H, W, C)``."""
        params = jnp.asarray([self.lam, self.t_lo, self.t_hi], dtype=jnp.float32)
        cond = None
        if self.model.cond_channels:
            if self.masks is None:
                raise ValueError("a conditioned model needs masks")
            idx = (self._pos + jnp.arange(len(x_0))) % len(self.masks)
            self._pos = (self._pos + len(x_0)) % len(self.masks)
            cond = cond_token(self.masks[idx], (*x_0.shape[:3], self.masks.shape[-1]))
        return self._solve(x_0, cond, params)


def main(
    checkpoint: str = "runs/exp_long/wide_noaux/model.eqx",
    n_fid: int = 5000,
    lams: str = "0,0.02,0.05",
    intervals: str = "0-1,0.5-1",
    n_steps: int = 16,
    seed: int = 0,
    outdir: str = "runs/ablation/guidance",
    scale: str = "const",
):
    """Sweep ``lam`` x interval; print FID and edge mass; write grids + JSON."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    arr = preprocess_all("./data/anime-faces")
    real_stats = compute_real_stats(arr)
    real = jnp.asarray(arr[np.random.default_rng(seed).choice(len(arr), 512, False)])
    m_edge_real = float(jax.vmap(edge_mass)(real).mean())
    print(f"real: edge mass {m_edge_real:.4f}")

    model = ImageFM.load(checkpoint, UNet.from_hparams)
    model.n_steps = n_steps
    masks = None
    if model.cond_channels:
        masks = LayoutPrior.load().sample_masks(n_fid, seed)[..., : model.cond_channels]
    sampler = GuidedSampler(
        model,
        lambda x: jax.nn.relu(m_edge_real - edge_mass(x)) ** 2,
        masks=masks,
        scale=scale,
    )
    windows = [tuple(float(v) for v in w.split("-")) for w in intervals.split(",")]
    lam_list = [float(v) for v in lams.split(",")]
    configs = [(0.0, (0.0, 1.0))] + [
        (lam, w) for lam in lam_list if lam > 0 for w in windows
    ]
    grid_noise = jax.random.normal(jax.random.key(seed), (8, 64, 64, 3))
    results, rows = [], []
    print(f"\n{'lam':>7}{'window':>12}{'FID':>9}{'edge/real':>11}")
    for lam, (t_lo, t_hi) in configs:
        sampler.lam, sampler.t_lo, sampler.t_hi = lam, t_lo, t_hi
        sampler._pos = 0
        fid = evaluate_fid(sampler, real_stats, jax.random.key(seed + 1), n_fid)
        noise = jax.random.normal(jax.random.key(seed + 2), (256, 64, 64, 3))
        s = jnp.clip(sampler.generate(noise), -1, 1)
        edge = float(jax.vmap(edge_mass)(s).mean()) / m_edge_real
        results.append(
            {"lam": lam, "t_lo": t_lo, "t_hi": t_hi, "fid": fid, "edge": edge}
        )
        rows.append(to_uint8(sampler.generate(grid_noise)))
        print(f"{lam:>7.2f}{f'{t_lo:.2f}-{t_hi:.2f}':>12}{fid:>9.1f}{edge:>11.3f}")
    grid = np.concatenate([np.concatenate(list(g), axis=1) for g in rows], axis=0)
    Pilimage.fromarray(grid).resize(
        (grid.shape[1] * 2, grid.shape[0] * 2), Pilimage.Resampling.NEAREST
    ).save(out / "grid_edge.png")
    with (out / "results.json").open("w") as f:
        json.dump(
            {"real_edge_mass": m_edge_real, "scale": scale, "runs": results},
            f,
            indent=2,
        )
    print(f"\nwrote {out}/results.json and grid_edge.png (rows = configs in order)")


if __name__ == "__main__":
    typer.run(main)
