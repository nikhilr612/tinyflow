"""Re-noise-and-re-solve refinement: fix a sample with the model's own prior.

A finished sample ``x`` is re-noised to an intermediate time,

    x_{t0} = t0 * x + (1 - t0) * eps,   eps ~ N(0, I),

and the probability-flow ODE is solved again from ``t0`` to ``1 - t_eps`` (the
SDEdit recipe, Meng et al. 2022, applied to the model's own output; one round of
Restart sampling, Xu et al. 2023, if repeated).  Everything decided before ``t0``
(layout, hair colour, composition) survives; what is re-decided is the detail
formed after ``t0`` -- the part the bottleneck analysis found deficient.  Unlike an
external upscaler this corrects toward the training distribution.  For a
layout-conditioned model the same mask is used in both passes.

Reports FID (5000 samples, 16 steps per pass) and edge mass relative to real for
``t0`` in ``--t0s``; ``t0 = 0`` is the unrefined reference.

Usage::

    uv run python experiments/refine.py [--checkpoint ...] [--t0s 0,0.5,0.7,0.85]
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
from experiments.guidance import edge_mass  # noqa: E402
from metrics import compute_real_stats, evaluate_fid  # noqa: E402
from models.imagefm import ImageFM  # noqa: E402
from models.unet import UNet  # noqa: E402


class RefinedSampler:
    """``generate(noise)``: sample, re-noise to ``t0``, solve again from ``t0``."""

    def __init__(self, model: ImageFM, t0: float, masks: np.ndarray | None, seed: int):
        """``masks``: prior-sampled bank for a conditioned model (cycled), else None."""
        self.model, self.t0 = model, t0
        self.masks = None if masks is None else jnp.asarray(masks)
        self._pos, self._key = 0, jax.random.key(seed)
        t1 = 1.0 - model.t_eps
        term = diffrax.ODETerm(
            lambda t, x, c: ImageFM.velocity(
                model.net_theta, x, t, model.denom_floor, c
            )
        )

        def one(x_i, c_i, t_start):
            sol = diffrax.diffeqsolve(
                term,
                diffrax.Dopri5(),
                t0=t_start,
                t1=t1,
                y0=x_i,
                args=c_i,
                dt0=(t1 - t_start) / model.n_steps,
                saveat=diffrax.SaveAt(t1=True),
            )
            return sol.ys[0]

        self._solve_from = eqx.filter_jit(jax.vmap(one, in_axes=(0, 0, None)))

    def generate(self, x_0):
        """Two-pass sampling; the second pass starts from the re-noised first result."""
        masks = None
        if self.masks is not None:
            idx = (self._pos + jnp.arange(len(x_0))) % len(self.masks)
            self._pos = (self._pos + len(x_0)) % len(self.masks)
            masks = self.masks[idx]
        x = self.model.generate(x_0, masks)
        if self.t0 <= 0:
            return x
        self._key, sk = jax.random.split(self._key)
        eps = jax.random.normal(sk, x.shape)
        x_t0 = self.t0 * x + (1 - self.t0) * eps
        cond = self.model._cond(x_0, masks)
        return self._solve_from(x_t0, cond, jnp.asarray(self.t0, jnp.float32))


def main(
    checkpoint: str = "runs/wide_rp_300/best_model.eqx",
    t0s: str = "0,0.5,0.7,0.85",
    n_fid: int = 5000,
    n_steps: int = 16,
    seed: int = 0,
    outdir: str = "runs/ablation/refine",
):
    """Sweep ``t0``; print FID and edge mass; write a grid and ``results.json``."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    arr = preprocess_all("./data/anime-faces")
    real_stats = compute_real_stats(arr)
    real = jnp.asarray(arr[np.random.default_rng(seed).choice(len(arr), 512, False)])
    m_edge_real = float(jax.vmap(edge_mass)(real).mean())
    model = ImageFM.load(checkpoint, UNet.from_hparams)
    model.n_steps = n_steps
    masks = None
    if model.cond_channels:
        masks = LayoutPrior.load().sample_masks(n_fid, seed)[..., : model.cond_channels]
    grid_noise = jax.random.normal(jax.random.key(seed), (8, 64, 64, 3))
    results, rows = [], []
    print(f"{'t0':>6}{'FID':>9}{'edge/real':>11}")
    for t0 in (float(v) for v in t0s.split(",")):
        sampler = RefinedSampler(model, t0, masks, seed + 3)
        fid = evaluate_fid(sampler, real_stats, jax.random.key(seed + 1), n_fid)
        sampler._pos = 0
        s = jnp.clip(
            sampler.generate(
                jax.random.normal(jax.random.key(seed + 2), (256, 64, 64, 3))
            ),
            -1,
            1,
        )
        edge = float(jax.vmap(edge_mass)(s).mean()) / m_edge_real
        results.append({"t0": t0, "fid": fid, "edge": edge})
        sampler._pos = 0
        rows.append(to_uint8(np.clip(np.asarray(sampler.generate(grid_noise)), -1, 1)))
        print(f"{t0:>6.2f}{fid:>9.1f}{edge:>11.3f}")
    grid = np.concatenate([np.concatenate(list(g), axis=1) for g in rows], axis=0)
    Pilimage.fromarray(grid).resize(
        (grid.shape[1] * 2, grid.shape[0] * 2), Pilimage.Resampling.NEAREST
    ).save(out / "grid_t0.png")
    with (out / "results.json").open("w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out}/results.json and grid_t0.png (rows = t0 values in order)")


if __name__ == "__main__":
    typer.run(main)
