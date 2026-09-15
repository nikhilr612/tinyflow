"""Autoguidance: guide the model with a worse version of itself (Karras et al. 2024).

    v = v_good + w * (v_good - v_bad),   optionally only for t_lo < t < t_hi

where ``v_bad`` comes from a checkpoint of the same architecture that is smaller
or less trained.  The extrapolation pushes samples away from the errors the two
models *share* and that the bad one makes more strongly -- hedged detail,
washed-out regions -- without any condition or extra training; on EDM2 it
improved FID by 1.5-2x.  For a layout-conditioned model both velocities see the
same mask.  ``w = 0`` is the unguided reference.  Reports FID (5000 samples,
16 steps) and edge mass relative to real.  ``--refine-t0 > 0`` adds one round
of re-noise-and-re-solve (``experiments/refine.py``) with the guided velocity
in both passes; ``--mask-source real`` conditions on training masks instead
of the prior (the layout-prior-free number).

Usage::

    uv run python experiments/autoguide.py --good runs/wide_rp_300/best_model.eqx
        --bad runs/confirm_rp/model.eqx --ws 0,0.5,1,2 --intervals 0-1,0.3-1
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

from data.animefaces import load_masks, preprocess_all, to_uint8  # noqa: E402
from data.layouts import LayoutPrior  # noqa: E402
from experiments.guidance import edge_mass  # noqa: E402
from metrics import compute_real_stats, evaluate_fid  # noqa: E402
from models.imagefm import ImageFM  # noqa: E402
from models.unet import UNet  # noqa: E402


class AutoguidedSampler:
    """``generate(noise)`` with ``v = v_good + w (v_good - v_bad)`` in a t-window."""

    def __init__(
        self,
        good: ImageFM,
        bad: ImageFM,
        w: float = 0.0,
        t_lo: float = 0.0,
        t_hi: float = 1.0,
        masks: np.ndarray | None = None,
        refine_t0: float = 0.0,
        seed: int = 0,
    ):
        """``masks``: mask bank (cycled); ``refine_t0``: see the module docstring."""
        self.good, self.w, self.t_lo, self.t_hi = good, w, t_lo, t_hi
        self.masks = None if masks is None else jnp.asarray(masks)
        self.refine_t0, self._key = refine_t0, jax.random.key(seed)
        self._pos = 0
        t1 = 1.0 - good.t_eps

        def velocity(t, x, args):
            (w, lo, hi), c = args
            v_g = ImageFM.velocity(good.net_theta, x, t, good.denom_floor, c)
            v_b = ImageFM.velocity(bad.net_theta, x, t, bad.denom_floor, c)
            gate = jnp.where((t > lo) & (t < hi), 1.0, 0.0)
            return v_g + w * gate * (v_g - v_b)

        term = diffrax.ODETerm(velocity)

        def one(x_i, c_i, params, t_start):
            sol = diffrax.diffeqsolve(
                term,
                diffrax.Dopri5(),
                t0=t_start,
                t1=t1,
                y0=x_i,
                args=(params, c_i),
                dt0=(t1 - t_start) / good.n_steps,
                saveat=diffrax.SaveAt(t1=True),
            )
            return sol.ys[0]

        self._solve = eqx.filter_jit(jax.vmap(one, in_axes=(0, 0, None, None)))

    def generate(self, x_0):
        """Solve the guided ODE from noise ``x_0``."""
        masks = None
        if self.masks is not None:
            idx = (self._pos + jnp.arange(len(x_0))) % len(self.masks)
            self._pos = (self._pos + len(x_0)) % len(self.masks)
            masks = self.masks[idx]
        params = jnp.asarray([self.w, self.t_lo, self.t_hi], dtype=jnp.float32)
        cond = self.good._cond(x_0, masks)
        x = self._solve(x_0, cond, params, jnp.float32(0.0))
        if self.refine_t0 <= 0:
            return x
        self._key, sk = jax.random.split(self._key)
        x_t0 = self.refine_t0 * x + (1 - self.refine_t0) * jax.random.normal(
            sk, x.shape
        )
        return self._solve(x_t0, cond, params, jnp.float32(self.refine_t0))


def main(
    good: str = "runs/wide_rp_300/best_model.eqx",
    bad: str = "runs/confirm_rp/model.eqx",
    ws: str = "0,0.5,1,2",
    intervals: str = "0-1,0.3-1",
    n_fid: int = 5000,
    n_steps: int = 16,
    seed: int = 0,
    outdir: str = "runs/ablation/autoguide",
    refine_t0: float = 0.0,
    mask_source: str = "prior",
    dataset: str = "anime",
):
    """Sweep ``w`` x interval; print FID and edge mass; write a grid and JSON.

    ``--dataset celeba`` measures against ``celebamask_faces.npy``; the
    ``prior`` mask source is unavailable (no layout prior) and ``real``
    resolves to the held-out ``celebamask_eval_masks.npy`` bank.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    if dataset not in ("anime", "celeba"):
        raise ValueError(f"unknown dataset {dataset!r}")
    if dataset == "celeba":
        arr = np.load("./.preprocessed/celebamask_faces.npy")
        real_stats = compute_real_stats(
            arr, cache_path="./.preprocessed/celebamask_stats.npz"
        )
    else:
        arr = preprocess_all("./data/anime-faces")
        real_stats = compute_real_stats(arr)
    real = jnp.asarray(arr[np.random.default_rng(seed).choice(len(arr), 512, False)])
    m_edge_real = float(jax.vmap(edge_mass)(real).mean())
    g = ImageFM.load(good, UNet.from_hparams)
    b = ImageFM.load(bad, UNet.from_hparams)
    if g.cond_channels != b.cond_channels:
        raise SystemExit("good and bad models must share the conditioning")
    g.n_steps = n_steps
    h = int(g.hparams.get("image_size", 64))
    masks = None
    if g.cond_channels and mask_source == "prior" and dataset == "anime":
        masks = LayoutPrior.load().sample_masks(n_fid, seed)[..., : g.cond_channels]
    elif g.cond_channels and mask_source == "prior":
        raise SystemExit("--dataset celeba has no layout prior; use --mask-source real")
    elif g.cond_channels:
        if dataset == "celeba" and mask_source == "real":
            default_bank = "./.preprocessed/celebamask_eval_masks.npy"
        else:
            default_bank = "./.preprocessed/anime_faces_masks4.npy"
        bank = load_masks(mask_source if mask_source != "real" else default_bank)
        idx = np.random.default_rng(seed).choice(
            len(bank), n_fid, replace=len(bank) < n_fid
        )
        masks = bank[idx, ..., : g.cond_channels].astype(np.float32) / 255.0
    sampler = AutoguidedSampler(g, b, masks=masks, refine_t0=refine_t0, seed=seed + 3)
    windows = [tuple(float(v) for v in x.split("-")) for x in intervals.split(",")]
    configs = [(0.0, (0.0, 1.0))] + [
        (w, win) for w in (float(v) for v in ws.split(",")) if w > 0 for win in windows
    ]
    grid_noise = jax.random.normal(jax.random.key(seed), (8, h, h, 3))
    results, rows = [], []
    print(f"bad: {bad}  masks: {mask_source}  refine_t0: {refine_t0}  steps: {n_steps}")
    print(f"{'w':>6}{'window':>12}{'FID':>9}{'edge/real':>11}")
    for w, (lo, hi) in configs:
        sampler.w, sampler.t_lo, sampler.t_hi = w, lo, hi
        sampler._pos = 0
        fid = evaluate_fid(sampler, real_stats, jax.random.key(seed + 1), n_fid)
        noise = jax.random.normal(jax.random.key(seed + 2), (256, h, h, 3))
        s = jnp.clip(sampler.generate(noise), -1, 1)
        edge = float(jax.vmap(edge_mass)(s).mean()) / m_edge_real
        results.append({"w": w, "t_lo": lo, "t_hi": hi, "fid": fid, "edge": edge})
        sampler._key = jax.random.key(seed + 3)
        sampler._pos = 0
        rows.append(to_uint8(np.clip(np.asarray(sampler.generate(grid_noise)), -1, 1)))
        print(f"{w:>6.2f}{f'{lo:.2f}-{hi:.2f}':>12}{fid:>9.1f}{edge:>11.3f}")
    grid = np.concatenate([np.concatenate(list(r), axis=1) for r in rows], axis=0)
    Pilimage.fromarray(grid).resize(
        (grid.shape[1] * 2, grid.shape[0] * 2), Pilimage.Resampling.NEAREST
    ).save(out / "grid_w.png")
    with (out / "results.json").open("w") as f:
        json.dump(
            {
                "good": good,
                "bad": bad,
                "mask_source": mask_source,
                "refine_t0": refine_t0,
                "n_steps": n_steps,
                "runs": results,
            },
            f,
            indent=2,
        )
    print(f"wrote {out}/results.json and grid_w.png (rows = configs in order)")


if __name__ == "__main__":
    typer.run(main)
