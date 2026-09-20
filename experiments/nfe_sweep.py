"""FID against the number of network evaluations (NFE) for fixed-step samplers.

The model is trained as a velocity field and sampled with Dopri5 (6 evaluations
per step; the 16-step evaluation setting is ~96 NFE).  This sweep measures the
*low*-NFE end -- Euler and midpoint at 1 to 32 steps -- which is (i) the baseline
any few-step method (flow map, consistency, shortcut, MeanFlow distillation) has
to beat, and (ii) a direct read of how curved the learned trajectories are: a
straight flow is exact in one Euler step.

Usage::

    uv run python experiments/nfe_sweep.py \
        runs/archive/9aef10c/ctrl_sched/best_model.eqx
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import PIL.Image as PilImage  # noqa: E402
import typer  # noqa: E402

from data.animefaces import preprocess_all, to_uint8  # noqa: E402
from data.layouts import LayoutPrior  # noqa: E402
from metrics import compute_real_stats, evaluate_fid  # noqa: E402
from models.imagefm import ImageFM  # noqa: E402
from models.unet import UNet  # noqa: E402


class FixedStepSampler:
    """``generate(noise)`` with an explicit Euler or midpoint scheme on the FM ODE."""

    def __init__(self, model: ImageFM, scheme: str, n_steps: int, masks=None):
        """``masks``: a bank cycled across batches for a conditioned model."""
        self.model, self.scheme, self.n_steps, self.masks = (
            model,
            scheme,
            n_steps,
            masks,
        )
        self._i = 0
        net = model.net_theta
        floor = model.denom_floor

        def v(x, t, c):  # batched: ``velocity`` is per-sample
            if c is None:
                return jax.vmap(lambda xi: ImageFM.velocity(net, xi, t, floor))(x)
            return jax.vmap(lambda xi, ci: ImageFM.velocity(net, xi, t, floor, ci))(
                x, c
            )

        @eqx.filter_jit
        def run(x, c):
            dt = 1.0 / n_steps
            for k in range(n_steps):
                t = k * dt
                if scheme == "euler":
                    x = x + dt * v(x, t, c)
                else:  # midpoint (2 NFE per step)
                    x_mid = x + 0.5 * dt * v(x, t, c)
                    x = x + dt * v(x_mid, t + 0.5 * dt, c)
            return x

        self._run = run

    @property
    def nfe(self) -> int:
        """Network evaluations per sample."""
        return self.n_steps * (1 if self.scheme == "euler" else 2)

    def generate(self, noise):
        """Sample ``len(noise)`` images."""
        c = None
        if self.masks is not None:
            n = len(noise)
            idx = (self._i + np.arange(n)) % len(self.masks)
            self._i += n
            c = self.model._cond(noise, jnp.asarray(self.masks[idx]))
        return self._run(noise, c)


def main(
    checkpoint: str,
    schemes: str = "euler,midpoint",
    steps: str = "1,2,4,8,16,32",
    n_fid: int = 5000,
    seed: int = 0,
    outdir: str = "runs/ablation/nfe",
):
    """FID per (scheme, steps) plus the Dopri5-16 reference, and a sample grid."""
    model = ImageFM.load(checkpoint, UNet.from_hparams)
    arr = preprocess_all("./data/anime-faces")
    real_stats = compute_real_stats(arr)
    masks = None
    if model.cond_channels:
        masks = LayoutPrior.load().sample_masks(n_fid, seed=seed)[
            ..., : model.cond_channels
        ]
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    key = jax.random.key(seed)
    grid_noise = jax.random.normal(jax.random.key(seed + 1), (8, 64, 64, 3))
    grid_masks = None if masks is None else jnp.asarray(masks[:8])

    rows = []
    model.n_steps = 16
    fid_ref = evaluate_fid(model, real_stats, key, n_samples=n_fid)
    rows.append({"scheme": "dopri5", "steps": 16, "nfe": 96, "fid": fid_ref})
    print(f"dopri5 16 steps (~96 NFE): FID {fid_ref:.1f}")
    grid_rows = [np.asarray(model.generate(grid_noise, grid_masks))]
    for scheme in schemes.split(","):
        for n in (int(s) for s in steps.split(",")):
            sampler = FixedStepSampler(model, scheme, n, masks)
            f = evaluate_fid(sampler, real_stats, key, n_samples=n_fid)
            rows.append({"scheme": scheme, "steps": n, "nfe": sampler.nfe, "fid": f})
            print(f"{scheme:>9} {n:>3} steps ({sampler.nfe:>3} NFE): FID {f:.1f}")
            sampler._i = 0
            grid_rows.append(np.asarray(sampler.generate(grid_noise)))
    (out / "results.json").write_text(json.dumps(rows, indent=2))
    grid = np.concatenate([np.concatenate(list(r), axis=1) for r in grid_rows], axis=0)
    PilImage.fromarray(to_uint8(np.clip(grid, -1, 1))).save(out / "grid_nfe.png")
    print(f"wrote {out}/results.json and grid_nfe.png (rows: dopri5-16, then each arm)")


if __name__ == "__main__":
    typer.run(main)
