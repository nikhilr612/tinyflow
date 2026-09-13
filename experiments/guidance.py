"""Sampling-time guidance with unary image energies -- no retraining.

The training-time auxiliary losses compare ``x_hat`` to ``x_1`` and are mostly
irreducible at the noise levels where they act (see ``METHODS.md``).  A prior
about *finished samples* belongs at sampling time instead: at every ODE step
the prediction is nudged down the gradient of an energy ``E(x_hat)`` that
needs no ``x_1``,

    x_hat' = x_hat - lam * g / rms(g),   g = d E(x_hat(x_t, t)) / d x_t,
    v      = (x_hat' - x_t) / max(1 - t, floor),

with the gradient taken through the network and RMS-normalised per sample so
``lam`` is in pixel units.  The learned field is untouched, so none of the
bias arguments against training-time losses apply; the cost is one backward
pass per step.  Energies (all scalar targets measured on real data offline):

* ``edge``: ``relu(m_edge_real - mean|Sobel(x_hat)|)^2`` -- penalise an
  edge-mass deficit only;
* ``ink``:  the same with the ink map;
* ``eye``:  ``mean(m_eye * (x_hat - hflip(x_hat))^2)`` -- left/right eye
  disagreement under the dataset-mean eye mask (a symmetry prior; aligned
  faces make it approximately valid).

For each (energy, lam) the script reports FID and the sampling proxies
(edge / ink mass relative to real, eye asymmetry) and writes a sample grid.

Usage::

    uv run python experiments/guidance.py [--checkpoint ...] [--n-fid 2000]
        [--lams 0,0.02,0.05,0.1,0.2] [--energies edge,ink,eye]
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
from metrics import compute_real_stats, evaluate_fid  # noqa: E402
from models.imagefm import EYE_CHANNEL, ImageFM  # noqa: E402
from models.unet import UNet  # noqa: E402

OUTDIR = Path("runs/ablation/guidance")


def edge_mass(x):
    """Mean Sobel magnitude of a single ``(H, W, C)`` image."""
    gx, gy = ImageFM._sobel(x[None])
    return jnp.sqrt(gx**2 + gy**2).mean()


def ink_mass(x):
    """Mean ink map of a single ``(H, W, C)`` image."""
    return ImageFM._ink(x[None]).mean()


def eye_asym(x, m_eye):
    """Left/right disagreement of one image under the mean eye mask."""
    return (m_eye[..., None] * (x - x[:, ::-1]) ** 2).sum() / (m_eye.sum() * 3)


class GuidedSampler:
    """``ImageFM``-compatible sampler whose velocity is guided by an energy.

    The guided ODE is compiled once per energy; ``lam`` is a traced argument
    so a sweep over its values reuses the executable (closing over it
    recompiles per value, and the accumulated executables exhausted GPU
    memory in a 15-point sweep).
    """

    def __init__(
        self,
        model: ImageFM,
        energy,
        lam: float = 0.0,
        t_lo: float = 0.0,
        t_hi: float = 1.0,
    ):
        """Wrap ``model``; ``energy(x_hat) -> scalar`` on one ``(H, W, C)`` image.

        ``t_lo``/``t_hi`` restrict guidance to ``t_lo < t < t_hi`` (Kynkaanniemi
        et al. 2024: guidance in a limited interval of noise levels).  The
        correction enters the velocity as ``(x_hat' - x_t) / (1 - t)``, so a
        constant ``lam`` is amplified toward ``t -> 1`` and, at low ``t``, acts
        on the blurry mean where structure is not yet decided; an interval
        keeps it where the edge/ink mass is actually formed.
        """
        self.model, self.lam = model, lam
        self.t_lo, self.t_hi = t_lo, t_hi
        net, floor = model.net_theta, model.denom_floor
        t1 = 1.0 - model.t_eps

        def x_hat_and_grad(x, t):
            def e(x):
                xh = net(x, t)
                return energy(xh), xh

            g, xh = jax.grad(e, has_aux=True)(x)
            return xh, g

        def velocity(t, x, args):
            lam, t_lo, t_hi = args
            xh, g = x_hat_and_grad(x, t)
            rms = jnp.sqrt((g**2).mean()) + 1e-12
            gate = jnp.where((t > t_lo) & (t < t_hi), 1.0, 0.0)
            xh = xh - lam * gate * g / rms
            return (xh - x) / jnp.maximum(1 - t, floor)

        term = diffrax.ODETerm(velocity)

        def one(x_i, args):
            sol = diffrax.diffeqsolve(
                term,
                diffrax.Dopri5(),
                t0=0.0,
                t1=t1,
                y0=x_i,
                args=args,
                dt0=t1 / model.n_steps,
                saveat=diffrax.SaveAt(t1=True),
            )
            return sol.ys[0]

        self._solve = eqx.filter_jit(jax.vmap(one, in_axes=(0, None)))

    def generate(self, x_0):
        """Solve the guided ODE from noise ``x_0`` of shape ``(B, H, W, C)``."""
        args = jnp.asarray([self.lam, self.t_lo, self.t_hi], dtype=jnp.float32)
        return self._solve(x_0, args)


def main(
    checkpoint: str = "runs/exp_edge/edge0/model.eqx",
    n_fid: int = 2000,
    lams: str = "0,0.02,0.05,0.1,0.2",
    energies: str = "edge,ink,eye",
    intervals: str = "0-1",
    seed: int = 0,
    outdir: str = str(OUTDIR),
):
    """Sweep energies x lams x intervals; print a table, write grids + JSON.

    ``intervals`` is a comma list of ``lo-hi`` guidance windows in ``t``.
    """
    global OUTDIR  # noqa: PLW0603
    OUTDIR = Path(outdir)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    arr = preprocess_all("./data/anime-faces")
    real_stats = compute_real_stats(arr)
    real = jnp.asarray(arr[np.random.default_rng(seed).choice(len(arr), 512, False)])
    m_eye = jnp.asarray(load_masks()[..., EYE_CHANNEL].mean(0) / 255.0)
    m_edge_real = float(jax.vmap(edge_mass)(real).mean())
    m_ink_real = float(jax.vmap(ink_mass)(real).mean())
    eye_real = float(jax.vmap(lambda x: eye_asym(x, m_eye))(real).mean())
    print(
        f"real: edge mass {m_edge_real:.4f}  ink mass {m_ink_real:.4f}  "
        f"eye asym {eye_real:.4f}"
    )

    model = ImageFM.load(checkpoint, lambda key, **hp: UNet(**hp, key=key))
    energy_fns = {
        "edge": lambda x: jax.nn.relu(m_edge_real - edge_mass(x)) ** 2,
        "ink": lambda x: jax.nn.relu(m_ink_real - ink_mass(x)) ** 2,
        "eye": lambda x: eye_asym(x, m_eye),
    }
    lam_list = [float(v) for v in lams.split(",")]
    windows = [tuple(float(v) for v in w.split("-")) for w in intervals.split(",")]
    grid_noise = jax.random.normal(jax.random.key(seed), (8, 64, 64, 3))
    results = []
    samplers: dict[str, GuidedSampler] = {}
    print(
        f"\n{'energy':>8}{'lam':>7}{'window':>12}{'FID':>9}{'edge/real':>11}"
        f"{'ink/real':>10}{'eye asym':>10}"
    )
    for name in energies.split(","):
        rows = []
        configs = [
            (lam, w) for lam in lam_list for w in (windows if lam > 0 else [(0, 1)])
        ]
        for lam, (t_lo, t_hi) in configs:
            if lam == 0 and results:  # lam=0 is the same for every energy
                r = dict(results[0], energy=name)
            else:
                sampler = samplers.setdefault(
                    name, GuidedSampler(model, energy_fns[name])
                )
                sampler.lam, sampler.t_lo, sampler.t_hi = lam, t_lo, t_hi
                fid = evaluate_fid(sampler, real_stats, jax.random.key(seed + 1), n_fid)
                s = sampler.generate(
                    jax.random.normal(jax.random.key(seed + 2), (256, 64, 64, 3))
                )
                s = jnp.clip(s, -1, 1)
                r = {
                    "energy": name,
                    "lam": lam,
                    "t_lo": t_lo,
                    "t_hi": t_hi,
                    "fid": fid,
                    "edge": float(jax.vmap(edge_mass)(s).mean()) / m_edge_real,
                    "ink": float(jax.vmap(ink_mass)(s).mean()) / m_ink_real,
                    "eye": float(jax.vmap(lambda x: eye_asym(x, m_eye))(s).mean()),
                }
                rows.append(to_uint8(sampler.generate(grid_noise)))
            results.append(r)
            win = f"{r.get('t_lo', 0):.2f}-{r.get('t_hi', 1):.2f}"
            print(
                f"{name:>8}{lam:>7.2f}{win:>12}{r['fid']:>9.1f}"
                f"{r['edge']:>11.3f}{r['ink']:>10.3f}{r['eye']:>10.4f}"
            )
        if rows:
            grid = np.concatenate(
                [np.concatenate(list(g), axis=1) for g in rows], axis=0
            )
            img = Pilimage.fromarray(grid)
            img = img.resize(
                (grid.shape[1] * 2, grid.shape[0] * 2), Pilimage.Resampling.NEAREST
            )
            img.save(OUTDIR / f"grid_{name}.png")
    with (OUTDIR / "results.json").open("w") as f:
        json.dump(
            {
                "real": {"edge": m_edge_real, "ink": m_ink_real, "eye": eye_real},
                "runs": results,
            },
            f,
            indent=2,
        )
    print(f"\nwrote {OUTDIR}/results.json and grid_<energy>.png (rows = lams in order)")


if __name__ == "__main__":
    typer.run(main)
