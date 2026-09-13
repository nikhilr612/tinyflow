"""Evaluate a layout-conditioned model in its three sampling modes.

The conditional model is trained on real ``(x_1, mask(x_1))`` pairs, which
is required to learn ``p(x | m)`` -- but a FID measured with *real* masks as
conditions inherits the data's layout marginal and is not an unconditional
number.  So three modes, reported separately:

1. ``uncond``  -- null token; the apples-to-apples comparison with
   unconditioned models (no information about any ``x_1`` enters);
2. ``prior``   -- masks sampled from ``LayoutPrior`` (no real image
   involved); the two-stage generator's headline number, and where the
   left/right eye-consistency metric is meaningful;
3. ``real``    -- masks of held-out real images; an *upper bound* on what
   layout knowledge is worth, never compared with unconditional FIDs.

Each mode reports FID (``metrics.evaluate_fid``, same sampler and sample
count as training-time FID) and the eye-consistency statistics of
``experiments/eye_consistency.py``.

Usage::

    uv run python experiments/cond_eval.py runs/exp_cond/trial/model.eqx
        [--n-fid 5000] [--modes uncond,prior,real]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import typer  # noqa: E402

from data.animefaces import load_masks, preprocess_all  # noqa: E402
from data.layouts import LayoutPrior  # noqa: E402
from experiments.eye_consistency import eye_distance  # noqa: E402
from metrics import compute_real_stats, evaluate_fid  # noqa: E402
from models.imagefm import ImageFM  # noqa: E402
from models.unet import UNet  # noqa: E402

EYE_CHANNEL = 1  # mask channels: 0 face, 1 eyes, 2 mouth


class MaskedSampler:
    """``generate(noise)`` that pairs every batch with masks from a source."""

    def __init__(self, model: ImageFM, mask_fn):
        """``mask_fn(n, key) -> (n, H, W, K)`` masks in [0, 1], or ``None``."""
        self.model, self.mask_fn = model, mask_fn
        self._key = jax.random.key(0)

    def generate(self, x_0):
        """Sample ``len(x_0)`` images with freshly drawn masks."""
        self._key, sk = jax.random.split(self._key)
        return self.model.generate(x_0, self.mask_fn(len(x_0), sk))


def main(
    checkpoint: str,
    n_fid: int = 5000,
    modes: str = "uncond,prior,real",
    n_eyes: int = 256,
    seed: int = 0,
    n_steps: int = 16,
):
    """Print FID and eye-consistency per mode; write ``<run>/cond_eval.json``.

    ``n_steps`` sets the sampler; 16 Dopri5 steps score within 0.5 FID of 64
    on this data (``experiments/bottleneck.py``) at a quarter of the cost.
    """
    arr = preprocess_all("./data/anime-faces")
    real_stats = compute_real_stats(arr)
    model = ImageFM.load(checkpoint, UNet.from_hparams)
    model.n_steps = n_steps
    k = model.cond_channels
    if k == 0:
        raise SystemExit("not a conditioned model (cond_channels == 0)")
    masks_real = load_masks()
    prior = LayoutPrior.load()

    def prior_masks(n, key):
        s = int(jax.random.randint(key, (), 0, 2**31 - 1))
        return jnp.asarray(prior.sample_masks(n, seed=s)[..., :k])

    def real_masks(n, key):
        idx = np.asarray(jax.random.choice(key, len(masks_real), (n,), replace=False))
        return jnp.asarray(masks_real[idx, ..., :k].astype(np.float32) / 255.0)

    sources = {"uncond": None, "prior": prior_masks, "real": real_masks}

    # eye-consistency reference
    m_eye = masks_real[..., EYE_CHANNEL].astype(np.float32).mean(0) / 255.0
    left, right = m_eye.copy(), m_eye.copy()
    left[:, 32:] = 0
    right[:, :32] = 0
    idx = np.random.default_rng(seed).choice(len(arr), n_eyes, replace=False)
    d_real = eye_distance(arr[idx], left, right)
    thresh = float(np.quantile(d_real, 0.95))

    results = {"real_images": {"eye_mean": float(d_real.mean()), "hetero": 5.0}}
    print(f"{'mode':>8}{'FID':>9}{'eye dist':>10}{'hetero %':>10}")
    noise = jax.random.normal(jax.random.key(seed + 7), (n_eyes, 64, 64, 3))
    for mode in modes.split(","):
        src = sources[mode]
        sampler = model if src is None else MaskedSampler(model, src)
        fid = evaluate_fid(sampler, real_stats, jax.random.key(seed + 1), n_fid)
        gen = np.clip(np.asarray(sampler.generate(noise)), -1, 1)
        d = eye_distance(gen, left, right)
        results[mode] = {
            "fid": fid,
            "eye_mean": float(d.mean()),
            "hetero": float(100 * (d > thresh).mean()),
        }
        print(f"{mode:>8}{fid:>9.1f}{d.mean():>10.4f}{results[mode]['hetero']:>10.1f}")
    print(f"{'real':>8}{'':>9}{d_real.mean():>10.4f}{5.0:>10.1f}")
    out = Path(checkpoint).parent / "cond_eval.json"
    with out.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    typer.run(main)
