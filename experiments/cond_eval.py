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
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import typer  # noqa: E402

from data.animefaces import load_masks, preprocess_all  # noqa: E402
from data.layouts import LayoutPrior  # noqa: E402
from experiments.checkpoints import CleanFM, load_any  # noqa: E402
from experiments.eye_consistency import eye_distance  # noqa: E402
from metrics import compute_real_stats, evaluate_fid  # noqa: E402

EYE_CHANNEL = 1  # mask channels: 0 face, 1 eyes, 2 mouth


class MaskedSampler:
    """``generate(noise)`` that pairs every batch with masks from a source."""

    def __init__(self, generate_fn, mask_fn):
        """``generate_fn(x_0, masks)`` and ``mask_fn(n, key) -> (n, H, W, K)`` masks."""
        self.generate_fn, self.mask_fn = generate_fn, mask_fn
        self._key = jax.random.key(0)

    def generate(self, x_0):
        """Sample ``len(x_0)`` images with freshly drawn masks."""
        self._key, sk = jax.random.split(self._key)
        return self.generate_fn(x_0, self.mask_fn(len(x_0), sk))


def main(
    checkpoint: str,
    n_fid: int = 5000,
    modes: str = "uncond,prior,real",
    n_eyes: int = 256,
    seed: int = 0,
    n_steps: int = 16,
    dataset: str = "anime",
    n_jumps: int = 0,
):
    """Print FID and eye-consistency per mode; write ``<run>/cond_eval.json``.

    ``n_steps`` sets the sampler; 16 Dopri5 steps score within 0.5 FID of 64
    on this data (``experiments/bottleneck.py``) at a quarter of the cost.
    ``n_jumps > 0`` samples with a clean-branch checkpoint's distilled flow map
    instead of the ODE solver.
    ``--dataset celeba`` evaluates against ``celebamask_faces.npy`` with the
    held-out ``celebamask_eval_masks.npy`` bank as the ``real`` masks (there
    is no layout prior, so ``prior`` is unavailable).
    """
    if dataset not in ("anime", "celeba"):
        raise SystemExit(f"unknown --dataset {dataset!r}")
    if dataset == "celeba":
        arr = np.load("./.preprocessed/celebamask_faces.npy")
        real_stats = compute_real_stats(
            arr, cache_path="./.preprocessed/celebamask_stats.npz"
        )
    else:
        arr = preprocess_all("./data/anime-faces")
        real_stats = compute_real_stats(arr)
    model = load_any(checkpoint)
    model.n_steps = n_steps
    generate_fn = model.generate
    if n_jumps:
        if not isinstance(model, CleanFM):
            raise SystemExit("--n-jumps needs a checkpoint with a distilled flow map")
        generate_fn = partial(model.jump, n_jumps=n_jumps)
    h = int(model.hparams.get("image_size", 64))
    k = model.cond_channels
    if k == 0:
        raise SystemExit("not a conditioned model (cond_channels == 0)")
    if dataset == "celeba":
        masks_real = np.load("./.preprocessed/celebamask_eval_masks.npy")
        prior = None
    else:
        masks_real = load_masks()
        prior = LayoutPrior.load()

    def prior_masks(n, key):
        assert prior is not None  # unavailable with --dataset celeba
        s = int(jax.random.randint(key, (), 0, 2**31 - 1))
        return jnp.asarray(prior.sample_masks(n, seed=s)[..., :k])

    def real_masks(n, key):
        idx = np.asarray(
            jax.random.choice(key, len(masks_real), (n,), replace=len(masks_real) < n)
        )
        return jnp.asarray(masks_real[idx, ..., :k].astype(np.float32) / 255.0)

    sources = {"uncond": None, "prior": prior_masks, "real": real_masks}

    # eye-consistency reference
    m_eye = masks_real[..., EYE_CHANNEL].astype(np.float32).mean(0) / 255.0
    left, right = m_eye.copy(), m_eye.copy()
    w2 = m_eye.shape[1] // 2
    left[:, w2:] = 0
    right[:, :w2] = 0
    idx = np.random.default_rng(seed).choice(len(arr), n_eyes, replace=False)
    d_real = eye_distance(arr[idx], left, right)
    thresh = float(np.quantile(d_real, 0.95))

    results = {"real_images": {"eye_mean": float(d_real.mean()), "hetero": 5.0}}
    print(f"{'mode':>8}{'FID':>9}{'eye dist':>10}{'hetero %':>10}")
    noise = jax.random.normal(jax.random.key(seed + 7), (n_eyes, h, h, 3))
    for mode in modes.split(","):
        if mode == "prior" and dataset == "celeba":
            raise SystemExit("--dataset celeba has no layout prior; use --modes real")
        src = sources[mode]
        sampler = model if src is None else MaskedSampler(generate_fn, src)
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
