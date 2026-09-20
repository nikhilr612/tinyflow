"""Reproduce the one-epoch divergence at LR 1e-3 with per-module optimiser telemetry.

Four runs of the 37M model with added modules diverged in a single epoch at a
peak learning rate of 1e-3 (anime region-pool at epoch 80, mid-attention at 12,
standard skips at 3, CelebA region-pool at 25 -- the last one *on* the warm-up
+ cosine schedule); the plain model never did.  Gradient clipping at global
norm 1 was in place every time, so it is not one large gradient.  Working
hypothesis: Adam's normalised step -- ~lr per parameter regardless of gradient
size -- on the zero-initialised modules, whose second moments stay tiny while
their gradients are tiny.

This script trains the CelebA-64 region-pool config (the cheapest reproduction,
~5k steps) with the same optimiser as ``train_on_image`` and logs per step, to
``<outdir>/telemetry.jsonl``: loss, pre-clip gradient norm, and for each
top-level module (in_conv, blocks[i], region_pools, time_mlp, out_conv) the
gradient norm, the update-to-weight ratio and the mean Adam second moment.
Raw (non-EMA) weights are saved every ``--save-every`` epochs.  Flags bisect
the hypothesis: ``--beta2``, ``--eps``, ``--lr``, ``--rp-init`` (std of a random
RegionPool init instead of zero), ``--clip``.

Usage::

    uv run python experiments/instability.py --outdir runs/instab/base
    uv run python experiments/instability.py --outdir runs/instab/b2_099 --beta2 0.99
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
import optax  # noqa: E402
import typer  # noqa: E402
from tqdm import tqdm  # noqa: E402

import data.animefaces  # noqa: E402
from models.imagefm import ImageFM  # noqa: E402
from models.unet import UNet  # noqa: E402


def group_of(path) -> str:
    """Top-level module name of a pytree leaf path, e.g. ``blocks[2]``."""
    name = jax.tree_util.keystr(path[:1]).lstrip(".")
    if name == "blocks" and len(path) > 1:
        return f"blocks{jax.tree_util.keystr(path[1:2])}"
    return name


def group_norms(tree) -> dict[str, jax.Array]:
    """Squared L2 norm of every leaf, summed per top-level module."""
    out: dict[str, jax.Array] = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        g = group_of(path)
        out[g] = out.get(g, 0.0) + jnp.sum(jnp.square(leaf))
    return {k: jnp.sqrt(v) for k, v in out.items()}


def group_means(tree) -> dict[str, jax.Array]:
    """Mean of every leaf, weighted by size, per top-level module."""
    s: dict[str, jax.Array] = {}
    n: dict[str, int] = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        g = group_of(path)
        s[g] = s.get(g, 0.0) + jnp.sum(leaf)
        n[g] = n.get(g, 0) + leaf.size
    return {k: s[k] / n[k] for k in s}


def main(
    outdir: str = "runs/instab/base",
    n_epochs: int = 40,
    lr: float = 1e-3,
    warmup_steps: int = 500,
    lr_end_frac: float = 0.01,
    beta2: float = 0.999,
    eps: float = 1e-8,
    clip: float = 1.0,
    rp_init: float = 0.0,
    save_every: int = 1,
    seed: int = 49,
    batch_size: int = 128,
):
    """Train the CelebA-64 region-pool config with telemetry (see module doc)."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    arr = np.load("./.preprocessed/celebamask_faces.npy")
    masks = np.load("./.preprocessed/celebamask_masks.npy")
    keep = np.flatnonzero(np.load("./.preprocessed/celebamask_keep.npy"))
    dataset, batches_per_epoch = data.animefaces.wrap_dataset(
        arr[keep], masks[keep], batch_size=batch_size, seed=seed
    )
    hparams = {
        "base_channels": 64,
        "time_embedding_dim": 128,
        "n_blocks": 4,
        "in_channels": 8,
        "out_channels": 3,
        "cond_channels": 4,
        "region_pool": 1,
        "image_size": 64,
    }
    key = jax.random.key(seed)
    key, sk = jax.random.split(key)
    net = UNet(key=sk, **hparams)
    if rp_init > 0:
        key, sk = jax.random.split(key)
        net = eqx.tree_at(
            lambda m: [rp.weight for rp in m.region_pools.values()],
            net,
            [
                rp_init * jax.random.normal(jax.random.fold_in(sk, i), rp.weight.shape)
                for i, rp in enumerate(net.region_pools.values())
            ],
        )
    model = ImageFM(net, hparams=hparams)

    total = n_epochs * batches_per_epoch
    warm = min(warmup_steps, total // 10)
    schedule = optax.warmup_cosine_decay_schedule(
        0.0 if warm else lr, lr, warm, total, lr * lr_end_frac
    )
    opt = optax.chain(
        optax.clip_by_global_norm(clip) if clip > 0 else optax.identity(),
        optax.adam(schedule, b2=beta2, eps=eps),
    )
    params, static = eqx.partition(net, eqx.is_inexact_array)
    opt_state = opt.init(params)
    floor = model.denom_floor

    @eqx.filter_jit
    def step(params, opt_state, key, batch, m):
        k1, k2 = jax.random.split(key)
        t = jax.nn.sigmoid(-0.8 + jax.random.normal(k1, (batch.shape[0],)))
        x0 = jax.random.normal(k2, batch.shape)

        def loss_fn(p):
            return ImageFM.train_step(eqx.combine(p, static), t, x0, batch, floor, m, 4)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        gnorm = optax.global_norm(grads)
        updates, opt_state = opt.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        nu = opt_state[1][0].nu  # ty: ignore  # adam = chain(scale_by_adam, lr)
        tele = {
            "loss": loss,
            "grad_norm": gnorm,
            "grad": group_norms(grads),
            "update_over_weight": {
                k: v / (w + 1e-12)
                for (k, v), w in zip(
                    group_norms(updates).items(), group_norms(params).values()
                )
            },
            "adam_nu_mean": group_means(nu),
        }
        return new_params, opt_state, tele

    log = (out / "telemetry.jsonl").open("w")
    it = iter(dataset)
    for epoch in range(n_epochs):
        losses = []
        for _ in tqdm(range(batches_per_epoch), desc=f"ep {epoch}", leave=False):
            batch, m = next(it)
            key, sk = jax.random.split(key)
            params, opt_state, tele = step(params, opt_state, sk, batch, m)
            tele = jax.device_get(tele)
            rec: dict = {
                "epoch": epoch,
                "loss": float(tele["loss"]),
                "grad_norm": float(tele["grad_norm"]),
            }
            for name in ("grad", "update_over_weight", "adam_nu_mean"):
                rec[name] = {k: float(v) for k, v in tele[name].items()}
            log.write(json.dumps(rec) + "\n")
            losses.append(rec["loss"])
        log.flush()
        mean = float(np.mean(losses))
        print(f"epoch {epoch}: loss {mean:.4f}  max step loss {max(losses):.4f}")
        if save_every and (epoch + 1) % save_every == 0:
            model.net_theta = eqx.combine(params, static)
            model.save(str(out / f"raw_epoch_{epoch:04d}.eqx"))
        if not np.isfinite(mean) or mean > 1.0:
            print("diverged; stopping")
            break


if __name__ == "__main__":
    typer.run(main)
