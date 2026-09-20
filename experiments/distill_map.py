"""Distil the ODE sampler into a one/two-jump flow map with a fixed teacher.

Self-consistency objectives (MeanFlow, shortcut) bootstrap the map from
itself and, fine-tuned from this velocity model, either diverged or saturated
far above the ODE sampler (METHODS.md 3.4).  This is the plain alternative:
the *teacher* -- the trained velocity model under the midpoint-8 sampler -- is
a deterministic function of (noise, layout), so its endpoint is a regression
target with no conditional spread, and the L2 does not blur.

``pairs``   draws ``n`` (noise, prior layout) inputs, runs the teacher and
          caches ``x_0``, the state at ``t = 1/2`` and ``x_1`` (float16).
``train``   fine-tunes a ``flow_map=1`` copy of the teacher on the three
          deterministic jumps 0 -> 1, 0 -> 1/2 and 1/2 -> 1 (uniform per
          sample), logging 1- and 2-jump FID; writes ``<outdir>/model.eqx``.

Usage::

    TEACHER=runs/archive/9aef10c/wide_rp_200/best_model.eqx
    uv run python experiments/distill_map.py pairs $TEACHER
    uv run python experiments/distill_map.py train $TEACHER --n-epochs 40
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
import PIL.Image as PilImage  # noqa: E402
import typer  # noqa: E402
from tqdm import tqdm  # noqa: E402

from data.animefaces import preprocess_all, to_uint8  # noqa: E402
from data.layouts import LayoutPrior  # noqa: E402
from metrics import compute_real_stats, evaluate_fid  # noqa: E402
from models.imagefm import ImageFM, cond_token  # noqa: E402
from models.unet import UNet  # noqa: E402
from training import MapSampler  # noqa: E402

app = typer.Typer()
PAIRS = Path(".preprocessed/distill_pairs.npz")


def midpoint_trajectory(model: ImageFM, x_0, masks, n_steps: int = 8):
    """Midpoint sampler; returns the states at ``t = 1/2`` and ``t = 1``."""
    cond = model._cond(x_0, masks)
    net, floor = model.net_theta, model.denom_floor

    def v(x, t):
        return jax.vmap(lambda xi, ci: ImageFM.velocity(net, xi, t, floor, ci))(x, cond)

    x, dt, half = x_0, 1.0 / n_steps, None
    for k in range(n_steps):
        t = k * dt
        x = x + dt * v(x + 0.5 * dt * v(x, t), t + 0.5 * dt)
        if k + 1 == n_steps // 2:
            half = x
    return half, x


@app.command()
def pairs(teacher: str, n: int = 100_000, batch: int = 250, seed: int = 0):
    """Cache ``n`` teacher trajectories (noise, layout, x_half, x_1) at ``PAIRS``."""
    model = ImageFM.load(teacher, UNet.from_hparams)
    masks = LayoutPrior.load().sample_masks(n, seed=seed)[..., : model.cond_channels]
    run = eqx.filter_jit(lambda x, m: midpoint_trajectory(model, x, m))
    key = jax.random.key(seed)
    x_0, x_h, x_1 = [], [], []
    for i in tqdm(range(0, n, batch), desc="teacher"):
        key, sk = jax.random.split(key)
        noise = jax.random.normal(sk, (batch, 64, 64, 3))
        half, end = run(noise, jnp.asarray(masks[i : i + batch]))
        x_0.append(np.asarray(noise, np.float16))
        x_h.append(np.asarray(half, np.float16))
        x_1.append(np.asarray(end, np.float16))
    PAIRS.parent.mkdir(exist_ok=True)
    np.savez(
        PAIRS,
        x_0=np.concatenate(x_0),
        x_half=np.concatenate(x_h),
        x_1=np.concatenate(x_1),
        masks=(masks * 255).round().astype(np.uint8),
    )
    print(f"wrote {PAIRS}")


@app.command()
def train(
    teacher: str,
    outdir: str = "runs/distill_rp",
    n_epochs: int = 40,
    batch_size: int = 128,
    lr: float = 2e-4,
    ema_decay: float = 0.999,
    eval_every: int = 5,
    seed: int = 0,
):
    """Regress the flow map onto the cached teacher jumps; log 1/2-jump FID."""
    d = np.load(PAIRS)
    x_0, x_h, x_1, masks = d["x_0"], d["x_half"], d["x_1"], d["masks"]
    n = len(x_0)
    t_model = ImageFM.load(teacher, UNet.from_hparams)
    hparams = {**t_model.hparams, "flow_map": 1}
    student = UNet(key=jax.random.key(seed), **hparams)
    student = eqx.tree_at(
        lambda m: (m.in_conv, m.out_conv, m.time_mlp, m.blocks, m.region_pools),
        student,
        tuple(
            getattr(t_model.net_theta, a)
            for a in ("in_conv", "out_conv", "time_mlp", "blocks", "region_pools")
        ),
    )
    model = ImageFM(student, hparams=hparams)
    k = model.cond_channels
    steps = n_epochs * (n // batch_size)
    schedule = optax.warmup_cosine_decay_schedule(0.0, lr, 200, steps, lr * 0.01)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))
    params, static = eqx.partition(student, eqx.is_inexact_array)
    opt_state = opt.init(params)
    ema = params

    def loss_fn(params, xs, ts, ss, targets, cond):
        net = eqx.combine(params, static)

        def jump(x, t, s, c):
            u = ImageFM.mean_velocity(net, x, t, s, model.denom_floor, c)
            return x + (s - t) * u

        pred = jax.vmap(jump)(xs, ts, ss, cond)
        return jnp.mean((pred - targets) ** 2)

    @eqx.filter_jit
    def step(params, ema, opt_state, xs, ts, ss, targets, cond, i):
        loss, grads = jax.value_and_grad(loss_fn)(params, xs, ts, ss, targets, cond)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        decay = jnp.minimum(ema_decay, (1 + i) / (10 + i))
        ema = optax.incremental_update(params, ema, 1 - decay)
        return params, ema, opt_state, loss

    real_stats = compute_real_stats(preprocess_all("./data/anime-faces"))
    eval_masks = LayoutPrior.load().sample_masks(5000, seed=seed + 1)[..., :k]
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    grid_noise = jax.random.normal(jax.random.key(seed + 3), (8, 64, 64, 3))
    grid_masks = jnp.asarray(eval_masks[:8])
    history, i = [], 0
    for epoch in range(n_epochs):
        perm = rng.permutation(n)
        losses = []
        for b in tqdm(range(0, n - batch_size + 1, batch_size), desc=f"ep {epoch}"):
            idx = np.sort(perm[b : b + batch_size])
            # One of the three teacher jumps per sample: 0->1, 0->1/2, 1/2->1.
            which = rng.integers(0, 3, batch_size)
            src = np.where(which[:, None, None, None] == 2, x_h[idx], x_0[idx])
            tgt = np.where(which[:, None, None, None] == 1, x_h[idx], x_1[idx])
            ts = np.where(which == 2, 0.5, 0.0).astype(np.float32)
            ss = np.where(which == 1, 0.5, 1.0).astype(np.float32)
            m = jnp.asarray(masks[idx], jnp.float32) / 255.0
            cond = cond_token(m, m.shape[:3] + (k,))
            params, ema, opt_state, loss = step(
                params,
                ema,
                opt_state,
                jnp.asarray(src, jnp.float32),
                jnp.asarray(ts),
                jnp.asarray(ss),
                jnp.asarray(tgt, jnp.float32),
                cond,
                jnp.asarray(i),  # an int would be static and recompile each step
            )
            losses.append(float(loss))
            i += 1
        record = {"epoch": epoch, "loss": float(np.mean(losses))}
        model.net_theta = eqx.combine(ema, static)
        model.save(str(out / "model.eqx"))
        # Fixed noise + layouts, 1-jump row over 2-jump row, one PNG per epoch.
        grid = np.concatenate(
            [
                np.concatenate(
                    list(np.asarray(model.generate_map(grid_noise, grid_masks, k))), 1
                )
                for k in (1, 2)
            ]
        )
        PilImage.fromarray(to_uint8(np.clip(grid, -1, 1))).save(
            out / f"sample_epoch_{epoch:04d}.png"
        )
        if (epoch + 1) % eval_every == 0 or epoch + 1 == n_epochs:
            for n_jumps in (1, 2):
                f = evaluate_fid(
                    MapSampler(model, eval_masks, n_jumps),
                    real_stats,
                    jax.random.key(seed + 2),
                    n_samples=5000,
                )
                record[f"fid_map{n_jumps}"] = round(f, 2)
            print(f"epoch {epoch}: loss {record['loss']:.4f}  " + str(record))
        history.append(record)
        (out / "losses.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    app()
