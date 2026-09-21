"""Distil the sampler into a flow map that renders an image in one or two evaluations.

The trained velocity model under the midpoint sampler is a *deterministic*
function of (noise, layout).  Its outputs are therefore a regression target
with no conditional spread -- unlike training on data pairs, where the
minimiser ``E[x_1 | x_t]`` is a blur -- so a plain L2 on the map's jumps
learns them without loss of detail, and nothing is bootstrapped from the
student itself (self-consistency objectives -- MeanFlow, shortcut -- either
diverged or saturated when fine-tuned from this model).

Stage 1 runs the teacher on ``n_pairs`` (noise, prior layout) inputs and keeps
the states at ``t = 1/2`` and ``t = 1``.  Stage 2 fine-tunes a copy of the
teacher, which already carries the second time input, on the three jumps

    x_0 -> x_1,   x_0 -> x_1/2,   x_1/2 -> x_1,

    L = E || x_t + (s - t) u_theta(x_t, m, t, s) - x_s^teacher ||^2.

Sampling with the result is ``ImageFM.jump`` (one or two evaluations): FID 53
in one jump, 38 in two, against the teacher's 32 at sixteen evaluations.
``export_onnx`` writes the two-jump sampler as a single ONNX graph for CPU
inference (``onnxruntime`` + numpy, ~0.1 s per image on two cores).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

from data.layouts import LayoutPrior, rasterize
from models.animefaces.flow import ImageFM, mean_velocity
from models.animefaces.train import save_grid

if TYPE_CHECKING:
    from pathlib import Path


def teacher_pairs(
    model: ImageFM, n_pairs: int, seed: int = 0, batch: int = 250
) -> dict:
    """Noise, layout masks, and the teacher's states at ``t = 1/2`` and ``t = 1``."""
    from models.animefaces.flow import (
        N_STEPS,
        T_END,
        velocity,
    )  # the midpoint sampler, unrolled

    @eqx.filter_jit
    def trajectory(x, masks):
        def v(x, t):
            return jax.vmap(lambda xi, mi: velocity(model.net, xi, mi, t))(x, masks)

        dt, half = T_END / N_STEPS, None
        for k in range(N_STEPS):
            t = k * dt
            x = x + dt * v(x + 0.5 * dt * v(x, t), t + 0.5 * dt)
            if k + 1 == N_STEPS // 2:
                half = x
        return half, x

    layouts = LayoutPrior.load().sample(n_pairs, seed=seed)
    masks = rasterize(layouts)
    key = jax.random.key(seed)
    out = {"x_0": [], "x_half": [], "x_1": []}
    for i in tqdm(range(0, n_pairs, batch), desc="teacher"):
        key, sk = jax.random.split(key)
        noise = jax.random.normal(sk, (batch, 64, 64, 3))
        half, end = trajectory(noise, jnp.asarray(masks[i : i + batch]))
        out["x_0"].append(np.asarray(noise, np.float16))
        out["x_half"].append(np.asarray(half, np.float16))
        out["x_1"].append(np.asarray(end, np.float16))
    return {k: np.concatenate(v) for k, v in out.items()} | {"masks": masks}


def train_map(
    model: ImageFM,
    pairs: dict,
    outdir: Path,
    n_epochs: int = 40,
    batch_size: int = 128,
    lr: float = 2e-4,
    ema_decay: float = 0.999,
    seed: int = 0,
) -> ImageFM:
    """Fine-tune ``model.net`` in place on the teacher's jumps, checkpointing."""
    x_0, x_h, x_1, masks = pairs["x_0"], pairs["x_half"], pairs["x_1"], pairs["masks"]
    n = len(x_0)
    steps = n_epochs * (n // batch_size)
    warmup = min(200, steps // 10)
    schedule = optax.warmup_cosine_decay_schedule(0.0, lr, warmup, steps, lr * 0.01)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))
    params, static = eqx.partition(model.net, eqx.is_inexact_array)
    opt_state = opt.init(params)
    ema = params

    @eqx.filter_jit
    def step(params, ema, opt_state, i, xs, ms, ts, ss, targets):
        def loss_fn(p):
            net = eqx.combine(p, static)
            jump = lambda x, m, t, s: x + (s - t) * mean_velocity(net, x, m, t, s)  # noqa: E731
            return jnp.mean((jax.vmap(jump)(xs, ms, ts, ss) - targets) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        decay = jnp.minimum(ema_decay, (1 + i) / (10 + i))
        return params, optax.incremental_update(params, ema, 1 - decay), opt_state, loss

    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    grid_noise = jax.random.normal(jax.random.key(seed), (8, 64, 64, 3))
    grid_masks = jnp.asarray(masks[:8])
    history, i = [], 0
    for epoch in range(n_epochs):
        perm = rng.permutation(n)
        total = jnp.zeros(())
        for b in tqdm(
            range(0, n - batch_size + 1, batch_size), desc=f"epoch {epoch}", leave=False
        ):
            idx = np.sort(perm[b : b + batch_size])
            which = rng.integers(0, 3, batch_size)  # 0: 0->1, 1: 0->1/2, 2: 1/2->1
            src = np.where(which[:, None, None, None] == 2, x_h[idx], x_0[idx])
            tgt = np.where(which[:, None, None, None] == 1, x_h[idx], x_1[idx])
            ts = np.where(which == 2, 0.5, 0.0).astype(np.float32)
            ss = np.where(which == 1, 0.5, 1.0).astype(np.float32)
            i += 1
            params, ema, opt_state, loss = step(
                params,
                ema,
                opt_state,
                jnp.asarray(i),
                jnp.asarray(src, jnp.float32),
                jnp.asarray(masks[idx]),
                jnp.asarray(ts),
                jnp.asarray(ss),
                jnp.asarray(tgt, jnp.float32),
            )
            total += loss
        history.append({"epoch": epoch, "loss": float(total) / (n // batch_size)})
        model.net = eqx.combine(ema, static)
        model.save(str(outdir / "model.eqx"))
        (outdir / "losses.json").write_text(json.dumps(history, indent=1))
        rows = [np.asarray(model.jump(grid_noise, grid_masks, k)) for k in (1, 2)]
        save_grid(
            np.concatenate(rows, axis=1), outdir / f"sample_epoch_{epoch:04d}.png"
        )
    return model


def export_onnx(model: ImageFM, path: Path, n_jumps: int = 2) -> None:
    """Write the ``n_jumps`` sampler ``(noise, masks) -> image`` (one image) as ONNX.

    Needs the ``export`` extra (``jax2onnx``).  Weights go to ``path.data``.
    """
    import jax2onnx

    def sampler(noise, masks):
        x = noise
        for k in range(n_jumps):
            t, s = k / n_jumps, (k + 1) / n_jumps
            x = x + (s - t) * mean_velocity(model.net, x, masks, t, s)
        return x

    for stale in (path, path.with_suffix(path.suffix + ".data")):
        stale.unlink(missing_ok=True)
    spec = jax.ShapeDtypeStruct((64, 64, 3), jnp.float32)
    jax2onnx.to_onnx(
        sampler,
        [spec, spec],
        return_mode="file",
        output_path=str(path),
        input_names=["noise", "masks"],
        output_names=["image"],
    )
