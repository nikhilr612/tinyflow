"""Export a flow-map checkpoint's sampler to ONNX and time it on the CPU.

The exported graph is the *whole* sampler, ``(noise, masks) -> image``: the
conditioning token, ``n_jumps`` network evaluations and the jump arithmetic,
with the weights baked in.  Nothing from the training stack is needed to run
it -- ``onnxruntime`` and numpy suffice, which is what a CPU demo wants.

Runs in the project environment plus ``jax2onnx`` and ``onnxruntime``::

    uv run --with jax2onnx --with onnxruntime python experiments/export_onnx.py
        runs/distill_rp/model.eqx --jumps 1,2      (one line)

Writes ``<run dir>/sampler_<k>jump.onnx`` per jump count (one image per call,
inputs ``noise (64, 64, 3)`` and ``masks (64, 64, K)``), checks each against
the JAX sampler on the same input, and prints the CPU latency per image.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import typer  # noqa: E402

from data.layouts import LayoutPrior  # noqa: E402
from models.imagefm import ImageFM, cond_token  # noqa: E402
from models.unet import UNet  # noqa: E402


def make_sampler(model: ImageFM, n_jumps: int):
    """``f(noise, masks) -> image`` for ONE sample (no batch axis), weights closed over.

    Per-sample rather than ``generate_map`` because the converter has no
    batching rule for ``jax.image.resize`` (the U-Net's upsampling) under
    ``vmap``; a demo samples one image at a time anyway.
    """
    k = model.cond_channels

    def f(noise, masks):
        cond = cond_token(masks[None], (1, *masks.shape[:2], k))[0]
        x = noise
        for j in range(n_jumps):
            t, s = j / n_jumps, (j + 1) / n_jumps
            u = ImageFM.mean_velocity(model.net_theta, x, t, s, model.denom_floor, cond)
            x = x + (s - t) * u
        return x

    return f


def main(
    checkpoint: str,
    jumps: str = "1,2",
    n_timing: int = 20,
    threads: int = 0,
):
    """Export, verify and time the sampler for each jump count in ``--jumps``."""
    import jax2onnx  # ty: ignore[unresolved-import]  # export env only
    import onnxruntime as ort  # ty: ignore[unresolved-import]

    model = ImageFM.load(checkpoint, UNet.from_hparams)
    k = model.cond_channels
    masks = LayoutPrior.load().sample_masks(1, seed=0)[0, ..., :k].astype(np.float32)
    noise = np.asarray(jax.random.normal(jax.random.key(0), (64, 64, 3)))
    outdir = Path(checkpoint).parent
    for n_jumps in (int(j) for j in jumps.split(",")):
        f = make_sampler(model, n_jumps)
        path = outdir / f"sampler_{n_jumps}jump.onnx"
        # The converter appends to an existing weight sidecar; start clean.
        for stale in (path, path.with_suffix(".onnx.data")):
            stale.unlink(missing_ok=True)
        jax2onnx.to_onnx(
            f,
            [
                jax.ShapeDtypeStruct(noise.shape, jnp.float32),
                jax.ShapeDtypeStruct(masks.shape, jnp.float32),
            ],
            model_name=f"tinyflow_{n_jumps}jump",
            return_mode="file",
            output_path=str(path),
            input_names=["noise", "masks"],
            output_names=["image"],
        )
        ref = np.asarray(f(jnp.asarray(noise), jnp.asarray(masks)))

        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        feed = {"noise": noise, "masks": masks}
        out = sess.run(["image"], feed)[0]
        err = float(np.abs(out - ref).max())
        for _ in range(3):
            sess.run(["image"], feed)
        t0 = time.perf_counter()
        for _ in range(n_timing):
            sess.run(["image"], feed)
        ms = 1000 * (time.perf_counter() - t0) / n_timing
        size = path.stat().st_size / 1e6
        print(
            f"{n_jumps}-jump: {path.name} {size:.0f} MB | max |onnx - jax| = {err:.2e}"
            f" | CPU latency, one image: {ms:.0f} ms"
        )


if __name__ == "__main__":
    typer.run(main)
