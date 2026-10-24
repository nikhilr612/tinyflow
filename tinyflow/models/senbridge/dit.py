"""The radar-to-optical denoiser: ``x_hat = DiT(x_t, sar, meta, t)``.

A diffusion transformer -- JiT (Li & He 2025) adapted to pixel-aligned image
conditioning -- with one sample in, no batch axis, and ``(H, W, C)`` at the
boundary like the rest of the code base.  It predicts the optical image;
``bridge.py`` derives the velocity from it.

* **Patches of 8 px**: 256 tokens at 128x128.  Large pixel patches work only
  because the denoiser predicts the clean image (JiT's argument).
* **Two patch embeddings, summed**: ``x_t`` and ``sar`` are pixel-aligned, so
  they share token positions; each goes through its own low-rank bottleneck
  (patch -> 128 -> width), so each reaches the tokens through a rank-128 map.
  One bottleneck on the concatenated channels would be rank 128 in total:
  the clean radar would compete with the noisy ``x_t`` for those directions.
* **adaLN-Zero** on ``c = time(t) + meta(meta)``; every residual branch and
  the output start at zero, so the denoiser starts as ``x_hat = 0``.
* **Blocks**: RMSNorm (affine-free: adaLN supplies the scale), attention with
  qk-norm and 2-D axial RoPE, SwiGLU.  No absolute position embedding: a crop
  of the ground has no privileged position, so only relative positions enter.
* **Precision follows the weights.**  The inputs are cast to the weights'
  dtype and the output back to the input's, so casting the weights to bf16
  is the whole mixed-precision policy.  Norm statistics, RoPE and the time
  embedding are computed in float32 regardless; attention runs on cuDNN's
  fused kernel in half precision and on XLA otherwise.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from einops import rearrange
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped

PATCH = 8
BOTTLENECK = 128
TIME_SCALE = 1000.0
ROPE_THETA = 100.0  # wavelengths of 1 to ~100 patches on a 16x16 grid
EPS = 1e-6


def rms_norm(x: Float[Array, "... D"]) -> Float[Array, "... D"]:
    """Scale the last axis to unit root-mean-square; statistics in float32."""
    x32 = x.astype(jnp.float32)
    return (x32 * jax.lax.rsqrt(jnp.mean(x32**2, -1, keepdims=True) + EPS)).astype(
        x.dtype
    )


def rope_2d(x: Float[Array, " T N D"], side: int) -> Float[Array, " T N D"]:
    """Axial rotary positions: half of each head rotates by row, half by column."""
    n_freq = x.shape[-1] // 4
    freqs = ROPE_THETA ** -(jnp.arange(n_freq) / n_freq)
    rows, cols = jnp.divmod(jnp.arange(side * side), side)
    angle = jnp.concatenate([rows[:, None] * freqs, cols[:, None] * freqs], -1)
    cos, sin = jnp.cos(angle)[:, None], jnp.sin(angle)[:, None]  # T 1 D/2
    a, b = rearrange(x.astype(jnp.float32), "t n (d two) -> two t n d", two=2)
    out = rearrange([a * cos - b * sin, a * sin + b * cos], "two t n d -> t n (d two)")
    return out.astype(x.dtype)


def zero[L: eqx.Module](layer: L) -> L:
    """The same layer with every weight and bias at zero (adaLN-Zero)."""
    return jax.tree.map(jnp.zeros_like, layer)


class Block(eqx.Module):
    """Pre-norm transformer block, modulated and gated by ``c`` (adaLN-Zero)."""

    qkv: eqx.nn.Linear
    out: eqx.nn.Linear
    q_scale: Float[Array, " D"]
    k_scale: Float[Array, " D"]
    w12: eqx.nn.Linear
    w3: eqx.nn.Linear
    ada: eqx.nn.Linear
    heads: int = eqx.field(static=True)

    def __init__(self, width: int, heads: int, key: PRNGKeyArray):
        """``width`` channels in ``heads`` heads; SwiGLU at 4x width in parameters."""
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        hidden = int(4 * width * 2 / 3)
        self.heads = heads
        self.qkv = eqx.nn.Linear(width, 3 * width, key=k1)
        self.out = eqx.nn.Linear(width, width, key=k2)
        self.q_scale = jnp.ones(width // heads)
        self.k_scale = jnp.ones(width // heads)
        self.w12 = eqx.nn.Linear(width, 2 * hidden, key=k3)
        self.w3 = eqx.nn.Linear(hidden, width, key=k4)
        self.ada = zero(eqx.nn.Linear(width, 6 * width, key=k5))

    def attend(self, x: Float[Array, "T D"], side: int) -> Float[Array, "T D"]:
        """Multi-head self-attention with qk-norm and 2-D RoPE."""
        q, k, v = rearrange(
            jax.vmap(self.qkv)(x), "t (three n d) -> three t n d", three=3, n=self.heads
        )
        q = rope_2d(rms_norm(q) * self.q_scale, side)
        k = rope_2d(rms_norm(k) * self.k_scale, side)
        half = x.dtype in (jnp.bfloat16, jnp.float16)
        impl = "cudnn" if half and jax.default_backend() == "gpu" else "xla"
        o = jax.nn.dot_product_attention(q, k, v, implementation=impl)
        return jax.vmap(self.out)(rearrange(o, "t n d -> t (n d)"))

    def mlp(self, x: Float[Array, "T D"]) -> Float[Array, "T D"]:
        """SwiGLU."""
        a, b = jnp.split(jax.vmap(self.w12)(x), 2, axis=-1)
        return jax.vmap(self.w3)(jax.nn.silu(a) * b)

    def __call__(
        self, x: Float[Array, "T D"], c: Float[Array, " D"], side: int
    ) -> Float[Array, "T D"]:
        """One block; ``side`` is the token grid's side length."""
        shift1, scale1, gate1, shift2, scale2, gate2 = jnp.split(
            self.ada(jax.nn.silu(c)), 6
        )
        x = x + gate1 * self.attend(rms_norm(x) * (1 + scale1) + shift1, side)
        return x + gate2 * self.mlp(rms_norm(x) * (1 + scale2) + shift2)


class DiT(eqx.Module):
    """``x_hat = DiT(x_t, sar, meta, t)`` -- see the module docstring."""

    embed_x: eqx.nn.Sequential
    embed_sar: eqx.nn.Sequential
    time_mlp: eqx.nn.MLP
    meta_mlp: eqx.nn.MLP
    blocks: list[Block]
    final_ada: eqx.nn.Linear
    final: eqx.nn.Linear
    t_dim: int = eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        width: int = 512,
        depth: int = 12,
        heads: int = 8,
        n_meta: int = 7,
        t_dim: int = 256,
    ):
        """A ``depth``-block transformer of ``width`` channels for 128x128x3 images."""
        keys = iter(jax.random.split(key, 8 + depth))
        p = PATCH * PATCH * 3

        def bottleneck():
            return eqx.nn.Sequential(
                [
                    eqx.nn.Linear(p, BOTTLENECK, use_bias=False, key=next(keys)),
                    eqx.nn.Linear(BOTTLENECK, width, key=next(keys)),
                ]
            )

        self.t_dim = t_dim
        self.embed_x, self.embed_sar = bottleneck(), bottleneck()
        self.time_mlp = eqx.nn.MLP(t_dim, width, width, 1, jax.nn.silu, key=next(keys))
        self.meta_mlp = eqx.nn.MLP(n_meta, width, width, 1, jax.nn.silu, key=next(keys))
        self.blocks = [Block(width, heads, next(keys)) for _ in range(depth)]
        self.final_ada = zero(eqx.nn.Linear(width, 2 * width, key=next(keys)))
        self.final = zero(eqx.nn.Linear(width, p, key=next(keys)))

    def embed_time(self, t: Float[Array, ""]) -> Float[Array, " T"]:
        """Sinusoidal features of ``TIME_SCALE * t``, always in float32."""
        scales = jnp.geomspace(1, 1e-4, self.t_dim // 2)
        freqs = TIME_SCALE * t.astype(jnp.float32) * scales
        return jnp.concatenate([jnp.sin(freqs), jnp.cos(freqs)])

    @jaxtyped(typechecker=beartype)
    def __call__(
        self,
        x: Float[Array, " H W 3"],
        sar: Float[Array, " H W 3"],
        meta: Float[Array, " M"],
        t: Float[Array, ""],
    ) -> Float[Array, " H W 3"]:
        """Denoise ``x`` at time ``t`` given the radar image and its metadata."""
        dtype = self.final.weight.dtype
        side = x.shape[0] // PATCH

        def tokens(embed, img):
            patches = rearrange(img, "(h p) (w q) c -> (h w) (p q c)", p=PATCH, q=PATCH)
            return jax.vmap(embed)(patches.astype(dtype))

        h = tokens(self.embed_x, x) + tokens(self.embed_sar, sar)
        c = self.time_mlp(self.embed_time(t).astype(dtype)) + self.meta_mlp(
            meta.astype(dtype)
        )
        for block in self.blocks:
            h = block(h, c, side)
        shift, scale = jnp.split(self.final_ada(jax.nn.silu(c)), 2)
        out = jax.vmap(self.final)(rms_norm(h) * (1 + scale) + shift)
        img = rearrange(out, "(h w) (p q c) -> (h p) (w q) c", h=side, p=PATCH, q=PATCH)
        return img.astype(x.dtype)
