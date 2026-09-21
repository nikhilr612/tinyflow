"""The denoising U-Net: image in, denoised image out, layout through region pooling.

Layout: Equinox ``Conv2d`` is channels-first, so everything inside this module
works on ``(C, H, W)``; ``UNet.__call__`` transposes from and to the ``(H, W, C)``
that the rest of the code base uses.

Design, every point of it measured against the alternative:

* **x-prediction.**  The network returns the denoised image ``x_hat``, not a
  velocity; the velocity is derived in ``flow.py``.  There is no bounded output
  activation: the regression target is unbounded either way and a squashing
  function would only add vanishing gradients.
* **GroupNorm (8 groups)** rather than InstanceNorm: per-channel statistics
  would strip the per-image brightness and colour cast a generator must keep.
* **adaLN-Zero.**  Time enters every ResBlock as a FiLM ``(1 + gamma, beta)``
  from a zero-initialised linear on a shared time embedding, so each block
  starts as a plain ResBlock.
* **Skips** run from the *input* of each encoder level to the decoder level
  after its upsample -- the standard pattern shifted down one level.  Tested
  against the standard wiring: neutral at 9 M parameters, more stable at 37 M.
* **RegionPool** after the decoder levels producing 16x16 and 32x32: the
  layout masks (face, eyes, mouth, and hair/background = 1 - face) enter the
  network *only* here.  Features are mean-pooled inside each region, projected
  by a zero-initialised per-region matrix and broadcast back into that region.
  Both irises are painted from one pooled eye feature, which takes the
  left/right iris-colour mismatch rate from 35 % to the data's own 4 %.
  Feeding the masks at the input as well was measured to change nothing.
* **A second time input ``s``**, through a zero-initialised projection of the
  embedding of ``s - t``, turns the same network into a flow map
  ``x_hat(x_t, m, t, s)`` for distillation (``distill.py``); with ``s`` omitted
  it is exactly the velocity model.
"""

from __future__ import annotations

import equinox as eqx
import jax
from beartype import beartype
from einops import einsum, pack, rearrange, reduce
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped

N_GROUPS = 8
GN_EPS = 1e-5
N_REGIONS = 4  # face, eyes, mouth, hair/background
POOL_RESOLUTIONS = (16, 32)
TIME_SCALE = 1000.0


def group_norm(x: Float[Array, " C H W"]) -> Float[Array, " C H W"]:
    """Normalise over the channels of each group and over space (no affine)."""
    g = rearrange(x, "(G g) h w -> G g h w", G=N_GROUPS)
    g = g - reduce(g, "G g h w -> G 1 1 1", "mean")
    var = reduce(g**2, "G g h w -> G 1 1 1", "mean")
    return rearrange(g * jax.lax.rsqrt(var + GN_EPS), "G g h w -> (G g) h w")


class ResBlock(eqx.Module):
    """GroupNorm -> SiLU -> conv -> adaLN(t) -> SiLU -> conv, plus a skip."""

    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    skip: eqx.nn.Conv2d | eqx.nn.Identity
    ada: eqx.nn.Linear
    gn_scale: Float[Array, " C 1 1"]
    gn_shift: Float[Array, " C 1 1"]

    def __init__(self, c_in: int, c_out: int, t_dim: int, key: PRNGKeyArray):
        """``c_in -> c_out`` channels; ``t_dim`` is the time-embedding width."""
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.gn_scale = jax.numpy.ones((c_in, 1, 1))
        self.gn_shift = jax.numpy.zeros((c_in, 1, 1))
        self.conv1 = eqx.nn.Conv2d(c_in, c_out, 3, padding=1, key=k1)
        self.conv2 = eqx.nn.Conv2d(c_out, c_out, 3, padding=1, key=k2)
        # adaLN-Zero: the FiLM projection starts at zero, so (1 + gamma, beta)
        # = (1, 0) and the block begins as a plain ResBlock.
        ada = eqx.nn.Linear(t_dim, 2 * c_out, key=k3)
        self.ada = jax.tree.map(jax.numpy.zeros_like, ada)
        self.skip = (
            eqx.nn.Conv2d(c_in, c_out, 1, key=k4)
            if c_in != c_out
            else eqx.nn.Identity()
        )

    @jaxtyped(typechecker=beartype)
    def __call__(
        self, x: Float[Array, " C H W"], t_emb: Float[Array, " T"]
    ) -> Float[Array, " C_out H W"]:
        """Forward ``x`` with the time embedding ``t_emb``."""
        h = self.conv1(jax.nn.silu(group_norm(x) * self.gn_scale + self.gn_shift))
        gamma, beta = rearrange(
            self.ada(jax.nn.silu(t_emb)), "(two c) -> two c 1 1", two=2
        )
        h = self.conv2(jax.nn.silu(group_norm(h) * (1 + gamma) + beta))
        return h + self.skip(x)


class UBlock(eqx.Module):
    """One U-Net level: ``down`` doubles the channels and halves the resolution."""

    down_res: ResBlock
    down_conv: eqx.nn.Conv2d
    up_conv: eqx.nn.Conv2d
    up_res: ResBlock

    def __init__(self, c: int, t_dim: int, key: PRNGKeyArray):
        """``c`` channels enter ``down`` and leave ``up``."""
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.down_res = ResBlock(c, 2 * c, t_dim, k1)
        self.down_conv = eqx.nn.Conv2d(2 * c, 2 * c, 2, stride=2, groups=2 * c, key=k2)
        self.up_conv = eqx.nn.Conv2d(2 * c, c, 3, padding=1, key=k3)
        self.up_res = ResBlock(2 * c, c, t_dim, k4)  # 2c: the skip is concatenated

    def down(self, x: Float[Array, " C H W"], t_emb) -> Float[Array, " C2 H2 W2"]:
        """ResBlock, then a stride-2 depthwise convolution."""
        return self.down_conv(jax.nn.silu(self.down_res(x, t_emb)))

    def up(self, x, skip: Float[Array, " C H W"], t_emb) -> Float[Array, " C H W"]:
        """Bilinear x2, conv, concatenate the skip, ResBlock."""
        c, h, w = x.shape
        x = jax.image.resize(
            x, (c, 2 * h, 2 * w), jax.image.ResizeMethod.LINEAR, antialias=False
        )
        x, _ = pack([jax.nn.silu(self.up_conv(x)), skip], "* h w")
        return self.up_res(x, t_emb)


class RegionPool(eqx.Module):
    """``h <- h + sum_k m_k (W_k pooled_k + b_k)``, ``pooled_k`` the masked mean."""

    weight: Float[Array, " K C C"]
    bias: Float[Array, " K C"]

    def __init__(self, channels: int):
        """Zero-initialised: the layer is the identity at the start of training."""
        self.weight = jax.numpy.zeros((N_REGIONS, channels, channels))
        self.bias = jax.numpy.zeros((N_REGIONS, channels))

    @jaxtyped(typechecker=beartype)
    def __call__(
        self, h: Float[Array, " C H W"], masks: Float[Array, " K H W"]
    ) -> Float[Array, " C H W"]:
        """Pool inside each region, project, broadcast back into the region."""
        mass = reduce(masks, "k h w -> k 1", "sum") + 1e-6
        pooled = einsum(masks, h, "k h w, c h w -> k c") / mass
        proj = einsum(self.weight, pooled, "k d c, k c -> k d") + self.bias
        return h + einsum(masks, proj, "k h w, k d -> d h w")


class UNet(eqx.Module):
    """``x_hat = UNet(x_t, masks, t[, s])`` -- see the module docstring."""

    in_conv: eqx.nn.Conv2d
    out_conv: eqx.nn.Conv2d
    time_mlp: eqx.nn.Sequential
    span_proj: eqx.nn.Linear
    blocks: list[UBlock]
    pools: dict[int, RegionPool]
    t_dim: int = eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        base_channels: int = 64,
        t_dim: int = 128,
        n_blocks: int = 4,
    ):
        """A ``n_blocks``-level U-Net for 64x64 RGB images."""
        k1, k2, k3, k4, key = jax.random.split(key, 5)
        self.t_dim = t_dim
        self.in_conv = eqx.nn.Conv2d(3, base_channels, 3, padding=1, key=k1)
        self.out_conv = eqx.nn.Conv2d(base_channels, 3, 1, key=k2)
        self.time_mlp = eqx.nn.Sequential(
            [
                eqx.nn.Linear(t_dim, t_dim, key=k3),
                eqx.nn.Lambda(jax.nn.silu),
                eqx.nn.Linear(t_dim, t_dim, key=k4),
            ]
        )
        self.span_proj = jax.tree.map(
            jax.numpy.zeros_like, eqx.nn.Linear(t_dim, t_dim, key=k4)
        )
        self.blocks = []
        for i in range(n_blocks):
            key, sk = jax.random.split(key)
            self.blocks.append(UBlock(base_channels * 2**i, t_dim, sk))
        # RegionPool after the decoder levels whose output is 16x16 and 32x32.
        self.pools = {
            i: RegionPool(base_channels * 2**i)
            for i in range(n_blocks)
            if 64 // 2**i in POOL_RESOLUTIONS
        }

    def embed(self, t: Float[Array, ""]) -> Float[Array, " T"]:
        """Sinusoidal features of ``TIME_SCALE * t``: fine and coarse resolution."""
        freqs = TIME_SCALE * t * jax.numpy.geomspace(1, 1e-4, self.t_dim // 2)
        return jax.numpy.concatenate([jax.numpy.sin(freqs), jax.numpy.cos(freqs)])

    @jaxtyped(typechecker=beartype)
    def __call__(
        self,
        x: Float[Array, " H W 3"],
        masks: Float[Array, " H W 3"],
        t: Float[Array, ""],
        s: Float[Array, ""] | None = None,
    ) -> Float[Array, " H W 3"]:
        """Denoise ``x`` at time ``t`` given the layout ``masks`` (face, eyes, mouth).

        ``s`` is the target time of a flow-map jump; ``None`` is the velocity model.
        """
        feats = self.embed(t)
        if s is not None:
            feats = feats + self.span_proj(self.embed(s - t))
        t_emb = self.time_mlp(feats)
        m = rearrange(masks, "h w k -> k h w")
        regions = jax.numpy.concatenate([m, 1 - m[:1]], axis=0)  # + hair/background

        h = self.in_conv(rearrange(x, "h w c -> c h w"))
        skips = []
        for block in self.blocks:
            skips.append(h)
            h = block.down(h, t_emb)
        for i in reversed(range(len(self.blocks))):
            h = self.blocks[i].up(h, skips[i], t_emb)
            if i in self.pools:
                size = h.shape[1]
                h = self.pools[i](
                    h, reduce(regions, "k (h a) (w b) -> k h w", "mean", h=size, w=size)
                )
        return rearrange(self.out_conv(h), "c h w -> h w c")
