"""Implement a denoising U-net for image generation via flow matching.

Note on data format:
    Equinox ``Conv2d`` (and ``lax.conv_general_dilated`` in JAX 0.10+)
    uses channels-first layout ``(C, H, W)`` for unbatched data.
    All internal operations in this module use ``(C, H, W)`` accordingly.
    The public interface (``UNet.__call__``) transposes to/from ``(H, W, C)``
    so that callers supplying image data in standard ``(H, W, C)`` format
    (e.g., ``ImageFM``) do not need to know about this internal detail.
"""

import equinox as eqx
import jax
from beartype import beartype
from einops import einsum, pack, rearrange, reduce
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped


def _group_norm(
    x: Float[Array, " C H W"], n_groups: int, eps: float
) -> Float[Array, " C H W"]:
    """Normalize over channel groups (no affine; callers apply their own).

    Statistics are taken over the ``g`` channels of each group *and* space
    (``G g h w -> G 1 1 1``).  Keeping ``g`` in the output would give one
    mean per channel, i.e. InstanceNorm, which strips per-image brightness
    and colour cast -- content a generator needs to keep.
    """
    grouped = rearrange(x, "(G g) h w -> G g h w", G=n_groups)
    mean = reduce(grouped, "G g h w -> G 1 1 1", "mean")
    shifted = grouped - mean
    var = reduce(shifted**2, "G g h w -> G 1 1 1", "mean")
    scaled = shifted * (1.0 / jax.lax.sqrt(var + eps))
    return rearrange(scaled, "G g h w -> (G g) h w")


class ResBlock(eqx.Module):
    """ResNet-inspired block.

    This module applies adaptive group normalization
    along with skip connections with learnable parameters.
    """

    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    skip_proj: eqx.nn.Conv2d | eqx.nn.Identity
    ada_proj: eqx.nn.Linear
    gn_scale: Float[Array, " C 1 1"]
    gn_shift: Float[Array, " C 1 1"]
    n_groups: int = eqx.field(static=True, default=8)
    gn_eps: float = eqx.field(static=True, default=1e-5)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embedding_dim: int,
        key: PRNGKeyArray,
    ):
        """Initialize a ResBlock which also consumes sinusoidal time embeddings.

        Args:
            in_channels: The number of input channels
            out_channels: The number of output channels
            time_embedding_dim: The number of channels in time embeddings.
            key: PRNG Key used for random initialization.
        """
        if in_channels % self.n_groups or out_channels % self.n_groups:
            raise ValueError(
                f"channels ({in_channels}, {out_channels}) must be divisible by "
                f"n_groups={self.n_groups}"
            )
        sk1, sk2, sk3, sk4 = jax.random.split(key, num=4)
        # Affine for the first norm only; the second norm's affine is the FiLM
        # (gamma, beta) from ada_proj.
        self.gn_scale = jax.numpy.ones((in_channels, 1, 1))
        self.gn_shift = jax.numpy.zeros((in_channels, 1, 1))
        self.conv1 = eqx.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            key=sk1,
        )

        self.conv2 = eqx.nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            key=sk2,
        )

        # adaLN-Zero (Peebles & Xie 2023; the FiLM analogue of the zero-gamma
        # trick in Wu & He 2018, Sec. 4.1): the projection starts at zero so
        # (1 + gamma, beta) = (1, 0) and the block begins as a plain ResBlock.
        # A default Linear would give gamma ~ U(+-1/sqrt(T_emb)), a random
        # near-zero gate that attenuates the main path ~10x at init.
        ada = eqx.nn.Linear(
            in_features=time_embedding_dim,
            out_features=out_channels * 2,
            key=sk3,
        )
        self.ada_proj = jax.tree.map(jax.numpy.zeros_like, ada)

        self.skip_proj = (
            eqx.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                key=sk4,
            )
            if in_channels != out_channels
            else eqx.nn.Identity()
        )

    def _group_norm(self, x: Float[Array, " C H W"]) -> Float[Array, " C H W"]:
        """Normalize over channel groups; see the module-level ``_group_norm``."""
        return _group_norm(x, self.n_groups, self.gn_eps)

    @jaxtyped(typechecker=beartype)
    def __call__(
        self, x: Float[Array, " C H W"], t_emb: Float[Array, " Temb"]
    ) -> Float[Array, " C_out H W"]:
        """Forward `x` through this block, with time embeddings `t`."""
        x_norm = self._group_norm(x) * self.gn_scale + self.gn_shift
        h1 = self.conv1(jax.nn.silu(x_norm))
        gamma, beta = rearrange(
            self.ada_proj(jax.nn.silu(t_emb)), "(two p) -> two p () ()", two=2
        )
        h2 = self._group_norm(h1) * (1 + gamma) + beta
        y = self.conv2(jax.nn.silu(h2))
        return y + self.skip_proj(x)


class UBlock(eqx.Module):
    """Module to implement Down and Up blocks in U-net.

    Implements the:
    - "down" ResNet-like blocks, 2-strided convolutions,
    - "up" interpolation and ResNet-like blocks
    with concatenation-based skip connections.
    The channels increase two-fold after `down`.

    Note:
    The skip connections in this Unet are slightly unconventional.
    This is partly due to a misunderstanding about their original purpose.
    The "U"-skip connections in this implementation rougly circumvent one encoder block.
    Indeed, the standard Unet has a skip-connection post-encoder to pre-decoder.
    Here, the skips are from pre-encoder to post-upsample.
    The fix is relatively straightforward;

    Traced precisely: the encoder ResBlock of level ``i`` *does* reach the
    decoder, as the skip of level ``i + 1`` (its output is what the strided
    conv downsamples into the next level's input).  So this pattern is the
    standard one shifted down one level: the top decoder level receives the
    raw ``in_conv`` features instead of a ResBlock output, and every other
    level receives the previous level's output (``channels`` wide) instead
    of its own (``2 * channels`` wide).  Tested (experiments/METHODS.md 6.1):
    the standard layout is neutral at 9M parameters and needs a lower
    learning rate at 37M; it is kept as is.
    """

    # TODO(n): Check if current skip connection pattern is detrimental to performance.
    #          https://github.com/nikhilr612/tinyflow/issues/16
    #          Switch over to proper skip connections.

    down_res_block: ResBlock
    down_conv: eqx.nn.Conv2d
    up_conv: eqx.nn.Conv2d
    up_res_block: ResBlock

    def __init__(
        self,
        channels: int,
        time_embedding_dim: int,
        key: PRNGKeyArray,
    ):
        """Initialize a single down and up block of the U-Net.

        Args:
            channels: The number of channels passing through this block.
                'down' doubles the number of channels, while 'up' restores it.
            time_embedding_dim: The number of dimensions for time embedding.
            key: PRNG Key for random initialization.
        """
        sk1, sk2, sk3, sk4 = jax.random.split(key, num=4)
        self.down_res_block = ResBlock(
            channels,
            channels * 2,
            time_embedding_dim,
            key=sk1,
        )

        self.down_conv = eqx.nn.Conv2d(
            in_channels=channels * 2,
            out_channels=channels * 2,
            groups=channels * 2,  # depthwise convolution.
            kernel_size=2,
            stride=2,
            key=sk2,
        )

        self.up_conv = eqx.nn.Conv2d(
            in_channels=channels * 2,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
            key=sk3,
        )

        self.up_res_block = ResBlock(
            channels * 2,  # twice, due to skip concatenation
            channels,
            time_embedding_dim,
            key=sk4,
        )

    @jaxtyped(typechecker=beartype)
    def down(
        self, x: Float[Array, " C H W"], t_emb: Float[Array, " Temb"]
    ) -> Float[Array, " C_out H_out W_out"]:
        """Forward data through the block, with downsampling.

        Pass `x` through a res block which increases channels.
        Then, downsample by half with a learned layer.
        """
        x_rb = self.down_res_block(x, t_emb)
        return self.down_conv(jax.nn.silu(x_rb))

    @jaxtyped(typechecker=beartype)
    def up(
        self,
        x: Float[Array, " C_in H W"],
        x_skip: Float[Array, " C_skip H_out W_out"],
        t_emb: Float[Array, " Temb"],
    ) -> Float[Array, " C_out H_out W_out"]:
        """Forward data through the block, with upsampling.

        Upsample `x` using bilinear interpolation.
        Then, forward through a res block with concatenated `x_skip`.
        """
        c_in, h, w = x.shape
        x_up = jax.image.resize(
            x,
            (c_in, h * 2, w * 2),
            method=jax.image.ResizeMethod.LINEAR,
        )
        x_conv = jax.nn.silu(self.up_conv(x_up))
        concatenated_skip, _packing = pack([x_conv, x_skip], "* h w")
        return self.up_res_block(concatenated_skip, t_emb)


class RegionPool(eqx.Module):
    """Mask-guided region pooling: one shared feature per semantic region.

    For every region ``k`` with soft mask ``m_k`` (``(H, W)`` in ``[0, 1]``)::

        pooled_k = sum_p m_k(p) h(p) / sum_p m_k(p)          one vector per region
        h_out    = h + sum_k m_k * (W_k pooled_k + b_k)

    The features inside a region are pooled to a single vector, projected by
    a zero-initialised ``1x1`` conv ``W_k`` and broadcast back into that region
    only.  Two irises are then painted from one shared iris feature -- "the
    eyes are one entity" stated structurally, at the resolution where hue is
    decided -- and the same holds for hair colour across strands.  It is
    masked attention with a single fixed query per region and uniform
    weights: the routing is given by the layout, only the map on the
    aggregate is learned.  All regions pool from the same input ``h`` (order
    independent) and the whole layer is three einsums.  Zero init makes it
    the identity at initialisation; all-zero masks (the null token, or an
    undetected face) contribute nothing.  ``W_k`` and ``b_k`` are stored as
    ``(K, C, C)`` and ``(K, C)`` arrays.
    """

    weight: Float[Array, " K C C"]
    bias: Float[Array, " K C"]

    def __init__(self, channels: int, n_regions: int):
        """Zero-initialised ``channels -> channels`` map per region."""
        self.weight = jax.numpy.zeros((n_regions, channels, channels))
        self.bias = jax.numpy.zeros((n_regions, channels))

    @jaxtyped(typechecker=beartype)
    def __call__(
        self, h: Float[Array, " C H W"], masks: Float[Array, " K H W"]
    ) -> Float[Array, " C H W"]:
        """Add every region's projected pooled feature back into that region."""
        mass = reduce(masks, "k h w -> k 1", "sum") + 1e-6
        pooled = einsum(masks, h, "k h w, c h w -> k c") / mass
        proj = einsum(self.weight, pooled, "k d c, k c -> k d") + self.bias
        return h + einsum(masks, proj, "k h w, k d -> d h w")


class UNet(eqx.Module):
    """Denoising UNet.

    Comprises of a fixed number of `UBlocks` applied recursively.
    Input channels are expanded using a preliminary convolution.
    Output is collapsed with a pointwise convolution to ``out_channels``
    (the image), which is fewer than ``in_channels`` when the input carries
    conditioning channels.

    Layout conditioning (``cond_channels > 0``):
        ``ImageFM`` concatenates ``cond_channels`` semantic-mask channels plus
        one indicator channel to the image (see ``imagefm.cond_token``); the
        U-Net only sees a wider input.  ``region_pool`` adds mask-guided
        ``RegionPool`` layers in the decoder, which read those mask channels
        straight from the input (experiments/METHODS.md, section 7.3: this is
        what fixes left/right iris colour agreement).
    """

    in_conv: eqx.nn.Conv2d
    blocks: list[UBlock]
    out_conv: eqx.nn.Conv2d
    time_mlp: eqx.nn.Sequential
    region_pools: dict[int, RegionPool]
    cond_channels: int = eqx.field(static=True)
    time_embedding_dim: int
    time_scale: float = eqx.field(static=True)

    def __init__(
        self,
        base_channels: int,
        time_embedding_dim: int,
        key: PRNGKeyArray,
        n_blocks: int = 4,
        in_channels: int = 3,
        out_channels: int = 0,
        cond_channels: int = 0,
        region_pool: int = 0,
        time_scale: float = 1000.0,
    ):
        """Initialize a denoising U-Net with multiple blocks conditioned on time.

        Args:
            base_channels: The number of input channels to the first down block.
            time_embedding_dim: The size of 1d time embedding.
            key: PRNG key for random initialization
            n_blocks: The number of blocks
            in_channels: Channels of the network input: the image plus, for a
                conditioned model, ``cond_channels + 1`` conditioning channels.
            out_channels: Output channels; ``0`` means the same as
                ``in_channels``.
            cond_channels: Number of layout-mask channels in the input (after
                the image).  Accepted so it can live in the shared ``hparams``
                dict; checked against ``in_channels - out_channels - 1`` and
                used by ``region_pool``.
            region_pool: ``1`` adds ``RegionPool`` layers after the up blocks
                that produce 16x16 and 32x32 features, using the layout masks
                in the conditioning channels (regions: face, eyes, mouth, and
                hair/background = 1 - face).  Needs ``cond_channels = 3``.
                Identity at init.  ``0`` leaves it out.
            time_scale: Multiplier applied to ``t`` before the sinusoidal
                embedding; see ``sinusoidal_embeddings``.
        """
        n_out = out_channels or in_channels
        if cond_channels and in_channels != n_out + cond_channels + 1:
            raise ValueError(
                f"in_channels={in_channels} must equal out_channels + cond_channels + 1"
                f" = {n_out + cond_channels + 1}"
            )
        sk1, sk2, key = jax.random.split(key, num=3)
        self.in_conv = eqx.nn.Conv2d(in_channels, base_channels, 3, padding=1, key=sk1)
        self.out_conv = eqx.nn.Conv2d(base_channels, n_out, 1, key=sk2)
        self.time_embedding_dim = time_embedding_dim
        self.time_mlp = eqx.nn.Sequential(
            [
                eqx.nn.Linear(time_embedding_dim, time_embedding_dim, key=sk1),
                eqx.nn.Lambda(jax.nn.silu),
                eqx.nn.Linear(time_embedding_dim, time_embedding_dim, key=sk2),
            ]
        )
        self.blocks = []
        self.time_scale = time_scale
        for _ in range(n_blocks):
            sk, key = jax.random.split(key)
            self.blocks.append(
                UBlock(base_channels * 2 ** len(self.blocks), time_embedding_dim, sk)
            )

        self.cond_channels = cond_channels
        self.region_pools = {}
        if region_pool:
            if cond_channels != 3:
                raise ValueError(
                    "region_pool needs cond_channels = 3 (face, eyes, mouth)"
                )
            # levels whose up block outputs 16x16 and 32x32 (64 / 2**i)
            for level in (2, 1):
                if level < n_blocks:
                    self.region_pools[level] = RegionPool(base_channels * 2**level, 4)

    @classmethod
    def from_hparams(cls, key: PRNGKeyArray, **hparams) -> "UNet":
        """Build a skeleton from a saved ``hparams`` dict, ignoring unknown keys.

        Checkpoints from the archived experiment branch carry keys for
        options that no longer exist (``mid_block``, ``skip_mode``, ...); those
        options were all off in the checkpoints worth loading, so dropping the
        keys reproduces the right architecture.
        """
        known = {
            "base_channels",
            "time_embedding_dim",
            "n_blocks",
            "in_channels",
            "out_channels",
            "cond_channels",
            "region_pool",
            "time_scale",
        }
        return cls(key=key, **{k: v for k, v in hparams.items() if k in known})

    def sinusoidal_embeddings(self, t: Float[Array, ""]) -> Float[Array, " Tembed"]:
        """Sinusoidal embeddings for a given scalar time `t`.

        The frequencies follow the transformer/DDPM convention, spanning
        ``1 .. 1e-4``, which presumes an integer-scale position.  ``t`` here is in
        ``[0, 1]``, so without ``time_scale`` no frequency would complete even a
        fraction of a cycle and the embedding would be a near-constant function
        of ``t``.  Scaled, the fastest component oscillates ~160 times across the
        unit interval and the slowest ~0.016 times, giving both fine and coarse
        resolution in ``t``.
        """
        freqs = (self.time_scale * t) * jax.numpy.geomspace(
            1, 1 / 10_000, num=self.time_embedding_dim // 2
        )
        return jax.numpy.concat([jax.numpy.sin(freqs), jax.numpy.cos(freqs)])

    def _forward(
        self, x: Float[Array, " H W C"], t: Float[Array, ""]
    ) -> Float[Array, " C H W"]:
        """Run the U-Net; return the channels-first output."""
        t_embed = self.time_mlp(self.sinusoidal_embeddings(t))
        x_c = rearrange(x, "h w c -> c h w")
        x_in = self.in_conv(x_c)
        skip_values: list[Array] = []
        for i in range(len(self.blocks)):
            skip_values.append(x_in)
            x_in = self.blocks[i].down(x_in, t_embed)
        x_out = x_in
        regions = None
        if self.region_pools:
            # Conditioning channels sit after the image: [x_t, masks, indicator].
            m = x_c[3 : 3 + self.cond_channels]  # (3, H, W): face, eyes, mouth
            regions = jax.numpy.concatenate([m, 1 - m[:1]], axis=0)  # + hair/bg
        for i in range(len(self.blocks) - 1, -1, -1):
            x_out = self.blocks[i].up(x_out, skip_values[i], t_embed)
            if regions is not None and i in self.region_pools:
                h = x_out.shape[1]
                m_level = reduce(regions, "k (h a) (w b) -> k h w", "mean", h=h, w=h)
                x_out = self.region_pools[i](x_out, m_level)
        return self.out_conv(x_out)

    @jaxtyped(typechecker=beartype)
    def __call__(
        self, x: Float[Array, " H W C_in"], t: Float[Array, ""]
    ) -> Float[Array, " H W C_out"]:
        """Forward an image to denoise through the U-net.

        Denoise an image `x`, at timestep `t`.  ``C_in`` and ``C_out`` agree
        for an unconditioned model; a conditioned one takes extra layout
        channels in (``C_in = C_out + cond_channels + 1``) and emits the image.

        The final bounded activation is omitted.  With the velocity field
        parametrised through the denoised image (Li & He, 2025) the regression
        target is unbounded either way, and a bounded output would add
        vanishing gradients without gaining anything.
        """
        x_out = self._forward(x, t)
        return rearrange(x_out, "c h w -> h w c")
