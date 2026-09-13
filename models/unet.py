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


class AttnBlock(eqx.Module):
    """Single self-attention layer over the spatial positions of a feature map.

    Every position attends to every other, so one layer gives the whole image
    a global receptive field -- the thing a stack of 3x3 convolutions lacks.
    Follows DDPM/ADM: GroupNorm, one 1x1 conv producing (q, k, v), softmax
    attention per head, a 1x1 output projection, residual add.  No positional
    encoding; the convolutions before it already break permutation symmetry.

    The output projection starts at zero, so the block is the identity at
    init and training switches it on gradually (the same trick as adaLN-Zero
    in ``ResBlock``).
    """

    qkv: eqx.nn.Conv2d
    proj: eqx.nn.Conv2d
    gn_scale: Float[Array, " C 1 1"]
    gn_shift: Float[Array, " C 1 1"]
    n_heads: int = eqx.field(static=True)
    n_groups: int = eqx.field(static=True, default=8)
    gn_eps: float = eqx.field(static=True, default=1e-5)

    def __init__(self, channels: int, key: PRNGKeyArray, head_dim: int = 64):
        """Initialize attention over ``channels`` with ``channels // head_dim`` heads.

        ``head_dim = 64`` is ADM's convention (Dhariwal & Nichol 2021).
        """
        if channels % head_dim:
            raise ValueError(
                f"channels={channels} must be divisible by head_dim={head_dim}"
            )
        self.n_heads = channels // head_dim
        sk1, sk2 = jax.random.split(key)
        self.gn_scale = jax.numpy.ones((channels, 1, 1))
        self.gn_shift = jax.numpy.zeros((channels, 1, 1))
        self.qkv = eqx.nn.Conv2d(channels, 3 * channels, kernel_size=1, key=sk1)
        proj = eqx.nn.Conv2d(channels, channels, kernel_size=1, key=sk2)
        self.proj = eqx.tree_at(
            lambda m: (m.weight, m.bias),
            proj,
            jax.tree.map(jax.numpy.zeros_like, (proj.weight, proj.bias)),
        )

    @jaxtyped(typechecker=beartype)
    def __call__(self, x: Float[Array, " C H W"]) -> Float[Array, " C H W"]:
        """Attend over the ``H * W`` positions of ``x`` and add the result to ``x``."""
        x_norm = _group_norm(x, self.n_groups, self.gn_eps) * self.gn_scale
        x_norm = x_norm + self.gn_shift
        # Split the tripled channels into (q, k, v) and heads; flatten space
        # into a token axis so each of q, k, v is (heads, tokens, d).
        q, k, v = rearrange(
            self.qkv(x_norm),
            "(three heads d) h w -> three heads (h w) d",
            three=3,
            heads=self.n_heads,
        )
        # scores[head, i, j]: how much query token i attends to key token j.
        scores = einsum(q, k, "heads i d, heads j d -> heads i j")
        attn = jax.nn.softmax(scores / jax.numpy.sqrt(q.shape[-1]), axis=-1)
        out = einsum(attn, v, "heads i j, heads j d -> heads i d")
        out = rearrange(out, "heads (h w) d -> (heads d) h w", h=x.shape[1])
        return x + self.proj(out)


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
    conv downsamples into the next level's input).  So the legacy pattern is
    the standard one shifted down one level: the top decoder level receives
    the raw ``in_conv`` features instead of a ResBlock output, and every
    other level receives the previous level's output (``channels`` wide)
    instead of its own (``2 * channels`` wide).

    ``skip_mode`` selects the pattern: ``0`` keeps the legacy one (default, so
    saved checkpoints load unchanged), ``1`` is the standard post-encoder to
    pre-decoder skip, which makes ``up_res_block`` take ``3 * channels``.
    """

    # TODO(n): Check if current skip connection pattern is detrimental to performance.
    #          https://github.com/nikhilr612/tinyflow/issues/16
    #          Switch over to proper skip connections.

    down_res_block: ResBlock
    down_conv: eqx.nn.Conv2d
    up_conv: eqx.nn.Conv2d
    up_res_block: ResBlock
    skip_mode: int = eqx.field(static=True)

    def __init__(
        self,
        channels: int,
        time_embedding_dim: int,
        key: PRNGKeyArray,
        skip_mode: int = 0,
    ):
        """Initialize a single down and up block of the U-Net.

        Args:
            channels: The number of channels passing through this block.
                'down' doubles the number of channels, while 'up' restores it.
            time_embedding_dim: The number of dimensions for time embedding.
            key: PRNG Key for random initialization.
            skip_mode: ``0`` legacy skips (pre-encoder features, ``channels``
                wide), ``1`` standard skips (encoder ResBlock output,
                ``2 * channels`` wide).  See the class docstring.
        """
        if skip_mode not in (0, 1):
            raise ValueError(f"skip_mode must be 0 or 1, got {skip_mode}")
        self.skip_mode = skip_mode
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

        # up_conv output (channels) concatenated with the skip: `channels` wide
        # for the legacy pattern, `2 * channels` for the standard one.
        self.up_res_block = ResBlock(
            channels * (2 if skip_mode == 0 else 3),
            channels,
            time_embedding_dim,
            key=sk4,
        )

    @jaxtyped(typechecker=beartype)
    def down(
        self, x: Float[Array, " C H W"], t_emb: Float[Array, " Temb"]
    ) -> tuple[Float[Array, " C_out H_out W_out"], Float[Array, " C_skip H W"]]:
        """Forward data through the block, with downsampling.

        Pass `x` through a res block which increases channels.
        Then, downsample by half with a learned layer.

        Returns:
            The downsampled features and the tensor `up` expects as its skip:
            the block input (legacy) or the res block output (standard).
        """
        x_rb = self.down_res_block(x, t_emb)
        skip = x if self.skip_mode == 0 else x_rb
        return self.down_conv(jax.nn.silu(x_rb)), skip

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

        h <- h + m_k * W_k mean_{m_k}(h),    mean_m(h) = sum_p m(p) h(p) / sum_p m(p)

    i.e. the features inside a region are pooled to a single vector, projected
    by a zero-initialised 1x1 conv ``W_k`` and broadcast back into that region
    only.  Two irises are then painted from one shared iris feature -- "the
    eyes are one entity" stated structurally, at the resolution where hue is
    decided -- and the same holds for hair colour across strands.  It is
    masked attention with a single fixed query, so it is cheap, and the zero
    init makes it the identity at initialisation.  All-zero masks (the null
    token, or an undetected face) contribute nothing.
    """

    proj: list[eqx.nn.Conv2d]

    def __init__(self, channels: int, n_regions: int, key: PRNGKeyArray):
        """One zero-initialised ``channels -> channels`` 1x1 conv per region."""
        self.proj = []
        for sk in jax.random.split(key, n_regions):
            conv = eqx.nn.Conv2d(channels, channels, kernel_size=1, key=sk)
            self.proj.append(jax.tree.map(jax.numpy.zeros_like, conv))

    @jaxtyped(typechecker=beartype)
    def __call__(
        self, h: Float[Array, " C H W"], masks: Float[Array, " K H W"]
    ) -> Float[Array, " C H W"]:
        """Add the broadcast pooled feature of every region to ``h``."""
        for k, conv in enumerate(self.proj):
            m = masks[k]
            pooled = reduce(h * m, "c h w -> c 1 1", "sum") / (m.sum() + 1e-6)
            h = h + m * conv(jax.numpy.broadcast_to(pooled, h.shape))
        return h


class UNet(eqx.Module):
    """Denoising UNet.

    Comprises of a fixed number of `UBlocks` applied recursively.
    Input channels are expanded using a preliminary convolution.
    Output is collapsed to the same shape as input via pointwise convolution.

    Optional mid-block (``mid_block > 0``):
        ``ResBlock -> [AttnBlock] -> ResBlock`` at the bottleneck, the standard
        DDPM/ADM mid-block.  Without it the bottleneck is the bare output of
        the last down-convolution.  ``mid_attention`` toggles the attention
        layer so the two can be ablated separately: the mid-block adds depth
        and parameters whether or not it attends.

    Optional auxiliary head (``n_aux_classes > 0``):
        A pointwise MLP reading the *encoder* feature map after ``aux_level``
        downsamplings and emitting ``n_aux_classes`` logit maps at that
        resolution.  It is used only by ``forward_aux`` during training, to
        supervise the encoder with semantic masks of the clean image (face,
        eyes, ...) while the input is the noisy ``x_t``.  This is the U-Net
        analogue of REPA (Yu et al. 2025): the hidden state of a noisy input
        is pushed toward semantics of the clean target, which the denoising
        objective alone only discovers slowly.  The head is tapped on the
        encoder rather than the decoder because decoder features sit next to
        the reconstruction and would carry the masks for free; the point is
        to shape the representation, not to read it out.  ``__call__`` never
        touches the head, so sampling is unchanged.
    """

    in_conv: eqx.nn.Conv2d
    blocks: list[UBlock]
    out_conv: eqx.nn.Conv2d
    time_mlp: eqx.nn.Sequential
    mid_res1: ResBlock | None
    mid_attn: AttnBlock | None
    mid_res2: ResBlock | None
    global_mlp: eqx.nn.Sequential | None
    region_pools: dict[int, RegionPool]
    cond_channels: int = eqx.field(static=True)
    aux_head: eqx.nn.Sequential | None
    time_embedding_dim: int
    aux_level: int = eqx.field(static=True)
    time_scale: float = eqx.field(static=True)

    def __init__(
        self,
        base_channels: int,
        time_embedding_dim: int,
        key: PRNGKeyArray,
        n_blocks: int = 4,
        in_channels: int = 3,
        mid_block: int = 0,
        mid_attention: int = 0,
        n_aux_classes: int = 0,
        aux_level: int = 2,
        time_scale: float = 1000.0,
        skip_mode: int = 0,
        out_channels: int = 0,
        cond_channels: int = 0,
        global_code: int = 0,
        region_pool: int = 0,
    ):
        """Initialize a denoising U-Net with multiple blocks conditioned on time.

        Args:
            base_channels: The number of input channels to the first down block.
            in_channels: The number of input channels to this UNet module.
            n_blocks: The number of blocks
            key: PRNG key for random initialization
            time_embedding_dim: The size of 1d time embedding.
            mid_block: ``1`` adds the ``ResBlock -> [AttnBlock] -> ResBlock``
                mid-block at the bottleneck; ``0`` leaves it out.
            skip_mode: ``0`` legacy skip connections (default), ``1`` standard
                U-Net skips; see ``UBlock``.
            out_channels: Output channels; ``0`` means the same as
                ``in_channels``.  A conditioned model takes the image plus
                conditioning channels in and emits only the image.
            global_code: ``1`` adds a global code to the decoder's conditioning:
                ``g = W2 silu(W1 LN(mean_hw z) + b1) + b2`` with ``z`` the
                bottleneck, ``W2 = b2 = 0`` at init, and ``t_dec = t_emb + g``
                for the mid-block and every up block (the ADM / DiT recipe for
                class embeddings).  Pooling over space keeps the channel
                identity and discards position, so every decoder layer sees
                one shared summary of *what* is in the image -- the
                StyleGAN-style route to globally consistent attributes such as
                matching iris colour -- while the encoder is unchanged.  Zero
                init makes the model identical to ``global_code=0`` at
                initialisation.  ``0`` leaves it out.
            region_pool: ``1`` adds ``RegionPool`` layers after the up blocks
                that produce 16x16 and 32x32 features, using the layout masks
                found in the conditioning channels of the input (regions:
                face, eyes, mouth, and hair/background = 1 - face).  Needs
                ``cond_channels = 3``.  Identity at init.  ``0`` leaves it out.
            cond_channels: Number of layout-mask channels ``ImageFM``
                concatenates to the image (plus one indicator channel).  The
                U-Net itself only sees ``in_channels``; this is accepted so it
                can live in the shared ``hparams`` dict, and is checked against
                ``in_channels - out_channels``.
            mid_attention: ``1`` puts the attention layer inside the mid-block.
                Ints rather than bools so they can live in ``hparams``.
            n_aux_classes: Number of semantic mask channels predicted by the
                auxiliary head; ``0`` disables the head (see class docstring).
            aux_level: Number of downsamplings between the input and the
                feature map the auxiliary head reads.  ``n_blocks`` selects the
                bottleneck.
            time_scale: Multiplier taking ``t`` from ``[0, 1]`` into the range
                the sinusoidal frequencies were designed for (DDPM's 1000 steps).
                Not an ``hparam``: ``ImageFM.load`` coerces those with ``int()``.
        """
        sk1, sk2, sk3, sk4, key = jax.random.split(key, num=5)
        n_out = out_channels or in_channels
        if cond_channels and in_channels != n_out + cond_channels + 1:
            raise ValueError(
                f"in_channels={in_channels} must equal out_channels + cond_channels + 1"
                f" = {n_out + cond_channels + 1}"
            )
        self.in_conv = eqx.nn.Conv2d(in_channels, base_channels, 3, padding=1, key=sk1)
        self.out_conv = eqx.nn.Conv2d(
            base_channels, out_channels or in_channels, 1, key=sk2
        )
        self.time_mlp = eqx.nn.Sequential(
            [
                eqx.nn.Linear(time_embedding_dim, time_embedding_dim, key=sk3),
                eqx.nn.Lambda(jax.nn.silu),
                eqx.nn.Linear(time_embedding_dim, time_embedding_dim, key=sk4),
            ]
        )
        self.blocks = []
        self.time_embedding_dim = time_embedding_dim
        self.time_scale = time_scale
        for i in range(n_blocks):
            sk, key = jax.random.split(key)
            self.blocks.append(
                UBlock(base_channels * 2**i, time_embedding_dim, sk, skip_mode)
            )

        self.mid_res1 = self.mid_attn = self.mid_res2 = None
        if mid_block:
            sk_m1, sk_m2, sk_m3, key = jax.random.split(key, num=4)
            c_mid = base_channels * 2**n_blocks
            self.mid_res1 = ResBlock(c_mid, c_mid, time_embedding_dim, key=sk_m1)
            self.mid_attn = AttnBlock(c_mid, key=sk_m2) if mid_attention else None
            self.mid_res2 = ResBlock(c_mid, c_mid, time_embedding_dim, key=sk_m3)

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
                    sk_r, key = jax.random.split(key)
                    c_level = base_channels * 2**level
                    self.region_pools[level] = RegionPool(c_level, 4, sk_r)

        self.global_mlp = None
        if global_code:
            sk_g1, sk_g2, key = jax.random.split(key, num=3)
            c_mid = base_channels * 2**n_blocks
            out = eqx.nn.Linear(time_embedding_dim, time_embedding_dim, key=sk_g2)
            out = eqx.tree_at(
                lambda m: (m.weight, m.bias),
                out,
                jax.tree.map(jax.numpy.zeros_like, (out.weight, out.bias)),
            )
            self.global_mlp = eqx.nn.Sequential(
                [
                    eqx.nn.LayerNorm(c_mid),  # pooled residual stream: fix its scale
                    eqx.nn.Linear(c_mid, time_embedding_dim, key=sk_g1),
                    eqx.nn.Lambda(jax.nn.silu),
                    out,
                ]
            )

        if not 0 <= aux_level <= n_blocks:
            raise ValueError(f"aux_level={aux_level} must be in [0, {n_blocks}]")
        self.aux_level = aux_level
        self.aux_head = None
        if n_aux_classes > 0:
            sk5, sk6 = jax.random.split(key)
            c_aux = base_channels * 2**aux_level
            self.aux_head = eqx.nn.Sequential(
                [
                    eqx.nn.Conv2d(c_aux, c_aux, 1, key=sk5),
                    eqx.nn.Lambda(jax.nn.silu),
                    eqx.nn.Conv2d(c_aux, n_aux_classes, 1, key=sk6),
                ]
            )

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
    ) -> tuple[Float[Array, " C H W"], list[Array]]:
        """Run the U-Net; return the channels-first output and encoder features.

        The feature list holds the input to each down block followed by the
        bottleneck, i.e. entry ``i`` has been downsampled ``i`` times.  It is
        what the auxiliary head reads and is independent of ``skip_mode``;
        the decoder's skips are whatever each block's ``down`` hands back.
        """
        t_embed = self.time_mlp(self.sinusoidal_embeddings(t))
        x_c = rearrange(x, "h w c -> c h w")
        x_in = self.in_conv(x_c)
        features: list[Array] = []
        skip_values: list[Array] = []
        for i in range(len(self.blocks)):
            features.append(x_in)
            x_in, skip = self.blocks[i].down(x_in, t_embed)
            skip_values.append(skip)
        x_out = x_in
        t_dec = t_embed
        if self.global_mlp is not None:
            # One vector for the whole image, added to the decoder's conditioning.
            t_dec = t_embed + self.global_mlp(reduce(x_in, "c h w -> c", "mean"))
        if self.mid_res1 is not None and self.mid_res2 is not None:
            x_out = self.mid_res1(x_out, t_dec)
            if self.mid_attn is not None:
                x_out = self.mid_attn(x_out)
            x_out = self.mid_res2(x_out, t_dec)
        regions = None
        if self.region_pools:
            # Conditioning channels sit after the image: [x_t, masks, indicator].
            m = x_c[3 : 3 + self.cond_channels]  # (3, H, W): face, eyes, mouth
            regions = jax.numpy.concatenate([m, 1 - m[:1]], axis=0)  # + hair/bg
        for i in range(len(self.blocks) - 1, -1, -1):
            x_out = self.blocks[i].up(x_out, skip_values[i], t_dec)
            if regions is not None and i in self.region_pools:
                h = x_out.shape[1]
                m_level = reduce(regions, "k (h a) (w b) -> k h w", "mean", h=h, w=h)
                x_out = self.region_pools[i](x_out, m_level)
        return self.out_conv(x_out), [*features, x_in]

    @jaxtyped(typechecker=beartype)
    def forward_aux(
        self, x: Float[Array, " H W C_in"], t: Float[Array, ""]
    ) -> tuple[Float[Array, " H W C_out"], Float[Array, " h w K"]]:
        """Denoise `x` at time `t` and also emit the auxiliary mask logits.

        Training-only companion to ``__call__``; requires ``n_aux_classes > 0``.
        The logits are at ``1 / 2**aux_level`` of the input resolution.
        """
        if self.aux_head is None:
            raise ValueError("forward_aux needs a UNet built with n_aux_classes > 0")
        x_out, feats = self._forward(x, t)
        logits = self.aux_head(feats[self.aux_level])
        return rearrange(x_out, "c h w -> h w c"), rearrange(logits, "k h w -> h w k")

    @jaxtyped(typechecker=beartype)
    def __call__(
        self, x: Float[Array, " H W C_in"], t: Float[Array, ""]
    ) -> Float[Array, " H W C_out"]:
        """Forward an image to denoise through the U-net.

        Denoise an image `x`, at timestep `t`.  ``C_in`` and ``C_out`` agree
        for an unconditioned model; a conditioned one takes extra layout
        channels in (``C_in = C_out + cond_channels + 1``) and emits the image.
        """
        x_out, _feats = self._forward(x, t)
        #
        # It appears that a reasonable number of papers dealing with
        # pixel-space generative modelling omit the final bounded activation
        #
        # Presumably, this may be because of the associated vanishing gradients.
        # In the context of Flow Matching models,
        # the regression target is the velocity field which is unbounded.
        # Consequently, the omission of final activation can be justified.
        # However, the velocity field can be parametrized in terms of the denoised image
        # as in (Li et. al, 2025). An advantage of this approach,
        # other than it's synergy with the objectives driving the original Unet design,
        # is the possibility of including an auxiliary perceptual loss
        # to enhance the learning signal.
        # Notwithstanding, owing to the bounded input, potential vanishing gradients
        # and the fact that the dervied velocity field is still unbounded,
        # The final activation is omitted here.
        #
        return rearrange(x_out, "c h w -> h w c")
