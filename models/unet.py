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
from einops import pack, rearrange, reduce
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped


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
        """Normalize over channel groups.

        Statistics are taken over the ``g`` channels of each group *and* space
        (``G g h w -> G 1 1 1``).  Keeping ``g`` in the output would give one
        mean per channel, i.e. InstanceNorm, which strips per-image brightness
        and colour cast -- content a generator needs to keep.
        """
        grouped = rearrange(x, "(G g) h w -> G g h w", G=self.n_groups)
        mean = reduce(grouped, "G g h w -> G 1 1 1", "mean")
        shifted = grouped - mean
        var = reduce(shifted**2, "G g h w -> G 1 1 1", "mean")
        scaled = shifted * (1.0 / jax.lax.sqrt(var + self.gn_eps))
        return rearrange(scaled, "G g h w -> (G g) h w")

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


class UNet(eqx.Module):
    """Denoising UNet.

    Comprises of a fixed number of `UBlocks` applied recursively.
    Input channels are expanded using a preliminary convolution.
    Output is collapsed to the same shape as input via pointwise convolution.

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
        n_aux_classes: int = 0,
        aux_level: int = 2,
        time_scale: float = 1000.0,
    ):
        """Initialize a denoising U-Net with multiple blocks conditioned on time.

        Args:
            base_channels: The number of input channels to the first down block.
            in_channels: The number of input channels to this UNet module.
            n_blocks: The number of blocks
            key: PRNG key for random initialization
            time_embedding_dim: The size of 1d time embedding.
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
        self.in_conv = eqx.nn.Conv2d(in_channels, base_channels, 3, padding=1, key=sk1)
        self.out_conv = eqx.nn.Conv2d(base_channels, in_channels, 1, key=sk2)
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
            self.blocks.append(UBlock(base_channels * 2**i, time_embedding_dim, sk))

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
        bottleneck, i.e. entry ``i`` has been downsampled ``i`` times.
        """
        t_embed = self.time_mlp(self.sinusoidal_embeddings(t))
        x_c = rearrange(x, "h w c -> c h w")
        x_in = self.in_conv(x_c)
        skip_values: list[Array] = []
        for i in range(len(self.blocks)):
            skip_values.append(x_in)
            x_in = self.blocks[i].down(x_in, t_embed)
        x_out = x_in
        for i in range(len(self.blocks) - 1, -1, -1):
            x_out = self.blocks[i].up(x_out, skip_values[i], t_embed)
        return self.out_conv(x_out), [*skip_values, x_in]

    @jaxtyped(typechecker=beartype)
    def forward_aux(
        self, x: Float[Array, " H W C"], t: Float[Array, ""]
    ) -> tuple[Float[Array, " H W C"], Float[Array, " h w K"]]:
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
        self, x: Float[Array, " H W C"], t: Float[Array, ""]
    ) -> Float[Array, " H W C"]:
        """Forward an image to denoise through the U-net.

        Denoise an image `x`, at timestep `t`.
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
