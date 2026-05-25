"""Implement a denoising U-net for image generation via flow matching."""

import beartype
import equinox as eqx
import jax
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
    n_groups: int = 8
    gn_eps: float = 1e-5

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
        sk1, sk2, sk3, sk4 = jax.random.split(key, num=4)
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

        self.ada_proj = eqx.nn.Linear(
            in_features=time_embedding_dim,
            out_features=out_channels * 2,
            key=sk3,
        )

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

    def _group_norm(self, x: Float[Array, " H W C"]) -> Float[Array, " H W C"]:
        """Normalize over channel groups."""
        grouped = rearrange(x, "h w (G g) -> G g h w", G=self.n_groups)
        mean = reduce(grouped, "G g h w -> G g 1 1", "mean")
        shifted = grouped - mean
        var = reduce(shifted**2, "G g h w -> G g 1 1", "mean")
        scaled = shifted * (1.0 / jax.lax.sqrt(var + self.gn_eps))
        return rearrange(scaled, "G g h w -> h w (G g)")

    @jaxtyped(typechecker=beartype)
    def __call__(
        self, x: Float[Array, " H W C"], t_emb: Float[Array, " Temb"]
    ) -> Float[Array, " H W C_out"]:
        """Forward `x` through this block, with time embeddings `t`."""
        x_norm = self._group_norm(x)
        h1 = self.conv1(jax.nn.silu(x_norm))
        gamma, beta = rearrange(self.ada_proj(t_emb), "(two p) -> two () () p", two=2)
        h2 = self._group_norm(h1) * gamma + beta
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
    Here, the skips are from pre-decoder to post-upsample.
    The fix is relatively straightforward;
    """

    # TODO(n): Check if current skip connection pattern is detrimental to performance.
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
        self, x: Float[Array, " H W C"], t_emb: Float[Array, " Temb"]
    ) -> Float[Array, " H_out W_out C_out"]:
        """Forward data through the block, with downsampling.

        Pass `x` through a res block which increases channels.
        Then, downsample by half with a learned layer.
        """
        x_rb = self.down_res_block(x, t_emb)
        return self.down_conv(jax.nn.silu(x_rb))

    @jaxtyped(typechecker=beartype)
    def up(
        self,
        x: Float[Array, " H W C_in"],  # C_in should be 2 * C_skip
        x_skip: Float[Array, " H_out W_out C_skip"],
        t_emb: Float[Array, " Temb"],
    ) -> Float[Array, " H_out W_out C_out"]:
        """Forward data through the block, with upsampling.

        Upsample `x` using bilinear interpolation.
        Then, forward through a res block with concatenated `x_skip`.
        """
        h, w, c_in = x.shape
        x_up = jax.image.resize(
            x,
            (h * 2, w * 2, c_in),
            method=jax.image.ResizeMethod.LINEAR,
        )
        x_conv = jax.nn.silu(self.up_conv(x_up))
        concatenated_skip, _packing = pack([x_conv, x_skip], "h w *")
        return self.up_res_block(concatenated_skip, t_emb)


class UNet(eqx.Module):
    """Denoising UNet.

    Comprises of a fixed number of `UBlocks` applied recursively.
    Input channels are expanded using a preliminary convolution.
    Output is collapsed to the same shape as input via pointwise convolution.
    """

    in_conv: eqx.nn.Conv2d
    blocks: list[UBlock]
    out_conv: eqx.nn.Conv2d
    time_embedding_dim: int

    def __init__(
        self,
        base_channels: int,
        time_embedding_dim: int,
        key: PRNGKeyArray,
        n_blocks: int = 4,
        in_channels: int = 3,
    ):
        """Initialize a denoising U-Net with multiple blocks conditioned on time.

        Args:
            base_channels: The number of input channels to the first down block.
            in_channels: The number of input channels to this UNet module.
            n_blocks: The number of blocks
            key: PRNG key for random initialization
            time_embedding_dim: The size of 1d time embedding.
        """
        sk1, sk2, key = jax.random.split(key, num=3)
        self.in_conv = eqx.nn.Conv2d(in_channels, base_channels, 3, padding=1, key=sk1)
        self.out_conv = eqx.nn.Conv2d(base_channels, in_channels, 1, key=sk2)
        self.blocks = []
        self.time_embedding_dim = time_embedding_dim
        for i in range(n_blocks):
            sk, key = jax.random.split(key)
            self.blocks.append(UBlock(base_channels * 2**i, time_embedding_dim, sk))

    def sinusoidal_embeddings(self, t: Float[Array, ""]) -> Float[Array, " Tembed"]:
        """Sinusoidal embeddings for a given scalar time `t`."""
        freqs = t * jax.numpy.geomspace(1, 1 / 10_000, num=self.time_embedding_dim // 2)
        return jax.numpy.concat([jax.numpy.sin(freqs), jax.numpy.cos(freqs)])

    @jaxtyped(typechecker=beartype)
    def __call__(
        self, x: Float[Array, " H W C"], t: Float[Array, ""]
    ) -> Float[Array, " H W C"]:
        """Forward an image to denoise through the U-net.

        Denoise an image `x`, at timestep `t`.
        """
        t_embed = self.sinusoidal_embeddings(t)
        x_in = self.in_conv(x)
        skip_values: list[Array] = []
        for i in range(len(self.blocks)):
            skip_values.append(x_in)
            x_in = self.blocks[i].down(x_in, t_embed)
        x_out = x_in
        for i in range(len(self.blocks) - 1, -1, -1):
            x_out = self.blocks[i].up(x_out, skip_values[i], t_embed)
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
        return self.out_conv(x_out)
