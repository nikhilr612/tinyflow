"""Grain wrapper for huggan/anime-faces dataset."""

import random
from dataclasses import dataclass

import datasets
import grain
import jax
import numpy as np
import PIL.Image as Pilimage
from jaxtyping import Array
from PIL import ImageEnhance, ImageOps
from PIL.Image import Image


@dataclass
class PreprocessingConfig:
    """Configuration parameters for preprocessing pipeline.

    Attributes:
        p_flip: Bernoulli trial probability for flipping an image.
    """

    p_flip: float = 0.5
    j_brightness: float = 0.1
    j_contrast: float = 0.1
    j_saturation: float = 0.1
    j_hue: float = 0.05


def image_to_mat(
    image: Image, prng: random.Random, pconfig: PreprocessingConfig
) -> Array:
    """Convert PIL image to Array, applying random flips and jitter.

    Args:
        image: The image to conert to Array
        prng: The `random.Random` instance used for randomness
        pconfig: Preprocessing parameters

    Returns:
        The image processed and converted into jax Array.
    """
    # sample factors
    brightness_factor = 1 + pconfig.j_brightness * prng.uniform(-1, 1)
    contrast_factor = 1 + pconfig.j_contrast * prng.uniform(-1, 1)
    saturation_factor = 1 + pconfig.j_saturation * prng.uniform(-1, 1)
    hue_factor = pconfig.j_hue * prng.uniform(-0.5, 0.5)

    h, s, v = image.convert("HSV").split()
    ha = np.array(h, dtype=np.int32)
    ha += np.int32(hue_factor * 255)
    ha = ha.astype(np.uint8)
    h1 = Pilimage.fromarray(ha, mode="L")

    coloured = Pilimage.merge("HSV", (h1, s, v)).convert("RGB")

    # apply jitter
    brigthened = ImageEnhance.Brightness(coloured).enhance(brightness_factor)
    constrasted = ImageEnhance.Contrast(brigthened).enhance(contrast_factor)
    jittered: Image = ImageEnhance.Color(constrasted).enhance(saturation_factor)

    if prng.uniform(0, 1) < pconfig.p_flip:
        jittered = ImageOps.mirror(jittered)

    jxa = jax.numpy.array(np.array(jittered, dtype=np.float32))
    return 2 * (jxa / 255) - 1


def wrap_dataset(
    local_dir: str,
    pconfig: PreprocessingConfig = PreprocessingConfig(),
    seed: int = 42,
    batch_size: int = 32,
) -> grain.MapDataset[Array]:
    """Load the dataset from `local_dir` and wrap it with Grain.

    Args:
        local_dir: Path to load dataset from.
            Use `huggan/animefaces` if not available locally.
        pconfig: Preprocessing configuration to use with dataset.
        seed: Seed for all random number generation and related operations
            including shuffling, flipping and jitter transformations.
        batch_size: Batch size to use when fetching.

    Returns:
        A grain `MapDataset` yielding batch arrays.
    """
    prng = random.Random(seed)
    shuffle_seed = prng.getrandbits(7)
    hf_dataset = datasets.load_dataset(local_dir, split="train", cache_dir="./.hfcache")
    images = hf_dataset["image"]
    return (
        grain.MapDataset.source(images)
        .shuffle(shuffle_seed)
        .map(lambda x: image_to_mat(x, prng, pconfig))
        .batch(
            batch_size=batch_size, batch_fn=lambda x_ls: jax.numpy.stack(x_ls, axis=0)
        )
    )
