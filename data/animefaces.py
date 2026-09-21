"""The anime-faces training set: images, their layout masks, and a Grain pipeline.

Offline, once: the 64x64 PNGs of ``huggan/anime-faces`` are loaded into one
array in ``[-1, 1]`` and the layout masks are rasterised from the shipped
landmarks (``data/landmarks.npz``); both are cached under ``.preprocessed/``.
Online, per batch, on Grain worker threads: a horizontal flip applied to the
image and its mask together.  That is the only augmentation: a photometric
jitter (brightness, contrast, saturation) was measured to cost 3 FID, since the
model then learns the jittered marginals and FID is against the clean data.

Every batch is an ``(images, masks)`` pair, ``(B, 64, 64, 3)`` each, images in
``[-1, 1]`` and masks in ``[0, 1]``.
"""

from __future__ import annotations

import random
from pathlib import Path

import grain
import grain.transforms as gt
import numpy as np
from PIL import Image

from data.layouts import LANDMARKS_PATH, rasterize

CACHE = Path(".preprocessed")


def to_uint8(x) -> np.ndarray:
    """``[-1, 1]`` floats to ``uint8`` pixels, clipped: the model may overshoot."""
    return np.clip((np.asarray(x) + 1) * 127.5, 0, 255).astype(np.uint8)


def load_images(data_dir: str = "./data/anime-faces") -> np.ndarray:
    """The PNGs in ``data_dir/images`` as ``(N, 64, 64, 3)`` in ``[-1, 1]``, cached.

    Sorted by path, which is the order the shipped landmarks follow.
    """
    cache = CACHE / "anime_faces.npy"
    if cache.exists():
        return np.load(cache)
    paths = sorted((Path(data_dir) / "images").glob("*.png"))
    arr = np.empty((len(paths), 64, 64, 3), np.float32)
    for i, p in enumerate(paths):
        arr[i] = np.asarray(Image.open(p).convert("RGB"), np.float32) / 127.5 - 1
    CACHE.mkdir(exist_ok=True)
    np.save(cache, arr)
    return arr


def load_masks() -> np.ndarray:
    """Layout masks ``(N, 64, 64, 3)`` in ``[0, 1]``, aligned with ``load_images``."""
    cache = CACHE / "anime_faces_masks.npy"
    if cache.exists():
        return np.load(cache)
    masks = rasterize(np.load(LANDMARKS_PATH)["landmarks"])
    CACHE.mkdir(exist_ok=True)
    np.save(cache, masks)
    return masks


def curated_indices() -> np.ndarray:
    """Indices of the images the detector recognised as faces.

    ``landmarks.npz`` carries a ``keep`` flag: mean landmark confidence at least
    0.3.  Below that the set is dominated by non-faces (torsos, hands,
    duplicated crops; 1.5 % of the images), which are excluded from training.
    """
    return np.flatnonzero(np.load(LANDMARKS_PATH)["keep"])


Pair = tuple[np.ndarray, np.ndarray]


class RandomHorizontalFlip(gt.RandomMap):
    """Mirror an ``(image, mask)`` pair about its width axis with probability ``p``."""

    def __init__(self, p: float = 0.5):
        """Store the flip probability."""
        self.p = p

    def random_map(self, element: Pair, rng: np.random.Generator) -> Pair:
        """Flip both arrays, or neither."""
        if rng.random() >= self.p:
            return element
        image, mask = element
        return np.flip(image, axis=1), np.flip(mask, axis=1)


def make_dataset(
    images: np.ndarray, masks: np.ndarray, batch_size: int = 128, seed: int = 42
) -> tuple[grain.IterDataset, int]:
    """An endless, shuffled, flip-augmented stream of ``(images, masks)`` batches.

    Grain derives shuffle order and augmentation draws from the element index,
    so the stream is made infinite with ``repeat()``: each pass over the data
    gets a fresh permutation and fresh draws.  Returns the dataset and the
    number of batches in one pass (the caller decides how many passes).
    """
    if len(masks) != len(images):
        raise ValueError(f"{len(masks)} masks for {len(images)} images")
    source = grain.MapDataset.source(range(len(images))).map(
        lambda i: (images[i], masks[i])
    )
    prng = random.Random(seed)  # independent seeds for shuffling and augmentation
    dataset = (
        source.seed(prng.getrandbits(32))
        .shuffle(prng.getrandbits(32))
        .repeat()
        .random_map(RandomHorizontalFlip())
        .batch(
            batch_size=batch_size, batch_fn=lambda ps: tuple(map(np.stack, zip(*ps)))
        )
    )
    batches_per_epoch = -(-len(images) // batch_size)
    return (
        dataset.to_iter_dataset(
            read_options=grain.ReadOptions(num_threads=4, prefetch_buffer_size=4)
        ),
        batches_per_epoch,
    )
