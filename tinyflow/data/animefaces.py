"""The anime-faces training set: images, their layout masks, and a Grain pipeline.

Offline, once: the 64x64 PNGs of ``huggan/anime-faces`` and their face
landmarks (``nikhilr01/anime-faces-landmarks``) are downloaded (``download``),
the PNGs loaded into one array in ``[-1, 1]`` and the layout masks rasterised
from the landmarks; both arrays are cached under ``.preprocessed/``.
Online, per batch, on Grain worker threads: a horizontal flip applied to the
image and its mask together.  That is the only augmentation: a photometric
jitter (brightness, contrast, saturation) was measured to cost 3 FID, since the
model then learns the jittered marginals and FID is against the clean data.

Every batch is an ``(images, masks)`` pair, ``(B, 64, 64, 3)`` each, images in
``[-1, 1]`` and masks in ``[0, 1]``.
"""

from __future__ import annotations

import random
import zipfile
from pathlib import Path

import grain
import grain.transforms as gt
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image

from tinyflow.data.layouts import rasterize

CACHE = Path(".preprocessed")
DATA_DIR = "./data/anime-faces"
# Pinned revisions of the two Hub datasets: (repository, commit).
IMAGES = ("huggan/anime-faces", "67ebcf8c69b45feb3883d695f04227078a6c9da9")
LANDMARKS = (
    "nikhilr01/anime-faces-landmarks",
    "c8a4d6c634ab019e65b574b164662f21c76c643d",
)


def download(data_dir: str = DATA_DIR) -> Path:
    """Fetch the PNGs to ``data_dir/images`` and their landmarks beside them, once."""
    root = Path(data_dir)
    if not (root / "images").exists():
        repo, revision = IMAGES
        archive = hf_hub_download(
            repo, "data.zip", repo_type="dataset", revision=revision
        )
        with zipfile.ZipFile(archive) as z:
            z.extractall(root)
        (root / "data").rename(root / "images")
    if not (root / "landmarks.npz").exists():
        repo, revision = LANDMARKS
        hf_hub_download(
            repo,
            "landmarks.npz",
            repo_type="dataset",
            revision=revision,
            local_dir=root,
        )
    return root


def load_images(data_dir: str = DATA_DIR) -> np.ndarray:
    """The PNGs in ``data_dir/images`` as ``(N, 64, 64, 3)`` in ``[-1, 1]``, cached.

    Sorted by path, which is the order the landmarks follow.
    """
    cache = CACHE / "anime_faces.npy"
    if cache.exists():
        return np.load(cache)
    paths = sorted((download(data_dir) / "images").glob("*.png"))
    arr = np.empty((len(paths), 64, 64, 3), np.float32)
    for i, p in enumerate(paths):
        arr[i] = np.asarray(Image.open(p).convert("RGB"), np.float32) / 127.5 - 1
    CACHE.mkdir(exist_ok=True)
    np.save(cache, arr)
    return arr


def load_masks(data_dir: str = DATA_DIR) -> np.ndarray:
    """Layout masks ``(N, 64, 64, 3)`` in ``[0, 1]``, aligned with ``load_images``."""
    cache = CACHE / "anime_faces_masks.npy"
    if cache.exists():
        return np.load(cache)
    masks = rasterize(np.load(download(data_dir) / "landmarks.npz")["landmarks"])
    CACHE.mkdir(exist_ok=True)
    np.save(cache, masks)
    return masks


def curated_indices(data_dir: str = DATA_DIR) -> np.ndarray:
    """Indices of the images the detector recognised as faces.

    ``landmarks.npz`` carries a ``keep`` flag: mean landmark confidence at least
    0.3.  Below that the set is dominated by non-faces (torsos, hands,
    duplicated crops; 1.5 % of the images), which are excluded from training.
    """
    return np.flatnonzero(np.load(download(data_dir) / "landmarks.npz")["keep"])


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
