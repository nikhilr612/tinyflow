"""Grain wrapper for huggan/anime-faces dataset."""

import random
import time
from dataclasses import dataclass
from pathlib import Path

import grain
import grain.transforms as gt
import numpy as np
from PIL import Image


@dataclass
class PreprocessingConfig:
    """Configuration for offline preprocessing (loading, caching)."""


@dataclass
class AugmentationConfig:
    """Configuration for online augmentation via Grain thread workers.

    Attributes:
        p_flip: Probability of random horizontal flip per image.
        brightness_max: Maximum absolute brightness shift (±).
        contrast_range: (low, high) multiplier range for contrast.
        saturation_range: (low, high) multiplier range for saturation.
    """

    p_flip: float = 0.5
    brightness_max: float = 0.2
    contrast_range: tuple[float, float] = (0.8, 1.2)
    saturation_range: tuple[float, float] = (0.8, 1.2)


def to_uint8(x) -> np.ndarray:
    """Denormalize images from ``[-1, 1]`` back to ``uint8``, the inverse of loading.

    Clipping is mandatory: the model is free to predict outside ``[-1, 1]``
    (``models/unet.py`` omits the final bounded activation), and a bare ``uint8``
    cast would wrap those values around to the opposite end of the range.

    Args:
        x: Image array in ``[-1, 1]``, any shape.

    Returns:
        The same shape as ``uint8`` in ``[0, 255]``.
    """
    return np.clip((np.asarray(x) + 1) * 127.5, 0, 255).astype(np.uint8)


def load_all_pngs(data_dir: str) -> np.ndarray:
    """Load all PNG files from ``data_dir / images`` into a numpy array.

    Args:
        data_dir: Path to the dataset directory containing ``images/``.

    Returns:
        Array of shape ``(N, 64, 64, 3)`` in [-1, 1].
    """
    img_dir = Path(data_dir) / "images"
    paths = sorted(img_dir.rglob("*.png"))
    n = len(paths)
    result = np.empty((n, 64, 64, 3), dtype=np.float32)
    for i, p in enumerate(paths):
        img = Image.open(p)
        arr = np.array(img, dtype=np.float32)
        result[i] = 2 * (arr / 255) - 1
        if (i + 1) % 20000 == 0:
            print(f"  loaded {i + 1}/{n}")
    return result


def preprocess_all(
    data_dir: str,
    cache_path: str | None = "./.preprocessed/anime_faces.npy",
) -> np.ndarray:
    """Preprocess entire dataset into a single numpy array (no flips).

    Random flips are now applied on-the-fly by :class:`RandomHorizontalFlip`
    in the Grain pipeline.  This function only loads, rescales, and caches.

    Args:
        data_dir: Path to the dataset directory containing ``images/``.
        cache_path: If set, save/load from this path for fast subsequent runs.

    Returns:
        Array of shape ``(N, 64, 64, 3)`` in [-1, 1].
    """
    if cache_path is not None:
        cache = Path(cache_path)
        if cache.exists():
            print(f"Loading preprocessed dataset from {cache_path}")
            return np.load(str(cache))

    print("Loading PNGs from disk...")
    t0 = time.perf_counter()
    raw = load_all_pngs(data_dir)
    print(f"  loaded {raw.shape[0]} images in {time.perf_counter() - t0:.1f}s")

    if cache_path is not None:
        cache = Path(cache_path)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(cache), raw)
        print(f"Cached to {cache_path}")

    return raw


class RandomHorizontalFlip(gt.RandomMap):
    """Randomly flip images horizontally with probability ``p_flip``."""

    def __init__(self, p_flip: float = 0.5):
        """Store the per-image flip probability."""
        self.p_flip = p_flip

    def random_map(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Mirror ``image`` about its width axis with probability ``p_flip``."""
        if rng.random() < self.p_flip:
            return np.flip(image, axis=1)
        return image


class ColorJitter(gt.RandomMap):
    """Random brightness, contrast, and saturation jitter."""

    def __init__(
        self,
        brightness_max: float = 0.2,
        contrast_range: tuple[float, float] = (0.8, 1.2),
        saturation_range: tuple[float, float] = (0.8, 1.2),
    ):
        """Store the brightness shift bound and the contrast/saturation ranges."""
        self.brightness_max = brightness_max
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range

    def random_map(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Apply one random brightness, contrast and saturation draw to ``image``."""
        b = rng.uniform(-self.brightness_max, self.brightness_max)
        c = rng.uniform(*self.contrast_range)
        s = rng.uniform(*self.saturation_range)

        image = image + b
        mean = image.mean()
        image = mean + c * (image - mean)
        gray = image.mean(axis=-1, keepdims=True)
        image = gray + s * (image - gray)

        return np.clip(image, -1.0, 1.0)


def wrap_dataset(
    array_or_dir: str | np.ndarray,
    pconfig: PreprocessingConfig = PreprocessingConfig(),
    aug_config: AugmentationConfig = AugmentationConfig(),
    seed: int = 42,
    batch_size: int = 32,
):
    """Wrap preprocessed image data with Grain for fast batch iteration.

    Grain derives ``shuffle`` and ``random_map`` randomness from the *element
    index*, not from an iteration counter, so re-iterating a finite dataset would
    replay the same order and the same augmentation draws every time.  The stream
    is therefore made infinite with ``repeat()``: indices keep climbing, and
    because ``shuffle`` is epoch-aware (it derives the epoch from the index) each
    pass gets a fresh permutation, fresh flips and fresh jitter.

    How long to train is the caller's business, so the dataset itself has no
    notion of epoch count; it just never runs out.

    Args:
        array_or_dir: Either a path to a dataset directory (str) or a preprocessed
            numpy array. If a string, ``preprocess_all`` is called internally.
        pconfig: Preprocessing configuration (loading / caching).
        aug_config: Online augmentation configuration.
        seed: Seed for shuffling and random augmentations.
        batch_size: Batch size to use when fetching.

    Returns:
        ``(dataset, batches_per_epoch)``: an endless grain ``IterDataset`` of
        batches, and the number of batches making up one pass over the data.
    """
    if isinstance(array_or_dir, str):
        arr = preprocess_all(array_or_dir)
    else:
        arr = array_or_dir

    # Independent streams: reusing one integer for both would tie the batch
    # composition to the augmentation draws for an element.
    prng = random.Random(seed)
    shuffle_seed = prng.getrandbits(32)
    augment_seed = prng.getrandbits(32)
    dataset: grain.MapDataset = (
        grain.MapDataset.source(arr)
        .seed(augment_seed)
        .shuffle(shuffle_seed)
        .repeat()
        .random_map(RandomHorizontalFlip(p_flip=aug_config.p_flip))
        .random_map(
            ColorJitter(
                brightness_max=aug_config.brightness_max,
                contrast_range=aug_config.contrast_range,
                saturation_range=aug_config.saturation_range,
            )
        )
        .batch(batch_size=batch_size, batch_fn=lambda x_ls: np.stack(x_ls, axis=0))
    )
    # Ceiling division: the stream is continuous, so batches straddle epoch
    # boundaries rather than a short batch ending each pass.
    batches_per_epoch = -(-len(arr) // batch_size)
    return (
        dataset.to_iter_dataset(
            read_options=grain.ReadOptions(num_threads=4, prefetch_buffer_size=4),
        ),
        batches_per_epoch,
    )
