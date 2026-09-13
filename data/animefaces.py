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


def load_masks(path: str = "./.preprocessed/anime_faces_masks.npy") -> np.ndarray:
    """Load cached semantic masks aligned index-for-index with ``preprocess_all``.

    The masks are produced offline from ``hysts/anime-face-detector`` landmarks
    (face hull, eyes, mouth) and stored as ``uint8`` in ``[0, 255]``; see the
    README section on auxiliary supervision for the generating script.  They
    are returned as-is and rescaled to ``[0, 1]`` inside the pipeline.

    Args:
        path: Location of the cached ``(N, 64, 64, K)`` ``uint8`` array.

    Returns:
        The ``uint8`` mask array.
    """
    return np.load(path)


def load_landmark_scores(
    path: str = "./.preprocessed/anime_faces_landmark_scores.npy",
) -> np.ndarray:
    """Load per-landmark detector confidences aligned with ``preprocess_all``.

    Shape ``(N, 28)``, one score per ``anime-face-detector`` keypoint; the
    same offline script that makes the masks writes them.

    Args:
        path: Location of the cached ``(N, 28)`` float array.

    Returns:
        The score array.
    """
    return np.load(path)


def curated_indices(scores: np.ndarray, min_mean_score: float) -> np.ndarray:
    """Indices of images whose mean landmark confidence is at least the threshold.

    The threshold is a content filter, not a quality bar: below a mean score
    of about 0.3 the anime-faces set is dominated by non-faces (torsos,
    clothing, hands, duplicated crops -- 317 images, 1.5%), while the
    0.3-0.65 band is mostly legitimate but hard faces (closed eyes, glasses,
    masks, mascots) that a generator should keep.  See
    ``runs/ablation/curation_*.png`` for the bands.

    Args:
        scores: ``(N, 28)`` landmark confidences from ``load_landmark_scores``.
        min_mean_score: Keep images with ``scores.mean(1) >= min_mean_score``;
            ``0`` keeps everything.

    Returns:
        Sorted integer indices of the kept images.
    """
    return np.flatnonzero(scores.mean(axis=1) >= min_mean_score)


Element = np.ndarray | tuple[np.ndarray, np.ndarray]
"""A pipeline element: an image, or an ``(image, mask)`` pair."""


class RandomHorizontalFlip(gt.RandomMap):
    """Randomly flip images horizontally with probability ``p_flip``.

    An ``(image, mask)`` pair is flipped as one so the mask stays aligned.
    """

    def __init__(self, p_flip: float = 0.5):
        """Store the per-image flip probability."""
        self.p_flip = p_flip

    def random_map(self, element: Element, rng: np.random.Generator) -> Element:
        """Mirror ``element`` about its width axis with probability ``p_flip``."""
        if rng.random() >= self.p_flip:
            return element
        if isinstance(element, tuple):
            image, mask = element
            return np.flip(image, axis=1), np.flip(mask, axis=1)
        return np.flip(element, axis=1)


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

    def random_map(self, element: Element, rng: np.random.Generator) -> Element:
        """Apply one random brightness, contrast and saturation draw to the image.

        A mask riding along in an ``(image, mask)`` pair is passed through.
        """
        if isinstance(element, tuple):
            image, mask = element
            return self._jitter(image, rng), mask
        return self._jitter(element, rng)

    def _jitter(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
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
    masks: np.ndarray | None = None,
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
        masks: Optional ``(N, H, W, K)`` ``uint8`` masks (see ``load_masks``).
            When given, the stream yields ``(images, masks)`` batch pairs with
            the masks in ``[0, 1]`` and augmented in lockstep with the images.
        pconfig: Preprocessing configuration (loading / caching).
        aug_config: Online augmentation configuration.
        seed: Seed for shuffling and random augmentations.
        batch_size: Batch size to use when fetching.

    Returns:
        ``(dataset, batches_per_epoch)``: an endless grain ``IterDataset`` of
        batches (or batch pairs), and the number of batches making up one pass
        over the data.
    """
    if isinstance(array_or_dir, str):
        arr = preprocess_all(array_or_dir)
    else:
        arr = array_or_dir

    if masks is None:
        # ndarray satisfies Grain's RandomAccessDataSource protocol (__len__ and
        # __getitem__) but is not typed as one.
        source: grain.MapDataset = grain.MapDataset.source(arr)  # ty: ignore[invalid-argument-type]

        def batch_fn(x_ls):
            return np.stack(x_ls, axis=0)
    else:
        if len(masks) != len(arr):
            raise ValueError(f"{len(masks)} masks for {len(arr)} images")
        # Pair by index so that shuffling permutes images and masks together.
        source = grain.MapDataset.source(range(len(arr))).map(
            lambda i: (arr[i], masks[i].astype(np.float32) / 255.0)
        )

        def batch_fn(x_ls):
            images, mask_ls = zip(*x_ls)
            return np.stack(images, axis=0), np.stack(mask_ls, axis=0)

    # Independent streams: reusing one integer for both would tie the batch
    # composition to the augmentation draws for an element.
    prng = random.Random(seed)
    shuffle_seed = prng.getrandbits(32)
    augment_seed = prng.getrandbits(32)
    dataset = (
        source.seed(augment_seed)
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
        .batch(batch_size=batch_size, batch_fn=batch_fn)
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
