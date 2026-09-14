"""FID evaluation for generative image models using fidax (pure JAX)."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from fidax.fid import FrechetInceptionDistance
from tqdm import tqdm


def _to_inception_input(images) -> jax.Array:
    """Map a ``(B, H, W, 3)`` batch in ``[-1, 1]`` to what ``fidax`` expects.

    ``FrechetInceptionDistance.update`` documents its input range as ``[0, 1]``
    and rescales to ``[-1, 1]`` internally, so feeding it ``[-1, 1]`` directly
    would give Inception ``[-3, 1]``.  Generated images are clipped first: the
    model has no bounded output activation, and the metric should score image
    content rather than overshoot.
    """
    images = jnp.clip(jnp.asarray(images), -1.0, 1.0)
    images = (images + 1.0) * 0.5
    return jax.image.resize(images, (images.shape[0], 299, 299, 3), "bilinear")


def compute_real_stats(
    real_images: np.ndarray,
    batch_size: int = 128,
    cache_path: str | None = "./.preprocessed/real_stats.npz",
) -> dict:
    """Compute and cache Inception feature statistics for real images.

    Args:
        real_images: ``(N, H, W, 3)`` array in [-1, 1].
        batch_size: Batch size for feature extraction.
        cache_path: Path to save/load cached stats.  ``None`` disables caching.
            The cache is keyed only by path: delete it after changing the
            preprocessing, or stale statistics will be reused silently.

    Returns:
        Dict with ``'mu'`` (2048,) and ``'sigma'`` (2048, 2048).
    """
    if cache_path is not None:
        cache = Path(cache_path)
        if cache.exists():
            print(f"Loading real stats from {cache_path}")
            data = np.load(str(cache))
            return {"mu": data["mu"], "sigma": data["sigma"]}

    print("Computing real dataset Inception stats...")
    fid = FrechetInceptionDistance()
    n = len(real_images)
    for i in tqdm(range(0, n, batch_size), desc="Real Inception feats"):
        fid.update(_to_inception_input(real_images[i : i + batch_size]), real=True)

    mu, sigma = fid.get_real_stats()
    result = {"mu": np.asarray(mu), "sigma": np.asarray(sigma)}

    if cache_path is not None:
        cache = Path(cache_path)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(cache), mu=result["mu"], sigma=result["sigma"])
        print(f"Cached real stats to {cache_path}")

    return result


def evaluate_fid(
    model,
    real_stats: dict,
    key: jax.Array,
    n_samples: int = 5000,
    batch_size: int = 128,
    image_size: int = 64,
) -> float:
    """Generate images from model and compute FID against real stats.

    Uses ``FrechetInceptionDistance._fid_from_stats`` directly to avoid
    NNX pytree issues with passing ``real_stats`` into the constructor.

    Args:
        model: ImageFM model with a ``.generate(noise)`` method.
        real_stats: Dict with ``'mu'`` and ``'sigma'`` from :func:`compute_real_stats`.
        key: JAX PRNG key.
        n_samples: Number of images to generate.
        batch_size: Batch size for generation and feature extraction.
        image_size: Side of the square images the model generates.

    Returns:
        FID score (lower is better).
    """
    mu_real = jnp.asarray(real_stats["mu"])
    sigma_real = jnp.asarray(real_stats["sigma"])

    fid = FrechetInceptionDistance()

    for i in tqdm(range(0, n_samples, batch_size), desc="FID"):
        bs = min(batch_size, n_samples - i)
        key, sample_key = jax.random.split(key)
        noise = jax.random.normal(sample_key, (bs, image_size, image_size, 3))
        fid.update(_to_inception_input(model.generate(noise)), real=False)

    mu_fake, sigma_fake = fid.get_fake_stats()
    return float(
        FrechetInceptionDistance._fid_from_stats(
            mu_fake, sigma_fake, mu_real, sigma_real
        )
    )
