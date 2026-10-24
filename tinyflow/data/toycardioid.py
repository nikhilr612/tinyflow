"""A toy dataset that generates points along a cardioid curve."""

import random
from dataclasses import dataclass
from typing import SupportsIndex

import grain
import jax.numpy as jnp
from jax import Array
from jax import random as jax_random


@dataclass
class CardioidDataParams:
    """Parameters for generating cardioid data.

    Attributes:
        n_images: Number of "images" in the dataset
        n_t: Number of uniform (x,y) samples from carioid per "image"
        r: Radial parameter of cardioid
        sigma_r: Variance of `r`, serves as epistemic uncertainty in parameter
        sigma_xy: Observation error, serves as aleotoric uncertainty
        seed: The random seed to use for generation
    """

    n_images: int = 32
    n_t: int = 64
    r: float = 2.0
    sigma_r: float = 0.05  # parameter noise
    sigma_xy: float = 0.025  # observation noise
    sigma_theta_deg: float = 4.0  # small random rotation
    seed: int = 42


def generate_cardioid_data(params: CardioidDataParams) -> Array:
    """Generates points along a cardioid curve with added noise.

    Args:
        params: Parameters for generating cardioid data.

    Returns:
        Array of shape (n_images, n_t, 2) containing the generated points.
    """
    key = jax_random.PRNGKey(params.seed)
    t = jnp.linspace(0, 2 * jnp.pi, params.n_t).reshape(1, -1)  # Shape (1, n_t)

    key, subkey = jax_random.split(key)
    r = params.r + params.sigma_r * jax_random.normal(
        subkey, shape=(params.n_images, 1)
    )

    key, subkey = jax_random.split(key)
    theta = (
        params.sigma_theta_deg
        * jnp.pi
        / 180.0
        * jax_random.normal(
            subkey,
            shape=(params.n_images, 1),
        )
    )

    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)

    # Parametric equations for a cardioid
    xp = 2 * r * jnp.sin(t) - r * jnp.sin(2 * t)
    yp = 2 * r * jnp.cos(t) - r * jnp.cos(2 * t)

    x = xp * cos_theta - yp * sin_theta
    y = xp * sin_theta + yp * cos_theta

    print(f"debug: x-variance:{x.var()}")
    print(f"debug: v-variance: {y.var()}")

    # Stack x and y coordinates
    points = jnp.stack([x, y], axis=2)

    noise = params.sigma_xy * jax_random.normal(key, shape=points.shape)
    return points + noise


class PointsImage(grain.sources.RandomAccessDataSource):
    """Grain data source for loading cardioid point images."""

    def __init__(self, fpath: str):
        """Initializes data source from the given file path."""
        loaded = jnp.load(fpath)
        print(f"Loaded data of shape: {loaded.shape}")
        self._data = loaded.reshape(loaded.shape[0] * loaded.shape[1], -1)
        print(f"Reshaped to {self._data.shape}")

    def __len__(self) -> int:
        """Returns total number of data points."""
        return self._data.shape[0]  # (N*M, ...)

    def __getitem__(self, index: SupportsIndex) -> Array:
        """Returns data point at the specified index."""
        return self._data[index]  # (1, ...)


def cardioid_dataset(
    fpath: str,
    batch_size: int = 32,
    seed: int | None = None,
) -> grain.MapDataset:
    """Creates a Grain dataset for cardioid training data.

    Args:
        fpath: The path to data
        batch_size: Batch size to use in loader
        seed: The seed to use for shuffling the dataset. If None, a random seed is used.
    """
    if seed is None:
        seed = random.getrandbits(16)

    def batch_fn(x):
        return jnp.stack(x, axis=0)  # use stack instead of concat

    return (
        grain.MapDataset.source(PointsImage(fpath))
        .shuffle(seed)
        .batch(batch_size, batch_fn=batch_fn)
    )


if __name__ == "__main__":
    a = generate_cardioid_data(CardioidDataParams())
    assert a.shape == (32, 64, 2)
    jnp.save("toycardioid", a)
