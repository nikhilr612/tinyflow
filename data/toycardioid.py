"""
A toy dataset that generates points along a cardioid curve.
"""

from dataclasses import dataclass
from typing import SupportsIndex

import grain
import jax.numpy as jnp
from jax import Array
from jax import random as jax_random


@dataclass
class CardioidDataParams:
    n_images: int = 32
    n_t: int = 64
    r: float = 2.0
    sigma_r: float = 0.5
    sigma_xy: float = 0.05
    sigma_theta_deg: float = 5.0
    seed: int = 42


def generate_cardioid_data(params: CardioidDataParams) -> Array:
    """
    Generates points along a cardioid curve with added noise.

    Args:
        num_points (int): Number of data points to generate.
        noise_level (float): Standard deviation of Gaussian noise to add.
        seed (int): Random seed for reproducibility.

    Returns:
        jnp.ndarray: Array of shape (num_points, 2) containing the generated points.
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

    # Stack x and y coordinates
    points = jnp.stack([x, y], axis=2)

    noise = params.sigma_xy * jax_random.normal(key, shape=points.shape)
    noisy_points = points + noise

    return noisy_points


class PointsImage(grain.sources.RandomAccessDataSource):
    def __init__(self, fpath: str):
        loaded = jnp.load(fpath)
        self._data = loaded.reshape(loaded.shape[0] * loaded.shape[1], -1)

    def __len__(self) -> int:
        return self._data.shape[0]  # (N*M, ...)

    def __getitem__(self, index: SupportsIndex) -> Array:
        return self._data[index]  # (1, ...)


def cardioid_dataset(
    fpath: str,
    batch_size: int = 32,
    seed: int | None = None,
) -> grain.MapDataset:
    return (
        grain.MapDataset.source(PointsImage(fpath))
        .shuffle(seed)
        .batch(
            batch_size, batch_fn=lambda x: jnp.concat(x)
        )  # use concat instead of stack
    )


if __name__ == "__main__":
    a = generate_cardioid_data(CardioidDataParams())
    assert a.shape == (32, 64, 2)
    jnp.save("toycardioid", a)
