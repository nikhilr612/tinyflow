"""Face layouts: 28 landmarks, their rasterised region masks, and a sampling prior.

The image model is conditioned on a three-channel layout mask -- the convex
hulls of the face, the two eyes and the mouth -- rasterised from 28 face
landmarks (``hysts/anime-face-detector`` on the training images; the arrays
ship with the repository, see ``data/README.md``).  At sampling time layouts
are drawn from ``LayoutPrior``, a Gaussian mixture over a point-distribution
model of those landmarks, so no real image is involved in generation.

Landmark layout, image coordinates in ``[-1, 1]``: 0-4 chin contour, 5-7 / 8-10
brows, 11-16 / 17-22 eyes, 23 nose, 24-27 mouth.  Everything here is numpy;
fitting the prior needs scikit-learn and was done once (``data/README.md``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

PRIOR_PATH = Path(__file__).parent / "landmark_prior.npz"
LANDMARKS_PATH = Path(__file__).parent / "landmarks.npz"

# Landmark groups.  Region masks are the hulls of these point sets.
FACE = slice(0, 11)  # chin contour (0-4) + brows (5-10)
EYE_A = slice(11, 17)
EYE_B = slice(17, 23)
MOUTH = slice(24, 28)


def to_pose_shape(lm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split ``(N, 28, 2)`` layouts into pose ``(N, 4)`` and shape ``(N, 56)``.

    Pose is ``(cx, cy, log s, theta)``: centroid, log inter-ocular distance and
    roll; shape is the similarity-normalised point set, flattened.
    """
    ea, eb = lm[:, EYE_A].mean(1), lm[:, EYE_B].mean(1)
    c = lm.mean(1)
    d = eb - ea
    s = np.linalg.norm(d, axis=-1)
    theta = np.arctan2(d[:, 1], d[:, 0])
    cos, sin = np.cos(-theta), np.sin(-theta)
    rel = lm - c[:, None]
    rot = np.stack(
        [
            cos[:, None] * rel[..., 0] - sin[:, None] * rel[..., 1],
            sin[:, None] * rel[..., 0] + cos[:, None] * rel[..., 1],
        ],
        -1,
    )
    shape = rot / s[:, None, None]
    pose = np.stack([c[:, 0], c[:, 1], np.log(s), theta], -1)
    return pose, shape.reshape(len(lm), -1)


def from_pose_shape(pose: np.ndarray, shape: np.ndarray) -> np.ndarray:
    """Inverse of :func:`to_pose_shape`."""
    cx, cy, log_s, theta = pose.T
    pts = shape.reshape(len(pose), 28, 2) * np.exp(log_s)[:, None, None]
    cos, sin = np.cos(theta), np.sin(theta)
    rot = np.stack(
        [
            cos[:, None] * pts[..., 0] - sin[:, None] * pts[..., 1],
            sin[:, None] * pts[..., 0] + cos[:, None] * pts[..., 1],
        ],
        -1,
    )
    return rot + np.stack([cx, cy], -1)[:, None]


def _hull(points: np.ndarray) -> np.ndarray:
    """Convex hull of ``(k, 2)`` points, counter-clockwise (monotone chain)."""
    pts = sorted(map(tuple, points))
    if len(pts) <= 2:
        return np.asarray(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1])


def _fill_polygon(canvas: np.ndarray, poly: np.ndarray) -> None:
    """Fill a convex polygon (``(k, 2)`` in pixel coordinates) into ``canvas``."""
    if len(poly) < 3:
        return
    h, w = canvas.shape
    ys, xs = np.mgrid[0:h, 0:w]
    px, py = xs + 0.5, ys + 0.5
    inside = np.ones((h, w), bool)
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        inside &= (x1 - x0) * (py - y0) - (y1 - y0) * (px - x0) >= 0
    canvas[inside] = 1.0


def rasterize(lm: np.ndarray, size: int = 64, hi: int = 256) -> np.ndarray:
    """Rasterise ``(N, 28, 2)`` layouts in ``[-1, 1]`` to ``(N, size, size, 3)`` masks.

    Channels: face, eyes, mouth -- the convex hulls of their landmarks, filled at
    ``hi`` resolution and area-averaged down to ``size`` for anti-aliased edges.
    """
    out = np.zeros((len(lm), size, size, 3), np.float32)
    px = (np.clip(lm, -1.2, 1.2) + 1) / 2 * hi
    f = hi // size
    for i, p in enumerate(px):
        canvas = np.zeros((hi, hi, 3), np.float32)
        _fill_polygon(canvas[..., 0], _hull(p[FACE]))
        _fill_polygon(canvas[..., 1], _hull(p[EYE_A]))
        _fill_polygon(canvas[..., 1], _hull(p[EYE_B]))
        _fill_polygon(canvas[..., 2], _hull(p[MOUTH]))
        out[i] = canvas.reshape(size, f, size, f, 3).mean((1, 3))
    return out


class LayoutPrior:
    """Gaussian mixture over ``[pose, PCA(shape)]``; samples layouts in ``[-1, 1]``.

    Sampling needs only numpy: pick a component by weight, draw from its
    Gaussian, invert the PCA, undo the pose normalisation.
    """

    def __init__(
        self,
        pca_components: np.ndarray,
        pca_mean: np.ndarray,
        shape_mean: np.ndarray,
        weights: np.ndarray,
        means: np.ndarray,
        covariances: np.ndarray,
    ):
        """Store the fitted arrays (see :meth:`load`)."""
        self.pca_components, self.pca_mean = pca_components, pca_mean
        self.shape_mean = shape_mean
        self.weights, self.means, self.covariances = weights, means, covariances

    @classmethod
    def load(cls, path: Path = PRIOR_PATH) -> LayoutPrior:
        """Load the arrays written by the fitting script."""
        d = np.load(path)
        keys = (
            "pca_components",
            "pca_mean",
            "shape_mean",
            "weights",
            "means",
            "covariances",
        )
        return cls(*(d[k] for k in keys))

    def sample(self, n: int, seed: int = 0) -> np.ndarray:
        """Draw ``n`` layouts ``(n, 28, 2)``."""
        rng = np.random.default_rng(seed)
        comp = rng.choice(
            len(self.weights), size=n, p=self.weights / self.weights.sum()
        )
        z = np.stack(
            [rng.multivariate_normal(self.means[c], self.covariances[c]) for c in comp]
        )
        pose, coef = z[:, :4], z[:, 4:]
        shape = coef @ self.pca_components + self.pca_mean + self.shape_mean
        return from_pose_shape(pose, shape)

    def sample_masks(self, n: int, seed: int = 0) -> np.ndarray:
        """``n`` rasterised layout masks ``(n, 64, 64, 3)`` in ``[0, 1]``."""
        return rasterize(self.sample(n, seed))
