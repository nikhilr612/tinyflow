"""Colour fidelity: mean CIEDE2000 between a sample and its target at an 8-pixel scale.

Inception-FID is nearly blind to colour on this data (desaturating real images by
30 % moves it from 7.0 to 8.6), yet colour is most of what the radar leaves
ambiguous.  ``colour_error`` measures it on its own: both images are rounded to
8 bits like the benchmark's metrics, averaged over ``CELL`` x ``CELL`` pixels
(texture and the ~1 px radar-optical misregistration drop out), converted from
sRGB to CIELAB (D65) and compared cell by cell with CIEDE2000 (Sharma, Wu &
Dalal 2005).  About 1 is a just-noticeable difference.  numpy only.
"""

from __future__ import annotations

import numpy as np

CELL = 8
_M = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
_WHITE = np.array([0.95047, 1.0, 1.08883])


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB in ``[0, 1]`` (``(..., 3)``) to CIELAB under D65."""
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = lin @ _M.T / _WHITE
    eps = (6 / 29) ** 3
    f = np.where(xyz > eps, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    return np.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], -1)


def ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """The CIEDE2000 difference of ``(..., 3)`` CIELAB arrays (Sharma et al. 2005)."""
    l1, a1, b1 = np.moveaxis(lab1, -1, 0)
    l2, a2, b2 = np.moveaxis(lab2, -1, 0)
    c_bar = (np.hypot(a1, b1) + np.hypot(a2, b2)) / 2
    g = 0.5 * (1 - np.sqrt(c_bar**7 / (c_bar**7 + 25.0**7)))
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    c1p, c2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    chroma = c1p * c2p
    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    dh = np.where(chroma == 0, 0.0, dh)
    d_l, d_c = l2 - l1, c2p - c1p
    d_h = 2 * np.sqrt(chroma) * np.sin(np.radians(dh) / 2)
    l_bar, c_bar_p = (l1 + l2) / 2, (c1p + c2p) / 2
    h_sum = h1p + h2p
    h_bar = np.where(
        np.abs(h1p - h2p) <= 180,
        h_sum / 2,
        np.where(h_sum < 360, (h_sum + 360) / 2, (h_sum - 360) / 2),
    )
    h_bar = np.where(chroma == 0, h_sum, h_bar)
    t = (
        1
        - 0.17 * np.cos(np.radians(h_bar - 30))
        + 0.24 * np.cos(np.radians(2 * h_bar))
        + 0.32 * np.cos(np.radians(3 * h_bar + 6))
        - 0.20 * np.cos(np.radians(4 * h_bar - 63))
    )
    d_theta = 30 * np.exp(-(((h_bar - 275) / 25) ** 2))
    r_c = 2 * np.sqrt(c_bar_p**7 / (c_bar_p**7 + 25.0**7))
    s_l = 1 + 0.015 * (l_bar - 50) ** 2 / np.sqrt(20 + (l_bar - 50) ** 2)
    s_c = 1 + 0.045 * c_bar_p
    s_h = 1 + 0.015 * c_bar_p * t
    r_t = -np.sin(np.radians(2 * d_theta)) * r_c
    return np.sqrt(
        (d_l / s_l) ** 2
        + (d_c / s_c) ** 2
        + (d_h / s_h) ** 2
        + r_t * (d_c / s_c) * (d_h / s_h)
    )


def colour_error(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mean CIEDE2000 per image pair, ``(B, H, W, 3)`` in ``[-1, 1]`` -> ``(B,)``."""

    def cells(images):
        u8 = np.round((np.clip(np.asarray(images, np.float64), -1, 1) + 1) * 127.5)
        b, h, w, _ = u8.shape
        pooled = u8.reshape(b, h // CELL, CELL, w // CELL, CELL, 3).mean((2, 4))
        return srgb_to_lab(pooled / 255)

    return ciede2000(cells(x), cells(y)).mean((1, 2))
