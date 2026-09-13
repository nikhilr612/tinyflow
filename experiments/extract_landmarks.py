"""Run ``hysts/anime-face-detector`` on the cached dataset; save landmarks.

Writes ``.preprocessed/anime_faces_landmarks.npy`` of shape ``(N, 28, 2)`` --
keypoint ``(x, y)`` in the image's ``[-1, 1]`` coordinates (``-1`` = left /
top edge, ``+1`` = right / bottom edge, the same convention as the pixel
values), aligned index-for-index with ``anime_faces.npy`` -- and refreshes
``anime_faces_landmark_scores.npy`` ``(N, 28)``.  Points the detector places
outside the crop are kept as they are (slightly beyond +-1) rather than
clipped, so the prior fitted on them is not biased toward faces that fit;
clip at rasterisation.  Images the detector rejects get NaN landmarks and
zero scores.

The detector needs its own environment (mmcv/mmdet, torch); it is *not* a
project dependency.  Run with that interpreter, e.g.::

    ~/.claude/jobs/211c1fe7/tmp/det/bin/python experiments/extract_landmarks.py

Landmark layout (28 points): 0-10 chin contour and brows, 11-16 left eye,
17-22 right eye, 23 nose, 24-27 mouth -- the same slices ``make_masks``
rasterised into the semantic masks.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2  # ty: ignore[unresolved-import]  # detector env only
import numpy as np
from anime_face_detector import create_detector  # ty: ignore[unresolved-import]

ROOT = Path(__file__).resolve().parent.parent
S = 256  # detector input size; landmarks are mapped back to [-1, 1]


def main() -> None:
    """Detect landmarks for every cached image and save them."""
    arr = np.load(ROOT / ".preprocessed/anime_faces.npy", mmap_mode="r")
    det = create_detector("yolov3", device="cuda:0")
    n = len(arr)
    landmarks = np.full((n, 28, 2), np.nan, np.float32)
    scores = np.zeros((n, 28), np.float32)
    box = [np.array([0, 0, S - 1, S - 1, 1.0], dtype=np.float32)]
    t0 = time.time()
    for i in range(n):
        img = ((arr[i] + 1) * 127.5).clip(0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(
            cv2.resize(img, (S, S), interpolation=cv2.INTER_CUBIC), cv2.COLOR_RGB2BGR
        )
        res = det(bgr, boxes=box)
        if not res:
            continue
        kp = res[0]["keypoints"]
        # pixel centre convention: pixel j spans [j, j + 1) of S, so the map
        # to [-1, 1] is (p + 0.5) / S * 2 - 1.
        landmarks[i] = (kp[:, :2] + 0.5) / S * 2.0 - 1.0
        scores[i] = kp[:, 2]
        if (i + 1) % 2000 == 0:
            print(i + 1, f"{time.time() - t0:.0f}s", flush=True)
    np.save(ROOT / ".preprocessed/anime_faces_landmarks.npy", landmarks)
    np.save(ROOT / ".preprocessed/anime_faces_landmark_scores.npy", scores)
    print("done", n, "undetected", int(np.isnan(landmarks[:, 0, 0]).sum()))


if __name__ == "__main__":
    sys.exit(main())
