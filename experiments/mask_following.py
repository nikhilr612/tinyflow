"""Mask-following score: does a conditioned model draw the face where the layout says?

Two stages, because the landmark detector lives in its own environment:

``sample`` (this repo's env) draws ``n`` layouts from the prior, renders one image
per layout with each ``name=checkpoint`` given, and writes ``<out>/<name>.npz``
with the uint8 images and the layouts, plus ``<out>/real.npz`` with real images
and their stored landmarks (the ceiling).

``score`` (detector env, e.g. ``~/.claude/jobs/211c1fe7/tmp/det/bin/python``)
runs ``hysts/anime-face-detector`` on every image and reports, per checkpoint:
detection rate, mean landmark error in ``[-1, 1]`` units against the layout the
image was rendered from (all 28 points; eyes, nose and mouth separately), the
detector's mean keypoint confidence for the nose and mouth points (low
confidence = the feature is not legibly there), and the IoU between the
rasterised detected hulls and the given face / eye masks.

Usage::

    uv run python experiments/mask_following.py sample --arms "rp=runs/x/model.eqx" ...
    <det-python> experiments/mask_following.py score
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import typer  # noqa: E402

from data.layouts import EYE_A, EYE_B, MOUTH, NOSE, LayoutPrior, rasterize  # noqa: E402

app = typer.Typer()
OUT = Path("runs/ablation/mask_following")


@app.command()
def sample(arms: list[str], n: int = 512, seed: int = 0, n_steps: int = 16):
    """Render ``n`` prior layouts with each ``name=checkpoint``; save images/layouts."""
    import jax
    import jax.numpy as jnp

    from data.animefaces import preprocess_all, to_uint8
    from models.imagefm import ImageFM
    from models.unet import UNet

    OUT.mkdir(parents=True, exist_ok=True)
    layouts = LayoutPrior.load().sample(n, seed)
    masks = jnp.asarray(rasterize(layouts))
    noise = jax.random.normal(jax.random.key(seed), (n, 64, 64, 3))
    for arm in arms:
        name, path = arm.split("=", 1)
        model = ImageFM.load(path, UNet.from_hparams)
        model.n_steps = n_steps
        imgs = []
        for i in range(0, n, 64):
            x = model.generate(
                noise[i : i + 64], masks[i : i + 64, ..., : model.cond_channels]
            )
            imgs.append(to_uint8(np.clip(np.asarray(x), -1, 1)))
        np.savez(OUT / f"{name}.npz", images=np.concatenate(imgs), layouts=layouts)
        print(f"wrote {OUT / f'{name}.npz'}")
    arr = preprocess_all("./data/anime-faces")
    lm = np.load(".preprocessed/anime_faces_landmarks.npy")
    sc = np.load(".preprocessed/anime_faces_landmark_scores.npy").mean(1)
    idx = np.random.default_rng(seed).choice(
        np.flatnonzero(sc >= 0.7), n, replace=False
    )
    np.savez(OUT / "real.npz", images=to_uint8(arr[idx]), layouts=lm[idx])
    print(f"wrote {OUT / 'real.npz'} (real images with their own landmarks)")


def _hull_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.minimum(a, b).sum()
    union = np.maximum(a, b).sum() + 1e-6
    return float(inter / union)


@app.command()
def score(size: int = 256):
    """Detect landmarks on every saved image set and score them against the layouts."""
    import cv2  # ty: ignore[unresolved-import]
    from anime_face_detector import create_detector  # ty: ignore[unresolved-import]

    det = create_detector("yolov3", device="cuda:0")
    box = [np.array([0, 0, size - 1, size - 1, 1.0], dtype=np.float32)]
    groups = {
        "eyes": np.r_[np.arange(28)[EYE_A], np.arange(28)[EYE_B]],
        "nose": np.array([NOSE]),
        "mouth": np.arange(28)[MOUTH],
    }
    cols = ["detected", "err all", *[f"err {g}" for g in groups]]
    cols += ["conf nose", "conf mouth", "IoU face", "IoU eyes"]
    print(f"{'set':>12}" + "".join(f"{c:>11}" for c in cols))
    for f in sorted(OUT.glob("*.npz")):
        d = np.load(f)
        imgs, layouts = d["images"], d["layouts"]
        rows, found = [], 0
        for img, lm in zip(imgs, layouts):
            bgr = cv2.cvtColor(
                cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC),
                cv2.COLOR_RGB2BGR,
            )
            res = det(bgr, boxes=box)
            if not res:
                continue
            found += 1
            kp_all = res[0]["keypoints"]
            kp = (kp_all[:, :2] + 0.5) / size * 2.0 - 1.0
            e = np.linalg.norm(kp - lm, axis=-1)
            m_det, m_giv = rasterize(kp[None])[0], rasterize(lm[None])[0]
            rows.append(
                [
                    e.mean(),
                    *[e[idx].mean() for idx in groups.values()],
                    kp_all[groups["nose"], 2].mean(),
                    kp_all[groups["mouth"], 2].mean(),
                    _hull_iou(m_det[..., 0], m_giv[..., 0]),
                    _hull_iou(m_det[..., 1], m_giv[..., 1]),
                ]
            )
        mean = np.mean(rows, axis=0)
        print(
            f"{f.stem:>12}{found / len(imgs):>11.3f}"
            + "".join(f"{v:>11.4f}" for v in mean)
        )


if __name__ == "__main__":
    app()
