"""CelebAMask-HQ at 64x64: images and four-channel semantic masks.

Offline preprocessing for the second dataset (see ``experiments/CELEBAMASK_SCOPE.md``).
The 256px mirror ships ``images/<id>.jpg`` and 19-class label maps
``mask-anno-256/<id>.png`` for ``id`` in ``0..29999`` (its ``*_label`` folders
renumber every split from 0 and cannot be joined to the images); this module
turns them into the arrays the image track consumes, in the same ``[-1, 1]`` /
``uint8`` conventions as ``data/animefaces.py`` and ``data/layouts.rasterize``:

- ``celebamask_faces.npy``      ``(N, 64, 64, 3)`` float32 in ``[-1, 1]``
- ``celebamask_masks.npy``      ``(N, 64, 64, 4)`` uint8: face, eyes, mouth, nose
- ``celebamask_eval_masks.npy`` masks of the last ``N_EVAL`` ids: a layout bank
  for sampling; training uses ids ``< N - N_EVAL`` so no layout is shared
- ``celebamask_keep.npy``       ``(N,)`` bool training curation (see ``curate``)

Curation drops what makes the conditioning wrong or the FID reference
double-counted, nothing else (~1.8%): near-duplicate photos (later copy of a
pair with cosine > 0.995 between mean-centred 16x16 grey thumbnails), images
with no eye label and no glasses (the eyes channel is empty), likewise no lip
or no nose label (empty mouth / nose channel), face areas
outside ``[600, 2200]`` px at 64x64 (alignment failures) and near-greyscale
images.  Glasses, hats and single-eye profiles stay: their masks describe
them.  FID reference statistics always use the full set.

Masks are binary at 256 and area-averaged to 64, which gives the same
anti-aliased edges as the anime hulls.  Channel definitions (label ids):
face = skin, nose, glasses, eyes, brows, mouth, lips (1-7, 10-12); eyes =
l_eye, r_eye, glasses (3-5); mouth = mouth, u_lip, l_lip (10-12); nose = 2.

Usage::

    uv run python data/celebamask.py [--data-dir ...] [--out-dir .preprocessed]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import typer
from PIL import Image

CHANNELS = {
    "face": (1, 2, 3, 4, 5, 6, 7, 10, 11, 12),
    "eyes": (3, 4, 5),
    "mouth": (10, 11, 12),
    "nose": (2,),
}
N_EVAL = 3000  # the last ids are the layout bank for sampling, never trained on
EYE_IDS, GLASSES_ID, LIP_IDS, NOSE_ID = (4, 5), 3, (11, 12), 2
FACE_AREA = (600.0, 2200.0)


def near_duplicates(images: np.ndarray, threshold: float = 0.995) -> np.ndarray:
    """Bool mask of images that are a later near-copy of an earlier one."""
    n = len(images)
    th = images.mean(-1).reshape(n, 16, 4, 16, 4).mean((2, 4)).reshape(n, -1)
    th = th - th.mean(1, keepdims=True)
    th /= np.linalg.norm(th, axis=1, keepdims=True) + 1e-8
    later = np.zeros(n, bool)
    for i in range(0, n, 2000):
        sim = th[i : i + 2000] @ th.T
        a, b = np.where(sim > threshold)
        later[b[b > a + i]] = True
    return later


def curate(images: np.ndarray, masks: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Training keep-mask; ``classes`` is ``(N, 19)`` bool label presence."""
    face = (masks[..., 0] / 255.0).sum((1, 2))
    no_eyes = ~classes[:, EYE_IDS].any(1) & ~classes[:, GLASSES_ID]
    no_mouth = ~classes[:, LIP_IDS].any(1) | ~classes[:, NOSE_ID]
    grey = (images.max(-1) - images.min(-1)).mean((1, 2)) < 0.05
    bad_size = (face < FACE_AREA[0]) | (face > FACE_AREA[1])
    drop = near_duplicates(images) | no_eyes | no_mouth | grey | bad_size
    print(
        f"curation: dup {near_duplicates(images).sum()}, no eyes {no_eyes.sum()}, "
        f"no mouth/nose {no_mouth.sum()}, grey {grey.sum()}, size {bad_size.sum()} "
        f"-> keep {(~drop).sum()}/{len(drop)}"
    )
    return ~drop


def build_masks(
    label_paths: list[Path], size: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    """``(N, size, size, 4)`` uint8 soft masks and ``(N, 19)`` label presence."""
    out = np.empty((len(label_paths), size, size, len(CHANNELS)), np.uint8)
    classes = np.zeros((len(label_paths), 19), bool)
    for i, p in enumerate(label_paths):
        lab = np.asarray(Image.open(p))
        classes[i, np.unique(lab)] = True
        f = lab.shape[0] // size
        for c, ids in enumerate(CHANNELS.values()):
            m = np.isin(lab, ids).astype(np.float32)
            out[i, ..., c] = np.round(m.reshape(size, f, size, f).mean((1, 3)) * 255)
        if (i + 1) % 5000 == 0:
            print(f"  masks {i + 1}/{len(label_paths)}")
    return out, classes


def load_images(image_dir: Path, ids: list[int], size: int = 64) -> np.ndarray:
    """``(N, size, size, 3)`` float32 in ``[-1, 1]``, area-resampled from the JPEGs."""
    out = np.empty((len(ids), size, size, 3), np.float32)
    for i, n in enumerate(ids):
        img = Image.open(image_dir / f"{n}.jpg").convert("RGB")
        img = img.resize((size, size), Image.Resampling.BOX)
        out[i] = np.asarray(img, np.float32) / 127.5 - 1
        if (i + 1) % 5000 == 0:
            print(f"  images {i + 1}/{len(ids)}")
    return out


def main(
    data_dir: str = "data/celebamask-hq/CelebAMask-HQ-Dataset/resized256",
    out_dir: str = ".preprocessed",
):
    """Write the three arrays described in the module docstring."""
    src, out = Path(data_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ids = sorted(int(p.stem) for p in (src / "mask-anno-256").glob("*.png"))
    if ids != list(range(len(ids))):
        raise ValueError("expected label maps 0..N-1 in mask-anno-256/")
    print(f"{len(ids)} label maps; the last {N_EVAL} are the eval bank")
    masks, classes = build_masks([src / "mask-anno-256" / f"{n}.png" for n in ids])
    np.save(out / "celebamask_masks.npy", masks)
    np.save(out / "celebamask_eval_masks.npy", masks[-N_EVAL:])
    images = load_images(src / "images", ids)
    np.save(out / "celebamask_faces.npy", images)
    keep = curate(images, masks, classes)
    keep[-N_EVAL:] = False  # the eval bank is never trained on
    np.save(out / "celebamask_keep.npy", keep)
    print(
        f"wrote {out}/celebamask_{{faces,masks,eval_masks,keep}}.npy; mean mask area "
        f"(px @64): {dict(zip(CHANNELS, (masks / 255).sum((1, 2)).mean(0).round(1)))}"
    )


if __name__ == "__main__":
    typer.run(main)
