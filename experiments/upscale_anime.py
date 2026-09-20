"""Upscale the anime-faces *training images* 64 -> 128 for a 128 px run.

The dataset is native 64x64, so a 128 px model needs super-resolved targets.  This
is the opposite situation from the presentation upscalers tried on *generated*
samples (METHODS.md 7.4): here the inputs are clean real images, which is what
these networks were trained on.  Each model runs at its native x4 (64 -> 256) and
the result is area-averaged to 128, which removes most of the x4 hallucinated
texture and keeps the edge definition.

Runs in the detector environment (torch + CUDA, no ``basicsr`` needed: the RRDBNet
used by Real-ESRGAN and APISR-RRDB is re-implemented below, key-compatible with
their checkpoints in ``runs/upscale/``).

    <det-python> experiments/upscale_anime.py preview          # comparison grid
    <det-python> experiments/upscale_anime.py build --model anime6b   # full dataset

``build`` writes ``.preprocessed/anime_faces_128_<model>.npy`` (N, 128, 128, 3)
uint8 in the same image order as ``data/animefaces.preprocess_all``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import PIL.Image as PilImage
import torch  # ty: ignore[unresolved-import]  # detector env only
import torch.nn.functional as tf  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "data/anime-faces/images"
WEIGHTS = {
    "anime6b": ("runs/upscale/RealESRGAN_x4plus_anime_6B.pth", 6),
    "x4plus": ("runs/upscale/RealESRGAN_x4plus.pth", 23),
    "apisr": ("runs/upscale/4x_APISR_RRDB_GAN_generator.pth", 6),
}


class _RDB(nn.Module):
    """Residual dense block: five 3x3 convs with dense connections, residual 0.2."""

    def __init__(self, nf: int, gc: int):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):  # noqa: D102
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class _RRDB(nn.Module):
    """Residual-in-residual dense block: three ``_RDB`` with a residual 0.2."""

    def __init__(self, nf: int, gc: int):
        super().__init__()
        self.rdb1, self.rdb2, self.rdb3 = _RDB(nf, gc), _RDB(nf, gc), _RDB(nf, gc)

    def forward(self, x):  # noqa: D102
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    """Real-ESRGAN x4 generator (basicsr layout, key-compatible)."""

    def __init__(self, nb: int, nf: int = 64, gc: int = 32):
        """``nb`` RRDB blocks, ``nf`` features, ``gc`` growth channels."""
        super().__init__()
        self.conv_first = nn.Conv2d(3, nf, 3, 1, 1)
        self.body = nn.Sequential(*[_RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):  # noqa: D102
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(
            self.conv_up1(tf.interpolate(feat, scale_factor=2, mode="nearest"))
        )
        feat = self.lrelu(
            self.conv_up2(tf.interpolate(feat, scale_factor=2, mode="nearest"))
        )
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


def load_model(name: str) -> RRDBNet:
    """Load one of ``WEIGHTS`` onto the GPU in eval mode."""
    path, nb = WEIGHTS[name]
    sd = torch.load(ROOT / path, map_location="cpu", weights_only=False)
    sd = sd.get("params_ema", sd.get("params", sd.get("model_state_dict", sd)))
    net = RRDBNet(nb)
    net.load_state_dict(sd, strict=True)
    return net.cuda().eval()


@torch.no_grad()
def upscale(
    net: RRDBNet, imgs: np.ndarray, out: int = 128, batch: int = 64
) -> np.ndarray:
    """uint8 ``(N, 64, 64, 3)`` -> ``(N, out, out, 3)``: x4 via ``net``, area-mean."""
    res = []
    for i in range(0, len(imgs), batch):
        x = (
            torch.from_numpy(imgs[i : i + batch]).cuda().permute(0, 3, 1, 2).float()
            / 255
        )
        y = net(x).clamp(0, 1)
        y = tf.interpolate(y, size=(out, out), mode="area")
        res.append((y.permute(0, 2, 3, 1) * 255).round().byte().cpu().numpy())
    return np.concatenate(res)


def lanczos(imgs: np.ndarray, out: int = 128) -> np.ndarray:
    """Classical baseline: Lanczos resampling per image."""
    lanczos_filter = PilImage.Resampling.LANCZOS
    return np.stack(
        [
            np.asarray(PilImage.fromarray(im).resize((out, out), lanczos_filter))
            for im in imgs
        ]
    )


def image_paths() -> list[Path]:
    """Same order as ``data/animefaces.preprocess_all`` (sorted by name)."""
    return sorted(IMAGES.glob("*.png"))


def load_uint8(paths: list[Path]) -> np.ndarray:
    """Stack the PNGs at ``paths`` as a uint8 ``(N, 64, 64, 3)`` array."""
    return np.stack([np.asarray(PilImage.open(p).convert("RGB")) for p in paths])


def preview(n: int = 8, seed: int = 0, out: str = "runs/upscale/dataset_preview.png"):
    """Grid: rows = images, columns = original (nearest x2), Lanczos, each model."""
    rng = np.random.default_rng(seed)
    paths = image_paths()
    pick = [paths[i] for i in rng.choice(len(paths), n, replace=False)]
    imgs = load_uint8(pick)
    cols = {
        "orig (nearest)": np.repeat(np.repeat(imgs, 2, 1), 2, 2),
        "lanczos": lanczos(imgs),
    }
    for name in WEIGHTS:
        cols[name] = upscale(load_model(name), imgs)
        torch.cuda.empty_cache()
    pad = 4
    w = 128 + pad
    canvas = np.full((n * w + 20, len(cols) * w, 3), 255, np.uint8)
    for c, (name, arr) in enumerate(cols.items()):
        for r in range(n):
            canvas[20 + r * w : 20 + r * w + 128, c * w : c * w + 128] = arr[r]
    im = PilImage.fromarray(canvas)
    from PIL import ImageDraw

    d = ImageDraw.Draw(im)
    for c, name in enumerate(cols):
        d.text((c * w + 2, 4), name, fill=(0, 0, 0))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    im.save(out)
    print(f"wrote {out}  columns: {list(cols)}")


def build(model: str = "anime6b", out_dir: str = ".preprocessed"):
    """Upscale the whole dataset -> ``<out_dir>/anime_faces_128_<model>.npy``."""
    paths = image_paths()
    net = load_model(model)
    chunks = []
    for i in range(0, len(paths), 512):
        chunks.append(upscale(net, load_uint8(paths[i : i + 512])))
        print(f"{i + len(chunks[-1])}/{len(paths)}", end="\r")
    arr = np.concatenate(chunks)
    out = Path(out_dir) / f"anime_faces_128_{model}.npy"
    np.save(out, arr)
    print(f"\nwrote {out} {arr.shape} {arr.dtype}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "preview"
    kwargs = dict(a.lstrip("-").split("=", 1) for a in sys.argv[2:])
    if cmd == "preview":
        preview(
            n=int(kwargs.get("n", 8)),
            seed=int(kwargs.get("seed", 0)),
            out=kwargs.get("out", "runs/upscale/dataset_preview.png"),
        )
    elif cmd == "build":
        build(
            model=kwargs.get("model", "anime6b"),
            out_dir=kwargs.get("out_dir", ".preprocessed"),
        )
    else:
        raise SystemExit(__doc__)
