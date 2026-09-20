"""Load a checkpoint from either code base into this branch's ``ImageFM``.

The clean rewrite (``../tinyflow-clean``, package ``models.animefaces``) saves
the same weight format but its network signature is ``net(x, masks, t[, s])``
with the image alone as input, while this branch's evaluators drive
``net(concat[x, masks, indicator], t)`` through ``ImageFM.velocity``.  The
adapter below bridges the two, so ``cond_eval.py``, ``eye_consistency.py`` and
the rest evaluate a clean-branch checkpoint with exactly the sampler and
metrics used for the reference numbers.

A checkpoint is recognised as clean when its ``.hparams`` lacks ``in_channels``.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

from models.imagefm import ImageFM
from models.unet import UNet

CLEAN_ROOT = Path(__file__).resolve().parents[2] / "tinyflow-clean"


class _CleanNet:
    """``net(x_in, t)`` for this branch, backed by a clean-branch ``UNet``."""

    def __init__(self, net):
        self.net = net

    def __call__(self, x_in, t):
        return self.net(x_in[..., :3], x_in[..., 3:6], t)


class CleanFM(ImageFM):
    """This branch's ``ImageFM`` over a clean-branch model, plus its flow map."""

    def __init__(self, clean, hparams):
        """Wrap the clean ``ImageFM`` and keep it for ``jump``."""
        super().__init__(_CleanNet(clean.net), hparams)
        self.clean = clean

    def jump(self, x_0, masks, n_jumps: int = 2):
        """Sample with the distilled flow map in ``n_jumps`` evaluations."""
        return self.clean.jump(x_0, masks, n_jumps)


def load_any(path: str) -> ImageFM:
    """``ImageFM.load`` that also accepts a clean-branch checkpoint."""
    hparams = json.loads(Path(path + ".hparams").read_text())
    if "in_channels" in hparams:
        return ImageFM.load(path, UNet.from_hparams)
    if str(CLEAN_ROOT) not in sys.path:
        sys.path.insert(0, str(CLEAN_ROOT))
    # The clean package shadows this branch's ``models`` and ``data`` names, so
    # import through a fresh module lookup and restore afterwards.
    saved = {
        k: v for k, v in sys.modules.items() if k.split(".")[0] in ("models", "data")
    }
    for k in saved:
        del sys.modules[k]
    try:
        clean_unet = importlib.import_module("models.animefaces.unet").UNet
        clean_flow = importlib.import_module("models.animefaces.flow").ImageFM
        clean = clean_flow.load(path, clean_unet)
    finally:
        for k in [k for k in sys.modules if k.split(".")[0] in ("models", "data")]:
            del sys.modules[k]
        sys.modules.update(saved)
        sys.path.remove(str(CLEAN_ROOT))
    return CleanFM(clean, {**hparams, "cond_channels": 3})
