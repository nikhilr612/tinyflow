"""Anime faces from a layout: sample a layout from the prior, tweak it, generate.

Everything runs on the CPU with numpy and onnxruntime.  The model is a 37M
parameter U-Net (flow matching, x-prediction, layout-conditioned with region
pooling) distilled into a two-jump flow map, exported whole as
``sampler_2jump.onnx``: ``(noise 64x64x3, masks 64x64x3) -> image``.  Layouts
are 28 face landmarks; ``LayoutPrior`` is a Gaussian mixture over a
point-distribution model of them, so no real image is involved.
"""

from __future__ import annotations

import gradio as gr
import numpy as np
import onnxruntime as ort
import spaces
from PIL import Image

from layouts import (
    EYE_A,
    EYE_B,
    MOUTH,
    LayoutPrior,
    from_pose_shape,
    rasterize,
    to_pose_shape,
)


@spaces.GPU
def _zerogpu_placeholder():
    """Never called.  ZeroGPU refuses to start a Space without a @spaces.GPU
    function; inference here is two CPU evaluations (~0.1 s), for which a GPU
    attach would only add seconds of overhead."""


PRIOR = LayoutPrior.load()
SESSION = ort.InferenceSession("sampler_2jump.onnx", providers=["CPUExecutionProvider"])
BROWS = slice(5, 11)
SLIDERS = (
    ("x", "left / right", -0.3, 0.3),
    ("y", "up / down", -0.3, 0.3),
    ("size", "face size", -0.4, 0.4),
    ("tilt", "tilt (degrees)", -20, 20),
    ("eye_gap", "eye spacing", -0.15, 0.15),
    ("eye_y", "eye height", -0.15, 0.15),
    ("eye_size", "eye size", -0.5, 0.5),
    ("mouth_w", "mouth width", -0.5, 0.5),
    ("mouth_h", "mouth open", -0.5, 1.0),
)


def sample_layout(seed: int) -> np.ndarray:
    """One layout ``(28, 2)`` from the prior."""
    return PRIOR.sample(1, seed=int(seed))[0]


def edit(lm: np.ndarray, x, y, size, tilt, eye_gap, eye_y, eye_size, mouth_w, mouth_h):
    """Apply the slider edits to a base layout; every slider at 0 is the identity."""
    lm = lm[None].copy()
    # Whole-face pose: translate, scale, rotate about the face centre.
    pose, shape = to_pose_shape(lm)
    pose += np.array([[x, y, size, np.deg2rad(tilt)]])
    lm = from_pose_shape(pose, shape)
    pts = lm[0]
    # Eyes (with brows): spread apart, move vertically, scale about each centre.
    for group, sign in ((EYE_A, -1), (EYE_B, +1)):
        c = pts[group].mean(0)
        pts[group] = c + (pts[group] - c) * (1 + eye_size)
        pts[group] += [sign * eye_gap, eye_y]
    pts[BROWS] += [0, eye_y]
    # Mouth: scale width and height about its centre.
    c = pts[MOUTH].mean(0)
    pts[MOUTH] = c + (pts[MOUTH] - c) * [1 + mouth_w, 1 + mouth_h]
    return pts


def mask_image(lm: np.ndarray) -> Image.Image:
    """Render the layout mask (face grey, eyes cyan, mouth red) at 256 px."""
    m = rasterize(lm[None], size=64)[0]  # (64, 64, 4): face, eyes, mouth, nose
    rgb = np.zeros((64, 64, 3))
    rgb += m[..., :1] * 0.45
    rgb += m[..., 1:2] * [0.0, 0.5, 0.6]
    rgb += m[..., 2:3] * [0.6, -0.2, -0.2]
    img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    return img.resize((256, 256), Image.Resampling.NEAREST)


def generate(lm: np.ndarray, noise_seed: int) -> Image.Image:
    """Two network evaluations: noise + layout mask -> 64x64 image, shown at 256 px."""
    masks = rasterize(lm[None], size=64)[0, ..., :3].astype(np.float32)
    noise = np.random.default_rng(int(noise_seed)).standard_normal(
        (64, 64, 3), np.float32
    )
    (x,) = SESSION.run(["image"], {"noise": noise, "masks": masks})
    img = Image.fromarray(((np.clip(x, -1, 1) + 1) * 127.5).round().astype(np.uint8))
    return img.resize((256, 256), Image.Resampling.LANCZOS)


with gr.Blocks(title="tinyflow anime faces") as demo:
    gr.Markdown(
        "**Anime faces from a layout.** Sample a face layout from the prior, tweak it, "
        "generate. 64x64, two network evaluations, CPU."
    )
    base = gr.State(sample_layout(0))
    with gr.Row():
        with gr.Column():
            layout_seed = gr.Number(value=0, label="layout seed", precision=0)
            sample_btn = gr.Button("Sample layout")
            sliders = [
                gr.Slider(lo, hi, value=0, label=label, step=(hi - lo) / 100)
                for _, label, lo, hi in SLIDERS
            ]
            mask_view = gr.Image(
                mask_image(base.value), label="layout mask", type="pil"
            )
        with gr.Column():
            noise_seed = gr.Number(value=0, label="noise seed", precision=0)
            gen_btn = gr.Button("Generate", variant="primary")
            out = gr.Image(label="sample", type="pil")

    def on_sample(seed, *edits):
        lm = sample_layout(seed)
        return lm, mask_image(edit(lm, *edits))

    def on_edit(lm, *edits):
        return mask_image(edit(lm, *edits))

    def on_generate(lm, seed, *edits):
        return generate(edit(lm, *edits), seed)

    sample_btn.click(on_sample, [layout_seed, *sliders], [base, mask_view])
    for s in sliders:
        s.release(on_edit, [base, *sliders], mask_view)
    gen_btn.click(on_generate, [base, noise_seed, *sliders], out)

demo.launch(ssr_mode=False)
