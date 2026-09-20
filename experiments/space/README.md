---
title: tinyflow anime faces
emoji: 🎨
colorFrom: pink
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# Anime faces from a layout

A 64×64 anime-face generator that runs on the CPU in two network evaluations.

1. **Sample layout** draws a face layout (28 landmarks) from a Gaussian-mixture
   prior fitted to detector landmarks of the training set — no real image is
   involved.
2. The sliders move the sampled layout: position, size, tilt, eye spacing /
   height / size, mouth width / openness. The mask preview shows the face,
   eye and mouth regions the model is conditioned on.
3. **Generate** renders the image for that layout and a noise seed.

## Model

`tinyflow`: flow matching with x-prediction in pixel space, a 37M-parameter
U-Net conditioned on the layout mask with *mask-guided region pooling* (both
irises rendered from one shared feature — the iris-mismatch rate drops from
35 % to the data's own 4 %). The ODE sampler is distilled into a two-jump flow
map with a fixed teacher and exported whole to ONNX (`sampler_2jump.onnx`,
inputs `noise (64, 64, 3)`, `masks (64, 64, 3)`, output `image` in `[-1, 1]`).
Two evaluations take about 0.1 s on two CPU cores.

Files: `app.py` (the demo), `layouts.py` + `landmark_prior.npz` (the layout
prior, numpy only), `sampler_2jump.onnx` + `.data` (weights).
