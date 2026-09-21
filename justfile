# tinyflow: `just` lists these.

default:
    @just --list --unsorted

# Install the locked environment exactly as CI does.
install:
    uv sync --locked --all-extras --dev

# One-shot: locked env + anime-faces data, ready for `just anime`.
setup: install data

# Lint + type check (the CI gates).
check:
    uv run ruff check
    uv run ty check

# Fetch the anime-faces images (huggan/anime-faces on the Hub) into data/anime-faces/.
data:
    uv run hf download huggan/anime-faces data.zip --repo-type dataset --local-dir data/anime-faces
    unzip -q -o data/anime-faces/data.zip -d data/anime-faces
    mv data/anime-faces/data data/anime-faces/images

# The 2-D toy flow model: trains and writes an animated SVG of its samples.
toy:
    uv run python data/toycardioid.py
    uv run main.py toy ./toycardioid.npy ./runs/toymodel.eqx

# The anime-face model: 125 epochs on one 4090 in about an hour -> runs/anime/model.eqx.
anime *args:
    uv run main.py anime ./runs/anime {{args}}

# Distil runs/anime/model.eqx into a 2-evaluation flow map + ONNX -> runs/distil/.
distil *args:
    uv run --extra export main.py distil ./runs/anime/model.eqx ./runs/distil {{args}}

# Figures from a checkpoint: `just showcase` (the trained model, ODE sampler) or
# `just showcase ./runs/distil/model.eqx 2` (the distilled map, two jumps).
showcase ckpt="./runs/anime/model.eqx" n_jumps="0":
    uv run main.py showcase {{ckpt}} --n-jumps {{n_jumps}}
