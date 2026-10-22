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

# The 2-D toy flow model -> runs/toymodel.eqx.
toy:
    uv run python data/toycardioid.py
    uv run main.py toy ./toycardioid.npy ./runs/toymodel.eqx

# The recipes live in models/animefaces/justfile, which also runs from that directory.
# The anime-face experiment: `just anime train|distil|showcase [ARGS]`, bare to list.
anime *args="--list":
    @just --justfile models/animefaces/justfile {{args}}
