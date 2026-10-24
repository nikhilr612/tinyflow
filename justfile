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

# Fetch the anime-faces images and landmarks (pinned, from the Hub) into data/anime-faces/.
data:
    uv run python -c "from tinyflow.data import animefaces; animefaces.download()"

# The 2-D toy flow model -> runs/toymodel.eqx.
toy:
    uv run python -m tinyflow.data.toycardioid
    uv run main.py toy ./toycardioid.npy ./runs/toymodel.eqx

# The recipes live in tinyflow/models/animefaces/justfile, which also runs from that directory.
# The anime-face experiment: `just anime train|distil|showcase [ARGS]`, bare to list.
anime *args="--list":
    @just --justfile tinyflow/models/animefaces/justfile {{args}}

# The recipes live in tinyflow/models/senbridge/justfile, which also runs from that directory.
# The radar-to-optical experiment: `just senbridge train|evaluate [ARGS]`, bare to list.
senbridge *args="--list":
    @just --justfile tinyflow/models/senbridge/justfile {{args}}
