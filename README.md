# tinyflow
A short and sweet implementation of flow models for unconditional image generation in the JAX ecosystem
with an emphasis on readability, simplicity, cleanliness and reproducibility.

# Get started
1. This project uses `uv`, so make sure to install `uv`:
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone this repo and run:
```
uv sync
```

3. [OPTIONAL] Install platform specific JAX runtimes (eg. TPU):
```sh
uv add jax[tpu]
```
JAX will auto-select the appropriate runtime from all that are intialized.
By default `jax[cuda13]` is installed.

4. Run the main CLI
```sh
uv run main.py --help
```
