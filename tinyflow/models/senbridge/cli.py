"""The SEN-Bridge experiment's commands: ``train`` and ``evaluate``.

Mounted by ``main.py`` as ``main.py senbridge ...``.  Each command only wires the
pieces together: the data (``data/senbridge.py``), the denoiser (``dit.py``),
the maths (``bridge.py``) and the shared loop (``trainer.py``).  ``sen_bridge``
-- the data release and the benchmark's metrics, the ``senbridge`` extra -- is
imported inside the commands, so the rest of the CLI runs without it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import jax
import numpy as np
import typer
from tqdm import tqdm

import tinyflow.data.senbridge as sb
from tinyflow import trainer
from tinyflow.models.senbridge.bridge import Bridge, BridgeTrainer, to_unit
from tinyflow.models.senbridge.colour import colour_error
from tinyflow.models.senbridge.dit import DiT

app = typer.Typer(add_completion=False, help="Radar-to-optical translation, 128x128.")

STATS_REPO = "nikhilr01/sen-bridge"  # the real targets' FID statistics, per split
STATS_REVISION = "86f349284fbc0037a1cd97186387aab110cf6878"


def _split(name: str) -> Any:
    """``sen_bridge.data.load_dataset(name)``: downloads and verifies on first use."""
    from sen_bridge import data

    return data.load_dataset(name)


@app.command()
def train(
    outdir: Annotated[
        str, typer.Argument(help="Run directory: model.eqx, losses.json, samples")
    ] = "./runs/senbridge",
    n_epochs: int = 120,
    width: int = 512,
    depth: int = 12,
    heads: int = 8,
    batch_size: int = 128,
    n_train: Annotated[
        int, typer.Option(help="Train on the first N pairs only (0: all), for tests")
    ] = 0,
    seed: int = 0,
):
    """Train the radar-to-optical bridge (the recipe in bridge.py)."""
    if 0 < n_train < batch_size:
        raise typer.BadParameter(
            f"--n-train {n_train} is below --batch-size {batch_size}: no whole batch"
        )
    rows = np.arange(n_train) if n_train else None
    dataset, batches = sb.make_dataset(_split("train"), batch_size, seed, rows)
    hparams = {"width": width, "depth": depth, "heads": heads}
    key, k_net, k_z = jax.random.split(jax.random.key(seed), 3)
    model = Bridge(DiT(k_net, **hparams), hparams)

    # Eight fixed validation pairs and noise, the same every epoch; each tile is
    # the radar image, the sample and the optical image, top to bottom.
    val = _split("validation")
    idx = np.linspace(0, len(val) - 1, 8).astype(int)
    sar, optical, meta = (
        to_unit(val.sar[idx]),
        to_unit(val.optical[idx]),
        val.meta[idx],
    )
    z = jax.random.normal(k_z, sar.shape)

    def sample_fn(m: Bridge):
        return np.concatenate(
            [sar, np.asarray(m.sample(sar, meta, z)), optical], axis=1
        )

    recipe = BridgeTrainer(schedule_steps=n_epochs * batches)
    fits = recipe.fit(model.net, dataset, n_epochs, batches, key)
    trainer.run(fits, model, Path(outdir), sample_fn, n_epochs)


@app.command()
def evaluate(
    checkpoint: Annotated[
        str, typer.Argument(help="A trained model.eqx")
    ] = "./runs/senbridge/model.eqx",
    split: Annotated[
        str, typer.Option(help="validation (model selection) or test (the benchmark)")
    ] = "validation",
    n: Annotated[
        int, typer.Option(help="Pairs evaluated, a fixed random subset (0: all)")
    ] = 5000,
    offset: Annotated[
        int, typer.Option(help="Start of the subset in a fixed shuffle of the split")
    ] = 0,
    kappa: Annotated[
        float, typer.Option(help="0: the ODE sampler; > 0: the SDE's diffusion scale")
    ] = 0.0,
    n_steps: Annotated[
        int, typer.Option(help="Sampler steps: 2 denoiser evaluations each")
    ] = 16,
    batch_size: int = 250,
    seed: int = 0,
):
    """The benchmark's FID, LPIPS, PSNR and SSIM, and colour error, of one sample each.

    FID is against the split's shipped statistics of all its targets, so below
    the split's size it is biased upwards: compare runs at the same ``n``, and
    report test with ``--n 0``.  ``--offset`` picks disjoint subsets: ``0`` and
    ``5000`` are two independent 5,000-pair validation sets.  Writes the
    metrics beside the checkpoint, named after the split and the settings.
    """
    from huggingface_hub import hf_hub_download
    from sen_bridge import jax as metrics

    pairs = _split(split)
    if not 0 <= offset < len(pairs):
        raise typer.BadParameter(f"--offset {offset} is outside the {len(pairs)} pairs")
    model = Bridge.load(checkpoint, DiT)
    # Subsets are slices of one fixed shuffle, so different offsets are disjoint;
    # ``seed`` only draws the sampler's noise.  ``--n 0`` is everything from
    # ``offset`` on.
    order = np.random.default_rng(0).permutation(len(pairs))
    rows = np.sort(order[offset : offset + n] if n else order[offset:])
    stats = hf_hub_download(
        STATS_REPO,
        f"stats/fid_{split}.npz",
        repo_type="dataset",
        revision=STATS_REVISION,
    )
    fid = metrics.FID.from_stats(stats)
    paired = {
        f: jax.jit(jax.vmap(getattr(metrics, f))) for f in ("lpips", "psnr", "ssim")
    }
    scores: dict[str, list] = {f: [] for f in (*paired, "colour_de")}
    key = jax.random.key(seed)
    for i in tqdm(range(0, len(rows), batch_size), desc=split):
        r = rows[i : i + batch_size]
        sar, optical = to_unit(pairs.sar[r]), to_unit(pairs.optical[r])
        key, k_z, k_w = jax.random.split(key, 3)
        z = jax.random.normal(k_z, sar.shape)
        x = model.sample(sar, pairs.meta[r], z, kappa, k_w, n_steps)
        fid.update(x, real=False)
        for f, fn in paired.items():
            scores[f].append(np.asarray(fn(x, optical)))
        scores["colour_de"].append(colour_error(np.asarray(x), np.asarray(optical)))
    result: dict[str, object] = {"split": split, "n": len(rows), "offset": offset}
    result |= {"kappa": kappa, "n_steps": n_steps}
    # lpips, psnr, ssim and colour_de (mean CIEDE2000 at 8 px, colour.py)
    per_image = {f: np.concatenate(v) for f, v in scores.items()}
    result |= {"fid": float(fid.compute())}
    for f, v in per_image.items():  # mean and its standard error over images
        result |= {f: float(v.mean()), f"{f}_se": float(v.std() / np.sqrt(len(v)))}
    text = json.dumps(result, indent=1)
    typer.echo(text)
    tag = f"{split}_n{len(rows)}_o{offset}_k{kappa:g}_N{n_steps}_s{seed}"
    # Not ``with_suffix``: the dot in a fractional kappa would pass for an extension.
    stem = f"{Path(checkpoint).stem}_metrics_{tag}"
    Path(checkpoint).with_name(f"{stem}.json").write_text(text)
    np.savez(
        Path(checkpoint).with_name(f"{stem}.npz"),
        rows=rows,
        scores=np.stack(list(per_image.values())),
    )
