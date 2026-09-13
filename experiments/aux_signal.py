"""Offline signal tests for the auxiliary losses in ``models/imagefm.py``.

No training and no GPU: JAX is pinned to CPU before import so this can run
next to a training job.  Every test writes a table to stdout and a PNG next
to it under ``./runs/ablation/signal/`` (git-ignored), so each number has a
picture that can be checked by eye.  The losses under test are the pixel
L2 (``pixel_l2``: what the flow-matching term reduces to on ``x_hat``),
``_edge_loss`` (``sobel``: signed Sobel) and ``_line_loss`` (``line``: soft
top-hat ink map).

1. **Discrimination** (``discrimination.png``).  ``L(x, degrade(x))`` for a
   set of degradations, normalised by ``L(x, other_image)``.  A loss earns
   its keep by rising sharply under the degradation it is meant to see
   (blur, for both) and staying flat under the ones it should ignore
   (colour shifts, brightness).  The panel shows each degradation next to
   the ink and Sobel-magnitude maps it produces.
2. **Pixel-space descent** (``descent.png``).  Start from a blurred image
   and run Adam on the *pixels* minimising ``L2 + w * L_aux`` towards the
   clean image.  L2 alone recovers everything eventually (the problem is
   convex), so the question is the *rate* at which edges and ink come
   back: PSNR and the ink/edge mass along the trajectory.
3. **Blurry-mean conflict** (``xhat_vs_t.png``).  Every aux loss punishes the
   conditional mean ``E[x_1 | x_t]``, which at low ``t`` is blurry.  Measured
   two ways: ``L(blur(x), x)`` as a proxy, and the real thing -- the
   baseline checkpoint's ``x_hat`` at a grid of ``t``.  The panel shows
   ``x_t``, ``x_hat`` and their ink maps per ``t``.
   ``xhat_sampling.png`` repeats this along real ODE trajectories from noise
   (sampling-time view: states the sampler actually visits, errors compound).
   ``gate.png`` (``test_gate``) estimates, per ``t``, how much of each aux
   loss is irreducible (the loss of a blur PSNR-matched to ``x_hat(t)``) and
   fits a logistic gate ``g(t)`` to the reducible fraction.
4. **Redundancy** (``grad_cosine.png``).  Cosine similarity between the
   per-sample gradients of each loss with respect to ``x_hat``.  Two losses
   whose gradients point the same way are one loss with a bigger weight.

Usage::

    uv run python experiments/aux_signal.py [--n 256] [--seed 0]
        [--checkpoint runs/exp_aux/baseline/model.eqx]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
import PIL.Image as Pilimage  # noqa: E402
import typer  # noqa: E402
from einops import rearrange  # noqa: E402
from PIL import ImageDraw, ImageFont  # noqa: E402

from data.animefaces import preprocess_all, to_uint8  # noqa: E402
from models.imagefm import ImageFM, _solve  # noqa: E402
from models.unet import UNet  # noqa: E402

OUTDIR = Path("./runs/ablation/signal")
LUMA_W = jnp.array([0.299, 0.587, 0.114], dtype=jnp.float32)


# --------------------------------------------------------------------------
# Losses and image operations
# --------------------------------------------------------------------------


def l2(pred, target):
    """Pixel L2 (``optax.l2_loss`` mean), the flow-matching term on ``x_hat``."""
    return optax.l2_loss(pred, target).mean()


# Names follow ``models/imagefm.py``: ``sobel`` is ``_edge_loss`` (edge_weight),
# ``line`` is ``_line_loss`` (line_weight).  ``pixel_l2`` is not an auxiliary
# loss: it is the flow-matching term expressed on ``x_hat`` (up to the
# ``1/(1-t)^2`` weight) and serves as the reference every aux term is judged
# against.  The mask ``_aux_loss`` acts on hidden logits, not ``x_hat``, so
# none of these pixel-space tests apply to it.
LOSSES = {
    "pixel_l2": l2,
    "sobel": ImageFM._edge_loss,
    "line": ImageFM._line_loss,
}


def gaussian_blur(x, sigma: float):
    """Depthwise Gaussian blur of ``(B, H, W, C)`` with edge padding."""
    if sigma <= 0:
        return x
    r = int(3 * sigma + 0.5)
    k = jnp.exp(-0.5 * (jnp.arange(-r, r + 1) / sigma) ** 2)
    k = k / k.sum()
    n = x.shape[-1]
    xp = jnp.pad(x, ((0, 0), (r, r), (r, r), (0, 0)), mode="edge")

    def conv(kernel):
        return jax.lax.conv_general_dilated(
            xp,
            jnp.broadcast_to(kernel, kernel.shape[:2] + (1, n)),
            (1, 1),
            "VALID",
            feature_group_count=n,
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )

    return conv(rearrange(jnp.outer(k, k), "h w -> h w 1 1"))


def luma(x):
    """BT.601 luma of ``(B, H, W, C)``."""
    return x @ LUMA_W


def desaturate(x, amount: float):
    """Blend each pixel ``amount`` of the way towards its luma."""
    return x + amount * (luma(x)[..., None] - x)


def posterize(x, levels: int):
    """Quantise every channel to ``levels`` values (palette collapse proxy)."""
    return jnp.round((x + 1) / 2 * (levels - 1)) / (levels - 1) * 2 - 1


def erase_strokes(x):
    """Remove thin dark lines: replace luma valleys by their hard 3x3 closing.

    ``closing(L) = min3(max3(L))`` fills every valley narrower than the window
    -- exactly the pixels the ink map marks -- and leaves fill and shading
    untouched.  The RGB pixel is lifted by the luma difference so colour is
    kept.  Pixel L2 barely sees this (strokes are a few percent of pixels);
    a line-aware loss should.
    """
    lum = luma(x)

    def pool(z, op):
        zp = jnp.pad(z, ((0, 0), (1, 1), (1, 1)), mode="edge")
        h, w = z.shape[1:]
        shifts = [zp[:, i : i + h, j : j + w] for i in range(3) for j in range(3)]
        return op(jnp.stack(shifts, axis=0), axis=0)

    closing = pool(pool(lum, jnp.max), jnp.min)
    return x + (closing - lum)[..., None]


def jitter(x, dx: int = 1):
    """Shift the whole image ``dx`` pixels right (edge-replicated)."""
    xp = jnp.pad(x, ((0, 0), (0, 0), (dx, 0), (0, 0)), mode="edge")
    return xp[:, :, : x.shape[2]]


def sobel_mag(x):
    """Sobel magnitude averaged over channels, ``(B, H, W)``."""
    gx, gy = ImageFM._sobel(x)
    return jnp.sqrt(gx**2 + gy**2).mean(-1)


def ink(x):
    """Ink map, ``(B, H, W)``."""
    return ImageFM._ink(x)


def psnr(pred, target):
    """PSNR in dB on the ``[-1, 1]`` scale (peak 2)."""
    mse = ((pred - target) ** 2).mean(axis=(1, 2, 3))
    return 10 * jnp.log10(4.0 / mse)


# --------------------------------------------------------------------------
# Panel helpers
# --------------------------------------------------------------------------


def gray_panel(m, scale: float):
    """Map a non-negative ``(H, W)`` map to an RGB uint8 panel, ``scale -> 255``."""
    g = np.clip(np.asarray(m) / max(scale, 1e-6) * 255, 0, 255).astype(np.uint8)
    return np.repeat(g[..., None], 3, axis=-1)


def save_grid(
    rows: list[list[np.ndarray]],
    path: Path,
    col_labels: list[str],
    groups: list[tuple[str, int]] | None = None,
    scale: int = 2,
):
    """Write rows of equally sized RGB panels as one PNG with a labelled header.

    ``col_labels`` gives one caption per column; ``groups`` optionally adds a
    second header line of ``(label, span)`` pairs covering runs of columns.
    """
    grid = np.concatenate([np.concatenate(r, axis=1) for r in rows], axis=0)
    body = Pilimage.fromarray(grid).resize(
        (grid.shape[1] * scale, grid.shape[0] * scale), Pilimage.Resampling.NEAREST
    )
    cell = rows[0][0].shape[1] * scale
    font = ImageFont.load_default(size=11)
    line_h = 16
    header_h = line_h * (2 if groups else 1) + 4
    img = Pilimage.new("RGB", (body.width, body.height + header_h), (255, 255, 255))
    img.paste(body, (0, header_h))
    draw = ImageDraw.Draw(img)
    y = 2
    if groups:
        x = 0
        for label, span in groups:
            draw.rectangle(
                [x + 1, y, x + span * cell - 2, y + line_h - 2], fill=(225, 225, 225)
            )
            draw.text((x + 3, y), label, fill=(0, 0, 0), font=font)
            x += span * cell
        y += line_h
    for i, label in enumerate(col_labels):
        draw.text((i * cell + 3, y), label, fill=(60, 60, 60), font=font)
    img.save(path)


def table(title: str, header: list[str], rows: list[list], fmt="{:>9.3f}"):
    """Print a fixed-width table and return it as Markdown lines."""
    w = max(len(r[0]) for r in rows) + 2
    lines = [f"\n## {title}\n", "| " + " | ".join(["", *header]) + " |"]
    lines.append("|" + "---|" * (len(header) + 1))
    print(f"\n{title}")
    print(" " * w + "".join(f"{h:>10}" for h in header))
    for r in rows:
        cells = [fmt.format(v) if isinstance(v, float) else str(v) for v in r[1:]]
        print(f"{r[0]:<{w}}" + "".join(f"{c:>10}" for c in cells))
        lines.append("| " + " | ".join([r[0], *[c.strip() for c in cells]]) + " |")
    return lines


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_discrimination(x, key, n_show: int = 6):
    """Loss response to degradations, normalised by ``L(x, other image)``."""
    other = x[jax.random.permutation(key, x.shape[0])]
    degradations = {
        "identical": lambda z: z,
        "other image": lambda z: other,
        "blur 0.5": lambda z: gaussian_blur(z, 0.5),
        "blur 1.0": lambda z: gaussian_blur(z, 1.0),
        "blur 2.0": lambda z: gaussian_blur(z, 2.0),
        "noise 0.05": lambda z: z + 0.05 * jax.random.normal(key, z.shape),
        "desaturate 50%": lambda z: desaturate(z, 0.5),
        "colour shift": lambda z: z + jnp.array([0.1, -0.05, 0.05]),
        "brightness +0.2": lambda z: z + 0.2,
        "contrast 0.8": lambda z: 0.8 * z,
        "posterize 4": lambda z: posterize(z, 4),
        "h-flip": lambda z: z[:, :, ::-1],
        "erase strokes": erase_strokes,
        "jitter 1px": lambda z: jitter(z, 1),
    }
    ref = {n: float(f(x, other)) for n, f in LOSSES.items()}
    rows, panels = [], {}
    for dname, d in degradations.items():
        y = d(x)
        panels[dname] = y[:n_show]
        rows.append([dname, *[float(f(x, y)) / ref[n] for n, f in LOSSES.items()]])
    lines = table("Discrimination: L(x, degrade(x)) / L(x, other)", list(LOSSES), rows)
    lines.append(
        "\nReference L(x, other): " + ", ".join(f"{n}={v:.4f}" for n, v in ref.items())
    )

    # Panel: one row per shown image; per degradation a column triple
    # (degraded image, ink map, Sobel magnitude).
    ink_scale = float(jnp.quantile(ink(x), 0.995))
    sob_scale = float(jnp.quantile(sobel_mag(x), 0.995))
    grid = []
    for i in range(n_show):
        row = []
        for dname in degradations:
            y = panels[dname][i : i + 1]
            row += [
                to_uint8(y[0]),
                gray_panel(ink(y)[0], ink_scale),
                gray_panel(sobel_mag(y)[0], sob_scale),
            ]
        grid.append(row)
    save_grid(
        grid,
        OUTDIR / "discrimination.png",
        ["image", "ink map", "sobel mag"] * len(degradations),
        [(d, 3) for d in degradations],
    )
    lines.append(
        "\n`discrimination.png`: rows = images; per degradation, in the order of "
        "the table, a triple (image | ink map | Sobel magnitude)."
    )
    return lines


def test_descent(x, n_show: int = 6, steps: int = 200, lr: float = 0.02):
    """Recover a blurred image by gradient descent on its pixels."""
    target = x
    start = gaussian_blur(x, 1.0)
    configs = {
        "pixel_l2": {},
        "+sobel 0.1": {"sobel": 0.1},
        "+sobel 1.0": {"sobel": 1.0},
        "+line 1.0": {"line": 1.0},
        "+line 5.0": {"line": 5.0},
        "+sobel 0.1 +line 1.0": {"sobel": 0.1, "line": 1.0},
    }
    checkpoints = [25, 50, 100, steps]
    ref_ink = float(ink(target).mean())
    ref_edge = float(sobel_mag(target).mean())

    def metrics(z):
        return (
            float(psnr(z, target).mean()),
            float(ink(z).mean()) / ref_ink,
            float(sobel_mag(z).mean()) / ref_edge,
        )

    rows, finals = [], {}
    rows.append(["start (blur 1.0)", *metrics(start)])
    for cname, weights in configs.items():

        def total(z, weights=weights):
            return l2(z, target) + sum(
                w * LOSSES[n](z, target) for n, w in weights.items()
            )

        opt = optax.adam(lr)
        state = opt.init(start)
        z = start

        @jax.jit
        def step(z, state):
            g = jax.grad(total)(z)
            upd, state = opt.update(g, state)
            return optax.apply_updates(z, upd), state

        for s in range(1, steps + 1):
            z, state = step(z, state)
            if s in checkpoints:
                p, im, em = metrics(z)
                rows.append([f"{cname} @{s}", p, im, em])
        finals[cname] = z[:n_show]
    lines = table(
        "Pixel descent from blur 1.0, pixel_l2 (+aux): PSNR, ink/ref, edge/ref",
        ["psnr", "ink/ref", "edge/ref"],
        rows,
    )
    ink_scale = float(jnp.quantile(ink(target), 0.995))
    grid = []
    for i in range(n_show):
        row = [to_uint8(target[i]), to_uint8(start[i])]
        row += [to_uint8(finals[c][i]) for c in configs]
        row += [gray_panel(ink(target[i : i + 1])[0], ink_scale)]
        row += [gray_panel(ink(finals[c][i : i + 1])[0], ink_scale) for c in configs]
        grid.append(row)
    save_grid(
        grid,
        OUTDIR / "descent.png",
        ["target", "start", *configs, "target", *configs],
        [("images after descent", 2 + len(configs)), ("ink maps", 1 + len(configs))],
    )
    lines.append(
        f"\n`descent.png`: columns = target | start | final after {steps} steps "
        f"for {list(configs)} | then the ink maps of target and of each final."
    )
    return lines


def test_blurry_mean(x, model: ImageFM | None, key, n_show: int = 4):
    """How much each loss penalises the blurry conditional mean."""
    rows = []
    for sigma in [0.0, 0.5, 1.0, 2.0, 4.0]:
        y = gaussian_blur(x, sigma)
        rows.append([f"blur {sigma:.1f}", *[float(f(y, x)) for f in LOSSES.values()]])
    lines = table("Proxy: L(blur_sigma(x), x)", list(LOSSES), rows, "{:>9.4f}")
    if model is None:
        lines.append("\n(no checkpoint given; skipped the x_hat measurements)")
        return lines, None

    ts = [0.1, 0.3, 0.5, 0.7, 0.9]
    n = min(x.shape[0], 64)
    x1 = x[:n]
    x0 = jax.random.normal(key, x1.shape)
    forward = eqx.filter_jit(lambda net, xt, t: jax.vmap(lambda a, b: net(a, b))(xt, t))
    rows, panels, xhats = [], [], {}
    for t in ts:
        tb = jnp.full((n,), t)
        xt = t * x1 + (1 - t) * x0
        xh = forward(model.net_theta, xt, tb)
        xhats[t] = xh
        u = (xh - xt) / max(1 - t, model.denom_floor)
        vel = float(optax.l2_loss(u, x1 - x0).mean())
        rows.append(
            [
                f"t={t}",
                vel,
                *[float(f(xh, x1)) for f in LOSSES.values()],
                float(psnr(xh, x1).mean()),
            ]
        )
        panels.append((t, xt[:n_show], xh[:n_show]))
    lines += table(
        "Checkpoint x_hat vs x_1: velocity loss, aux losses, PSNR of x_hat",
        ["vel", *LOSSES, "psnr"],
        rows,
        "{:>9.4f}",
    )
    ink_scale = float(jnp.quantile(ink(x1), 0.995))
    grid = []
    for i in range(n_show):
        row = [to_uint8(x1[i]), gray_panel(ink(x1[i : i + 1])[0], ink_scale)]
        for _t, xt, xh in panels:
            row += [
                to_uint8(xt[i]),
                to_uint8(xh[i]),
                gray_panel(ink(xh[i : i + 1])[0], ink_scale),
            ]
        grid.append(row)
    save_grid(
        grid,
        OUTDIR / "xhat_vs_t.png",
        ["x_1", "ink(x_1)", *["x_t", "x_hat", "ink(x_hat)"] * len(ts)],
        [("data", 2), *[(f"t = {t}", 3) for t in ts]],
    )
    lines.append(
        f"\n`xhat_vs_t.png`: columns = x_1 | ink(x_1) | then for t in {ts}: "
        "(x_t | x_hat | ink(x_hat))."
    )
    return lines, xhats


def test_sampling(x, model: ImageFM | None, key, n_show: int = 4, n: int = 8):
    """Sampling-time view: x_hat along actual ODE trajectories from noise.

    Unlike ``test_blurry_mean`` (ground-truth interpolants, one query each)
    the states here are what the sampler really visits, so errors compound.
    Losses are measured against the trajectory's own final sample, since
    there is no ground truth for a generated image.
    """
    if model is None:
        return ["\n(no checkpoint given; skipped the sampling-time panel)"]

    ts = [0.1, 0.3, 0.5, 0.7, 0.9]
    t1 = 1.0 - model.t_eps
    x0 = jax.random.normal(key, (n, *x.shape[1:]))
    traj = _solve(
        model.net_theta, x0, jnp.array([*ts, t1]), t1, model.n_steps, model.denom_floor
    )  # (n, len(ts) + 1, H, W, C)
    final = traj[:, -1]
    forward = eqx.filter_jit(lambda net, xt, t: jax.vmap(lambda a, b: net(a, b))(xt, t))
    ref_ink = float(ink(x).mean())
    ref_edge = float(sobel_mag(x).mean())
    rows, panels = [], []
    for i, t in enumerate(ts):
        xt = traj[:, i]
        xh = forward(model.net_theta, xt, jnp.full((n,), t))
        rows.append(
            [
                f"t={t}",
                *[float(f(xh, final)) for f in LOSSES.values()],
                float(psnr(xh, final).mean()),
                float(ink(xh).mean()) / ref_ink,
                float(sobel_mag(xh).mean()) / ref_edge,
            ]
        )
        panels.append((t, xt[:n_show], xh[:n_show]))
    rows.append(
        [
            "final sample",
            *[0.0] * len(LOSSES),
            float("inf"),
            float(ink(final).mean()) / ref_ink,
            float(sobel_mag(final).mean()) / ref_edge,
        ]
    )
    lines = table(
        "Sampling-time x_hat(t) vs the trajectory's final sample; "
        "ink/edge mass relative to real data",
        [*LOSSES, "psnr", "ink/ref", "edge/ref"],
        rows,
        "{:>9.4f}",
    )
    ink_scale = float(jnp.quantile(ink(x), 0.995))
    grid = []
    for i in range(n_show):
        row = []
        for _t, xt, xh in panels:
            row += [
                to_uint8(xt[i]),
                to_uint8(xh[i]),
                gray_panel(ink(xh[i : i + 1])[0], ink_scale),
            ]
        row += [to_uint8(final[i]), gray_panel(ink(final[i : i + 1])[0], ink_scale)]
        grid.append(row)
    save_grid(
        grid,
        OUTDIR / "xhat_sampling.png",
        [*["x(t)", "x_hat", "ink(x_hat)"] * len(ts), "sample", "ink(sample)"],
        [*[(f"t = {t}", 3) for t in ts], (f"t = {t1:.3f}", 2)],
    )
    lines.append(
        f"\n`xhat_sampling.png`: ODE trajectories from noise; for t in {ts}: "
        "(state x(t) | x_hat predicted from it | ink(x_hat)), then the final sample."
    )
    return lines


def test_gate(x, model: ImageFM | None, key, n: int = 64):
    """Find a t-schedule for the aux terms from the loss at the optimum.

    At time ``t`` the best prediction is the blurry conditional mean, and an
    aux loss against the sharp ``x_1`` is nonzero there: that part of the
    penalty is *irreducible* and only biases the model.  The irreducible part
    is estimated by matching the checkpoint's ``x_hat(t)`` with a Gaussian
    blur of ``x_1`` at the same PSNR and reading the loss of that blur.  The
    reducible fraction ``(L - L_opt) / L`` is what a gate ``g(t)`` should
    track; a logistic ``g(t) = sigmoid((t - t0) / tau)`` is fitted to it.
    ``gate.png`` plots the raw and irreducible losses, the reducible fraction
    with candidate gates, and the effective weight ``w g(t) L(t) / vel(t)``.
    """
    if model is None:
        return ["\n(no checkpoint given; skipped the gate test)"]
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    aux = {k: v for k, v in LOSSES.items() if k != "pixel_l2"}
    weights = {"sobel": 0.1, "line": 1.0}  # current defaults / candidate
    ts = np.round(np.arange(0.05, 0.96, 0.05), 2)
    sigmas = np.arange(0.0, 8.01, 0.25)
    x1 = x[:n]
    x0 = jax.random.normal(key, x1.shape)
    forward = eqx.filter_jit(lambda net, xt, t: jax.vmap(lambda a, b: net(a, b))(xt, t))
    # PSNR and aux losses of blur_sigma(x_1) vs x_1: the "optimum" lookup.
    blur_psnr = np.array([float(psnr(gaussian_blur(x1, s), x1).mean()) for s in sigmas])
    blur_loss = {
        k: np.array([float(f(gaussian_blur(x1, s), x1)) for s in sigmas])
        for k, f in aux.items()
    }

    rows = []
    rec: dict[str, list[float]] = {"vel": [], "psnr": [], "sigma": []}
    rec.update({f"{k}_{c}": [] for k in aux for c in ("raw", "opt", "frac")})
    for t in ts:
        xt = t * x1 + (1 - t) * x0
        xh = forward(model.net_theta, xt, jnp.full((n,), float(t)))
        u = (xh - xt) / max(1 - t, model.denom_floor)
        vel = float(optax.l2_loss(u, x1 - x0).mean())
        p = float(psnr(xh, x1).mean())
        # blur_psnr decreases with sigma; interpolate sigma at the model's PSNR.
        sig = float(np.interp(-p, -blur_psnr, sigmas))
        rec["vel"].append(vel)
        rec["psnr"].append(p)
        rec["sigma"].append(sig)
        row = [f"t={t:.2f}", vel, p, sig]
        for k, f in aux.items():
            raw = float(f(xh, x1))
            opt = float(np.interp(sig, sigmas, blur_loss[k]))
            frac = max(raw - opt, 0.0) / max(raw, 1e-8)
            rec[f"{k}_raw"].append(raw)
            rec[f"{k}_opt"].append(opt)
            rec[f"{k}_frac"].append(frac)
            row += [raw, opt, frac]
        rows.append(row)
    header = ["vel", "psnr", "sigma*"]
    for k in aux:
        header += [k, f"{k} opt", f"{k} red."]
    lines = table(
        "Gate: loss of x_hat(t), of the PSNR-matched blur (irreducible), "
        "and reducible fraction",
        header,
        rows,
        "{:>9.3f}",
    )
    arr = {k: np.asarray(v) for k, v in rec.items()}

    # Fit sigmoid((t - t0) / tau) to the mean reducible fraction over losses.
    frac = np.mean([arr[f"{k}_frac"] for k in aux], axis=0)
    grid_t0 = np.arange(0.0, 1.0, 0.01)
    grid_tau = np.arange(0.01, 0.5, 0.01)
    best = min(
        (
            float(
                ((1 / (1 + np.exp(-(ts - t0) / tau))) - frac) ** 2 @ np.ones_like(ts)
            ),
            t0,
            tau,
        )
        for t0 in grid_t0
        for tau in grid_tau
    )
    _, t0, tau = best
    gates = {
        "none": np.ones_like(ts),
        "t^2": ts**2,
        "window [0.3, 0.9]": ((ts > 0.3) & (ts < 0.9)).astype(float),
    }
    if frac.max() > 0.2:
        gates[f"sigmoid(t0={t0:.2f}, tau={tau:.2f})"] = 1 / (
            1 + np.exp(-(ts - t0) / tau)
        )
        lines.append(
            f"\nFitted gate: sigmoid((t - {t0:.2f}) / {tau:.2f}) to the mean "
            f"reducible fraction of {list(aux)}."
        )
    else:
        lines.append(
            f"\nNo gate fitted: the mean reducible fraction never exceeds "
            f"{frac.max():.2f}, so there is no window worth targeting."
        )
    for k in aux:
        rows = [
            [g, *(weights[k] * gv * arr[f"{k}_raw"] / arr["vel"]).tolist()]
            for g, gv in gates.items()
        ]
        lines += table(
            f"Effective relative weight  w g(t) {k}(t) / vel(t)  (w={weights[k]})",
            [f"{t:.2f}" for t in ts],
            rows,
            "{:>9.2f}",
        )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    ax = axes[0]
    for k in aux:
        (line_,) = ax.plot(ts, arr[f"{k}_raw"], "-o", ms=3, label=f"{k}: L(x_hat, x_1)")
        ax.plot(
            ts,
            arr[f"{k}_opt"],
            "--",
            color=line_.get_color(),
            label=f"{k}: irreducible",
        )
    ax.set_xlabel("t")
    ax.set_ylabel("loss")
    ax.set_title("aux loss of x_hat(t) vs. its PSNR-matched blur")
    ax.legend(fontsize=8)
    ax = axes[1]
    for k in aux:
        ax.plot(ts, arr[f"{k}_frac"], "-o", ms=3, label=f"{k}: reducible fraction")
    for g, gv in gates.items():
        if g != "none":
            ax.plot(ts, gv, ":", label=f"gate {g}")
    ax.set_xlabel("t")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("reducible fraction and candidate gates")
    ax.legend(fontsize=8)
    ax = axes[2]
    for k in aux:
        for g, gv in gates.items():
            ax.plot(
                ts,
                weights[k] * gv * arr[f"{k}_raw"] / arr["vel"],
                "-" if g == "none" else ":",
                label=f"{k} w={weights[k]}, gate {g}",
            )
    ax.set_xlabel("t")
    ax.set_title("effective weight  w g(t) L(t) / vel(t)")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(OUTDIR / "gate.png", dpi=110)
    plt.close(fig)
    lines.append(
        "\n`gate.png`: left, raw vs irreducible aux loss per t; middle, reducible "
        "fraction with candidate gates; right, effective weight relative to the "
        "velocity term."
    )
    return lines


def test_redundancy(x, xhats: dict | None, key):
    """Mean cosine similarity between per-sample loss gradients wrt x_hat."""
    names = list(LOSSES)
    n = min(x.shape[0], 64)
    x1 = x[:n]
    points = {"blur 1.0": gaussian_blur(x1, 1.0)}
    if xhats is not None:
        points.update(
            {f"x_hat t={t}": xh for t, xh in xhats.items() if t in (0.5, 0.7)}
        )

    def per_sample_grads(z):
        def single(zi, xi, name):
            return jax.grad(lambda a: LOSSES[name](a[None], xi[None]))(zi)

        return {
            nm: jax.vmap(lambda a, b, nm=nm: single(a, b, nm))(z, x1) for nm in names
        }

    lines, mats = [], {}
    for pname, z in points.items():
        g = per_sample_grads(z)
        flat = {nm: rearrange(v, "b h w c -> b (h w c)") for nm, v in g.items()}
        unit = {
            nm: v / (jnp.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
            for nm, v in flat.items()
        }
        mat = np.array(
            [[float((unit[a] * unit[b]).sum(1).mean()) for b in names] for a in names]
        )
        mats[pname] = mat
        lines += table(
            f"Gradient cosine similarity at {pname}",
            names,
            [[a, *mat[i].tolist()] for i, a in enumerate(names)],
        )
        norms = {nm: float(jnp.linalg.norm(flat[nm], axis=1).mean()) for nm in names}
        lines.append(
            "\nMean per-sample gradient norm: "
            + ", ".join(f"{nm}={v:.4f}" for nm, v in norms.items())
        )

    # Panel: gradient direction of each loss on the first images at the first point.
    z = next(iter(points.values()))
    g = per_sample_grads(z)
    grid = []
    for i in range(4):
        row = [to_uint8(x1[i]), to_uint8(z[i])]
        for nm in names:
            gi = np.asarray(g[nm][i]).mean(-1)
            s = np.quantile(np.abs(gi), 0.995)
            # signed map: grey = 0, dark = push darker, bright = push brighter
            row.append(
                np.repeat(
                    np.clip((-gi / s + 1) * 127.5, 0, 255).astype(np.uint8)[..., None],
                    3,
                    -1,
                )
            )
        grid.append(row)
    save_grid(
        grid,
        OUTDIR / "grad_cosine.png",
        ["x_1", "current", *[f"-grad {nm}" for nm in names]],
        [(f"at {next(iter(points))}", 2 + len(names))],
    )
    lines.append(
        f"\n`grad_cosine.png`: at {next(iter(points))}: x_1 | current point | "
        f"negative gradient of {names} (bright = loss wants the pixel brighter)."
    )
    return lines


# --------------------------------------------------------------------------


def main(
    n: int = 256,
    seed: int = 0,
    checkpoint: str = "runs/exp_aux/baseline/model.eqx",
    outdir: str = str(OUTDIR),
):
    """Run every test and write ``REPORT.md`` plus the PNGs to ``outdir``."""
    global OUTDIR  # noqa: PLW0603 - simplest way to redirect every test's output
    OUTDIR = Path(outdir)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    arr = preprocess_all("./data/anime-faces")
    idx = np.sort(np.random.default_rng(seed).choice(len(arr), n, replace=False))
    x = jnp.asarray(arr[idx])
    key = jax.random.key(seed)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    model, n_params = None, 0
    if checkpoint and Path(checkpoint + ".hparams").exists():
        model = ImageFM.load(checkpoint, lambda key, **hp: UNet(**hp, key=key))
        n_params = sum(
            a.size for a in jax.tree.leaves(eqx.filter(model.net_theta, eqx.is_array))
        )
        print(f"loaded checkpoint {checkpoint} ({n_params:,} params)")
    else:
        print(f"checkpoint {checkpoint!r} not found; skipping x_hat tests")

    lines = [
        f"# Auxiliary-loss offline signal tests\n\n{n} images, seed {seed}, "
        f"checkpoint `{checkpoint if model else 'none'}`"
        + (f" ({n_params:,} params, hparams {model.hparams})" if model else "")
        + ", device "
        f"{jax.devices()[0].platform}.\n"
    ]
    lines += test_discrimination(x, k1)
    lines += test_descent(x)
    bl, xhats = test_blurry_mean(x, model, k2)
    lines += bl
    lines += test_sampling(x, model, k4)
    lines += test_gate(x, model, k5)
    lines += test_redundancy(x, xhats, k3)
    (OUTDIR / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUTDIR}/REPORT.md and PNGs")


if __name__ == "__main__":
    typer.run(main)
