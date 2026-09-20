"""Vector diagrams of the model for the paper: ``uv run python paper/figures.py``.

Writes ``paper/figures/{architecture,regionpool,skips,pipeline,distill}.{pdf,png}``.
Everything is drawn from the code's actual structure (channel and resolution
ladder, skip wiring, RegionPool placement, conditioning token, sampler); keep
it in step with ``models/`` when those change.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

matplotlib.rcParams.update(
    {
        "font.family": "STIXGeneral",  # Times-compatible, matches \usepackage{times}
        "mathtext.fontset": "stix",
        "font.size": 8,
        "pdf.fonttype": 42,
        "axes.linewidth": 0.6,
    }
)
OUT = Path(__file__).parent / "figures"
INK = "#222222"
GREY = "#8a8a8a"
LIGHT = "#f2f2f2"
BLUE = "#3b6ea8"  # encoder / features
ORANGE = "#d1782c"  # decoder
GREEN = "#3d8f5f"  # region pooling
RED = "#b8433f"  # conditioning / masks
PURPLE = "#6b4fa0"  # time path
REGION_COLORS = {
    "face": "#c9b58e",
    "eyes": "#5aa9c9",
    "mouth": "#d46a6a",
    "hair/bg": "#9c9c9c",
}


def box(ax, xy, w, h, text="", fc=LIGHT, ec=INK, lw=0.7, fs=8, r=0.06, color=INK, **kw):
    """Rounded box with centred text; returns (cx, cy)."""
    x, y = xy
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={r}",
            fc=fc,
            ec=ec,
            lw=lw,
            **kw,
        )
    )
    if text:
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fs,
            color=color,
        )
    return x + w / 2, y + h / 2


def arrow(ax, p, q, color=INK, lw=0.7, style="-|>", rad=0.0, ls="-", ms=6, shrink=0):
    """Arrow from p to q, optionally curved (``rad``)."""
    a = FancyArrowPatch(
        p,
        q,
        arrowstyle=style,
        mutation_scale=ms,
        color=color,
        lw=lw,
        ls=ls,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=shrink,
        shrinkB=shrink,
    )
    ax.add_patch(a)


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    return fig, ax


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=220)
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


# ----------------------------------------------------------------- 1. architecture
def architecture():
    fig, ax = canvas(5.6, 3.9)
    levels = [(64, 64), (32, 128), (16, 256), (8, 512)]
    ys = [2.62, 1.98, 1.34, 0.70]  # top of each level's boxes
    bw, bh = 0.66, 0.30
    enc_x, dec_x = 1.42, 3.52
    enc, dec = [], []
    for i, ((res, ch), y) in enumerate(zip(levels, ys)):
        enc.append(
            box(
                ax,
                (enc_x, y),
                bw,
                bh,
                f"ResBlock\n{ch}$\\to${2 * ch}",
                fc="#e3ebf5",
                ec=BLUE,
                fs=6.2,
            )
        )
        dec.append(
            box(
                ax,
                (dec_x, y),
                bw,
                bh,
                f"ResBlock\n{2 * ch}$\\to${ch}",
                fc="#f8e8dc",
                ec=ORANGE,
                fs=6.2,
            )
        )
        ax.text(
            enc_x - 0.04,
            y + bh + 0.02,
            f"{res}$^2$",
            ha="right",
            va="bottom",
            fontsize=6.5,
            color=GREY,
        )
    # encoder chain (strided depthwise conv between levels)
    for i in range(3):
        arrow(ax, (enc[i][0], ys[i]), (enc[i][0], ys[i + 1] + bh), color=BLUE)
        ax.text(
            enc[i][0] + 0.05,
            (ys[i] + ys[i + 1] + bh) / 2,
            "2$\\times$2 stride-2\ndepthwise",
            ha="left",
            va="center",
            fontsize=5,
            color=GREY,
        )
    # 4x4 tensor, no mid block
    tx0 = (enc[3][0] + dec[3][0]) / 2
    ax.text(
        tx0,
        0.40,
        "$4^2\\times1024$ (no mid block)",
        ha="center",
        va="center",
        fontsize=6,
        color=GREY,
    )
    arrow(ax, (enc[3][0], ys[3]), (tx0 - 0.55, 0.40), color=BLUE, rad=0.3)
    arrow(ax, (tx0 + 0.55, 0.40), (dec[3][0], ys[3]), color=ORANGE, rad=0.3)
    # decoder chain with RegionPool inline after the levels that output 16^2 and 32^2
    rp_nodes = []
    for i in range(3, 0, -1):
        y_from, y_to = ys[i] + bh, ys[i - 1]
        if i in (2, 1):
            rp_y = (y_from + y_to) / 2 - 0.08
            box(
                ax,
                (dec_x + 0.05, rp_y),
                bw - 0.10,
                0.16,
                "RegionPool",
                fc="#e2f0e7",
                ec=GREEN,
                fs=5.8,
                color=GREEN,
            )
            rp_nodes.append((dec_x + bw - 0.05, rp_y + 0.08))
            arrow(ax, (dec[i][0], y_from), (dec[i][0], rp_y), color=ORANGE, ms=4)
            arrow(ax, (dec[i][0], rp_y + 0.16), (dec[i][0], y_to), color=ORANGE)
        else:
            arrow(ax, (dec[i][0], y_from), (dec[i][0], y_to), color=ORANGE)
    # skips: what enters encoder level i (pre-ResBlock) -> decoder level i after its upsample
    for i in range(4):
        y_s = ys[i] + bh + 0.06 if i == 0 else (ys[i - 1] + ys[i] + bh) / 2 + 0.02
        src = (
            (enc[i][0] + bw / 2 + 0.03, y_s)
            if i > 0
            else (enc[0][0] + bw / 2 + 0.03, y_s)
        )
        dst = (dec[i][0] - bw / 2 - 0.03, y_s)
        arrow(ax, src, dst, color=GREY, ls=(0, (3, 2)), lw=0.6)
        lab = "skip: in-conv feats" if i == 0 else f"skip: level {i - 1} out"
        ax.text(
            dst[0] - 0.02,
            y_s + 0.03,
            lab,
            ha="right",
            va="bottom",
            fontsize=5.2,
            color=GREY,
        )
    # inputs
    ix = 0.08
    box(ax, (ix, ys[0] + 0.03), 0.92, 0.24, "in-conv 3$\\times$3 $\\to$ 64", fs=5.8)
    box(
        ax,
        (ix, ys[0] - 0.48),
        0.92,
        0.36,
        "$x_t$\n$64^2\\times3$",
        fc="#f7e3e2",
        ec=RED,
        fs=6,
    )
    arrow(ax, (ix + 0.46, ys[0] - 0.12), (ix + 0.46, ys[0] + 0.03))
    arrow(ax, (ix + 0.92, ys[0] + bh / 2), (enc_x, ys[0] + bh / 2))
    ax.text(
        ix + 0.46,
        ys[0] - 0.53,
        "image only",
        ha="center",
        va="top",
        fontsize=5.5,
        color=RED,
    )
    # outputs
    ox = 4.62
    box(ax, (ox, ys[0] + 0.03), 0.90, 0.24, "out-conv 1$\\times$1 $\\to$ 3", fs=5.8)
    box(
        ax,
        (ox, ys[0] - 0.48),
        0.90,
        0.36,
        "$\\hat{x}$  (no output\nactivation)",
        fc="#f8e8dc",
        ec=ORANGE,
        fs=6,
    )
    arrow(ax, (dec_x + bw, ys[0] + bh / 2), (ox, ys[0] + bh / 2), color=ORANGE)
    arrow(ax, (ox + 0.45, ys[0] + 0.03), (ox + 0.45, ys[0] - 0.12), color=ORANGE)
    ax.text(
        ox + 0.45,
        ys[0] - 0.53,
        "$v=\\dfrac{\\hat{x}-x_t}{\\max(1-t,\\,0.05)}$",
        ha="center",
        va="top",
        fontsize=6.5,
    )
    # the masks enter a second time, straight into the RegionPool layers
    mx_, my_ = ox, 0.62
    box(ax, (mx_, my_), 0.90, 0.40, "layout masks $m$\n$64^2\\times3$, area-downsampled\nto each level", fc="#f7e3e2", ec=RED, fs=5.2)
    xr = ox - 0.10
    ax.plot([mx_, xr, xr], [my_ + 0.20, my_ + 0.20, rp_nodes[-1][1]], color=RED, lw=0.7)
    for px_, py_ in rp_nodes:
        ax.plot(xr, py_, "o", ms=2.2, color=RED)
        arrow(ax, (xr, py_), (px_, py_), color=RED, lw=0.7, ms=4)
    # time path: a shared e(t) rail down the middle, tapped by every ResBlock's own adaLN
    ty, rail = 3.45, (enc_x + bw + dec_x) / 2
    box(ax, (rail - 1.05, ty), 0.36, 0.24, "$t$", fc="#ece6f4", ec=PURPLE, fs=7)
    box(
        ax,
        (rail - 0.60, ty),
        1.20,
        0.24,
        "sinusoid($10^3\\,t$) $\\to$ MLP $=e(t)$",
        fc="#ece6f4",
        ec=PURPLE,
        fs=5.8,
    )
    arrow(ax, (rail - 0.69, ty + 0.12), (rail - 0.60, ty + 0.12), color=PURPLE)
    ax.text(
        rail + 0.68,
        ty + 0.12,
        "each adaLN: its own zero-init\nlinear $e(t)\\mapsto(1+\\gamma,\\beta)$",
        ha="left",
        va="center",
        fontsize=5.4,
        color=PURPLE,
    )
    ax.plot([rail, rail], [ty, ys[3] + bh / 2], color=PURPLE, lw=0.8)
    nw, nh = 0.30, 0.15
    for i, y in enumerate(ys):
        yc = y + bh / 2
        for x_node, x_block in (
            (enc_x + bw + 0.06, enc_x + bw),
            (dec_x - 0.06 - nw, dec_x),
        ):
            box(
                ax,
                (x_node, yc - nh / 2),
                nw,
                nh,
                "adaLN",
                fc="#ece6f4",
                ec=PURPLE,
                fs=5,
                color=PURPLE,
                lw=0.5,
            )
        ax.plot([enc_x + bw + 0.06 + nw, rail], [yc, yc], color=PURPLE, lw=0.6)
        ax.plot([rail, dec_x - 0.06 - nw], [yc, yc], color=PURPLE, lw=0.6)
        ax.plot(rail, yc, "o", ms=2.2, color=PURPLE)
        arrow(ax, (enc_x + bw + 0.06, yc), (enc_x + bw, yc), color=PURPLE, lw=0.6, ms=4)
        arrow(ax, (dec_x - 0.06, yc), (dec_x, yc), color=PURPLE, lw=0.6, ms=4)
    ax.text(
        enc_x - 0.08,
        ys[0] + bh + 0.16,
        "encoder",
        ha="right",
        va="bottom",
        fontsize=7,
        color=BLUE,
    )
    ax.text(
        dec_x + bw + 0.08,
        ys[0] + bh + 0.16,
        "decoder\n(levels joined by $\\uparrow$2 bilinear\n+ 3$\\times$3 conv)",
        ha="left",
        va="bottom",
        fontsize=6,
        color=ORANGE,
    )
    ax.text(
        2.8,
        0.12,
        "37 M parameters, GroupNorm (8 groups) throughout.",
        ha="center",
        fontsize=5.6,
        color=GREY,
    )
    save(fig, "architecture")


# ----------------------------------------------------------------- 2. region pooling
def _mask_icon(ax, x, y, s, which, color):
    """A small face-mask glyph: the region ``which`` filled in ``color``."""
    from matplotlib.patches import Ellipse, Rectangle

    ax.add_patch(Rectangle((x, y), s, s, fc="white", ec=INK, lw=0.4))
    face = Ellipse((x + s / 2, y + s * 0.48), s * 0.62, s * 0.78)
    if which == "hair/bg":
        ax.add_patch(Rectangle((x, y), s, s, fc=color, ec="none"))
        face.set(fc="white", ec="none")
        ax.add_patch(face)
    elif which == "face":
        face.set(fc=color, ec="none")
        ax.add_patch(face)
    elif which == "eyes":
        for dx in (-0.16, 0.16):
            ax.add_patch(
                Ellipse(
                    (x + s * (0.5 + dx), y + s * 0.58),
                    s * 0.16,
                    s * 0.10,
                    fc=color,
                    ec="none",
                )
            )
    else:  # mouth
        ax.add_patch(
            Ellipse((x + s / 2, y + s * 0.27), s * 0.16, s * 0.06, fc=color, ec="none")
        )


def regionpool():
    from matplotlib.patches import Circle

    fig, ax = canvas(5.6, 2.15)

    def fmap(x, y, s, label):
        for k in range(3):
            ax.add_patch(
                Polygon(
                    [
                        (x + 0.05 * k, y + 0.05 * k),
                        (x + s + 0.05 * k, y + 0.05 * k),
                        (x + s + 0.05 * k, y + s + 0.05 * k),
                        (x + 0.05 * k, y + s + 0.05 * k),
                    ],
                    closed=True,
                    fc="#dde6f1",
                    ec=INK,
                    lw=0.5,
                )
            )
        ax.text(x + s / 2 + 0.05, y - 0.07, label, ha="center", va="top", fontsize=6.5)

    mid = 1.22  # y of the main data path
    fmap(0.22, mid - 0.31, 0.58, "$h\\in\\mathbb{R}^{C\\times H\\times W}$")
    # masks: a row directly under the "masked mean" arrow, feeding up into it
    mx, my, ms_ = 0.62, 0.30, 0.22
    for j, (name, c) in enumerate(REGION_COLORS.items()):
        _mask_icon(ax, mx + 0.25 * j, my, ms_, name, c)
    ax.text(
        mx + 0.47,
        my - 0.05,
        "$m_k$: face, eyes, mouth, hair/bg",
        ha="center",
        va="top",
        fontsize=5.4,
    )
    arrow(ax, (mx + 0.47, my + ms_ + 0.02), (mx + 0.47, mid - 0.04), color=GREY, lw=0.6)
    # masked mean -> pooled vectors
    px = 1.92
    arrow(ax, (0.85, mid), (px, mid), color=GREEN, lw=0.9)
    ax.text(
        1.38,
        mid + 0.05,
        "masked mean",
        ha="center",
        va="bottom",
        fontsize=6,
        color=GREEN,
    )
    ax.text(
        1.52,
        mid - 0.07,
        "$\\bar e_k=\\dfrac{\\sum_p m_k(p)\\,h(p)}{\\sum_p m_k(p)}$",
        ha="center",
        va="top",
        fontsize=6.2,
    )
    plus = (4.12, mid)
    for j, (name, c) in enumerate(REGION_COLORS.items()):
        y = mid + 0.48 - 0.32 * j - 0.085
        ax.add_patch(
            FancyBboxPatch(
                (px, y),
                0.46,
                0.17,
                boxstyle="round,pad=0,rounding_size=0.02",
                fc=c,
                ec=INK,
                lw=0.4,
            )
        )
        r = name.split("/")[0]
        ax.text(
            px + 0.23,
            y + 0.085,
            "$\\bar e_{\\mathrm{%s}}$" % r,
            ha="center",
            va="center",
            fontsize=6,
        )
        box(
            ax,
            (px + 0.72, y - 0.01),
            0.80,
            0.19,
            "$W_{\\mathrm{%s}}\\bar e_{\\mathrm{%s}}+b_{\\mathrm{%s}}$" % (r, r, r),
            fc=LIGHT,
            ec=GREEN,
            fs=5.6,
        )
        arrow(ax, (px + 0.46, y + 0.085), (px + 0.72, y + 0.085), color=GREEN)
        arrow(ax, (px + 1.52, y + 0.085), (plus[0] - 0.09, plus[1]), color=c, lw=0.8)
    # sum node and output
    ax.add_patch(Circle(plus, 0.09, fc="white", ec=INK, lw=0.7))
    ax.text(plus[0], plus[1], "+", ha="center", va="center", fontsize=9)
    ax.text(
        plus[0] + 0.05,
        mid - 0.16,
        "$\\sum_k m_k\\otimes(W_k\\bar e_k+b_k)$\nbroadcast into region",
        ha="center",
        va="top",
        fontsize=5.4,
    )
    fmap(4.80, mid - 0.31, 0.58, "$h' = h + \\sum_k m_k\\otimes$\n$(W_k\\bar e_k+b_k)$")
    arrow(ax, (plus[0] + 0.09, mid), (4.80, mid), color=INK)
    # residual: straight line along the top, from h's stack to the sum node
    top = mid + 0.62
    ax.plot(
        [0.55, 0.55, plus[0], plus[0]],
        [mid + 0.37, top, top, plus[1] + 0.09],
        color=GREY,
        lw=0.6,
        solid_capstyle="round",
    )
    arrow(ax, (plus[0], top - 0.2), (plus[0], plus[1] + 0.09), color=GREY, lw=0.6, ms=5)
    ax.text(
        3.1, top + 0.03, "residual", ha="center", va="bottom", fontsize=6, color=GREY
    )
    ax.text(
        2.85,
        0.10,
        "$W_k$: one $C\\times C$ matrix per region, zero-initialised (the layer is the identity at init).  "
        "After the decoder levels at $16^2$ and $32^2$; both irises are painted from the same $\\bar e_{\\mathrm{eyes}}$.",
        ha="center",
        va="center",
        fontsize=5.5,
        color=INK,
    )
    save(fig, "regionpool")


# ----------------------------------------------------------------- 3. skip wiring
def skips():
    fig, ax = canvas(5.6, 2.15)

    def unet(x0, title, shifted):
        ys = [1.55, 1.15, 0.75]
        bw, bh = 0.42, 0.22
        ex, dx = x0 + 0.35, x0 + 1.55
        E, D = [], []
        for i, y in enumerate(ys):
            E.append(
                box(ax, (ex, y), bw, bh, f"enc$_{i}$", fc="#e3ebf5", ec=BLUE, fs=6.5)
            )
            D.append(
                box(ax, (dx, y), bw, bh, f"dec$_{i}$", fc="#f8e8dc", ec=ORANGE, fs=6.5)
            )
        for i in range(2):
            arrow(
                ax,
                (E[i][0], E[i][1] - bh / 2),
                (E[i + 1][0], E[i + 1][1] + bh / 2),
                color=BLUE,
            )
            arrow(
                ax,
                (D[i + 1][0], D[i + 1][1] + bh / 2),
                (D[i][0], D[i][1] - bh / 2),
                color=ORANGE,
            )
        arrow(
            ax,
            (E[2][0], E[2][1] - bh / 2),
            (D[2][0], D[2][1] - bh / 2),
            color=GREY,
            rad=0.5,
        )
        ax.text(
            (ex + dx + bw) / 2,
            0.36,
            "bottleneck",
            ha="center",
            fontsize=5.5,
            color=GREY,
        )
        box(ax, (ex, 1.95), bw, 0.16, "in-conv", fs=5.5)
        arrow(ax, (E[0][0], 1.95), (E[0][0], ys[0] + bh))
        for i in range(3):
            if shifted:
                # skip taken *before* enc_i, delivered to dec_i after its upsample
                y_src = (
                    1.95 - 0.02
                    if i == 0
                    else (E[i - 1][1] - bh / 2 + E[i][1] + bh / 2) / 2
                )
                src = (E[i][0] + bw / 2 + 0.02, y_src)
                dst = (D[i][0] - bw / 2 - 0.02, y_src)
                lab = "in-conv features" if i == 0 else f"enc$_{i - 1}$ output"
            else:
                src = (E[i][0] + bw / 2 + 0.02, E[i][1])
                dst = (D[i][0] - bw / 2 - 0.02, D[i][1])
                lab = f"enc$_{i}$ output"
            arrow(ax, src, dst, color=RED if shifted else GREEN, ls=(0, (3, 2)), lw=0.7)
            ax.text(
                (src[0] + dst[0]) / 2,
                src[1] + 0.035,
                lab,
                ha="center",
                va="bottom",
                fontsize=5,
                color=GREY,
            )
        ax.text(x0 + 1.15, 0.12, title, ha="center", fontsize=7)

    unet(0.15, "standard: post-encoder $\\to$ pre-decoder", shifted=False)
    unet(
        3.05,
        "this work: pre-encoder $\\to$ post-upsample (shifted one level)",
        shifted=True,
    )
    save(fig, "skips")


# ----------------------------------------------------------------- 4. pipeline
def pipeline():
    fig, ax = canvas(5.6, 3.0)
    w, h = 1.0, 0.44

    def row(y, items, color, ec):
        cs = []
        for x, txt in items:
            cs.append(box(ax, (x, y), w, h, txt, fc=color, ec=ec, fs=5.8))
        for a, b in zip(cs[:-1], cs[1:]):
            arrow(ax, (a[0] + w / 2, a[1]), (b[0] - w / 2, b[1]), color=ec)
        return cs

    # branch A: the flow model, then its distillation
    ya = 2.30
    ax.text(
        0.10,
        ya + h + 0.05,
        "A. generator",
        ha="left",
        va="bottom",
        fontsize=6.5,
        color=BLUE,
    )
    A = row(
        ya,
        [
            (0.10, "training images\n+ detected layouts"),
            (1.35, "flow matching,\nx-prediction\n(U-Net + RegionPool)"),
            (2.60, "velocity model\n$\\hat x(x_t, m, t)$"),
            (3.85, "distil the midpoint-8\nsampler (fixed teacher)"),
        ],
        "#e3ebf5",
        BLUE,
    )
    box(
        ax,
        (5.05, ya),
        0.48,
        h,
        "flow map\n$u_\\theta$",
        fc="#f8e8dc",
        ec=ORANGE,
        fs=5.8,
    )
    arrow(ax, (A[-1][0] + w / 2, A[-1][1]), (5.05, ya + h / 2), color=BLUE)
    # branch B: the layout prior
    yb = 1.52
    ax.text(
        0.10,
        yb + h + 0.05,
        "B. layout prior",
        ha="left",
        va="bottom",
        fontsize=6.5,
        color=GREEN,
    )
    B = row(
        yb,
        [
            (0.10, "detected layouts\n(28 landmarks)"),
            (1.35, "PDM: pose\n$(c_x, c_y, \\log s, \\theta)$ + shape"),
            (2.60, "PCA on shape\n(99 % variance)"),
            (3.85, "Gaussian mixture\nover [pose, PCA]"),
        ],
        "#e2f0e7",
        GREEN,
    )
    box(ax, (5.05, yb), 0.48, h, "prior", fc="#e2f0e7", ec=GREEN, fs=6)
    arrow(ax, (B[-1][0] + w / 2, B[-1][1]), (5.05, yb + h / 2), color=GREEN)
    # inference
    yi = 0.55
    ax.text(
        0.10,
        yi + h + 0.05,
        "C. inference",
        ha="left",
        va="bottom",
        fontsize=6.5,
        color=ORANGE,
    )
    C = row(
        yi,
        [
            (0.10, "sample a layout\nfrom the prior"),
            (1.35, "(optional) edit pose,\neyes, mouth"),
            (2.60, "rasterise hulls\n$\\to$ masks $m$"),
            (3.85, "1–2 jumps:\n$x_0 + u_\\theta(x_0, m, 0, 1)$"),
        ],
        "#f8e8dc",
        ORANGE,
    )
    box(ax, (5.05, yi), 0.48, h, "image", fc="#f8e8dc", ec=ORANGE, fs=6)
    arrow(ax, (C[-1][0] + w / 2, C[-1][1]), (5.05, yi + h / 2), color=ORANGE)

    # what feeds inference: elbow routes in the gap between rows B and C
    def elbow(pts, color):
        ax.plot(
            [q[0] for q in pts],
            [q[1] for q in pts],
            color=color,
            lw=0.7,
            ls=(0, (2, 2)),
        )
        arrow(ax, pts[-2], pts[-1], color=color, lw=0.7, ms=5)

    elbow([(5.29, yb), (5.29, 1.32), (0.95, 1.32), (0.95, yi + h)], GREEN)
    elbow(
        [
            (5.53, ya + h / 2),
            (5.57, ya + h / 2),
            (5.57, 1.22),
            (4.35, 1.22),
            (4.35, yi + h),
        ],
        ORANGE,
    )
    ax.text(
        2.8,
        0.14,
        "A and B are fitted independently from the same detected layouts.  Only C runs at inference time: no real image is involved.",
        ha="center",
        fontsize=5.6,
        color=GREY,
    )
    save(fig, "pipeline")


# ----------------------------------------------------------------- 5. distillation
def distill():
    import numpy as np

    fig, ax = canvas(5.6, 2.55)
    x0, x1 = (0.5, 1.0), (5.1, 1.0)
    ts = np.linspace(0, 1, 200)
    px = x0[0] + (x1[0] - x0[0]) * ts
    py = x0[1] + 1.05 * np.sin(np.pi * ts) * (1 - 0.3 * ts)
    xh = (px[99], py[99])
    ax.plot(px, py, color=GREY, lw=1.0)
    for k in range(9):
        i = int(k / 8 * 199)
        ax.plot(px[i], py[i], "o", ms=2.6, color="white", mec=GREY, mew=0.6)
    ax.text(
        2.8,
        2.38,
        "teacher: midpoint sampler, 8 steps, on the velocity model",
        ha="center",
        fontsize=6,
        color=GREY,
    )
    for p_, lab, dy in (
        (x0, "$x_0$ (noise)", -0.13),
        (xh, "$x_{1/2}$", 0.1),
        (x1, "$x_1$", -0.13),
    ):
        ax.plot(*p_, "o", ms=4.5, color=INK)
        ax.text(
            p_[0],
            p_[1] + dy,
            lab,
            ha="center",
            va="top" if dy < 0 else "bottom",
            fontsize=7,
        )

    def jump(p, q, lab, rad, color, dy):
        arrow(ax, p, q, color=color, lw=1.1, rad=rad, ms=8)
        mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
        ax.text(
            mx,
            my + dy,
            lab,
            ha="center",
            va="center",
            fontsize=6.2,
            color=color,
            bbox={"fc": "white", "ec": "none", "pad": 1},
        )

    # A jump is a straight displacement: chords against the curved teacher path.
    jump(
        x0, x1, "1 jump:  $x_1 \\approx x_0 + u_\\theta(x_0, 0, 1)$", 0.0, ORANGE, -0.14
    )
    jump(
        x0,
        xh,
        "2 jumps:  $x_0+\\frac{1}{2}u_\\theta(x_0,0,\\frac{1}{2})$",
        0.0,
        GREEN,
        -0.16,
    )
    jump(
        xh,
        x1,
        "$x_{1/2}+\\frac{1}{2}u_\\theta(x_{1/2},\\frac{1}{2},1)$",
        0.0,
        GREEN,
        -0.16,
    )
    ax.text(
        2.8,
        0.30,
        r"$\mathcal{L}=\mathbb{E}\,\|\,x_t+(s-t)\,u_\theta(x_t,m,t,s)-x_s^{\mathrm{teacher}}\|^2,"
        r"\quad (t,s)\in\{(0,1),\,(0,\frac{1}{2}),\,(\frac{1}{2},1)\}$",
        ha="center",
        va="center",
        fontsize=7,
    )
    save(fig, "distill")


if __name__ == "__main__":
    architecture()
    regionpool()
    skips()
    pipeline()
    distill()
