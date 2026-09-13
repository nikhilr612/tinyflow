"""A generative prior over face layouts, fitted to detector landmarks.

Purpose: the mask-conditioned image model needs layouts at sampling time
that come from *no real image* (otherwise "unconditional" FID would inherit
the data's layout marginal).  This fits a small closed-form model to the
28-point landmark vectors and rasterises its samples exactly the way the
training masks were made, so sampled masks are in-distribution for the
conditional model.

Parametrisation -- a point-distribution model (Cootes et al.):

* **pose**: face centroid ``(cx, cy)``, scale ``s`` = inter-ocular distance
  (between the two eye centroids), roll ``theta`` = angle of the eye line;
* **shape**: every point expressed relative to the centroid, rotated by
  ``-theta`` and divided by ``s`` -- similarity-normalised coordinates in
  which "big face" and "wide eye" are different directions.

Mirror augmentation doubles the data (``x -> -x`` and the left/right eye and
brow groups swapped), so the prior is exactly left/right symmetric.

Model: PCA on the shape vector keeping 99% of the variance, then a
Dirichlet-process Gaussian mixture (variational, ``sklearn``) on
``[pose, pca coefficients]`` -- the number of active components is inferred,
which settles the open/closed-eye multimodality from the data.  A KDE
baseline (empirical layouts + small Gaussian jitter) is fitted alongside;
the mixture has to look at least as plausible as that.

Checks written to ``runs/ablation/prior/``: ``REPORT.md`` with marginal
statistics (eye spacing, eye size, openness, mouth width) for real, DP-GMM
and KDE samples plus the fraction of samples whose hulls leave the canvas;
``masks_real.png`` / ``masks_gmm.png`` / ``masks_kde.png`` grids of
rasterised masks over the corresponding images (real) or blank (sampled);
``landmarks_mean.png`` with the mean shape and point indices.

The fitted model is saved to ``.preprocessed/landmark_prior.npz``; the
conditional pipeline samples it through ``data.layouts.LayoutPrior`` (numpy
only).  The parametrisation and rasteriser live in ``data/layouts.py``.

Usage::

    uv run python experiments/landmark_prior.py [--min-score 0.7] [--n-show 32]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib as mpl  # noqa: E402
import numpy as np  # noqa: E402
import PIL.Image as Pilimage  # noqa: E402
import typer  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.mixture import BayesianGaussianMixture  # noqa: E402

from data.animefaces import load_masks, preprocess_all, to_uint8  # noqa: E402
from data.layouts import (  # noqa: E402
    EYE_A,
    EYE_B,
    PRIOR_PATH,
    LayoutPrior,
    from_pose_shape,
    mirror,
    rasterize,
    to_pose_shape,
)

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUTDIR = Path("runs/ablation/prior")


# --------------------------------------------------------------------------
# Prior
# --------------------------------------------------------------------------


class LandmarkPrior:
    """DP-GMM over ``[pose, pca(shape)]``; samples layouts in ``[-1, 1]``."""

    def __init__(self, pca: PCA, gmm: BayesianGaussianMixture, shape_mean: np.ndarray):
        """Store the fitted pieces."""
        self.pca, self.gmm, self.shape_mean = pca, gmm, shape_mean

    @classmethod
    def fit(
        cls, lm: np.ndarray, var: float = 0.99, max_components: int = 16, seed: int = 0
    ):
        """Fit on ``(N, 28, 2)`` layouts (mirror-augmented inside)."""
        lm = np.concatenate([lm, mirror(lm)])
        pose, shape = to_pose_shape(lm)
        shape_mean = shape.mean(0)
        pca = PCA(n_components=var, svd_solver="full").fit(shape - shape_mean)
        z = np.concatenate([pose, pca.transform(shape - shape_mean)], 1)
        gmm = BayesianGaussianMixture(
            n_components=max_components,
            weight_concentration_prior_type="dirichlet_process",
            covariance_type="full",
            max_iter=500,
            random_state=seed,
        ).fit(z)
        return cls(pca, gmm, shape_mean)

    def sample(self, n: int, seed: int = 0) -> np.ndarray:
        """Draw ``n`` layouts ``(n, 28, 2)``."""
        z, _ = self.gmm.sample(n)
        z = z[np.random.default_rng(seed).permutation(n)]
        pose, coef = z[:, :4], z[:, 4:]
        shape = self.pca.inverse_transform(coef) + self.shape_mean
        return from_pose_shape(pose, shape)

    def save(self, path: Path) -> None:
        """Serialise every array the sampler needs."""
        np.savez(
            path,
            pca_components=self.pca.components_,
            pca_mean=self.pca.mean_,
            shape_mean=self.shape_mean,
            weights=self.gmm.weights_,
            means=self.gmm.means_,
            covariances=self.gmm.covariances_,
        )

    @staticmethod
    def load(path: Path = PRIOR_PATH) -> LayoutPrior:
        """The sklearn-free sampler over the saved arrays (``data.layouts``)."""
        return LayoutPrior.load(path)


class KDEPrior:
    """Baseline: a real layout plus isotropic Gaussian jitter."""

    def __init__(self, lm: np.ndarray, sigma: float = 0.01):
        """Keep the layouts; ``sigma`` is the jitter in ``[-1, 1]`` units."""
        self.lm, self.sigma = np.concatenate([lm, mirror(lm)]), sigma

    def sample(self, n: int, seed: int = 0) -> np.ndarray:
        """Draw ``n`` jittered real layouts."""
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self.lm), n, replace=False)
        return self.lm[idx] + self.sigma * rng.standard_normal((n, 28, 2))


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


def layout_stats(lm: np.ndarray) -> dict[str, np.ndarray]:
    """Interpretable per-layout measures in ``[-1, 1]`` units."""
    ea, eb = lm[:, EYE_A], lm[:, EYE_B]

    def size(e):  # corner to corner along the upper lid
        return np.linalg.norm(e[:, 2] - e[:, 0], axis=-1)

    def openness(e):  # lower-lid minus upper-lid height, over the eye width
        return (e[:, 3:6, 1].mean(1) - e[:, 0:3, 1].mean(1)) / (size(e) + 1e-6)

    return {
        "eye spacing": np.linalg.norm(eb.mean(1) - ea.mean(1), axis=-1),
        "eye size A": size(ea),
        "eye size B": size(eb),
        "openness A": openness(ea),
        "openness B": openness(eb),
        "|open A - open B|": np.abs(openness(ea) - openness(eb)),
        "mouth width": np.linalg.norm(lm[:, 26] - lm[:, 24], axis=-1),
        "face width": lm[:, 0:5, 0].max(1) - lm[:, 0:5, 0].min(1),
        "off-canvas frac": (np.abs(lm) > 1).any((1, 2)).astype(float),
    }


def mask_grid(masks: np.ndarray, images: np.ndarray | None, path: Path, cols: int = 8):
    """Overlay masks (R=face, G=eyes, B=mouth) on images or on grey; save a grid."""
    n = len(masks)
    base = (
        np.full((n, 64, 64, 3), 128, np.uint8) if images is None else to_uint8(images)
    )
    over = np.clip(base * 0.5 + masks * 127, 0, 255).astype(np.uint8)
    rows = [
        np.concatenate(list(over[r * cols : (r + 1) * cols]), 1)
        for r in range(n // cols)
    ]
    g = np.concatenate(rows, 0)
    Pilimage.fromarray(g).resize(
        (g.shape[1] * 2, g.shape[0] * 2), Pilimage.Resampling.NEAREST
    ).save(path)


def main(min_score: float = 0.7, n_show: int = 32, n_stats: int = 4000, seed: int = 0):
    """Fit the prior, run the checks, save the model."""
    OUTDIR.mkdir(parents=True, exist_ok=True)
    lm = np.load(".preprocessed/anime_faces_landmarks.npy")
    sc = np.load(".preprocessed/anime_faces_landmark_scores.npy")
    ok = ~np.isnan(lm[:, 0, 0]) & (sc.mean(1) >= min_score)
    print(f"fitting on {ok.sum()} of {len(lm)} layouts (mean score >= {min_score})")
    lm_ok = lm[ok]

    # mean shape with indices, to confirm the point layout
    mean_lm = lm_ok.mean(0)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(mean_lm[:, 0], -mean_lm[:, 1], s=12)
    for i, (x, y) in enumerate(mean_lm):
        ax.annotate(
            str(i), (x, -y), fontsize=7, xytext=(2, 2), textcoords="offset points"
        )
    ax.set_aspect("equal")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_title("mean landmark layout (y flipped for display)")
    fig.savefig(OUTDIR / "landmarks_mean.png", dpi=110)
    plt.close(fig)

    # mirror-correspondence check: with the right L/R pairing the mirrored
    # mean layout coincides with the mean layout; a wrong pairing shows up as
    # a large residual at the mis-paired indices.
    resid = np.linalg.norm(mirror(lm_ok).mean(0) - mean_lm, axis=-1)
    worst = np.argsort(resid)[::-1][:5]
    print(
        "mirror check: max per-point residual "
        f"{resid.max():.4f} at points {worst.tolist()} (median {np.median(resid):.4f})"
    )

    prior = LandmarkPrior.fit(lm_ok, seed=seed)
    prior.save(PRIOR_PATH)
    # every check below samples through the sklearn-free path used at run time
    sampler = LayoutPrior.load(PRIOR_PATH)
    active = (prior.gmm.weights_ > 0.01).sum()
    print(
        f"PCA components: {prior.pca.n_components_}; DP-GMM active components: {active}"
    )
    kde = KDEPrior(lm_ok)

    samples = {
        "real": lm_ok[np.random.default_rng(seed).choice(len(lm_ok), n_stats, False)],
        "dp-gmm": sampler.sample(n_stats, seed),
        "kde": kde.sample(n_stats, seed),
    }
    stats = {k: layout_stats(v) for k, v in samples.items()}
    lines = [
        "# Landmark prior checks\n",
        f"Fitted on {ok.sum()} layouts (mean score >= {min_score}), mirror-augmented; "
        f"PCA {prior.pca.n_components_} comps (99% var); DP-GMM {active} active of "
        f"{len(prior.gmm.weights_)} components.\n",
        "| measure | real mean / std | dp-gmm mean / std | kde mean / std |",
        "|---|---|---|---|",
    ]
    for k in stats["real"]:
        cells = [f"{stats[s][k].mean():.3f} / {stats[s][k].std():.3f}" for s in samples]
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    print("\n".join(lines[2:]))

    # grids
    idx = np.random.default_rng(seed).choice(len(lm_ok), n_show, False)
    arr = preprocess_all("./data/anime-faces")
    real_idx = np.flatnonzero(ok)[idx]
    mask_grid(rasterize(lm_ok[idx]), arr[real_idx], OUTDIR / "masks_real.png")
    mask_grid(rasterize(samples["dp-gmm"][:n_show]), None, OUTDIR / "masks_gmm.png")
    mask_grid(rasterize(samples["kde"][:n_show]), None, OUTDIR / "masks_kde.png")
    # reconstruction sanity: rasterised real landmarks vs the stored masks
    stored = load_masks()[real_idx].astype(np.float32) / 255
    ours = rasterize(lm_ok[idx])
    inter = np.minimum(stored, ours).sum((1, 2))
    union = np.maximum(stored, ours).sum((1, 2)) + 1e-6
    iou = (inter / union).mean(0)
    lines.append(
        f"\nRasteriser check: IoU of re-rasterised real landmarks vs stored masks, per "
        f"channel (face, eyes, mouth) = {iou[0]:.3f}, {iou[1]:.3f}, {iou[2]:.3f}."
    )
    print(lines[-1])
    lines.append(
        f"\nMirror-correspondence check: max per-point residual {resid.max():.4f} "
        f"(median {np.median(resid):.4f}) at points {worst.tolist()}; a large "
        "value flags a wrong left/right pairing in `mirror`."
    )
    lines.append(
        "\nGrids: `masks_real.png` (over images), `masks_gmm.png`, `masks_kde.png` "
        "(R = face, G = eyes, B = mouth); `landmarks_mean.png` for the point layout."
    )
    (OUTDIR / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"saved prior to {PRIOR_PATH}; report in {OUTDIR}/")


if __name__ == "__main__":
    typer.run(main)
