# Methods and findings: inductive biases for anime-face flow matching

Dates: 2026-09-12 / 13.  Hardware: one RTX 4090 (24 GB).  Code: this repository at
the commit that includes this file; every experiment is reproducible from the
commands in each section (`just` recipes are in the root `justfile`).

This document is the record of one campaign: starting from a flow-matching image
model with two auxiliary "anime" losses (Sobel edge, dark-line), we asked whether
dataset-specific inductive biases help, found that *training-time pixel losses
cannot* (and why), located the model's actual failure modes, and found one
structural change that fixes a measured defect outright.  Negative results are
kept deliberately so they are not re-run.

Sections: 0 Setup · 1 Auxiliary-loss theory and tests · 2 Training ablations of
the losses · 3 Scaling, LR, and noise floor · 4 Sampling-time guidance · 5 Where
the model fails (bottleneck analysis) · 6 Architecture variants · 7 Layout masks:
conditioning and region pooling · 8 Auxiliary measurements · 9 Summary of what
is open and closed.

---

## 0. Setup and notation

**Data.** 21 551 anime faces, 64×64×3, values in `[-1, 1]` (`data/animefaces.py`).
Augmentation: horizontal flip (p = 0.5), colour jitter.  Semantic masks
`(N, 64, 64, 3)` = convex hulls of `hysts/anime-face-detector` landmarks (channel
0 face, 1 both eyes, 2 mouth); 28 landmarks per face with per-point confidences
(`experiments/extract_landmarks.py`).  **Curation** (§8.1): images with mean
landmark confidence < 0.3 (317, 1.5 %) are dropped from *training* where stated;
FID reference statistics always use the full set.

**Model.** `models/imagefm.py` + `models/unet.py`.  Flow matching with
x-prediction:

    x_t = t x_1 + (1 − t) x_0,   x_0 ~ N(0, I),   t ~ logit-normal(−0.8, 1)
    x̂  = net(x_t, t)
    u   = (x̂ − x_t) / max(1 − t, 0.05)
    L   = E ½ ‖u − (x_1 − x_0)‖²  =  E ½ ‖x̂ − x_1‖² / max(1 − t, 0.05)²

U-Net: 4 levels, `base_channels` 32 ("9M", 9.38 M params) or 64 ("wide", 36.7 M),
adaLN time conditioning (zero-initialised, no gate), legacy skip layout (§6.1).
Adam, LR 1e-3 constant unless stated, batch 128 (168 steps / epoch), EMA 0.999.
Sampler: Dopri5, 64 fixed steps unless stated.

**Metrics.** FID (`fidax`, Inception pool features) on 5000 samples against
statistics of the full data set; run-to-run noise at 30 epochs ≈ ±4 (§3.3).
Precision / recall (Kynkäänniemi et al. 2019, k = 3, 5000 vs 5000).  *Iris
mismatch rate*: per sample, the distance between the mean chroma `(Cb, Cr)` of the
left and right halves of the dataset-mean eye mask; "mismatched" = above the real
data's 95th percentile (0.1027), so real data scores ≈ 5 %
(`experiments/eye_consistency.py`; validated visually in §8.3).  Local-detail
proxies: mean Sobel magnitude and mean ink map of samples relative to real.

**Protocol caveat that applies to every 30/40-epoch number.** Screening runs are
5–7 k steps.  Seed-to-seed FID spread at this horizon is ±4; learning-rate 3e-4
gives *lower* loss but *worse* FID than 1e-3 here (§3.2); and effects can reverse
with length (§2.4).  Differences under ~5 FID at this horizon are not evidence.

---

## 1. Auxiliary losses on `x̂`: theory and offline tests

Losses under test (`ImageFM._edge_loss`, `ImageFM._line_loss`):

| name | definition |
|---|---|
| `sobel` | ½ (mean\|Δgₓ\| + mean\|Δg_y\|), `g = Sobel(x)` per channel, **L1** (gradient-difference loss, Mathieu et al. 2016) |
| `line` | mean \|ink(x̂) − ink(x_1)\|, `ink = relu(softclose₃ₓ₃(luma) − luma)`, β = 10 |
| `pixel_l2` | ½ MSE — reference; it is `L` without the `t`-weight |

Harness: `experiments/aux_signal.py` (`just signal CKPT`); 256 images, seed 0;
checkpoint `runs/exp_aux/baseline` (9M, 100 epochs, trained *with* `sobel 0.1`).

### 1.1 Discrimination — do the losses measure what they claim?

`L(x, d(x)) / L(x, x')`, `x'` a random other image (1.0 = "as wrong as an unrelated image").

| degradation | pixel_l2 | sobel | line |
|---|---|---|---|
| blur σ=1 | 0.064 | 0.379 | 0.552 |
| noise σ=0.05 | 0.005 | 0.112 | 0.103 |
| desaturate 50 % | 0.019 | 0.075 | 0.000 |
| brightness +0.2 | 0.073 | 0.000 | 0.000 |
| erase strokes (3×3 closing of luma) | 0.097 | 0.355 | 0.550 |
| jitter 1 px | 0.206 | 0.625 | 0.812 |

Both losses are 5–6× more sensitive than L2 to stroke erasure and 1-px
misplacement and blind to global colour.  They measure what they claim.

### 1.2 Pixel-space descent

From `blur₁(x)`, 200 Adam steps on the pixels minimising `pixel_l2 + w·L_aux`:
L2 alone recovers ink mass by step 25 and 113 dB PSNR by step 200; every aux term
is slower (sobel 0.1: 60 dB, line 1.0: 47 dB).  With a known target the aux terms
carry no information L2 lacks.

### 1.3 The minimiser argument (why these losses cannot help)

The population minimiser of `L` at every `(x_t, t)` is `x*(t) = E[x_1 | x_t]`.
Any aux loss `L_aux(x̂, x_1)` evaluated at `x*` is > 0 and **irreducible**: no
network can reduce it without leaving the FM optimum.  Only
`L_aux(x̂) − L_aux(x*)` is usable signal.

*Estimate of the irreducible part* (`test_gate`): match the checkpoint's `x̂(t)`
with a Gaussian blur of `x_1` at equal PSNR; the blur's aux loss approximates
`L_aux(x*)`.  Reducible fraction = `max(L(x̂) − L(blur), 0) / L(x̂)`, 64 pairs,
`t` on a 0.05 grid:

| t | σ* | sobel: x̂ / blur | line: x̂ / blur | reducible sobel / line |
|---|---|---|---|---|
| 0.10 | 8.0 | 0.807 / 0.820 | 0.099 / 0.101 | 0 / 0 |
| 0.30 | 2.5 | 0.677 / 0.744 | 0.090 / 0.098 | 0 / 0 |
| 0.50 | 1.1 | 0.496 / 0.516 | 0.071 / 0.085 | 0 / 0 |
| 0.65 | 0.7 | 0.367 / 0.344 | 0.054 / 0.068 | 0.06 / 0 |
| 0.85 | 0.5 | 0.205 / 0.184 | 0.029 / 0.041 | 0.10 / 0 |

≥ 90 % of both penalties is unreachable at every `t`; no `t`-window contains
signal worth gating for (the logistic gate fit degenerates).  With default
weights the aux penalty is 0.6–1.1× the velocity loss for `t < 0.5`.

*Why bias, not noise:* `sobel` is L1 on gradients, whose minimiser is a
per-pixel **median**; under uncertainty the median of a signed gradient is nearer
zero than the mean, so minimising an irreducible L1 penalty rewards flatter
predictions (hedging).  `line` is L1 on a nonlinear map and biased for the same
and a second reason.  Gradient geometry (`test_redundancy`): cos(∇L2, ∇sobel) ≈
0.65, cos(∇L2, ∇line) ≈ 0.40; the sobel-L1 gradient is a saturated ±1 stripe
pattern (`Sobelᵀ · sign`), visible in `grad_cosine.png`.

Caveats: a Gaussian blur is not `E[x_1|x_t]` (likely *over*-estimates the
irreducible part); the checkpoint was trained with `sobel 0.1`.  Both are why
§2 exists.

### 1.4 Weighted losses: the general statement

For a per-pixel weight `w` and prediction `c` at fixed `x_t`:

    J(c) = E[w (c − Y)² | x_t],  ∇J = 0  ⇒  c* = E[wY | x_t] / E[w | x_t]
         = E[Y | x_t] + Cov(w, Y | x_t) / E[w | x_t]

`c* = E[Y | x_t]` **iff** `w` is conditionally independent of `Y` given `x_t`,
i.e. a function of `(t, x_t)` only (Lipman et al. 2022, Thm 2, "any positive
λ(t)").  Equivalently (condition on the pair instead): a weight `w(x_0, x_1)` is
the unweighted CFM loss under the re-weighted coupling `p̃ ∝ w·p`, so the sampler
transports noise to the data distribution *tilted by `w`*.  The per-sample
gradient factoring `W ⊙ r` is correct but does not make the original optimum
stationary, because `E[w r | x_t] = Cov(w, r | x_t) ≠ 0` for stochastic
couplings (`r = 0` for all samples is unattainable).

---

## 2. Training ablations of the losses

All 9M unless stated; seed 49; FID every 10 epochs; `runs/exp_edge`, `runs/exp_fix`.

### 2.1 Shipped default vs off (paired, 40 epochs)

| epoch | `edge_weight 0.0` | `edge_weight 0.1` (shipped default) |
|---|---|---|
| 9 | 125.0 | 125.7 |
| 19 | 88.0 | 89.5 |
| 29 | 76.0 | 82.7 |
| 39 | **68.0** | 74.7 |

Sampling proxies (8 trajectories): ink mass / real 0.79 vs 0.68, edge mass / real
0.86 vs 0.76 — the model trained *with* the edge loss generates fewer edges.
On training-time interpolants the two are within 1.5 % (`sobel(x̂,x_1)` at t=0.1:
0.821 vs 0.808).  Epoch-0 loss 0.180 vs 0.249: the term was ≈ 0.8× the
velocity loss.  **The shipped loss is harmful; default changed to 0.**

### 2.2 Variants (40 epochs)

| arm | FID @39 | ink / real | edge / real | classification (§1.4) |
|---|---|---|---|---|
| no aux (reference) | 68.0 | 0.79 | 0.86 | — |
| `sobel_l2` 0.05 (L2 norm) | 68.0 | – | – | unbiased (Sobel linear) — inert |
| `fm_edge` 1.0 (FM weight `1 + \|Sobel(x_1)\|/mean`) | 65.3 | 0.89 | 0.91 | biased (tilt) |
| `fm_edge_sg` 1.0 (weight from `stop_grad(x̂)`) | 65.3 | 0.79 | 0.87 | unbiased |
| `fm_ink` 1.0 | 97.0 | | | biased; sparse map → ~50× per-pixel |
| `fm_eye` 1.0 / 0.2 (eye-mask weight) | 117.7 / 76.0 | | | ~unbiased; ~20× / 5× per-pixel |

`fm_edge` and `fm_edge_sg` end identical: the covariance/tilt term has no FID
effect (it shows only in the proxies: edgier samples).  Sparse-map weights are
harmful at every coefficient tried.  Iris mismatch (measured later, §8.3):
no-aux 41.8 %, `fm_eye` 0.2: 37.5 %, `fm_eye` 1.0: 32.8 %, `fm_edge` 43.0 % —
a dose-dependent effect on the eye metric at a prohibitive FID cost.

### 2.3 Mask-head auxiliary (`aux_weight`, BCE on hidden logits)

9M, 100 epochs: best FID 53.5 (0) / 57.6 (0.25) / 69.7 (1.0) — monotonically
worse.  Wide, 30 epochs: 62.4 vs 56.5.  The capacity hypothesis is not supported.

### 2.4 At length (wide, 200 epochs, FID every 50)

| run | 49 | 99 | 149 | 199 |
|---|---|---|---|---|
| `wide_noaux` | 43.1 | 36.3 | 36.5 | **32.0** |
| `wide_fm_edge_sg` | 42.9 | 37.7 | 35.9 | 33.4 |

The one loss-side candidate is a wash at length.  **The pixel-space aux family is
closed**: every form (L1, L2, ink, mask-weighted, FM re-weightings, gated) at 9M
and 37M, 30–200 epochs.

---

## 3. Scaling, learning rate, noise floor

### 3.1 Width and length

| model | epochs | FID |
|---|---|---|
| 9M | 40 | 68.0 |
| 9M | 100 | 53.5 (best) |
| 37M | 30 | 56.5 |
| 37M | 200 | 32.0 (still falling: 36.5 → 32.0 over the last 50) |

Rank analysis (§8.5): the wide model's layers use ≈ 1.9× the absolute effective
rank of the 9M model's (mean eff. rank / max 0.88 vs 0.85) — width is consumed,
not idle.

### 3.2 Learning rate (wide, 30 epochs, curated)

LR 3e-4 vs 1e-3: loss 0.0870 vs 0.0890 (lower), FID 63.9 vs 56.5 (worse).  Same
pattern for `skip1` (0.0874 / 63.6).  At this horizon the training loss and FID
disagree about the LR; the 200-epoch runs used 1e-3.  An LR *schedule* (warm-up
+ cosine decay) was not tested and is the obvious next lever for length.

### 3.3 Noise floor and evaluation cost

Seed 49 vs 50, wide, 30 epochs: 56.5 vs 52.7 (loss 0.0890 vs 0.0887).  **±4 FID;
loss is stable to ±0.0003 and is the sensitive statistic for architecture
changes.**  Sampler steps (wide_noaux): 16 → 31.6, 32 → 31.4, 64 → 32.0 — the
sampler is not limiting and 16-step FID is used from §7 on.  Training batch 256
is *slower* per epoch than 128 (39.9 vs 29.0 s): the GPU is saturated at 128.
Two concurrent jobs need `XLA_PYTHON_CLIENT_MEM_FRACTION=0.47` or the FID stage
OOMs.

---

## 4. Sampling-time guidance (`experiments/guidance.py`)

Unary energies on `x̂` (no `x_1`), gradient through the network, RMS-normalised:

    g = ∂E(x̂(x_t,t))/∂x_t,   x̂' = x̂ − λ·g/rms(g),   v = (x̂' − x_t)/max(1−t, floor)

`E_edge = relu(m_real − mean|Sobel(x̂)|)²` (one-sided deficit vs the data's mean
edge mass 1.30), `E_ink` likewise (0.101), `E_eye = mean(m_eye (x̂ − hflip x̂)²)`.

| model | λ (edge) | FID | edge/real | ink/real |
|---|---|---|---|---|
| 9M 40ep | 0 | 67.4 | 0.91 | 0.84 |
| | 0.02 | **59.0** | 1.11 | 1.05 |
| | 0.05 / 0.10 / 0.20 | 59.4 / 62.1 / 69.5 | 1.22 / 1.32 / 1.38 | |
| wide 200ep | 0 | 31.4 | 0.89 | 0.82 |
| | 0.02 | **28.1** | 1.11 | 1.07 |
| | 0.05 | 28.4 | 1.22 | 1.15 |
| | ink 0.02 | 28.5 | 1.11 | 1.18 |

Interval gating (9M, edge; Kynkäänniemi et al. 2024): λ=0.02 on [0,0.5] 61.4;
[0.3,0.9] 59.1; [0.5,1] 59.7; λ=0.05 on [0.5,1] **57.8** (best), [0.3,0.9] 58.2.
Guidance early in `t` hurts; a middle/late window tolerates a larger λ.  Effect on
precision/recall (§5): precision 0.52 → 0.58, recall 0.15 → 0.15 — a
mode-seeking correction, which bounds its value at a few FID.


### 4.1 Edge guidance on the region-pool model (`runs/ablation/guidance_final`)

`wide_rp_300/best_model.eqx`, prior layouts, 5000 samples, 16 steps:

| λ | window | FID | edge/real |
|---|---|---|---|
| 0 | — | 34.6 | 0.90 |
| 0.02 | [0, 1] | **32.5** | 1.11 |
| 0.02 | [0.5, 1] | 33.6 | 1.05 |
| 0.05 | [0, 1] | 33.0 | 1.24 |
| 0.05 | [0.5, 1] | 32.8 | 1.09 |

The same −2 to −3 FID as on the plain model; the late window no longer helps.
Precision / recall on this checkpoint (§5): 0.471 / 0.125 plain, 0.485 / 0.108
guided — again precision up, recall down.

### 4.2 Autoguidance (`experiments/autoguide.py`) — **positive result**

Karras et al. 2024: guide the model with a worse version of itself,

    v = v_good + w · (v_good − v_bad),   optionally only for t_lo < t < t_hi

where `v_bad` is a checkpoint of the same architecture that is less trained.  Both
velocities see the same layout mask.  No training, no external energy; cost is one
extra forward pass per ODE evaluation.  Good = `wide_rp_300/best_model.eqx`
(unguided 34.6), 5000 samples, 16 steps, prior layouts unless stated:

| bad model | w | window | FID | edge/real |
|---|---|---|---|---|
| `confirm_rp` (same recipe, 40 ep, FID 52) | 0.5 | [0, 1] | 24.9 | |
| | 0.75 | [0, 1] | 22.0 | 1.01 |
| | **1.0** | **[0, 1]** | **21.2** | 1.08 |
| | 1.25 | [0, 1] | 23.3 | 1.15 |
| | 1.5 | [0, 1] | 29.7 | 1.23 |
| | 2.0 | [0, 1] | 56.9 | |
| | 1.0 | [0.3, 1] | 23.1 | |
| `wide_rp_300_ep49` (same run, epoch 49) | 0.5 / 1.0 / 2.0 | [0, 1] | 28.9 / 26.5 / 51.6 | |

Checks at the best setting (w = 1, full window, bad = `confirm_rp`): 64 sampler
steps 21.2 (unchanged); real layouts 20.8 (vs 34.0 unguided: the prior-vs-real
gap, 0.5 FID on this checkpoint, is unchanged); adding one refinement round (§7.5, t0 = 0.85) *hurts*,
25.2.  On `wide_rp_200/best_model.eqx` (unguided 33.2) with bad =
`confirm_rp/best_model.eqx`: **18.1**.

Readings.  (i) −13 FID is 3–4× the gain of any energy guidance tried and takes the
region-pool model from 3.5 FID *behind* the schedule-matched plain control (§7.4:
29.3) to 11 FID *ahead* of it.  (ii) The bad model must be bad in the right way:
a 40-epoch checkpoint of the same recipe (a separately trained, under-fitted
model) is a far better guide than epoch 49 of the *same* run, which shares the
good model's initialisation and early trajectory.  (iii) w = 1 is the optimum
and the response is sharp above it (1.5 → 29.7, 2 → collapse); the t-window
restriction that helped energy guidance hurts here.  (iv) Edge mass rises from
0.90 to 1.08 of real at the optimum — the guidance is adding back the hedged
detail that the bottleneck analysis (§5) found missing, which is the mechanism
the paper describes.  (v) Reproduced on a second dataset (§10: 27.4 → 18.6).
Not yet measured under autoguidance: iris mismatch, precision / recall,
mask-following.

    just autoguide runs/wide_rp_300/best_model.eqx runs/confirm_rp/model.eqx runs/ablation/autoguide/fine16 --ws 0.75,1,1.25,1.5

---

## 5. Where the model fails (`experiments/bottleneck.py`, `wide_noaux`)

**FID floor** (5000 real vs real stats): 2.1.

**Precision / recall** (k = 3): plain 0.521 / **0.154**; edge guidance 0.581 /
0.147; real vs real ceiling 0.774 / 0.765.  **Coverage is the dominant gap**:
85 % of real images have no generated neighbour.  The worst samples by nearest
real feature are washed-out, low-contrast images with incoherent hair / clothing
/ background — averages of modes the model has not learned, not broken faces.

**On the region-pool model** (`wide_rp_300/best_model.eqx`,
`runs/ablation/bottleneck_final/REPORT.md`): precision 0.471 / recall 0.125
(plain sampler), 0.485 / 0.108 with edge guidance λ = 0.02; the same coverage
picture as the unconditioned model, slightly worse on both axes at a similar FID.
Region-relative errors at t = 0.3: face 0.32, eyes 0.30, mouth 0.33, hair/bg 0.23
(vs 0.41 / 0.43 / 0.50 / 0.25 unconditioned): the layout removes most of the
*where* uncertainty, evenly across regions.  Generated-vs-real local statistics
are −9 to −12 % on edge mass, luma std and chroma std in every region — the same
hedging, now uniform.

**Prediction error by region** (512 pairs; error ÷ per-pixel data variance):

| t | face | eyes | mouth | hair/bg |
|---|---|---|---|---|
| 0.3 | 0.41 | 0.43 | **0.50** | 0.25 |
| 0.5 | 0.20 | 0.19 | 0.28 | 0.12 |
| 0.7 | 0.086 | 0.076 | 0.129 | 0.048 |

Raw error is largest at the eyes and hair; *relative* to what is predictable,
the mouth is recovered worst (small, low-variance, high-frequency features get
the least loss weight).

**Generated vs real local statistics** (1024 each, relative difference): edge
mass −9 to −12 %, luma std −9 to −13 % (worst at the mouth / nose centre: −30 to
−40 % in the two central cells), chroma std −16 to −21 % everywhere.  Colour
statistics (§8.2) agree: means correct, chroma spread −13 %, histogram tails
thin, palette size *larger* than real (118 vs 101 colours / image).

---

## 6. Architecture variants

### 6.1 Skip connections (`skip_mode`)

Legacy layout: the skip at level `i+1` is level `i`'s encoder ResBlock output
after the strided conv, i.e. the standard pattern shifted down one level (top
decoder level gets `in_conv` features; each skip is half the standard width).
`skip_mode=1` = standard.  9M, 30 ep: 76.0 vs 76.0 (neutral).  Wide, 30 ep, LR
1e-3: **188.0** — an optimisation collapse (loss flat at 0.108 from epoch 3,
samples frozen, tiled texture); LR 3e-4: 63.6 with loss 0.0874 < 0.0890 — the
collapse is an LR effect, and at 3e-4 the layout is no better than legacy.
Legacy layout retained (documented TODO stays).

### 6.2 Mid-block ResBlock → [attention] → ResBlock (`mid_block`, `mid_attention`)

9M, 40 ep: 78.0 / 78.7 vs 77.1 (neutral).  Wide, 200 ep, LR 1e-3: 75.8 with a
loss spike to 0.635 at epoch 12 and a plateau above baseline — optimisation
failure, not a verdict on attention.  Iris mismatch 23.0 % (best of the
unconditioned models; §8.3).  Needs warm-up / lower LR to be judged; not rerun.

### 6.3 Global code (`global_code`)

`g = W₂ silu(W₁ LN(mean_hw z) + b₁) + b₂`, `W₂ = 0` at init, `t_dec = t_emb + g`
for the decoder (ADM/DiT recipe).  Wide, 30 ep, LR 3e-4, paired: loss 0.0869 vs
0.0870, FID 63.8 vs 63.9, iris mismatch 43.8 % vs 42.6 %.  The path *did*
switch on (`‖g‖` ≈ 0.6–0.8 `‖t_emb‖`, image-dependent) and changed nothing: the
decoder is not short of global *information*; the defect is in how two regions
are *rendered* (→ §7.3).

---

## 7. Layout masks: conditioning and region pooling

### 7.1 Landmark prior (`experiments/landmark_prior.py`)

Point-distribution model: pose (centroid, log inter-ocular distance, roll) +
similarity-normalised shape; mirror augmentation with the verified point
permutation; PCA (32 comps, 99 %) + Dirichlet-process GMM (16 active).  Fitted
on 20 519 layouts (mean confidence ≥ 0.7).  Samples match real marginals: eye
spacing 0.768 / 0.769, eye size 0.367 / 0.36–0.38, openness 0.58 / 0.60,
left–right openness difference 0.131 / 0.137, off-canvas 0.1 % / 0.3 %.
Rasteriser reproduces the stored masks (IoU face 0.99, eyes 0.95).

### 7.2 Conditioning (`cond_channels=3`)

`[x_t, mask, indicator]` concatenated at the input (`cond_token`), UNet
`in_channels 7 → out 3`; sampling with prior masks, real masks, or the null token.
Wide, curated:

| run | ep | loss | FID uncond | FID prior masks | FID real masks | mismatch (prior) |
|---|---|---|---|---|---|---|
| dropout 0.15 | 30 | 0.0837 | 62.6 | 57.8 | 57.2 | 29.7 % |
| no dropout | 40 | 0.0813 | — | 50.7 | 50.1 | 33.6 % |
| *unconditional refs* | 30 | 0.089 | 56.5 / 52.7 / 53.2 | | | ~29 % |

Layout information lowers the loss but not FID (real and prior masks agree, so
the prior is not the limitation) and does not touch iris agreement.  Dropout
costs ≈ 7 FID at this horizon.

### 7.3 Mask-guided region pooling (`region_pool=1`) — **positive result**

`RegionPool` after the up blocks producing 16×16 and 32×32 features; regions =
face, eyes, mouth, hair/background (= 1 − face), masks area-downsampled to the
level; for each region `k`:

    h ← h + m_k ⊙ W_k · mean_{m_k}(h),    mean_m(h) = Σ_p m(p) h(p) / Σ_p m(p)

`W_k` a zero-initialised 1×1 conv (identity at init; +5 k params at width 8,
negligible at width 64).  Null-token masks contribute nothing.  Same recipe as
"no dropout" above (wide, curated, 40 epochs, LR 1e-3, seed 49, 16-step FID):

| 40 ep, pure conditional | loss | FID prior / real masks | iris mismatch prior / real |
|---|---|---|---|
| no region pool | 0.0813 | 50.7 / 50.1 | 33.6 % / 38.7 % |
| **region pool** | **0.0807** | **48.2 / 48.0** | **6.6 % / 5.1 %** |
| real data | | | 5.0–5.9 % |

Iris mismatch falls to the real data's rate at slightly better FID and loss.
Verified visually with paired samples (same noise, same masks): the faces are
identical pair-for-pair and only the iris pairs change, becoming matched and
more saturated (`runs/exp_cond/rp_vs_nodrop.png`).  Mechanism: both irises are
rendered from one pooled feature at the resolution where hue is decided —
consistent with §6.3 (information was not missing) and with `fm_eye` (§2.2:
more gradient on the eyes helps a little; a shared computation fixes it).

Reproduction:

    uv run main.py anime --outpath runs/exp_cond/nodrop_rp/model.eqx --base-channels 64 \
        --edge-weight 0.0 --min-landmark-score 0.3 --cond-channels 3 --cond-dropout 0.0 \
        --region-pool 1 --n-epochs 40 --eval-every 0 --early-stop-patience 0 --seed 49
    uv run python experiments/cond_eval.py runs/exp_cond/nodrop_rp/model.eqx --modes prior,real --n-steps 16

Eye-region chroma of the region-pool model vs real (256 samples): saturation
0.121 ± 0.067 vs 0.137 ± 0.071; Cb/Cr std 85–95 % of real; the 8-bin hue histogram
has the same dominant band (32 % in both).  The layer enforces *agreement*, not a
particular colour.

**Mask-following** (`experiments/mask_following.py`; 512 prior layouts rendered
by each model, detector re-run on the samples, error against the layout the image
was rendered from, in `[-1, 1]` units; real images vs their own landmarks are the
ceiling):

| model | landmark err (all / eyes / nose / mouth) | detector conf. nose / mouth | IoU face / eyes |
|---|---|---|---|
| conditional, no RP (40 ep) | 0.052 / 0.027 / 0.101 / 0.062 | 0.57 / 0.71 | 0.918 / 0.830 |
| **conditional + RP** (`wide_rp_300/best`) | **0.046 / 0.026 / 0.091 / 0.053** | **0.64 / 0.75** | **0.931 / 0.850** |
| real | 0 | 0.82 / 0.85 | 1 |

Both models follow the layout closely (100 % detection; eye error 0.027 ≈ 0.9 px
at 64 px); region pooling tightens nose and mouth placement by ~10 % and raises the
detector's confidence in those features, but the remaining gap to real on nose /
mouth legibility (0.64 vs 0.82) is the larger effect and is not a layout problem.

**Definition change.** The first implementation updated `h` sequentially over
regions (later regions pooled features already offset by earlier ones; regions
overlap).  The parallel, order-independent form above differs by mean |Δx̂| =
0.008 (1.6 % of |x̂|) on the 40-epoch checkpoint; re-confirmed on the parallel
form (`runs/confirm_rp`, same recipe): loss 0.0815, FID 52.1 / 51.4, iris
mismatch **6.25 %** / 5.1 %.

### 7.4 Final run (`runs/wide_rp_300`): 300 epochs, three segments

Recipe: wide, curated, pure conditional (`cond_channels 3`, no dropout),
`region_pool 1`, batch 128, EMA 0.999, 16-step FID every 50 epochs with prior
layouts.  Course: the first attempt at constant LR 1e-3 diverged in one epoch at
epoch 80 (loss 0.0784 → 0.2837; FID 42.0 at epoch 49); a warm-up + cosine schedule
(500 steps, to 1 % of peak) was introduced and the run resumed from the epoch-49
EMA weights with a fresh optimiser (`--init-from`); the second segment (49 epochs,
loss 0.0770) was killed at its first FID by a cuSolver failure caused by a
concurrent GPU job; the third segment ran 200 epochs from those weights.
Checkpoints and loss logs of every segment are kept (`runs/wide_rp_300_ep49.eqx`,
`_ep98.eqx`, `wide_rp_300_losses_collapsed.json`, `_seg2.json`).

| cumulative epoch | plain unconditioned wide (constant LR) | this run |
|---|---|---|
| ~50 | 43.1 | 42.0 |
| ~150 | 36.5 | 47.8 (just after a warm restart) |
| ~200 | **32.0** | **34.9** (`best_model.eqx`) |
| ~250 | — | 36.2 |
| ~300 | — | 37.5 |

Final evaluation (16 steps, 5000 samples): **FID 37.8 prior layouts / 34.2 real
layouts; iris mismatch 4.7 % / 4.7 %** (real 5.0 %; mean L/R chroma distance 0.039
vs 0.037 real); final loss 0.0557.  Layout adherence is visibly strong
(`figures/layout_to_image.png`).

Readings.  (i) The eye result holds at length and reaches the data's own rate.
(ii) FID is not better than the plain model: ≈6 behind at the end, ≈3.5 at the
best point; the turning point is near 200 cumulative epochs (~33 k steps), after
which FID drifts up while the training loss keeps falling (0.077 → 0.056).  Note
that FID reference statistics are the *training* images, so this drift is not
memorisation of training images (that would lower FID); it is samples moving away
from, or collapsing within, the training distribution — precision/recall on this
checkpoint was not measured (the guided stage of `bottleneck.py` OOMs at batch
256; `bs=64` fixed in `9656891`, rerun pending).  (iii) The prior-vs-real layout
gap (3.5 FID), absent at 40 epochs, opened as adherence sharpened: a residual
mismatch between the prior's layouts and the data's becomes an FID cost once the
model follows layouts closely.  (iv) Confounds: two warm restarts; no
schedule-matched unconditioned control.

**The two confounds, resolved** (both runs in the overnight queue of 2026-09-14,
warm-up + cosine, 200 epochs in one segment, FID every 25 with prior layouts):

| epoch | `ctrl_sched` — plain unconditioned, same schedule | `wide_rp_200` — cond + RP, the recipe |
|---|---|---|
| 24 | 64.1 | 58.9 |
| 49 | 38.2 | 37.9 |
| 74 | 30.6 | 35.0 |
| 99 | 30.8 | 33.1 |
| 124 | 29.1 | **32.9** (`best_model.eqx`) |
| 149 | 29.5 | 34.5 |
| 174 | 29.5 | 36.5 |
| 199 | **29.3** | 37.8 |
| final loss | 0.0789 | 0.0537 |

`wide_rp_200/best_model.eqx` evaluated: FID 32.9 prior / 31.4 real layouts, iris
mismatch 4.3 % / 4.3 %.  So: (i) the schedule alone is worth ~2.7 FID on the
plain model (32.0 → 29.3) and its curve is flat from epoch 75 on; (ii) the
one-segment recipe reproduces the three-segment run (best 32.9 vs 34.9, final
37.8 vs 37.8) — the warm restarts were not the cause of the drift; (iii) the
conditioned model's drift after epoch ~125 is real and specific to it: its
training loss keeps falling (0.065 → 0.054) while the control's is flat and its
FID is flat.  The conditioned + region-pool model is therefore **3.5 FID behind
the matched plain model at its best point and 8.5 behind at the end**, at 4.3 %
iris mismatch vs 35 %.  The drift is consistent with the reading in §7.3 /
`CONTEXT.md`: the model's reliance on the region-pool broadcast grows with
training (its contribution to x̂ triples), so it renders the prior's imperfect
layouts more literally — the prior-vs-real gap is 1.5 FID at the best checkpoint.
Autoguidance (§4.2) more than recovers the deficit (18.1 on this checkpoint).

**Recipe going forward:** `just paper NAME 200 25` (one segment, cosine ending at
200, FID every 25), report `best_model.eqx` and the final; never chain warm
restarts.

**Presentation upscalers — tried and dropped.** Real-ESRGAN (x4plus general,
anime 6B, animevideov3) and APISR (RRDB, GRL, DAT) were run on the final samples
(`runs/upscale/compare_models.png`).  Trained on clean-then-degraded pairs, they
treat every irregularity of a 64×64 sample as signal and render the generator's
own errors as confident detail; the native 64×64 samples read better.  No figure
in the paper.

### 7.5 Re-noise-and-re-solve refinement (`experiments/refine.py`)

A finished sample is re-noised to `t0` (`x_{t0} = t0·x + (1 − t0)·ε`) and the ODE
solved again from `t0` with the same mask (SDEdit on the model's own output; one
round of Restart sampling).  `wide_rp_300/best_model.eqx`, 16 steps per pass,
prior layouts:

| t0 | 0 (reference) | 0.5 | 0.7 | 0.85 |
|---|---|---|---|---|
| FID | 34.6 | 33.8 | 32.8 | **31.8** |
| edge/real | 0.90 | 0.88 | 0.89 | 0.89 |

−3 FID at t0 = 0.85, monotone in t0 (re-deciding only late detail is best), edge
mass unchanged — so unlike guidance it is not adding contrast; it is a second
draw of the high-frequency detail.  Does **not** stack with autoguidance (§4.2:
21.2 → 25.2 with one round at t0 = 0.85 — the guided field is already off the
model's own manifold, and refining pulls it back).  Kept as the in-distribution
alternative to external upscalers (§7.4), superseded by autoguidance for FID.

---

## 8. Auxiliary measurements

**8.1 Curation.** Mean landmark confidence < 0.3 is non-faces (torsos, hands,
duplicated crops); 0.3–0.65 is legitimate hard faces (closed eyes, glasses,
masks).  Training on the 98.5 % kept: wide 30 ep FID 53.2, loss 0.0875 (vs 56.5 /
52.7, 0.0890) — neutral to slightly positive; kept on.

**8.2 Colour** (`experiments/color_stats.py`, `wide_noaux` 200 ep vs real, 1000
each): Cb/Cr means −0.039/0.096 vs −0.044/0.097; skin-region chroma −0.062/0.089
vs −0.070/0.098; saturation 0.266 vs 0.280; chroma std 0.109/0.115 vs
0.124/0.136; chroma-histogram L1 0.249 vs a real-vs-real floor of 0.116.  Colour
*means* are right; *spread* is 13 % narrow.  A palette loss has no target.

**8.3 Iris mismatch rate** (256 samples, threshold = real p95):

| checkpoint | mismatch |
|---|---|
| real | 5.1 % (2nd half 5.9 %) |
| 9M no-aux 40 ep | 41.8 % |
| wide 30 ep | 28.9 % |
| wide 200 ep (`wide_noaux`, FID 32) | 35.5 % |
| wide midattn 200 ep | 23.0 % |
| wide LR 3e-4 30 ep (ref / global code) | 42.6 % / 43.8 % |
| conditional, no dropout, 40 ep | 33.6 % |
| **conditional + region pool, 40 ep** | **6.6 %** |

Validated: the 32 most-flagged `wide_noaux` samples are all genuine red/blue,
yellow/blue, purple/green pairs; the least-flagged are matched
(`runs/ablation/eye_flagged.png`).  The metric is uncorrelated with FID.

**8.4 Guided-sampler cost.** One backward pass per ODE evaluation; ≈ 2× the
plain sampler.

**8.5 Weight spectra.** Effective rank (entropy of normalised singular values) of
every conv, reshaped `(out, in·k·k)`: mean eff. rank / max 0.877 (9M 40 ep),
0.853 (wide 30 ep), 0.825 (wide 200 ep); mean stable rank 13.5 / 13.8 / 17.5.
Rank scales with width (capacity is used); training sharpens a few directions
while spreading energy into the mid-spectrum.

---

## 9. State of the questions

**Closed (with evidence above).** Pixel-space auxiliary losses on `x̂` in any
form; gating them; mask-head auxiliary; global code; palette losses; layout
conditioning as a FID lever; sampler steps; batch 256; eye-weighted FM loss.

**Positive.** Removing the shipped edge loss (+7 FID); width (53 → 32);
warm-up + cosine schedule (32.0 → 29.3 on the plain model, and stability);
curation (small); edge guidance λ≈0.02 (−2 to −8 FID); refinement at t0 = 0.85
(−3); **mask-guided region pooling (iris mismatch 35 % → 4.3–4.7 %, at a 3.5-FID
cost against the matched control)**; **autoguidance (−13 FID on the region-pool
model, −9 on CelebA-64; the best lever found)**.

**Open, ranked.** (1) Autoguidance is under-explored: which "bad" model
(epoch, width, data fraction) guides best — a 40-epoch separately-trained
checkpoint beats an early checkpoint of the same run by 5 FID; whether the
region-pool model's iris rate, mask-following and recall survive w = 1; and
whether the plain model gains as much (if not, conditioning + RP + autoguidance
is the recipe; if so, autoguidance is orthogonal).  (2) The conditioned model's
late-training drift: a better layout prior (real landmarks + jitter) or early
stopping at the FID-optimal checkpoint.  (3) EDM-style augmentation
conditioning, including hair/eye hue rotation (a dataset-specific coverage
prior).  (4) Spatial mixing at 16×16 with warm-up, as a fine-tune from the
strong checkpoint.  (5) Hair-colour consistency metric for the region-pool
layer.  (6) Flip-equivariant sampling (free test).  (7) Not pursued: CelebAMask-HQ at
128 px (§10 — runs with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`, dropped as a cost
class above the question).

**Protocol from here.** Loss + FID (16 steps) + recall + iris mismatch on every
arm; screening only for collapse; anything within 5 FID needs a second seed or
a longer horizon.

---

## 10. Second dataset: CelebAMask-HQ (`data/celebamask.py`, `just celeba`)

Does the recipe transfer to real faces?  30 000 CelebA-HQ images (the 256 px
mirror) with 19-class label maps, area-averaged to 64×64 and to 128×128;
4-channel masks from label unions (face = skin ∪ nose ∪ brows ∪ eyes ∪ lips ∪
mouth ∪ glasses; eyes; mouth = lips ∪ interior; nose), region pooling adds
1 − face as before.  Curation drops ~1.8 % (duplicate thumbnails, empty eye /
mouth / nose masks, mis-aligned faces, greyscale; `celebamask_keep*.npy`),
26 487 kept.  No layout prior is fitted: sampling-time layouts are the real label
maps of a held-out bank (the last 3000 ids, never trained on), so the "real
layouts" mode of `cond_eval.py` is the only one.  FID reference statistics are
the full 30 000 images.  Details and the plug-and-play audit:
`experiments/CELEBAMASK_SCOPE.md`.

At 64 px the eyes are one 16×16 cell (24 px for both, 14× smaller than anime
eyes), so the iris mechanism is barely testable; the 128 px arrays are the real
test.

**64 px** (`--base-channels 64 --cond-channels 4 --region-pool 1`, curated, batch
128, 16-step FID on real held-out layouts):

| run | LR schedule | epochs | FID (ep) | final loss |
|---|---|---|---|---|
| `celeba_rp_100` | constant 1e-3 | 100 | 41.1 (24) → **diverged at ep 25** (loss 0.043 → 0.285; 228 / 196 / 194 after) | 0.065 |
| `celeba_rp_120` | 5e-4 peak, 2000-step warm-up, cosine to 1 % | 120 | 39.9 (39) / **27.3** (79) / 27.3 (119) | **0.036** |

The same divergence as the anime run at constant 1e-3, at a smaller loss and
earlier (ep 25); the `just celeba` recipe bakes in the lower peak and the longer
warm-up.  FID 27.3 is flat from epoch 80 (no late drift — with real layouts there
is no prior mismatch to render).

**Autoguidance on CelebA-64** (`runs/celeba_autoguide`, good =
`celeba_rp_120/best_model.eqx`, bad = `celeba_rp_100/best_model.eqx` — the
epoch-24 checkpoint of the diverged run, FID 41; real layouts, 16 steps):

| w | 0 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|
| FID | 27.4 | **18.6** | 19.1 | 20.3 |
| edge/real | 0.96 | 1.07 | 1.10 | 1.13 |

−9 FID, optimum slightly below w = 1, edge mass again crossing real at the
optimum: the anime result (§4.2) reproduces on a second dataset with a different
mask source and a different "bad" model.

**128 px** (`just celeba128`: `--image-size 128 --n-blocks 5`, RegionPool placed by
resolution) — **not pursued.**  The first launch (2026-09-14) failed at the first
training step with `RESOURCE_EXHAUSTED`: the compiled train step needs 17.1 GiB of
activations at batch 128 and JAX's default 75 % preallocation of the 24 GB card is
17.15 GiB.  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` fixes it (verified 2026-09-20:
one epoch trains, ≈4 min, and the 16-step FID loop runs at `--fid-batch-size 64`),
but the run was stopped by decision: the 64 px result already shows the recipe and
autoguidance transfer, and a 128 px model is a different cost class (≈10× per epoch)
for a question — iris agreement on real faces — that the 64 px anime result
already answers.

---

## 11. 128 px — tried, not pursued (2026-09-20)

**Upscaling the anime sources.**  The dataset is native 64×64, so a 128 px model
needs super-resolved targets.  `experiments/upscale_anime.py` (standalone RRDBNet,
runs in the detector env) compared Lanczos, Real-ESRGAN anime-6B, Real-ESRGAN
x4plus and APISR-RRDB on training images at ×4 then area-mean to 128
(`runs/upscale/dataset_preview.png`, `_zoom.png`): **anime-6B and x4plus are clean
and faithful** (sharper line-art, iris highlights and hair strands preserved, no
hallucinated texture, colours unchanged); APISR over-darkens outlines and shifts
contrast; Lanczos blurs.  The opposite verdict from §7.4 on *generated* samples,
for the same reason: clean sources are on-distribution for these networks,
generated samples are not.  `build --model=anime6b` writes
`.preprocessed/anime_faces_128_anime6b.npy` (kept, 1 GB).

**Plumbing** (`main.py --image-size 128`, committed): training masks are
`layouts.rasterize(landmarks, 128)` — verified to reproduce the stored 64 px masks
to within 0.2 % of pixels — and the prior bank is rasterised at 128
(`LayoutPrior.sample_masks(size=)`); Inception stats cached at
`anime_stats_128.npz`; `eye_consistency.py --image-size 128` scores at this size.
Note: the nose disc in `rasterize` has a pixel-unit radius, so channel 3 shrinks
4× at 128 — irrelevant for `cond_channels 3`, fix before using the nose at 128.
FID at 128 would be against the *upscaler's rendering* of the data and must be
labelled as such; iris mismatch and paired samples are the honest metrics.

**Screen pair, RP on/off, 40 epochs, batch 128** (`runs/chain_anime128_bs128_oom.log`):
both arms OOM on the 24 GB card — the compiled train step needs 16.05 GiB and the
per-epoch checkpoint / sample-PNG churn pushes the allocator over it even at
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` (the RP arm at step 0, the no-RP arm at
epoch 4 after 166 s/epoch, loss 0.075).  Batch 64 fits but at ≈3+ min/epoch is
a different cost class; **dropped by decision** along with CelebA-128 (§10, whose
arrays were deleted the same day; regenerable with `data/celebamask.py --size 128`).

