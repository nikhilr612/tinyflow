# Handoff — where things stand and what to do next

Written 2026-09-13 23:35 while the final run was still training.  Everything
below is on branch `feat/pixel-unet`; the full experiment tree (including every
dropped variant) is on `exp/aux-loss-campaign`.

## 1. What is running / where it ends up

`just paper wide_rp_300 200 50 --init-from runs/wide_rp_300_ep98.eqx` was launched
at 22:27 (third segment of the 300-epoch run; see §4).  At 23:34 it was at epoch
167/200, loss 0.0570, FID 47.8 / 34.9 / 36.2 at epochs 49 / 99 / 149 (16-step
sampler, prior layouts).  When training ends the recipe runs `just eval` and
`just figures` automatically.  Expected finish ≈ 23:55.

Outputs (all under `runs/wide_rp_300/`):

| file | what |
|---|---|
| `model.eqx`, `model.eqx.hparams` | final EMA weights (effective epoch ≈ 298) |
| `best_model.eqx` | best training-time FID checkpoint of this segment |
| `losses.json` | per-epoch loss + FID for this segment (epochs 0–199 of the segment) |
| `cond_eval.json`, `cond_eval.log` | final FID with prior / real layouts + iris-mismatch rate |
| `figures/showcase.png` + individual panels | the paper figures |
| `train.log`, `../paper.log` | logs |

Earlier segments of the same run are kept beside it: `runs/wide_rp_300_ep49.eqx`
(+ `runs/wide_rp_300_losses_collapsed.json`: the 100 epochs of the first attempt,
which diverged at epoch 80) and `runs/wide_rp_300_ep98.eqx`
(+ `runs/wide_rp_300_losses_seg2.json`: 49 epochs of the second segment).

**If it died:** `just watch` shows nothing and `runs/wide_rp_300/cond_eval.json`
is missing.  Check `runs/paper.log` and `runs/wide_rp_300/train.log` (grep for
`Error`).  A `cuSolver` error means the GPU was shared during FID — never run
anything else on the GPU while a `paper` run is going.  Resume with
`just paper wide_rp_300 <remaining> 50 --init-from runs/wide_rp_300/model.eqx`
(copy the checkpoint aside first).

## 2. Immediate to-dos (in order)

1. **Read the result**: `cat runs/wide_rp_300/cond_eval.json`, open
   `runs/wide_rp_300/figures/showcase.png`.  Reference numbers to compare with:
   200-epoch unconditioned wide model FID 32.0 (64-step) / 31.4 (16-step), iris
   mismatch 35.5 %; 40-epoch region-pool model FID 48–52, mismatch 6.3–6.6 %; real
   data mismatch 5–6 %.
2. **Run the extra evaluations** on the final checkpoint (GPU must be free):
   - `just bottleneck runs/wide_rp_300/model.eqx` → precision/recall, error maps,
     worst samples (`runs/ablation/bottleneck/REPORT.md`).  Compare recall with 0.154.
   - `just guidance runs/wide_rp_300/model.eqx --lams 0,0.02,0.05 --intervals 0-1,0.5-1`
     → guided FID (expect −2 to −4).  Report guided numbers separately.
   - `just eyes "final=runs/wide_rp_300/model.eqx" "wide200=runs/exp_long/wide_noaux/model.eqx"`
     → iris mismatch side by side.
   - Optional: eye-chroma distribution (the snippet is in the session log; ~10 lines:
     mask-weighted mean (Cb, Cr) inside the eye mask, saturation mean/std, 8-bin hue
     histogram, real vs generated).
3. **Fill in the paper**: `paper/main.tex` §13 ("Final training run → Results") has a
   red `\todo`; replace it with the numbers from step 1–2 and rebuild:
   `cd paper && latexmk -pdf main.tex && latexmk -c`.  The showcase figure is picked
   up automatically once `runs/wide_rp_300/figures/showcase.png` exists.
4. **Update `experiments/METHODS.md`**: add a §7.4 with the final-run numbers, the
   three-segment schedule story (§4 below), and the confirmation run
   (`runs/confirm_rp`: mismatch 6.25 %, FID 52.1/51.4, loss 0.0815 on the parallel
   RegionPool).  Also add the eye-chroma check (real sat 0.137±0.071, region-pool
   model 0.121±0.067; hue histograms match) under §7.3.
5. **Commit**: `git add paper experiments/METHODS.md HANDOFF.md && git commit`.
   `paper/` is untracked right now (the `.tex`, a `.gitignore` for LaTeX by-products,
   and the built PDF — commit the PDF or not as you prefer).

## 2b. Layout control — a second finding to quantify (do this early)

Visually, the region-pool model follows its layout mask far more closely than the
no-RP conditional model (`runs/exp_cond/rp_vs_nodrop.png`).  Mechanism: without RP
the layout enters only at the input and is washed out by four GroupNorm stages and
the bottleneck (the SPADE observation); RP re-injects the mask *shape* inside the
decoder at 16×16 and 32×32 (`m_k ⊗ W_k ē_k` is region-shaped), so the layer gives
both cross-region agreement (irises) and layout adherence.  Three measurements turn
this into a claim (all post-hoc on saved checkpoints; the detector needs the GPU):

1. **Mask-following score**: run `experiments/extract_landmarks.py`-style detection
   (detector env: `~/.claude/jobs/211c1fe7/tmp/det/bin/python`) on 512 samples from
   the no-RP model (`runs/exp_cond/nodrop`, archive branch code) and the RP model
   (`runs/wide_rp_300`), each rendered from known prior layouts; report mean
   landmark error and eye/face-hull IoU against the given layout, with real images
   vs their own masks as the ceiling.
2. **Layout-editing figure**: same noise, a sequence of edited layouts (eyes moved,
   spaced, face widened, mouth opened) → samples tracking the edit.  `data/layouts.py`
   has `to_pose_shape` / `from_pose_shape` / `rasterize`; edit in shape space.
3. **Change locality**: two layouts differing only in the eye region → |Δsample|
   should concentrate in the eye region for the RP model.

Add the results to METHODS.md §7.3 and the paper §12; "controllable layout" is a
stronger headline than the iris rate alone.

## 3. The codebase in one paragraph

Pixel-space flow matching with x-prediction (`models/imagefm.py`), a 4-level U-Net
(`models/unet.py`) with the legacy skip layout, conditioned on a 3-channel layout
mask (face / eyes / mouth hulls from detector landmarks) concatenated to `x_t`, plus
`RegionPool` layers at 16×16 and 32×32 that pool decoder features per region and
broadcast a zero-initialised projection back — this is what fixes iris-colour
agreement.  Layouts at sampling time come from `data/layouts.py:LayoutPrior`
(PDM + PCA + Gaussian mixture, fitted by `experiments/landmark_prior.py`, stored at
`.preprocessed/landmark_prior.npz`).  `training.py` owns bookkeeping (checkpoints,
sample PNGs, FID with a prior-mask bank at 16 sampler steps, `losses.json`).  The
loss has **no auxiliary terms** — read `experiments/METHODS.md` §1–2 before adding
any; the derivation and the ablations are there.  `CLAUDE.md` is up to date.

`just` recipes: `paper`, `train`, `train-bg`, `eval`, `eyes`, `samples`, `guidance`,
`bottleneck`, `figures`, `fid`, `watch`, `prior`, `landmarks`, `check`.

## 4. Things you need to know that are not obvious from the code

- **Learning rate.** Constant 1e-3 is at the edge of stability for the 37M model:
  three runs with added modules diverged in one epoch (mid-attention at epoch 12,
  standard skips from epoch 3, the first paper run at epoch 80: loss 0.078 → 0.284).
  The schedule (500-step warm-up, cosine to 1 % of peak) was added in response and is
  now the default.  Constant 3e-4 gives lower loss but worse 30-epoch FID; it was not
  adopted.
- **The 300-epoch run is three segments** (49 constant-LR + 49 cosine + 200 cosine),
  each resumed from the previous EMA weights with a fresh optimiser (`--init-from`).
  The LR was near its peak at both cut points, so it approximates one long cosine.
  Say so in the paper; the loss logs of all three segments are kept.
- **FID noise** at 30 epochs is ±4 (two seeds: 56.5 vs 52.7).  Loss is stable to
  ±0.0003 and is the sensitive statistic for architecture changes.  Anything within
  5 FID needs a second seed or a longer horizon.
- **16-step FID** is within 0.5 of 64-step on this model; training-time and
  screening FID use 16.  Reported "paper" numbers: state the step count.
- **Two GPU jobs at once** → OOM or cuSolver failure at the FID stage.  Don't.
- **`UNet.from_hparams`** ignores unknown hparam keys so checkpoints from the archive
  branch load; but checkpoints trained with the *sequential* RegionPool
  (`runs/exp_cond/nodrop_rp`) evaluate differently on the parallel layer (mean
  |Δx̂| = 0.008).  Load those on the archive branch if exact numbers matter.
- **The detector environment** (needed only to regenerate landmarks/masks) lives in
  `~/.claude/jobs/211c1fe7/tmp/det/` — a job scratch directory that can be deleted.
  If you need it long-term, copy it or note the recipe: `anime_face_detector`
  (hysts), `cv2`, torch+CUDA.
- **Iris-mismatch metric** is validated (flagged samples are genuine mismatches) but
  crude; it is uncorrelated with FID, so always report both.

## 5. What to try next (ranked; each is a 30–40-epoch screen on the recipe unless noted)

The measured bottleneck of the unconditioned 200-epoch model is **coverage**
(precision 0.52, recall 0.15 vs ceiling 0.77/0.77).  Re-measure on the final model
first (step 2 above); the ranking assumes it holds.

1. **Longer training with the schedule** — the curve was still falling at 200
   epochs.  Resume the final checkpoint for +200–300 epochs with a fresh cosine
   (`just paper wide_rp_600 300 50 --init-from runs/wide_rp_300/model.eqx`).
   Cheapest likely gain.
2. **Augmentation conditioning (EDM-style)** with hair/eye hue rotation — the one
   dataset-specific *coverage* prior (hair/iris colour is exchangeable in anime).
   Needs: the augmentation parameters fed to the network (concatenate to the time
   embedding) so they don't leak into samples.  Judge on recall.
3. **Spatial mixing at 16×16** (attention or a token-mixing MLP), zero-init, with
   the schedule — the mid-attention failure was optimisation, not attention.  Try as
   a fine-tune from the final checkpoint (all additions are zero-init, so the
   augmented model starts identical).
4. **Hair-colour consistency metric** (same construction as the iris metric with
   the hair/background region) to see whether region pooling also fixed hair.
5. **Flip-equivariant sampling** — `v_sym = ½[v(x) + flip(v(flip x))]`; 10-minute
   test, no training.
6. Width 128 (2× again) if compute allows; the rank analysis says width is being used.
7. **Schedule control for FID**: the final run is FID-neutral vs the plain 200-epoch
   unconditioned model so far (36.2 at cumulative epoch ~247 vs 32.0 at 200); the
   missing control is a plain unconditioned run on the same warm-up+cosine schedule
   and length (`just train ctrl_sched 300 --base-channels 64 --eval-every 50`).

Closed — do not re-run (evidence in METHODS.md): any pixel-space loss on x̂ vs x_1,
gated or not; FM re-weightings from x_1; the mask-prediction head; the global code;
standard skips; layout conditioning as a *FID* lever; palette losses; batch 256.

## 6. Paper

`paper/main.tex` (14 pages, builds clean with `latexmk -pdf`).  Sections: data,
model (layer-level), region pooling, layout prior, optimisation, protocol, aux-loss
theory (two propositions), offline tests, ablations, scaling, guidance, bottleneck,
failed variants, conditioning results, final run, decisions summary.  The only
placeholder is the final-run results paragraph.  If you extend it, the numbers in
every table are also in `experiments/METHODS.md` with the commands that produced
them.
