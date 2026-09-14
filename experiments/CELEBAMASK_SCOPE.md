# Scope: running the recipe on CelebAMask-HQ

Status 2026-09-14: data downloaded and inspected; nothing else done. This
document is the plan and the plug-and-play audit.

## Data (on disk)

`data/celebamask-hq/CelebAMask-HQ-Dataset/resized256/` (614 MB, from the HF
mirror `limsanky/celebamask-hq-256`, the original 1024/512 data resized to 256):

- `images/{0..29999}.jpg` — 256x256 aligned faces (CelebA-HQ alignment).
- `{train,val,test}_label/<id>.png` — 256x256 label maps, 19 classes
  (24184 / 2994 / 2825; the split is the standard one, ids match `images/`).

Class presence and mean area at 64x64 (500-image sample):

| id | class | present | px @64 | id | class | present | px @64 |
|---|---|---|---|---|---|---|---|
| 0 | bg | 1.00 | 1214 | 10 | mouth (interior) | 0.55 | 10 |
| 1 | skin | 1.00 | 981 | 11 | u_lip | 1.00 | 20 |
| 2 | nose | 1.00 | 93 | 12 | l_lip | 1.00 | 34 |
| 3 | eye_g (glasses) | 0.06 | 13 | 13 | hair | 0.98 | 1227 |
| 4/5 | l_eye / r_eye | 0.98 | 12 each | 14 | hat | 0.06 | 53 |
| 6/7 | l_brow / r_brow | 0.96 | 21 each | 17 | neck | 0.96 | 178 |
| 8/9 | l_ear / r_ear | 0.5 | 20 | 18 | cloth | 0.61 | 158 |

Compared with the anime masks (face 1335 px, eyes 165, mouth 10): eyes are
**14x smaller** (24 px for both — one 16x16 cell), mouth as a whole
(10+11+12) is 6x larger, the nose is a real region (93 px). The eye
consistency problem that motivated `RegionPool` is therefore much less
visible at 64x64 here; what the layer can buy is skin-tone / hair-colour
consistency across a region, and mouth/nose placement — the same
"one aggregate per region" argument, on regions that are large enough.

## Channel mapping (4 channels, matches `cond_channels = 4`)

| ours | CelebAMask classes |
|---|---|
| 0 face | skin ∪ nose ∪ brows ∪ eyes ∪ lips ∪ mouth ∪ eye_g (1,2,3,4,5,6,7,10,11,12) |
| 1 eyes | l_eye ∪ r_eye ∪ eye_g (3,4,5) |
| 2 mouth | mouth ∪ u_lip ∪ l_lip (10,11,12) |
| 3 nose | nose (2) |

Region pooling adds `1 - face` = hair ∪ background ∪ neck ∪ cloth ∪ ears ∪
hat as the last region, as it does now. Soft masks by area-averaging the
256 binary maps to 64 (same anti-aliasing as `data/layouts.rasterize`).

No layout prior: sampling-time layouts are **real label maps** (the
user's decision). Use the val+test label maps (5819) as the evaluation
bank so no training layout is reused; FID reference stats remain the
full image set, as for anime.

## Plug-and-play audit

What works unchanged: `models/`, `training.py`, `metrics.py`, `ImageFM`
conditioning (`cond_token` is channel-count agnostic), `RegionPool` with
`cond_channels = 4` (commit `5df1325`), `RandomHorizontalFlip` /
`ColorJitter` on `(image, mask)` pairs, `experiments/cond_eval.py --mode real`,
`refine.py`, `autoguide.py`, `guidance.py` (they take masks arrays).

What is hard-wired to anime and needs a switch (all small):

| place | issue | change |
|---|---|---|
| `data/animefaces.load_all_pngs` | `rglob("*.png")` under `data_dir/images`, no resize | accept jpg, resize 256 -> 64 with `Image.LANCZOS` (area average) |
| `main.py anime` | `preprocess_all("./data/anime-faces")`, cache paths `.preprocessed/anime_faces*.npy`, `real_stats.npz` | `--data-dir` and a `--name` prefix for the three caches (or a new `celeba` command that sets them; a separate command is clearer and keeps `anime` byte-identical) |
| `main.py anime` | `--min-landmark-score` curation reads `anime_faces_landmark_scores.npy` | skip when the file is absent (`min_landmark_score = 0`) |
| `main.py anime` | eval masks from `LayoutPrior` | `--eval-mask-path`: a `.npy` bank of real masks; `LayoutPrior` only when absent |
| `experiments/cond_eval.py`, `generate_figures.py`, `bottleneck.py`, `guidance.py`, `autoguide.py`, `refine.py` | `preprocess_all("./data/anime-faces")` and `LayoutPrior` for masks | same two flags; the layout figures (`layout_to_image`, `interpolation`) draw from the mask bank instead of the prior |
| `experiments/eye_consistency.py` (iris metric) | assumes anime eye masks; eyes are 12 px here | keep as-is, report but do not optimise; the meaningful region metric here is per-region chroma spread (skin, hair) — a 20-line generalisation of the same code |
| `metrics.compute_real_stats` | cache keyed by fixed path | pass `cache_path` |
| `data/layouts.py` | landmark-specific | untouched; unused for this dataset |

Preprocessing script (new, `experiments/celebamask_masks.py`, numpy + PIL,
~60 lines): read the 30k label maps, build the `(N, 64, 64, 4)` uint8 soft
masks in image-id order, write `.preprocessed/celebamask_masks.npy` and the
val+test bank `.preprocessed/celebamask_eval_masks.npy`; ~2 min.

Estimated work: ~half a day of plumbing, mostly threading the two paths
through the scripts. Training cost is the same as anime per epoch-image
(30k vs 21.5k curated images -> ~31 s/epoch at base 64); 200 epochs ≈ 1.7 h.

## Experiment plan

1. Plain (unconditional) control, cosine 200 ep — the FID reference.
2. `cond_channels 4, region_pool 1` (the anime recipe) — real masks at eval.
3. `cond_channels 4, region_pool 0` — is pooling still needed when eyes are
   one cell? Answers whether RegionPool is an anime-eye fix or a general one.
4. Autoguidance on (2) with its own epoch-40 checkpoint as the bad model.

Metrics: FID (train stats, 5000 samples, 16 steps); mask-following via
a face parser is out of scope (no parser env) — use the region chroma
spread and the given-vs-rendered mask IoU from a simple colour threshold
only if needed; qualitative grids with the layout overlay.

## Risks

- 64x64 CelebA is a well-trodden benchmark (FID ~ 3–10 for good models,
  1e5+ images at 64 px in the literature via full CelebA); at 30k images and
  our budget expect 15–30. Compare arms to each other, not to the literature.
- Label maps at 256 come from the 512 annotations upsampled; edges are
  blocky at 64 after averaging, which is fine for conditioning.
- Glasses (6%) and hats (6%) are not in any of our channels except via
  "face" / "rest"; the model must infer them — same as anime accessories.
