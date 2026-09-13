# tinyflow task runner.  `just` lists recipes.  Everything shells out to `uv run`.
#
#   just check                      lint + type check (the CI gates)
#   just paper NAME EPOCHS EVERY    the full run: train -> evaluate -> paper figures
#   just train NAME EPOCHS [ARGS]   one training run -> runs/NAME/, FID at the end
#   just figures RUN_DIR            paper figures + showcase page for a finished run
#   just eval CKPT                  FID (prior / real layouts) + iris-mismatch rate
#   just eyes "name=ckpt" ...       iris-mismatch rate across checkpoints
#   just samples OUT "name=ckpt"... labelled sample grid from shared noise + layouts
#   just guidance CKPT [ARGS]       sampling-time edge-guidance sweep
#   just bottleneck CKPT            where/why a checkpoint falls short (P/R, error maps)
#   just fid runs/exp_x             FID table across the runs under a directory
#   just watch                      progress of every running job

set positional-arguments := true

default:
    @just --list --unsorted

# ---------------------------------------------------------------- setup / gates

# Install the locked environment (same flags as CI).
install:
    uv sync --locked --all-extras --dev

# Lint and type check -- the two CI gates.
check:
    uv run ruff check
    uv run ty check

# Auto-fix lint findings and format.
fmt:
    uv run ruff check --fix
    uv run ruff format

# ---------------------------------------------------------------- the model

# The current best recipe (experiments/METHODS.md section 7.3): 37M-parameter U-Net,
# layout-conditioned on the detector masks with region pooling, curated data.
model_args := "--base-channels 64 --cond-channels 3 --cond-dropout 0.0 --region-pool 1 --min-landmark-score 0.3"

# runs/NAME/{model.eqx,losses.json,cond_eval.json,figures/showcase.png}; ~22 s/epoch
# of training plus a 16-step FID every EVAL_EVERY epochs.  Extra `main.py anime`
# flags go after the cadence, e.g. `just paper wide_rp_300 300 50 --init-lr 5e-4`.
# Full pipeline for the current best architecture: NAME EPOCHS EVAL_EVERY [ARGS].
paper name epochs eval_every *args:
    mkdir -p runs/{{name}}
    uv run main.py anime --outpath runs/{{name}}/model.eqx --n-epochs {{epochs}} \
        --eval-every {{eval_every}} --early-stop-patience 0 --seed 49 {{model_args}} {{args}} \
        2>&1 | tee runs/{{name}}/train.log
    just eval runs/{{name}}/model.eqx
    just figures runs/{{name}}

#   just train exp/x 30 --base-channels 64     (any `main.py anime` flag after EPOCHS)
# A single training run, FID at the end only.
train name epochs *args:
    mkdir -p runs/{{name}}
    uv run main.py anime --outpath runs/{{name}}/model.eqx --n-epochs {{epochs}} \
        --eval-every {{epochs}} --early-stop-patience 0 --seed 49 {{args}} \
        2>&1 | tee runs/{{name}}/train.log

# Same as `train` but detached (use `just watch`).
train-bg name epochs *args:
    mkdir -p runs/{{name}}
    nohup uv run main.py anime --outpath runs/{{name}}/model.eqx --n-epochs {{epochs}} \
        --eval-every {{epochs}} --early-stop-patience 0 --seed 49 {{args}} \
        > runs/{{name}}/train.log 2>&1 &
    @echo "started runs/{{name}}"

# Train the 2-D toy model (fast smoke test of the whole stack).
toy:
    uv run main.py toy ./data/toycardioid.npy

# Progress bar (epochs, or the FID batch loop during an evaluation) of every running job.
watch:
    @pgrep -af "main.py anime" | grep -v pgrep | sed 's/.*--outpath \([^ ]*\).*/\1/' | sort -u | \
        while read p; do printf '%-40s ' "$p"; \
        tail -c 300 "$(dirname $p)/train.log" 2>/dev/null | tr '\r' '\n' | grep -o '^[A-Za-z]*:.*[0-9]*/[0-9]* \[[^]]*\]' | tail -1 | tr -s ' '; done
    @nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader

# ---------------------------------------------------------------- evaluation

# FID with prior-sampled and real layouts plus the iris-mismatch rate (16 sampler steps).
eval ckpt *args:
    uv run python experiments/cond_eval.py {{ckpt}} --modes prior,real --n-steps 16 {{args}}

#   just eyes "wide=runs/exp_long/wide_noaux/model.eqx" "rp=runs/exp_cond/nodrop_rp/model.eqx"
# Left/right iris-colour agreement per checkpoint ("mismatch rate") vs. real data.
eyes *arms:
    uv run python experiments/eye_consistency.py {{arms}}

#   just samples runs/cmp.png "a=runs/x/model.eqx" "b=runs/y/model.eqx"
# Labelled sample grid from shared noise (and shared layouts), one row per NAME=CHECKPOINT.
samples out *arms:
    uv run python experiments/compare_samples.py {{out}} {{arms}}

#   just guidance runs/x/model.eqx --lams 0,0.02,0.05 --intervals 0-1,0.5-1
# Sampling-time edge-guidance sweep (lambda x t-window) with FID and edge mass.
guidance ckpt *args:
    uv run python experiments/guidance.py --checkpoint {{ckpt}} {{args}}

# Where and why a checkpoint falls short: error maps, texture deficits, FID floor,
# precision/recall, worst samples, FID vs sampler steps.
bottleneck ckpt *args:
    uv run python experiments/bottleneck.py --checkpoint {{ckpt}} {{args}}

# Paper figures and the showcase page for a finished run directory.
figures run_dir *args:
    uv run python generate_figures.py {{run_dir}} {{args}}

#   just fid runs/exp_x runs/exp_y
# FID and final loss for every run under the given directories.
fid *dirs:
    #!/usr/bin/env python3
    import json, sys
    from pathlib import Path
    dirs = sys.argv[1:] or ["runs"]
    arms = []
    for d in dirs:
        arms += sorted(p.parent for p in Path(d).glob("**/losses.json"))
    for a in arms:
        r = json.load(open(a / "losses.json"))
        fids = ", ".join(f"{x['epoch']}:{x['fid']:.1f}" for x in r if "fid" in x)
        print(f"{str(a):<36} ep {len(r):>3}  loss {r[-1]['loss']:.4f}  FID {fids}")

# ---------------------------------------------------------------- data

# Re-run the face detector to (re)build landmarks + confidences (needs its own env; see script).
landmarks:
    ~/.claude/jobs/211c1fe7/tmp/det/bin/python experiments/extract_landmarks.py

# Fit the layout prior from the landmarks and write its checks to runs/ablation/prior/.
prior *args:
    uv run python experiments/landmark_prior.py {{args}}

# Delete the image and Inception-statistics caches (keyed by path only, so stale
# after any change to preprocessing); keeps the detector masks and landmarks.
clean-caches:
    rm -f .preprocessed/anime_faces.npy .preprocessed/real_stats.npz
