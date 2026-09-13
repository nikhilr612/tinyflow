# tinyflow task runner.  `just` lists recipes; `just <recipe> --help`-style docs are
# the comments above each one.  Everything shells out to `uv run`.
#
# The experiment loop this encodes:
#   just check                      lint + type check (the CI gates)
#   just train NAME EPOCHS [ARGS]   one training arm -> runs/NAME/, FID at the end
#   just queue EPOCHS A "ARGS" B …  arms one after another (best for the 37M model)
#   just pair EPOCHS A "ARGS" B …   two arms concurrently, memory-capped (9M model)
#   just watch                      progress of every running arm
#   just fid runs/exp_x             FID table across the arms of an experiment
#   just samples OUT ARMS...        labelled sample grid across checkpoints
#   just signal CKPT [OUT]          offline aux-loss signal tests + panels
#   just colors CKPT                colour statistics vs. real data
#   just guidance CKPT [ARGS]       sampling-time guidance sweep
#   just figures                    paper figures from ./runs

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

# ---------------------------------------------------------------- training

# Train the 2-D toy model (fast smoke test of the whole stack).
toy:
    uv run main.py toy ./data/toycardioid.npy

# Output goes to runs/NAME/{model.eqx,losses.json,sample_epoch_*.png,train.log}.
# FID once, at the end (screening only needs the final ranking; each evaluation
# of the 37M model costs ~10 min).  Seed 49 matches experiments/METHODS.md.
# One screening arm: `just train exp_x/edge0 30 --edge-weight 0.0`  (NAME EPOCHS [ARGS])
train name epochs *args:
    mkdir -p runs/{{name}}
    uv run main.py anime --outpath runs/{{name}}/model.eqx --n-epochs {{epochs}} \
        --eval-every {{epochs}} --early-stop-patience 0 --seed 49 {{args}} \
        2>&1 | tee runs/{{name}}/train.log

# Same as `train` but detached (returns immediately; use `just watch`).
train-bg name epochs *args:
    mkdir -p runs/{{name}}
    nohup uv run main.py anime --outpath runs/{{name}}/model.eqx --n-epochs {{epochs}} \
        --eval-every {{epochs}} --early-stop-patience 0 --seed 49 {{args}} \
        > runs/{{name}}/train.log 2>&1 &
    @echo "started runs/{{name}} (pid $!)"

# Run several arms one after another, e.g. overnight:
#   just queue 30 exp_x/a "--edge-weight 0.0" exp_x/b "--edge-weight 0.1" ...
# Sequential arms, final-FID only; NAME ARGS pairs after EPOCHS.
queue epochs *pairs:
    #!/usr/bin/env bash
    set -e
    args=("$@"); epochs=${args[0]}; args=("${args[@]:1}")
    for ((i = 0; i < ${#args[@]}; i += 2)); do
        name=${args[i]}; extra=${args[i+1]}
        mkdir -p runs/$name
        uv run main.py anime --outpath runs/$name/model.eqx --n-epochs $epochs \
            --eval-every $epochs --early-stop-patience 0 --seed 49 $extra \
            > runs/$name/train.log 2>&1
    done
    just fid $(for ((i = 0; i < ${#args[@]}; i += 2)); do echo runs/${args[i]}; done)

#   just pair 30 exp_x/a "--edge-weight 0.0" exp_x/b "--edge-weight 0.1"
# GPU is split so both final FID evaluations fit (uncapped, the second job OOMs).
# Worth it for the 9M model; the 37M model is faster run sequentially (`queue`).
# Two arms concurrently, memory-capped; prints the FID table when done.
pair epochs name_a args_a name_b args_b:
    mkdir -p runs/{{name_a}} runs/{{name_b}}
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.47 uv run main.py anime \
        --outpath runs/{{name_a}}/model.eqx --n-epochs {{epochs}} --eval-every {{epochs}} \
        --early-stop-patience 0 --seed 49 --fid-batch-size 128 {{args_a}} \
        > runs/{{name_a}}/train.log 2>&1 & \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.47 uv run main.py anime \
        --outpath runs/{{name_b}}/model.eqx --n-epochs {{epochs}} --eval-every {{epochs}} \
        --early-stop-patience 0 --seed 49 --fid-batch-size 128 {{args_b}} \
        > runs/{{name_b}}/train.log 2>&1 & \
    wait
    @just fid runs/{{name_a}} runs/{{name_b}}

# Progress bar (epochs, or the FID batch loop during an evaluation) of every running job.
watch:
    @pgrep -af "main.py anime" | grep -v pgrep | sed 's/.*--outpath \([^ ]*\).*/\1/' | sort -u | \
        while read p; do printf '%-40s ' "$p"; \
        tail -c 300 "$(dirname $p)/train.log" 2>/dev/null | tr '\r' '\n' | grep -o '^[A-Za-z]*:.*[0-9]*/[0-9]* \[[^]]*\]' | tail -1 | tr -s ' '; done
    @nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader

# ---------------------------------------------------------------- reading results

#   just fid runs/exp_edge runs/exp_fix
# FID and final loss for every arm under the given directories.
fid *dirs:
    #!/usr/bin/env python3
    import json, sys
    from pathlib import Path
    dirs = sys.argv[1:] or ["runs"]
    arms = []
    for d in dirs:
        d = Path(d)
        arms += sorted(p.parent for p in d.glob("**/losses.json"))
    for a in arms:
        r = json.load(open(a / "losses.json"))
        fids = ", ".join(f"{x['epoch']}:{x['fid']:.1f}" for x in r if "fid" in x)
        print(f"{str(a):<36} ep {len(r):>3}  loss {r[-1]['loss']:.4f}  FID {fids}")

#   just samples runs/cmp.png "no aux=runs/exp_edge/edge0/model.eqx" "edge=runs/exp_edge/edge01/model.eqx"
# Labelled sample grid from shared noise, one row per NAME=CHECKPOINT.
samples out *arms:
    uv run python experiments/compare_samples.py {{out}} {{arms}}

# ---------------------------------------------------------------- analysis (no training)

#   just signal runs/exp_edge/edge0/model.eqx runs/ablation/signal_edge0
# Runs on CPU if the GPU is busy: `JAX_PLATFORMS=cpu just signal ...` (slow ODE part).
# Offline aux-loss signal tests (discrimination, descent, x_hat vs t, gate, gradients).
signal ckpt="runs/exp_aux/baseline/model.eqx" out="runs/ablation/signal":
    uv run python experiments/aux_signal.py --checkpoint {{ckpt}} --outdir {{out}}
    @echo "report: {{out}}/REPORT.md"

# Colour statistics of generated samples vs. real (chroma histogram, saturation, palette).
colors ckpt="runs/exp_edge/edge0/model.eqx":
    uv run python experiments/color_stats.py --checkpoint {{ckpt}}

#   just eyes "wide=runs/exp_long/wide_noaux/model.eqx" "9M=runs/exp_edge/edge0/model.eqx"
# Left/right eye colour agreement per checkpoint ("heterochromia rate") vs. real data.
eyes *arms:
    uv run python experiments/eye_consistency.py {{arms}}

#   just guidance runs/exp_edge/edge0/model.eqx --n-fid 2000 --lams 0,0.05,0.1
# Sampling-time guidance sweep (unary energies: edge/ink mass deficit, eye symmetry).
guidance ckpt *args:
    uv run python experiments/guidance.py --checkpoint {{ckpt}} {{args}}

# Dump dataset images next to their Sobel / ink maps into ./.inspect/.
inspect-losses *args:
    uv run python inspect_losses.py {{args}}

# Paper figures from ./runs into ./runs/figures.
figures:
    uv run python generate_figures.py

# ---------------------------------------------------------------- housekeeping

# Caches are keyed by path only, so they go stale silently after any change to
# data/animefaces.py preprocessing or metrics.compute_real_stats.
# Delete the image and Inception-statistics caches (keeps the detector masks).
clean-caches:
    rm -f .preprocessed/anime_faces.npy .preprocessed/real_stats.npz
    @echo "kept .preprocessed/anime_faces_masks.npy and landmark scores (detector output, expensive)"
