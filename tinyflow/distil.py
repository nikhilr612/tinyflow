"""Distil a sampler into a flow map that generates in one or two evaluations.

A velocity model under a fixed ODE sampler is a *deterministic* function of
its inputs: the source ``x_0`` and the conditioning ``c``.  Its outputs are
therefore a regression target with no conditional spread -- unlike training
on data pairs, where the minimiser ``E[x_1 | x_t]`` is a blur -- so a plain L2
on the map's jumps learns them without loss of detail, and nothing is
bootstrapped from the student itself (self-consistency objectives -- MeanFlow,
shortcut -- either diverged or saturated when fine-tuned from the anime-face
model).  The precondition is the determinism: a teacher sampled with an SDE
would bring the spread back, and the L2 would learn its blur.  Nothing here
assumes what ``x_0`` is (noise, or a source image); the caller draws it.

Stage 1 runs the teacher on the caller's ``(x_0, c)`` batches and keeps its
states at ``HALF`` and at the end.  Stage 2 fine-tunes the student -- the
caller starts it from the teacher's weights -- on every jump of the grid
``(0, HALF, 1)``,

    x_0 -> x_1,   x_0 -> x_half,   x_half -> x_1,

    L = E || jump(x_t, c, t, s) - x_s^teacher ||^2,

where ``jump(x_t, c, t, s) = x_t + (s - t) u_theta(x_t, c, t, s)`` reads the
student, given a second time ``s``, as the trajectory's velocity averaged over
``[t, s]``: a flow map.  How ``u`` is read from the student is the
experiment's parametrisation, so ``MapTrainer.jump_step`` is left to a
subclass; the student must take the second time input.  ``jump`` chains single
jumps along the same grid into the one- and two-jump samplers, and
``export_onnx`` writes a sampler as one ONNX graph.

The grid is the contract between the three: ``teacher`` returns its states at
the times the labels ``HALF`` and ``1`` stand for, ``jump_batches`` trains on
those labels, and ``jump`` queries only them.  A teacher whose solve stops
short of the labels (the anime one ends at ``t = 0.999``, its half state at
``0.4995``) is still distilled exactly: the labels are only inputs, used the
same way in training and sampling, so the student reproduces the teacher's
*sampler*.  They are not exact times for mixing the map with the teacher's
own velocity field.

Every batch of jumps is ``(source, cond, t, s, target)``: ``jump_batches``
produces it, ``MapTrainer.loss`` consumes it.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from tinyflow.trainer import Trainer

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence
    from pathlib import Path

    from jaxtyping import Array, Float, PRNGKeyArray

HALF = 0.5  # the grid's interior point: the jumps are on (0, HALF, 1)


def teacher_pairs(
    teacher: Callable[[Any, Any], tuple[Any, Any]],
    inputs: Iterable[tuple[Any, Any]],
    dtype: type = np.float16,
) -> dict[str, Any]:
    """The inputs and the teacher's states at ``HALF`` and the end, as numpy.

    ``inputs`` yields ``(x_0, cond)`` batches -- ``cond`` any pytree with a
    leading batch axis, ``()`` for none -- and ``teacher(x_0, cond)`` returns
    ``(x_half, x_1)``.  Sources and states are stored as ``dtype``; float16
    halves the memory and, for states in ``[-1, 1]``, rounds by ~1e-3, far
    below the distillation loss.  ``cond`` is stored as given.
    """
    states: dict[str, list] = {"x_0": [], "x_half": [], "x_1": []}
    conds = []
    for x_0, cond in inputs:
        half, end = teacher(x_0, cond)
        for k, v in zip(states, (x_0, half, end), strict=True):
            states[k].append(np.asarray(v, dtype))
        conds.append(jax.tree.map(np.asarray, cond))
    return {k: np.concatenate(v) for k, v in states.items()} | {
        "cond": jax.tree.map(lambda *c: np.concatenate(c), *conds)
    }


def jump_batches(
    pairs: dict[str, Any], batch_size: int, seed: int = 0
) -> tuple[Iterator[tuple[Any, ...]], int]:
    """An endless stream of jump batches, and the number of batches in one pass.

    Each pass over ``pairs`` (from ``teacher_pairs``) is a fresh permutation,
    dropping a trailing partial batch; each element is one of the grid's three
    jumps, chosen uniformly.  States are cast to float32 for training.
    """
    x_0, x_h, x_1, cond = pairs["x_0"], pairs["x_half"], pairs["x_1"], pairs["cond"]
    n = len(x_0)

    def stream():
        rng = np.random.default_rng(seed)
        while True:
            perm = rng.permutation(n)
            for b in range(0, n - batch_size + 1, batch_size):
                idx = np.sort(perm[b : b + batch_size])
                # 0: 0 -> 1,  1: 0 -> half,  2: half -> 1
                which = rng.integers(0, 3, batch_size)
                per_sample = which.reshape(-1, *[1] * (x_0.ndim - 1))
                src = np.where(per_sample == 2, x_h[idx], x_0[idx])
                tgt = np.where(per_sample == 1, x_h[idx], x_1[idx])
                ts = np.where(which == 2, HALF, 0.0).astype(np.float32)
                ss = np.where(which == 1, HALF, 1.0).astype(np.float32)
                yield (
                    jnp.asarray(src, jnp.float32),
                    jax.tree.map(lambda c, idx=idx: jnp.asarray(c[idx]), cond),
                    jnp.asarray(ts),
                    jnp.asarray(ss),
                    jnp.asarray(tgt, jnp.float32),
                )

    return stream(), n // batch_size


@dataclass(frozen=True)
class MapTrainer(Trainer):
    """Fit the teacher's jumps by L2; a subclass supplies ``jump_step``.

    The loss is the unweighted mean over the three jumps, so the long jump
    ``0 -> 1``, with the largest errors, dominates it.  The defaults -- a lower
    peak rate and a shorter warm-up than ``Trainer``'s, for fine-tuning a
    trained model -- are the anime-face recipe's.
    """

    peak_lr: float = 2e-4
    warmup_steps: int = 200

    @abc.abstractmethod
    def jump_step(self, net: Any, x, cond, t, s) -> jax.Array:
        """One jump of one sample, ``x_t -> x_s``: the experiment's parametrisation.

        ``t`` and ``s`` are traced scalars in training and Python floats in
        ``jump``; both must be accepted.
        """

    def loss(self, net: Any, batch: Any, key: PRNGKeyArray) -> Float[Array, ""]:
        """Mean squared error of the jumps; ``key`` is unused (fixed targets)."""
        src, cond, t, s, tgt = batch
        jumped = jax.vmap(lambda x, c, ti, si: self.jump_step(net, x, c, ti, si))(
            src, cond, t, s
        )
        return jnp.mean((jumped - tgt) ** 2)


def jump(step: Callable, net: Any, x_0: Any, cond: Any, n_jumps: int) -> jax.Array:
    """The sample from one source ``x_0`` in one or two jumps of ``step``.

    Only the grid's jumps were trained, so the samplers are ``0 -> 1`` and
    ``0 -> HALF -> 1``; any other ``n_jumps`` is an error.
    """
    grids = {1: (0.0, 1.0), 2: (0.0, HALF, 1.0)}
    if n_jumps not in grids:
        raise ValueError(f"n_jumps must be 1 or 2 (the trained jumps): {n_jumps}")
    times = grids[n_jumps]
    x = x_0
    for t, s in pairwise(times):
        x = step(net, x, cond, t, s)
    return x


def export_onnx(
    fn: Callable,
    specs: Sequence[jax.ShapeDtypeStruct],
    path: Path,
    input_names: Sequence[str],
    output_names: Sequence[str],
) -> None:
    """Write ``fn`` (one sample, no batch axis) as an ONNX graph at ``path``.

    ``fn`` closes over the weights, which go to ``path.data``.  Exported and
    checked so far: the jump samplers.  Needs the ``export`` extra (``jax2onnx``).
    """
    import jax2onnx

    for stale in (path, path.with_suffix(path.suffix + ".data")):
        stale.unlink(missing_ok=True)
    jax2onnx.to_onnx(
        fn,
        list(specs),
        return_mode="file",
        output_path=str(path),
        input_names=list(input_names),
        output_names=list(output_names),
    )
