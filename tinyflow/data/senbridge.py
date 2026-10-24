"""SEN-Bridge: paired Sentinel-1 radar and Sentinel-2 optical crops, as Grain batches.

The data is the ``.npy`` release of ``nikhilr01/sen-bridge``, opened with
``sen_bridge.data.load_dataset`` (pinned, digest-checked, fetched on first use):
per split, ``sar`` and ``optical`` as memory-mapped ``uint8`` ``(N, 128, 128,
3)`` and ``meta`` as ``(N, 7)`` float32 -- the season ``(sin, cos)`` of the day
of year, the location's unit vector, the radar-to-optical time gap over 48 h
and the orbit (+1 descending, -1 ascending).  A split pickles by path, so
workers reopen the memory maps instead of copying them.

No augmentation: a flip or a quarter turn would break the radar's side-looking
geometry against ``orbit``, and the direction of the sun's shadows in the
optical image -- pairs no satellite could have taken.  There are 125,000 real
ones.

Every batch is ``(optical, (sar, meta))``.  The images stay ``uint8`` up to the
device (a quarter of float32's bytes) and are scaled to ``[-1, 1]`` there
(``tinyflow.models.senbridge.bridge.to_unit``).
"""

from __future__ import annotations

from typing import Any

import grain
import numpy as np


def make_dataset(
    split: Any, batch_size: int = 128, seed: int = 0, rows: np.ndarray | None = None
) -> tuple[grain.IterDataset, int]:
    """An endless, shuffled stream of ``(optical, (sar, meta))`` batches of ``split``.

    ``split`` is a ``sen_bridge.data.Split``; ``rows`` restricts it to a subset
    of its positions (default: all of them).  The shuffle is over positions,
    each pass a fresh permutation (``repeat`` after ``shuffle``), and a batch
    is one gather from the memory maps, in sorted order for locality.  Returns
    the dataset and the number of whole batches in one pass.
    """
    rows = np.arange(len(split)) if rows is None else np.asarray(rows)

    def gather(batch: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        r = np.sort(batch)
        return split.optical[r], (split.sar[r], split.meta[r])

    dataset = (
        grain.MapDataset.source(rows.tolist())
        .shuffle(seed)
        .repeat()
        .batch(batch_size, drop_remainder=True)
        .map(gather)
    )
    return (
        dataset.to_iter_dataset(
            read_options=grain.ReadOptions(num_threads=4, prefetch_buffer_size=4)
        ),
        len(rows) // batch_size,
    )
