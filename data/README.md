# Data

**Images.** `just data` downloads `huggan/anime-faces` (21 551 PNGs, 64×64) from
the Hugging Face Hub and unpacks it to `data/anime-faces/images/`; `data/animefaces.py`
loads and caches the PNGs to `.preprocessed/anime_faces.npy` on first use. The
images are not redistributed here: they are anime character art scraped from
getchu.com for the Kaggle dataset by Soumik Rakshit (cropped with nagadomi's
`lbpcascade_animeface`), which the Hub mirror labels CC0 but whose underlying
artwork is not the uploader's to license. Please cite both sources when using
them, as the dataset card asks.

**`landmarks.npz`** (shipped): `landmarks` — `(21551, 28, 2)` face landmarks in
`[-1, 1]` image coordinates from `hysts/anime-face-detector`, one row per PNG in
sorted-filename order; `keep` — a boolean per image, true when the detector's
mean landmark confidence is at least 0.3 (the 1.5 % below it are non-faces and
are excluded from training). The layout masks used for conditioning are
rasterised from these landmarks (`data/layouts.py`) and cached.

**`landmark_prior.npz`** (shipped): the Gaussian-mixture layout prior — a
point-distribution model of the landmarks (pose + PCA shape) with a
Dirichlet-process mixture over it — that sampling draws layouts from, so no real
image is involved in generation. Fitted once with scikit-learn; sampling is numpy.

Both `.npz` files were produced once, offline: the landmarks with
`hysts/anime-face-detector` (its own environment: torch + OpenCV), the prior by
fitting the point-distribution model and mixture with scikit-learn. Neither
step is needed to train or sample.
