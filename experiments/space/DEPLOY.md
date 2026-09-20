# Deploying the demo Space

Source of `https://huggingface.co/spaces/nikhilr01/animefaces-distil-0.03b`.
The weights and the prior are not in the repo; assemble a staging directory:

    cp experiments/space/{app.py,requirements.txt,README.md,.gitattributes} STAGE/
    cp data/layouts.py STAGE/layouts.py           # then point PRIOR_PATH at the file beside it
    cp .preprocessed/landmark_prior.npz STAGE/
    cp runs/distill_rp/sampler_2jump.onnx{,.data} STAGE/    # experiments/export_onnx.py
    hf upload nikhilr01/animefaces-distil-0.03b STAGE . --repo-type space

Hosting constraint (2026-09): Hugging Face gates free Gradio Spaces to ZeroGPU
hardware, and ZeroGPU refuses to start without a `@spaces.GPU` function, hence
the never-called placeholder in `app.py`.  Inference is two CPU evaluations
(~0.1 s); no GPU is attached.  `ssr_mode=False` because the Gradio SSR proxy
crashed on startup there.
