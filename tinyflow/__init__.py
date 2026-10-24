"""tinyflow: flow matching in JAX, as a small library.

``trainer`` (the shared optimisation loop and checkpoints) and ``distil``
(flow-map distillation) are generic; ``models`` holds each experiment's
network and maths, and ``data`` its dataset pipeline (the ``datasets`` extra).
"""
