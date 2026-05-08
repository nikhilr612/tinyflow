"""A simple implementation of flow matching on the toy dataset.

This module contains an implementation of the Flow Matching Recipe,
using an MLP as the flow velocity field, to generatively model the toy dataset.
"""

import json
from pathlib import Path

import beartype
import diffrax
import equinox as eqx
import jax
import optax
from einops import pack, rearrange
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped
from tqdm import tqdm


class VelocityField(eqx.Module):
    """MLP implementing the velocity field for the toy model.

    The simplest flow matching variant is Gaussian Prior Conditional Flow Matching.
    Set: `X_t = tX_1 + (1-t)X_0`, `X_0 ~ N(0, I)` and `X_1 ~ Pdata`
    Set: `X_t ~ phi(t, X_0)`, require `phi(1, X_0) = Pdata`
    Flow matching evolves the ODE: `phi'(t, x) = u(t, phi(t, x))`
    Here, `u` is the velocity field. To fit FM, we need `u_theta` appox `u`.
    Direct regression is not tractable due to the unknown function `u`.
    One solution, is to consider regression when X_1 is known, i.e, conditional FM Loss.
    The tractable loss `E[u(t, X_t | X_1) - u_theta(t, X_t)]^2` can be simplified by:
    `X_(t|1) = t(x_1) + (1-t)X_0` (known x_1)
    x'_{t|1}(x_(t|1)) = u(t|1, x_(t|1)) (evolution from x_0)
    x_1 - x_0 = (x_1 - x_(t|1))/(1-t) = u(t|1, x_(t|1))

    `L_{CFM}^{Gaussian} = E[u_theta(t, X_t) - (X_1 - X_0)]^2`
    """

    in_layer: eqx.nn.Linear
    out_layer: eqx.nn.Linear

    def __init__(
        self,
        key: PRNGKeyArray,
        i_dim: int = 2,
        h_dim: int = 12,
    ) -> None:
        """Instantiate this velocity field MLP.

        Args:
            key: Jax PRNG key for initialization.
            i_dim: Input dimension.
            h_dim: Hidden dimension or output dimension of first layer.
        """
        sk_i, sk_o = jax.random.split(key, num=2)
        self.in_layer = eqx.nn.Linear(
            in_features=i_dim + 1, out_features=h_dim, key=sk_i
        )
        self.out_layer = eqx.nn.Linear(in_features=h_dim, out_features=i_dim, key=sk_o)

    @jaxtyped(typechecker=beartype.beartype)
    def __call__(
        self,
        t: Float[Array, " B"],
        x: Float[Array, " B T"],  # `T` is used as a symbol for data features.
        # We should have `T == self.i_dim`
    ) -> Float[Array, " B T"]:
        """Simple MLP as the velocity field."""
        packed, _ = pack([t, x], "B *")
        h = jax.nn.elu(jax.vmap(self.in_layer)(packed))
        return jax.vmap(self.out_layer)(h)


class ToyFM:
    """Class implementing training and inference of the toy model."""

    def __init__(self, velocity_field: VelocityField, i_dim: int, h_dim: int):
        """Initialize Toy FM with the specified velocity field.

        Args:
            velocity_field: The velocity field comprising this model.
            i_dim: Input dimensions of the velocity field
            h_dim: Hidden dimensions of the velocity field
        """
        self.i_dim = i_dim
        self.h_dim = h_dim
        self.u_theta = velocity_field

    @classmethod
    def from_key(cls, key: PRNGKeyArray, i_dim: int = 2, h_dim: int = 12):
        """Create a Toy FM model, with a velocity field of specified parameters.

        Args:
            key: The PRNG key used to initialize velocity field.
            i_dim: Input dimensions passed to the velocity field.
            h_dim: Hidden dimensions passed to the velocity field.
        """
        velocity_field = VelocityField(key, i_dim=i_dim, h_dim=h_dim)
        return cls(velocity_field, i_dim, h_dim)

    def save(self, path: str) -> None:
        """Save the learned velocity field parameters to a file at the specified path.

        Uses serialization mechanism provided by equinox.
        Hyperparameters (h_dim, i_dim) are dumped in a separate `.hparams` file.
        """
        hparams = {"h_dim": self.h_dim, "i_dim": self.i_dim}

        with Path(path + ".hparams").open("w") as f:
            json.dump(hparams, f)

        with Path(path).open("wb") as f:
            eqx.tree_serialise_leaves(f, self.u_theta)

    @staticmethod
    def load(path: str) -> "ToyFM":
        """Load a model from file."""
        with Path(path + ".hparams").open("r") as f:
            loaded = json.load(f)
            if "h_dim" in loaded and "i_dim" in loaded:
                hparams = {"h_dim": int(loaded["h_dim"]), "i_dim": int(loaded["i_dim"])}
            else:
                raise RuntimeError(f"Hyperparameters not found in file: {path}.hparams")
        key = jax.random.key(42)
        like: VelocityField = eqx.filter_eval_shape(
            VelocityField,
            key,
            i_dim=hparams["i_dim"],
            h_dim=hparams["h_dim"],
        )

        with Path(path).open("rb") as f:
            vf = eqx.tree_deserialise_leaves(f, like)  # deserialized velocity field
            return ToyFM(
                velocity_field=vf,
                i_dim=hparams["i_dim"],
                h_dim=hparams["h_dim"],
            )

    def sample(
        self, batch_size: int, key: PRNGKeyArray, ts: Float[Array, " L"]
    ) -> Float[Array, " {batch_size} L T"]:
        """Sample a batch of data from this model.

        Args:
            batch_size: Number of data points to sample
            ts: Timestamps at which to save the data.
                If only the final generated point is needed, set to 1.
            key: PRNG key
        """
        x_0 = jax.random.normal(key, shape=(batch_size, self.i_dim))
        ode_term = diffrax.ODETerm(
            # ugly `expand_dims` and `squeeze`
            # due to the fact that vmap, diffeqsolve expect _unbatched_ functions.
            # this is in line with jaxland, but not so much the convention elsewhere.
            lambda t, x, _args: self.u_theta(
                jax.numpy.expand_dims(t, 0),
                jax.numpy.expand_dims(x, 0),
            ).squeeze(axis=0),
        )

        @jax.vmap
        def solve(x_i):
            return diffrax.diffeqsolve(
                ode_term,
                diffrax.Dopri5(),
                t0=0.0,
                t1=1.0,
                y0=x_i,
                dt0=0.01,
                saveat=diffrax.SaveAt(ts=ts),
            )

        s = solve(x_0)
        return s.ys

    @jaxtyped(typechecker=beartype.beartype)
    @staticmethod
    def train_step(
        u_theta: VelocityField,
        t: Float[Array, " B"],
        x_0: Float[Array, " B T"],
        x_1: Float[Array, " B T"],
    ) -> Float[Array, ""]:
        """Perform a forward pass through the model and return L2 loss.

        Args:
            u_theta: The toy model to update
            t: Minibatch of timestamps t ~ U[0, 1]
            x_0: Noise input
            x_1: Minibatch of examples from dataset

        Note:
            This is a static method as opposed to an instance method because
            jax.value_and_grad computes Jacobian with respect to its first argument.
        """
        batch_size = t.shape[0]
        t_b = rearrange(t, "b -> b 1")  # get compatible shapes for broadcasting.
        x_t = t_b * x_1 + (1 - t_b) * x_0
        out = u_theta(t, x_t)
        target = x_1 - x_0
        return optax.l2_loss(out, target).sum() / batch_size


def train_on(
    key: PRNGKeyArray, model: ToyFM, dataset, n_epochs: int = 100, init_lr: float = 1e-3
) -> ToyFM:
    """Run a training loop for specified number of epochs on the dataset.

    Returns:
        The trained model.
    """
    optimizer = optax.adam(init_lr)
    optimizer_state = optimizer.init(eqx.filter(model.u_theta, eqx.is_inexact_array))

    # Wrap everything in jit so it can be as fast as possible.
    @eqx.filter_jit
    def make_update(
        key: PRNGKeyArray, u_theta: VelocityField, batch: jax.Array, optimizer_state
    ):
        newkey, sk1, sk2 = jax.random.split(key, num=3)
        batch_size = batch.shape[0]
        t = jax.random.uniform(sk1, shape=(batch_size,))
        rand_input = jax.random.normal(sk2, shape=batch.shape)
        loss, grad = jax.value_and_grad(ToyFM.train_step)(u_theta, t, rand_input, batch)
        updates, optimizer_state = optimizer.update(grad, optimizer_state)
        u_theta = eqx.apply_updates(u_theta, updates)
        return (newkey, u_theta, optimizer_state, loss)

    for epoch in (pbar := tqdm(desc="run", iterable=range(n_epochs), total=n_epochs)):
        net_loss = 0
        count = 0
        for batch in tqdm(iterable=dataset, desc="epoch", leave=False):
            key, new_utheta, optimizer_state, loss = make_update(
                key,
                model.u_theta,
                batch,
                optimizer_state,
            )
            model.u_theta = new_utheta
            net_loss += loss
            count += 1
        avg_loss = net_loss / count
        pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

    return model
