import jax
from flax import linen as nn
from jax import numpy as jnp


class TinyCNN(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(8, (3, 3), 1, 0)(x)
        x = nn.relu(x)
        x = nn.Conv(16, (3, 3), 2, 0)(x)
        x = nn.relu(x)
        x = nn.Conv(32, (3, 3), 1, 0)(x)
        x = nn.relu(x)
        x = jnp.reshape(x, (-1, 32))
        x = nn.Dense(self.num_classes)(x)
        return x


if __name__ == "__main__":
    model = TinyCNN(10)
    rng = jax.random.PRNGKey(1337)
    params = model.init(rng, jnp.empty((1, 8, 8, 1)))
