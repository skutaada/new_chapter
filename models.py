from typing import List

import jax
from jax import numpy as jnp
from flax import linen as nn


class MyMLP(nn.Module):
    layers: int
    neurons: int

    @nn.compact
    def __call__(self, x):
        for l in range(self.layers-1):
            x = nn.Dense(self.neurons, use_bias=False)(x)
            x = nn.relu(x)
        y = nn.Dense(self.neurons, use_bias=False)(x)
        return y
    

if __name__ == "__main__":
    model = MyMLP(1, 30)
    rng = jax.random.PRNGKey(0)
    print(model.tabulate(rng, jnp.ones((1, 10))))