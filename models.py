from typing import List

import jax
from jax import numpy as jnp
from flax import linen as nn


class MLP(nn.Module):
    features: list[int]

    @nn.compact
    def __call__(self, x):
        for feat in self.features:
            x = nn.Dense(feat, use_bias=False)(x)
            x = nn.relu(x)
        return nn.Dense(10, use_bias=False)(x)
    

class LSTM(nn.Module):
    hidden_sizes: list[int]

    @nn.compact
    def __call__(self, x):
        for hs in self.hidden_sizes:
            cell = nn.LSTMCell(hs)
            rnn = nn.RNN(cell)
            outputs = []
            for t in range(x.shape[1]):
                y = rnn(x[:,t,:])
                outputs.append(y)
            x = jnp.stack(outputs, axis=1)
        x = x[:, -1, :]
        return nn.Dense(10)(x)
    

class CNN(nn.Module):
    channels: list[int]

    @nn.compact
    def __call__(self, x):
        for ch in self.channels:
            x = nn.Conv(ch, kernel_size=(3, 3), use_bias=False)(x)
            x = nn.max_pool(x, (3, 3))
            x = nn.relu(x)
        return nn.Dense(10, use_bias=False)(x)
