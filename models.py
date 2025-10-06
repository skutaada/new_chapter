from typing import List

import jax
from flax import linen as nn
from jax import numpy as jnp


class MLP(nn.Module):
    features: list[int]

    @nn.compact
    def __call__(self, x):
        for feat in self.features:
            x = nn.Dense(feat, use_bias=False)(x)
            x = nn.relu(x)
        return x


class LSTM(nn.Module):
    hidden_sizes: list[int]

    @nn.compact
    def __call__(self, x):
        for hs in self.hidden_sizes:
            ScanLSTM = nn.scan(
                nn.LSTMCell,
                variable_broadcast="params",
                split_rngs={"params": False},
                in_axes=1,
                out_axes=1,
            )
            lstm = ScanLSTM(features=hs)
            batch_shape = x[:, 0].shape
            carry = lstm.initialize_carry(jax.random.key(0), batch_shape)
            carry, x = lstm(carry, x)
        return x


class CNN(nn.Module):
    channels: list[int]

    @nn.compact
    def __call__(self, x):
        for ch in self.channels:
            x = nn.Conv(ch, kernel_size=(3, 3), use_bias=False)(x)
            x = nn.max_pool(x, (3, 3))
            x = nn.relu(x)
        return x
