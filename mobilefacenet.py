from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp

from benchmark import count_parameters

DEFAULT_SETTING = [
    [2, 64, 5, 2],
    [4, 128, 1, 2],
    [2, 128, 6, 1],
    [4, 128, 1, 2],
    [2, 128, 2, 1],
]


# ReLU ~ 0.234317x² + 0.5x + 0.187265
# ReLU6 ~ 0.000288x² + 0.790275x + 0.625143


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Module):
    out_planes: int
    kernel_size: int = 3
    stride: int = 1
    groups: int = 1
    act_fn: Callable = jax.nn.relu6

    @nn.compact
    def __call__(self, x, train: bool = False):
        padding = (self.kernel_size - 1) // 2
        if padding > 0:
            x = jnp.pad(x, ((0, 0), (padding, padding), (padding, padding), (0, 0)))
        x = nn.Conv(
            self.out_planes,
            (self.kernel_size, self.kernel_size),
            strides=self.stride,
            padding=0,
            feature_group_count=self.groups,
            use_bias=False,
            name="0",
        )(x)
        x = nn.BatchNorm(use_running_average=not train, name="1")(x)
        return self.act_fn(x)


class DepthwiseSeparableConv(nn.Module):
    in_planes: int
    out_planes: int
    kernel_size: int
    padding: int
    bias: bool = False
    act_fn: Callable = jax.nn.relu

    @nn.compact
    def __call__(self, x, train: bool = False):
        # Depthwise
        if self.padding > 0:
            x = jnp.pad(
                x,
                (
                    (0, 0),
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                    (0, 0),
                ),
            )
        x = nn.Conv(
            self.in_planes,
            kernel_size=(self.kernel_size, self.kernel_size),
            padding=0,
            feature_group_count=self.in_planes,
            use_bias=self.bias,
            name="depthwise",
        )(x)
        x = nn.BatchNorm(use_running_average=not train, name="bn1")(x)
        x = self.act_fn(x)

        # Pointwise
        x = nn.Conv(
            self.out_planes,
            kernel_size=(1, 1),
            padding=0,
            use_bias=self.bias,
            name="pointwise",
        )(x)
        x = nn.BatchNorm(use_running_average=not train, name="bn2")(x)
        x = self.act_fn(x)

        return x


class GDConv(nn.Module):
    in_planes: int
    out_planes: int
    kernel_size: int
    padding: int
    bias: bool = False

    @nn.compact
    def __call__(self, x, train: bool = False):
        if self.padding > 0:
            x = jnp.pad(
                x,
                (
                    (0, 0),
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                    (0, 0),
                ),
            )
        x = nn.Conv(
            self.out_planes,
            (self.kernel_size, self.kernel_size),
            padding=0,
            feature_group_count=self.in_planes,
            use_bias=self.bias,
            name="depthwise",
        )(x)
        x = nn.BatchNorm(use_running_average=not train, name="bn")(x)
        return x


class InvertedResidual(nn.Module):
    in_planes: int
    out_planes: int
    stride: int
    expand_ratio: int
    act_fn: Callable = jax.nn.relu6

    @nn.compact
    def __call__(self, x, train: bool = False):
        hidden_dim = int(round(self.in_planes * self.expand_ratio))
        use_res_connect = self.stride == 1 and self.in_planes == self.out_planes
        layers = []
        if self.expand_ratio != 1:
            layers.append(
                ConvBNReLU(hidden_dim, kernel_size=1, act_fn=self.act_fn, name="conv.0")
            )
        layers.extend(
            [
                ConvBNReLU(
                    hidden_dim,
                    stride=self.stride,
                    groups=hidden_dim,
                    act_fn=self.act_fn,
                    name="conv.1",
                ),
                nn.Conv(
                    self.out_planes, (1, 1), 1, padding=0, use_bias=False, name="conv.2"
                ),
                nn.BatchNorm(use_running_average=not train, name="conv.3"),
            ]
        )
        y = nn.Sequential(layers)(x, train)
        if use_res_connect:
            return x + y
        return y


class MobileFaceNet(nn.Module):
    width_mult: float = 1.0
    round_nearest: int = 8
    act_fn1: Callable = jax.nn.relu6
    act_fn2: Callable = jax.nn.relu

    @nn.compact
    def __call__(self, x, train: bool = False):
        input_channel = 64
        last_channel = 512

        inverted_residual_setting = DEFAULT_SETTING

        last_channel = _make_divisible(
            last_channel * max(1.0, self.width_mult), self.round_nearest
        )
        x = ConvBNReLU(input_channel, stride=2, act_fn=self.act_fn1, name="conv1")(
            x, train
        )
        x = DepthwiseSeparableConv(
            in_planes=64,
            out_planes=64,
            kernel_size=3,
            padding=1,
            act_fn=self.act_fn2,
            name="dw_conv",
        )(x, train)
        features = []
        mod_name = 0
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * self.width_mult, self.round_nearest)
            for j in range(n):
                stride = s if j == 0 else 1
                features.append(
                    InvertedResidual(
                        input_channel,
                        output_channel,
                        stride,
                        expand_ratio=t,
                        act_fn=self.act_fn1,
                        name=f"features.{mod_name}",
                    )
                )
                mod_name += 1
                input_channel = output_channel
        x = nn.Sequential(features, name="features")(x, train)
        x = ConvBNReLU(last_channel, kernel_size=1, act_fn=self.act_fn1, name="conv2")(
            x, train
        )
        x = GDConv(
            in_planes=512, out_planes=512, kernel_size=7, padding=0, name="gdconv"
        )(x, train)
        x = nn.Conv(128, kernel_size=(1, 1), padding=0, name="conv3")(x)
        x = nn.BatchNorm(use_running_average=not train, name="bn")(x)
        x = jnp.squeeze(x)
        return x


if __name__ == "__main__":
    rng = jax.random.PRNGKey(0)
    mkey, xkey = jax.random.split(rng)

    model = MobileFaceNet()
    vars = model.init(mkey, jnp.ones((1, 112, 112, 3)))
    print(count_parameters(vars))
