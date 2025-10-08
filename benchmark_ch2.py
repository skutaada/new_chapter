import argparse
import json
import time
from statistics import mean, stdev

import jax
from flax import linen as nn
from jax import numpy as jnp
from spu.utils import distributed as ppd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--config")
parser.add_argument("--model")
args = parser.parse_args()

lenet = nn.Sequential(
    [
        nn.Conv(20, (5, 5)),
        lambda x: nn.relu(x),
        lambda x: nn.max_pool(x, (2, 2)),
        nn.Conv(50, (5, 5)),
        lambda x: nn.relu(x),
        lambda x: nn.max_pool(x, (2, 2)),
        lambda x: jnp.reshape(x, (1, -1)),
        lambda x: nn.relu(x),
        nn.Dense(500),
        lambda x: nn.relu(x),
        nn.Dense(10),
    ]
)
lenet_shape = (1, 28, 28, 1)

alexnet = nn.Sequential(
    [
        nn.Conv(64, (3, 3), 1, 2),
        lambda x: nn.relu(x),
        lambda x: nn.max_pool(x, (2, 2)),
        nn.Conv(96, (3, 3), padding=2),
        lambda x: nn.relu(x),
        lambda x: nn.max_pool(x, (2, 2)),
        nn.Conv(96, (3, 3), padding=1),
        lambda x: nn.relu(x),
        nn.Conv(64, (3, 3), padding=1),
        lambda x: nn.relu(x),
        nn.Conv(64, (3, 3), padding=1),
        lambda x: nn.relu(x),
        lambda x: nn.max_pool(x, (3, 3), strides=2),
        lambda x: jnp.reshape(x, (1, -1)),
        nn.Dense(128),
        lambda x: nn.relu(x),
        nn.Dense(256),
        lambda x: nn.relu(x),
        nn.Dense(10),
    ]
)
alexnet_shape = (1, 32, 32, 3)


with open(args.config, "r") as f:
    conf = json.load(f)
ppd.init(conf["nodes"], conf["devices"])


def main():
    model = None
    ishape = None
    match args.model:
        case "lenet":
            model = lenet
            ishape = lenet_shape
        case "alexnet":
            model = alexnet
            ishape = lenet_shape
        case _:
            raise NotImplementedError

    rng = jax.random.PRNGKey(0)
    params = model.init(rng, jnp.empty(ishape))
    params_s = ppd.device("P2")(lambda x: x)(params)

    time_s = []
    for _ in tqdm(range(100)):
        x = jax.random.normal(rng, ishape)
        x_s = ppd.device("P1")(lambda x: x)(x)

        start = time.time()
        y_s = ppd.device("SPU")(model.apply)(params_s, x_s)
        end = time.time()
        y = ppd.get(y_s)
        time_s.append(end - start)

    print(f"Mean for {args.model}: {mean(time_s)}")
    print(f"Stdev for {args.model}: {stdev(time_s)}")


if __name__ == "__main__":
    main()
