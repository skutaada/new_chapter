import argparse
import json
import time
from statistics import mean, stdev

import jax
from jax import numpy as jnp
from spu.utils import distributed as ppd
from tqdm import tqdm

from mobilefacenet import MobileFaceNet

parser = argparse.ArgumentParser()
parser.add_argument("--config")
parser.add_argument("--out-results")
parser.add_argument("--num-epochs", type=int, default=10)
args = parser.parse_args()


with open(args.config, "r") as f:
    conf = json.load(f)
ppd.init(conf["nodes"], conf["devices"])


rng = jax.random.PRNGKey(1337)
model = MobileFaceNet()
i_shape = jnp.empty((1, 112, 112, 3))
params = model.init(rng, i_shape)

params_s = ppd.device("P2")(lambda x: x)(params)


time_p = []
time_s = []
for _ in tqdm(range(args.num_epochs)):
    x = jax.random.normal(rng, (1, 112, 112, 3))
    x_s = ppd.device("P1")(lambda x: x)(x)

    start = time.time()
    y_s = ppd.device("SPU")(model.apply)(params_s, x_s)
    end = time.time()
    _ = ppd.get(y_s)
    time_s.append(end - start)

    start = time.time()
    _ = model.apply(params, x)
    end = time.time()
    time_p.append(end - start)


results = {
    "time_p": time_p,
    "time_s": time_s,
    "mean_p": mean(time_p),
    "mean_s": mean(time_s),
    "stdev_p": stdev(time_p),
    "stdev_s": stdev(time_s),
}
with open(args.out_results, "w") as f:
    json.dump(results, f)
