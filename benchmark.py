import time
import json
import argparse
import statistics as stat
from typing import Tuple

import jax
from jax import numpy as jnp
from flax import linen as nn
from tqdm import tqdm
from spu.utils import distributed as ppd
from tqdm import tqdm

from models import MLP, LSTM, CNN

parser = argparse.ArgumentParser()
parser.add_argument("--num-epochs", type=int)
parser.add_argument("--config", type=str)
#parser.add_argument("--settings", type=str)
#parser.add_argument("--model", type=str)
parser.add_argument("--results", type=str)


def count_parameters(params):
    return sum(jnp.size(p) for p in jax.tree_util.tree_leaves(params))


def benchmark_model(model_def: nn.Module, input_shape: Tuple[int], runs=1000):
    rng = jax.random.PRNGKey(0)
    x = jnp.ones(input_shape)
    params = model_def.init(rng, x)
    params_s = ppd.device("P2")(lambda x: x)(params)

    num_params = count_parameters(params)

    for _ in range(10):
        x = jax.random.normal(rng, input_shape)
        x_s = ppd.device("P1")(lambda x: x)(x)
        model_def.apply(params, x)
        y_s = ppd.device("SPU")(model_def.apply)(params_s, x_s)
        y = ppd.get(y_s)

    time_s = []
    time_p = []
    for _ in tqdm(range(runs)):
        x = jax.random.normal(rng, input_shape)
        x_s = ppd.device("P1")(lambda x: x)(x)
        start = time.time()
        _ = model_def.apply(params, x)
        time_p.append(time.time() - start)

        start = time.time()
        y_s = ppd.device("SPU")(model_def.apply)(params_s, x_s)
        time_s.append(time.time() - start)
        y = ppd.get(y_s)

    stats = {
        "num_params": num_params,
        "mean_p": stat.mean(time_p),
        "stdev_p": stat.stdev(time_p),
        "mean_s": stat.mean(time_s),
        "stdev_s": stat.stdev(time_s)
    }

    return stats


mlp_configs = {
    "Very Wide Shallow": [512],
    "Wide": [256, 256],
    "Balanced": [128]*4,
    "Deep": [64]*8,
    "Very Deep Narrow": [32]*16
}
mlp_inputs = [(1, 128), (1, 512), (1, 1024)]

lstm_configs = {
    "Very Wide Shallow": [32],
    "Wide": [16, 16],
    "Balanced": [8, 8, 8],
    "Deep": [4]*4,
    "Very Deep Narrow": [2]*6
}
lstm_inputs = [(1, 16, 32), (1, 32, 64), (1, 64, 128)]

cnn_configs = {
    "Very Wide Shallow": [64, 64, 64],
    "Wide": [32]*6,
    "Balanced": [16]*12,
    "Deep": [8]*24,
    "Very Deep Narrow": [4]*48
}
cnn_inputs = [(1, 32, 32, 3), (1, 64, 64, 3), (1, 128, 128, 3)]


def main(args):
    with open(args.config, 'r') as f:
        conf = json.load(f)

    ppd.init(conf["nodes"], conf["devices"])
    full_stats = []
    for model_name, configs, inputs, cls in [
#        ("LSTM", lstm_configs, lstm_inputs, LSTM),
        ("CNN", cnn_configs, cnn_inputs, CNN),
        ("MLP", mlp_configs, mlp_inputs, MLP),
    ]:
        print(f"\n===== {model_name} (Small size) =====")
        for ratio_name, config in configs.items():
            for in_shape in inputs:
                model = cls(config)
                stats = benchmark_model(model, in_shape, args.num_epochs)
                stats["type"] = model_name
                stats["input_shape"] = in_shape
                stats["model_config"] = config
                full_stats.append(stats)
    
    with open(args.results, 'w') as f:
        json.dump(full_stats, f)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

