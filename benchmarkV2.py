import argparse
import json
import statistics as stats
import time

from flax import linen as nn
from jax import numpy as jnp
from jax import random as jrng
from spu.utils import distributed as ppd
from tqdm import tqdm

from benchmark import benchmark_model
from models import CNN, LSTM, MLP

parser = argparse.ArgumentParser(description="Benchmark models.")
parser.add_argument("--config", default="3pc.json")
parser.add_argument("--model")
parser.add_argument("--num-epochs", type=int, default=10)
parser.add_argument("--results")
args = parser.parse_args()


with open(args.config, "r") as f:
    config = json.load(f)
ppd.init(config["nodes"], config["devices"])


def benchmark(model_def: nn.Module, params, x_shape, runs=100):
    rng = jrng.PRNGKey(42)
    x = jnp.ones(x_shape)
    params = model_def.init(rng, x)
    params_s = ppd.device("P2")(lambda x: x)(params)

    for _ in range(3):
        x = jrng.normal(rng, x_shape)
        x_s = ppd.device("P1")(lambda x: x)(x)
        y_s = ppd.device("SPU")(model_def.apply)(params_s, x_s)
        _ = ppd.get(y_s)

    time_s = []
    time_p = []
    for _ in tqdm(range(runs)):
        x = jrng.normal(rng, x_shape)
        x_s = ppd.device("P1")(lambda x: x)(x)

        start = time.time()
        y_s = ppd.device("SPU")(model_def.apply)(params_s, x_s)
        end = time.time()
        _ = ppd.get(y_s)
        time_s.append(end - start)

        start = time.time()
        _ = model_def.apply(params, x)
        end = time.time()
        time_p.append(end - start)

    res = {
        "mean_p": stats.mean(time_p),
        "mean_s": stats.mean(time_s),
        "stdev_p": stats.stdev(time_p),
        "stdev_s": stats.stdev(time_s),
    }
    return res


mlp_config = {
    "input_size": (1, 10),
    "model_config": {
        "1k": ([100], [28] * 2, [20] * 3, [17] * 4),
        "10k": ([1000], [95] * 2, [70] * 3, [56] * 4),
        "1M": ([10000], [995] * 2, [705] * 3, [575] * 4),
    },
}

cnn_config = {
    "input_size": (1, 112, 112, 3),
    "model_config": {
        "60k": ([25] * 12, [17] * 24, [14] * 36, [12] * 50),
        "1M": ([100] * 12, [70] * 24, [56] * 36, [48] * 50),
        "25M": ([500] * 12, [350] * 24, [280] * 36, [48] * 50),
    },
}

lstm_config = {
    "input_size": (1, 1, 14),
    "model_config": {
        "500": ([6], [4] * 3, [3] * 6, [2] * 10),
        "1k": ([10], [6] * 3, [4] * 6, [3] * 10),
        "5k": ([29], [14] * 3, [10] * 6, [8] * 10),
    },
}


def main():
    full_stats = []
    m_c = {}
    model = None
    match args.model:
        case "mlp":
            m_c = mlp_config
            model = MLP
        case "cnn":
            m_c = cnn_config
            model = CNN
        case "lstm":
            m_c = lstm_config
            model = LSTM
        case _:
            raise NotImplementedError

    input_size = m_c["input_size"]
    model_config = m_c["model_config"]

    for np, layers in model_config.items():
        for l in layers:
            m_d = model(l)
            results = benchmark_model(m_d, input_size, args.num_epochs)
            results["type"] = args.model
            results["input_shape"] = input_size
            results["model_config"] = l
            results["num_parameters"] = np
            full_stats.append(results)

    with open(args.results, "w") as f:
        json.dump(full_stats, f)


if __name__ == "__main__":
    main()
