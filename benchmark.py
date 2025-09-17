import time
import json
import argparse
import statistics as stat

import jax
from jax import numpy as jnp
from tqdm import tqdm
from spu.utils import distributed as ppd

from models import MyMLP

parser = argparse.ArgumentParser()
parser.add_argument("--num-epochs", type=int)
parser.add_argument("--config", type=str)
#parser.add_argument("--settings", type=str)
#parser.add_argument("--model", type=str)
parser.add_argument("--results", type=str)


def count_parameters(params):
    return sum(jnp.size(p) for p in jax.tree_util.tree_leaves(params))


def run(args):
    with open(args.config, 'r') as f:
        conf = json.load(f)

    ppd.init(conf["nodes"], conf["devices"])

    rng = jax.random.PRNGKey(0)

    results = []

    nl_list = [1, 10, 100]
    nn_list = [10, 100, 1000]
    i_shapes = [[1, 10], [1, 50], [1, 100]]

    total_iters = len(nl_list) * len(nn_list) * len(i_shapes) * args.num_epochs
    pbar = tqdm(total=total_iters)

    for nl in nl_list:
        for nn in nn_list:
            for i_shape in i_shapes:
                model = MyMLP(nl, nn)
                params = model.init(rng, jnp.ones(i_shape))
                params_s = ppd.device("P2")(lambda x: x)(params)
                r = {
                    "n_params": count_parameters(params),
                    "i_shape": i_shape,
                    "nn": nn,
                    "nl": nl,
                }
                time_p = []
                time_s = []
                for _ in range(args.num_epochs):
                    x = jax.random.normal(rng, i_shape)
                    x_s = ppd.device("P1")(lambda x: x)(x)

                    start = time.time()
                    model.apply(params, x)
                    end = time.time()
                    time_p.append(end-start)

                    start = time.time()
                    y_s = ppd.device("SPU")(model.apply)(params_s, x_s)
                    end = time.time()
                    ppd.get(y_s)
                    time_s.append(end-start)

                    pbar.set_description(f"#layers={nl}, #neurons={nn}, shape={i_shape}")
                    pbar.update(1)
                r["mean_p"] = stat.mean(time_p)
                r["mean_s"] = stat.mean(time_s)
                r["stdev_p"] = stat.stdev(time_p)
                r["stdev_s"] = stat.stdev(time_s)
                results.append(r)
    pbar.close()
            


    with open(args.results, 'w') as f:
        json.dump(results, f)


if __name__ == "__main__":
    args = parser.parse_args()
    run(args)

    
