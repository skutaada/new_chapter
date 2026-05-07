import argparse
import json
import statistics as stats
import time
import threading

from flax import linen as nn
from jax import numpy as jnp
from jax import random as jrng
from spu.utils import distributed as ppd
from tqdm import tqdm
import psutil
import pyRAPL

from models import CNN, LSTM, MLP

parser = argparse.ArgumentParser(description="Benchmark models.")
parser.add_argument("--config", default="3pc.json")
parser.add_argument("--model")
parser.add_argument("--num-epochs", type=int, default=10)
parser.add_argument("--results")
parser.add_argument("--use-spu", action="store_true")
args = parser.parse_args()


if args.use_spu:
    with open(args.config, "r") as f:
        config = json.load(f)
    ppd.init(config["nodes"], config["devices"])

# ── initialise pyRAPL once at module level ───────────────────────────────────
try:
    pyRAPL.setup()
    RAPL_AVAILABLE = True
except Exception:
    RAPL_AVAILABLE = False


def benchmark(model_def: nn.Module, x_shape, runs=100):
    rng = jrng.PRNGKey(1337)
    x = jnp.ones(x_shape)
    params = model_def.init(rng, x)
    if args.use_spu:
        params_s = ppd.device("P2")(lambda x: x)(params)

    if args.use_spu:
        for _ in range(3):
            x = jrng.normal(rng, x_shape)
            x_s = ppd.device("P1")(lambda x: x)(x)
            y_s = ppd.device("SPU")(model_def.apply)(params_s, x_s)
            _ = ppd.get(y_s)

    time_s = []
    time_p = []
    for _ in tqdm(range(runs)):
        x = jrng.normal(rng, x_shape)
        if args.use_spu:
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
        "mean_s": stats.mean(time_s) if args.use_spu else 0.0,
        "stdev_p": stats.stdev(time_p),
        "stdev_s": stats.stdev(time_s) if args.use_spu else 0.0,
    }
    return res


# ── helpers ──────────────────────────────────────────────────────────────────

def _loopback_io():
    """Return (bytes_sent, bytes_recv) on the loopback interface."""
    counters = psutil.net_io_counters(pernic=True)
    lo = counters.get("lo") or counters.get("lo0")  # Linux / macOS
    if lo is None:
        return 0, 0
    return lo.bytes_sent, lo.bytes_recv


def _process_mem_mb():
    """RSS memory of the current process in MB."""
    proc = psutil.Process()
    return proc.memory_info().rss / (1024 ** 2)


class _PowerSampler(threading.Thread):
    """Background thread that polls pyRAPL energy between start() and stop()."""
    def __init__(self, interval=0.05):
        super().__init__(daemon=True)
        self.interval = interval
        self._stop_event = threading.Event()
        self.samples_uj: list[float] = []   # microjoules per sample window

    def run(self):
        while not self._stop_event.is_set():
            if RAPL_AVAILABLE:
                meter = pyRAPL.Measurement("sample")
                meter.begin()
                time.sleep(self.interval)
                meter.end()
                # sum pkg + dram across all sockets
                energy = sum(meter.result.pkg or []) + sum(meter.result.dram or [])
                self.samples_uj.append(energy)
            else:
                time.sleep(self.interval)

    def stop(self):
        self._stop_event.set()
        self.join()

    def mean_power_w(self, elapsed_s: float) -> float:
        """Average power in Watts derived from accumulated energy samples."""
        if not self.samples_uj or elapsed_s <= 0:
            return float("nan")
        total_j = sum(self.samples_uj) / 1e6          # µJ → J
        return total_j / elapsed_s


# ── main benchmark ────────────────────────────────────────────────────────────

def benchmark_with_stats(model_def: nn.Module, x_shape, runs=100):
    rng = jrng.PRNGKey(1337)
    x = jnp.ones(x_shape)
    params = model_def.init(rng, x)

    if args.use_spu:
        params_s = ppd.device("P2")(lambda x: x)(params)

    # warm-up
    if args.use_spu:
        for _ in range(3):
            x = jrng.normal(rng, x_shape)
            x_s = ppd.device("P1")(lambda x: x)(x)
            y_s = ppd.device("SPU")(model_def.apply)(params_s, x_s)
            _ = ppd.get(y_s)

    time_s, time_p = [], []
    bw_sent_s, bw_recv_s = [], []   # bytes per SPU run
    bw_sent_p, bw_recv_p = [], []   # bytes per plain run
    mem_s, mem_p = [], []            # MB snapshots
    power_s, power_p = [], []        # Watts

    for _ in tqdm(range(runs)):
        x = jrng.normal(rng, x_shape)

        # ── SPU run ──────────────────────────────────────────────────────────
        if args.use_spu:
            x_s = ppd.device("P1")(lambda x: x)(x)

            lo_sent_0, lo_recv_0 = _loopback_io()
            mem_before = _process_mem_mb()
            sampler = _PowerSampler()
            sampler.start()
            start = time.time()

            y_s = ppd.device("SPU")(model_def.apply)(params_s, x_s)

            end = time.time()
            elapsed = end - start
            sampler.stop()
            lo_sent_1, lo_recv_1 = _loopback_io()
            mem_after = _process_mem_mb()

            _ = ppd.get(y_s)

            time_s.append(elapsed)
            bw_sent_s.append(lo_sent_1 - lo_sent_0)
            bw_recv_s.append(lo_recv_1 - lo_recv_0)
            mem_s.append(max(mem_before, mem_after))
            power_s.append(sampler.mean_power_w(elapsed))

        # ── plain run ─────────────────────────────────────────────────────────
        lo_sent_0, lo_recv_0 = _loopback_io()
        mem_before = _process_mem_mb()
        sampler = _PowerSampler()
        sampler.start()
        start = time.time()

        _ = model_def.apply(params, x)

        end = time.time()
        elapsed = end - start
        sampler.stop()
        lo_sent_1, lo_recv_1 = _loopback_io()
        mem_after = _process_mem_mb()

        time_p.append(elapsed)
        bw_sent_p.append(lo_sent_1 - lo_sent_0)
        bw_recv_p.append(lo_recv_1 - lo_recv_0)
        mem_p.append(max(mem_before, mem_after))
        power_p.append(sampler.mean_power_w(elapsed))

    def _safe_mean(lst):
        clean = [v for v in lst if v == v]  # drop NaN
        return stats.mean(clean) if clean else float("nan")
    
    def _safe_stdev(lst):
        clean = [v for v in lst if v == v]  # drop NaN
        return stats.stdev(clean) if len(clean) >= 2 else float("nan")

    res = {
        # timing
        "mean_p":   stats.mean(time_p),
        "mean_s":   stats.mean(time_s) if args.use_spu else 0.0,
        "stdev_p":  stats.stdev(time_p),
        "stdev_s":  stats.stdev(time_s) if args.use_spu else 0.0,
        # bandwidth sent (bytes)
        "mean_bw_sent_p":  _safe_mean(bw_sent_p),
        "mean_bw_sent_s":  _safe_mean(bw_sent_s) if args.use_spu else 0.0,
        "stdev_bw_sent_p": _safe_stdev(bw_sent_p),
        "stdev_bw_sent_s": _safe_stdev(bw_sent_s) if args.use_spu else 0.0,
        # bandwidth recv (bytes)
        "mean_bw_recv_p":  _safe_mean(bw_recv_p),
        "mean_bw_recv_s":  _safe_mean(bw_recv_s) if args.use_spu else 0.0,
        "stdev_bw_recv_p": _safe_stdev(bw_recv_p),
        "stdev_bw_recv_s": _safe_stdev(bw_recv_s) if args.use_spu else 0.0,
        # memory (MB)
        "mean_mem_p":  _safe_mean(mem_p),
        "mean_mem_s":  _safe_mean(mem_s) if args.use_spu else 0.0,
        "stdev_mem_p": _safe_stdev(mem_p),
        "stdev_mem_s": _safe_stdev(mem_s) if args.use_spu else 0.0,
        # power (Watts)
        "mean_power_p":  _safe_mean(power_p),
        "mean_power_s":  _safe_mean(power_s) if args.use_spu else 0.0,
        "stdev_power_p": _safe_stdev(power_p),
        "stdev_power_s": _safe_stdev(power_s) if args.use_spu else 0.0,
    }
    return res


mlp_config = {
    "input_size": (1, 10),
    "model_config": {
        "1k": ([100], [28] * 2, [20] * 3, [17] * 4),
        "5k": ([500], [66] * 2, [48] * 3, [39] * 4),
        "10k": ([1000], [95] * 2, [70] * 3, [56] * 4),
        #        "1M": ([10000], [995] * 2, [705] * 3, [575] * 4),
        #        "500k": ([50000], [702] * 2, [498] * 3, [407] * 4),
    },
}

cnn_config = {
    "input_size": (1, 112, 112, 3),
    "model_config": {
        "1k": ([7] * 3, [5] * 6, [4] * 9),
        "5k": ([16] * 3, [10] * 6, [8] * 9),
        "10k": ([23] * 3, [15] * 6, [12] * 9),
        #        "60k": ([25] * 12, [17] * 24, [14] * 36, [12] * 50),
        #        "1M": ([100] * 12, [70] * 24, [56] * 36, [48] * 50),
        # "25M": ([500] * 12, [350] * 24, [280] * 36, [240] * 50),
        # "25M": ([240] * 50,),
        #        "15M": ([389] * 12, [270] * 24, [218] * 36, [185] * 50),
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
            results = benchmark_with_stats(m_d, input_size, args.num_epochs)
            results["type"] = args.model
            results["input_shape"] = input_size
            results["model_config"] = l
            results["num_parameters"] = np
            full_stats.append(results)

    with open(args.results, "w") as f:
        json.dump(full_stats, f)


if __name__ == "__main__":
    main()
