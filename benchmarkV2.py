import argparse
import json
import statistics as stats
import threading
import time


from flax import linen as nn
from jax import numpy as jnp
from jax import random as jrng
from spu.utils import distributed as ppd
from tqdm import tqdm
import psutil
import pyRAPL

from models import CNN, LSTM, MLP, LinearRegression

parser = argparse.ArgumentParser(description="Benchmark models.")
parser.add_argument("--config", default="3pc.json")
parser.add_argument("--model")
parser.add_argument("--num-epochs", type=int, default=10)
parser.add_argument("--num-epochs-spu", type=int, default=None)
parser.add_argument("--num-epochs-plain", type=int, default=None)
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
    sq_diff_sum = 0.0
    element_count = 0
    for _ in tqdm(range(runs)):
        x = jrng.normal(rng, x_shape)
        if args.use_spu:
            x_s = ppd.device("P1")(lambda x: x)(x)

            start = time.time()
            y_s = ppd.device("SPU")(model_def.apply)(params_s, x_s)
            end = time.time()
            y_spu = ppd.get(y_s)
            time_s.append(end - start)

        start = time.time()
        y_plain = model_def.apply(params, x)
        end = time.time()
        time_p.append(end - start)

        if args.use_spu:
            sq_diff_sum += float(jnp.sum((y_plain - y_spu) ** 2))
            element_count += int(y_plain.size)

    rmse = float(jnp.sqrt(sq_diff_sum / element_count)) if args.use_spu else 0.0

    res = {
        "mean_p": stats.mean(time_p),
        "mean_s": stats.mean(time_s) if args.use_spu else 0.0,
        "stdev_p": stats.stdev(time_p),
        "stdev_s": stats.stdev(time_s) if args.use_spu else 0.0,
        "rmse": rmse,
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


class _ResourceSampler(threading.Thread):
    def __init__(self, interval=0.05):
        super().__init__(daemon=True)
        self.interval = interval
        self._stop_event = threading.Event()
        self._phase = None
        self.power_uj_spu: list[float] = []
        self.power_uj_plain: list[float] = []
        self.mem_mb_spu: list[float] = []
        self.mem_mb_plain: list[float] = []
        self.bw_sent_spu: list[float] = []
        self.bw_recv_spu: list[float] = []
        self.timestamps_spu: list[float] = []
        self.timestamps_plain: list[float] = []
        self._t0 = None
        self._prev_sent = 0.0
        self._prev_recv = 0.0

    def set_phase(self, phase):
        self._phase = phase

    def run(self):
        self._t0 = time.monotonic()
        while not self._stop_event.is_set():
            phase = self._phase

            if RAPL_AVAILABLE:
                meter = pyRAPL.Measurement("sample")
                meter.begin()
                time.sleep(self.interval)
                meter.end()
                energy = sum(meter.result.pkg or []) + sum(meter.result.dram or [])
            else:
                energy = float("nan")
                time.sleep(self.interval)

            mem = _process_mem_mb()
            sent, recv = _loopback_io()

            if phase == "spu":
                self.timestamps_spu.append(time.monotonic() - self._t0)
                self.power_uj_spu.append(energy)
                self.mem_mb_spu.append(mem)
                self.bw_sent_spu.append(sent - self._prev_sent)
                self.bw_recv_spu.append(recv - self._prev_recv)
            elif phase == "plain":
                self.timestamps_plain.append(time.monotonic() - self._t0)
                self.power_uj_plain.append(energy)
                self.mem_mb_plain.append(mem)

            self._prev_sent = sent
            self._prev_recv = recv

    def stop(self):
        self._stop_event.set()
        self.join()


# ── main benchmark ────────────────────────────────────────────────────────────

def benchmark_with_stats(model_def: nn.Module, x_shape, runs=100, sample_interval=0.05,
                         runs_spu=None, runs_plain=None):
    runs_spu = runs_spu if runs_spu is not None else runs
    runs_plain = runs_plain if runs_plain is not None else runs
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
    sq_diff_sum = 0.0
    element_count = 0

    n_shared = min(runs_spu, runs_plain) if args.use_spu else 0
    xs_spu = [jrng.normal(jrng.fold_in(rng, i), x_shape) for i in range(runs_spu)]
    xs_plain = (
        xs_spu[:n_shared] + [jrng.normal(jrng.fold_in(rng, i), x_shape)
                             for i in range(n_shared, runs_plain)]
    )
    y_spus: list = []
    y_plains: list = []

    sampler = _ResourceSampler(interval=sample_interval)
    sampler.start()

    if args.use_spu:
        sampler.set_phase("spu")
        for x in tqdm(xs_spu, desc="spu"):
            x_s = ppd.device("P1")(lambda x: x)(x)
            start = time.time()
            y_s = ppd.device("SPU")(model_def.apply)(params_s, x_s)
            end = time.time()
            y_spus.append(ppd.get(y_s))
            time_s.append(end - start)
        sampler.set_phase(None)

    sampler.set_phase("plain")
    for x in tqdm(xs_plain, desc="plain"):
        start = time.time()
        y_plain = model_def.apply(params, x)
        end = time.time()
        y_plains.append(y_plain)
        time_p.append(end - start)
    sampler.set_phase(None)

    if args.use_spu:
        for y_plain, y_spu in zip(y_plains[:n_shared], y_spus[:n_shared]):
            sq_diff_sum += float(jnp.sum((y_plain - y_spu) ** 2))
            element_count += int(y_plain.size)

    sampler.stop()

    def _safe_mean(lst):
        clean = [v for v in lst if v == v]
        return stats.mean(clean) if clean else float("nan")

    def _safe_stdev(lst):
        clean = [v for v in lst if v == v]
        return stats.stdev(clean) if len(clean) >= 2 else float("nan")

    power_w_spu = [e / (sampler.interval * 1e6) for e in sampler.power_uj_spu] if sampler.power_uj_spu else []
    power_w_plain = [e / (sampler.interval * 1e6) for e in sampler.power_uj_plain] if sampler.power_uj_plain else []

    rmse = float(jnp.sqrt(sq_diff_sum / element_count)) if args.use_spu else 0.0

    res = {
        "mean_p":              stats.mean(time_p),
        "mean_s":              stats.mean(time_s) if args.use_spu else 0.0,
        "stdev_p":             stats.stdev(time_p),
        "stdev_s":             stats.stdev(time_s) if args.use_spu else 0.0,
        "rmse":                rmse,
        "mean_power_w_spu":    _safe_mean(power_w_spu) if args.use_spu else float("nan"),
        "mean_power_w_plain":  _safe_mean(power_w_plain),
        "mean_mem_mb_spu":     _safe_mean(sampler.mem_mb_spu) if args.use_spu else float("nan"),
        "mean_mem_mb_plain":   _safe_mean(sampler.mem_mb_plain),
        "mean_bw_sent_spu":    _safe_mean(sampler.bw_sent_spu) if args.use_spu else float("nan"),
        "mean_bw_recv_spu":    _safe_mean(sampler.bw_recv_spu) if args.use_spu else float("nan"),
        "timeseries": {
            "timestamps_spu":   sampler.timestamps_spu,
            "timestamps_plain": sampler.timestamps_plain,
            "power_w_spu":      power_w_spu,
            "power_w_plain":    power_w_plain,
            "mem_mb_spu":       sampler.mem_mb_spu,
            "mem_mb_plain":     sampler.mem_mb_plain,
            "bw_sent_spu":      sampler.bw_sent_spu,
            "bw_recv_spu":      sampler.bw_recv_spu,
        },
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

lin_reg_config = {
    "input_size": [(1, 10), (1, 50), (1, 100), (1, 250), (1, 500)]
}


def main():
    full_stats = []
    m_c = {}
    model = None
    lin_reg_flag = False
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
        case "lin":
            lin_reg_flag = True
            m_c = lin_reg_config
            model = LinearRegression
        case _:
            raise NotImplementedError

    if lin_reg_flag:
        for i in m_c['input_size']:
            m_d = model()
            results = benchmark_with_stats(m_d, i, args.num_epochs,
                                              runs_spu=args.num_epochs_spu,
                                              runs_plain=args.num_epochs_plain)
            results["type"] = args.model
            results["input_shape"] = i
            full_stats.append(results)
    else:
        input_size = m_c["input_size"]
        model_config = m_c["model_config"]
        for np, layers in model_config.items():
            for l in layers:
                m_d = model(l)
                results = benchmark_with_stats(m_d, input_size, args.num_epochs,
                                                  runs_spu=args.num_epochs_spu,
                                                  runs_plain=args.num_epochs_plain)
                results["type"] = args.model
                results["input_shape"] = input_size
                results["model_config"] = l
                results["num_parameters"] = np
                full_stats.append(results)

    with open(args.results, "w") as f:
        json.dump(full_stats, f)


if __name__ == "__main__":
    main()
