"""Reconstruct and retain per-seed formal trajectories.

This reruns the three mechanism-focused variants over the complete formal
``p_HK`` grid using the original paired seeds.  The saved terminal metrics are
later checked against ``exp6_ablation.json`` before any path inference is used.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path



ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import NETWORK, TRUST, DYNAMICS, SIMULATION  # noqa: E402
from provenance import build_metadata, standard_seeds  # noqa: E402
from simulation import SingleRun  # noqa: E402


FORMAL_ROOT = ROOT / "output" / "formal_revision"
DATA_DIR = FORMAL_ROOT / "data"
DATA_PATH = DATA_DIR / "exp_formal_trajectory_raw.json"
META_PATH = DATA_DIR / "exp_formal_trajectory_raw.meta.json"

P_VALUES = [round(index * 0.05, 2) for index in range(21)]
MODELS = ["threshold_only", "no_triadic_feedback", "full_model"]
N_RUNS = 20
SEEDS = standard_seeds(N_RUNS)
TIME_METRICS = [
    "n_clusters",
    "variance",
    "order_param",
    "triangle_alignment_fixed",
    "triangle_admissibility_adaptive",
    "mean_abs_dx",
    "accepted_update_rate",
]
FINAL_METRICS = [
    "order_param",
    "n_clusters",
    "variance",
    "triangle_alignment_fixed",
    "triangle_admissibility_adaptive",
    "edge_disagreement_penalty",
]


def p_key(value):
    return "{:.2f}".format(float(value))


def write_atomic(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("{}.tmp-{}".format(path.name, os.getpid()))
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def control_block():
    return {
        "experiment": "formal_trajectory_reconstruction",
        "N": int(NETWORK["N"]),
        "m": int(NETWORK["m"]),
        "p_hk_values": P_VALUES,
        "models": MODELS,
        "n_runs": N_RUNS,
        "seeds": SEEDS,
        "main_sweeps": float(SIMULATION["main_sweeps"]),
        "record_sweeps": float(SIMULATION["record_sweeps"]),
        "epsilon_0": float(TRUST["epsilon_0_default"]),
        "alpha": float(TRUST["alpha"]),
        "mu": float(DYNAMICS["mu"]),
        "beta": float(DYNAMICS["beta_default"]),
        "update_mode": "asynchronous",
        "paired_on": "network, initial opinions, and update RNG seed",
    }


def compact_result(result):
    return {
        "seed": int(result["params"]["seed"]),
        "final": {
            metric: result["final"][metric] for metric in FINAL_METRICS
        },
        "convergence": dict(result["convergence"]),
        "time_series": {
            "step": list(result["time_series"]["step"]),
            "sweep": list(result["time_series"]["sweep"]),
            **{
                metric: list(result["time_series"][metric])
                for metric in TIME_METRICS
            },
        },
    }


def load_or_initialize(control):
    if not DATA_PATH.exists():
        return {"control": control, "data": {}}
    with DATA_PATH.open("r", encoding="utf-8") as handle:
        existing = json.load(handle)
    if existing.get("control") != control:
        raise RuntimeError(
            "existing trajectory checkpoint has incompatible controls; "
            "use a different output path instead of overwriting it")
    return existing


def main():
    started = time.time()
    control = control_block()
    payload = load_or_initialize(control)
    data = payload.setdefault("data", {})
    total_blocks = len(P_VALUES) * len(MODELS)
    complete_before = 0
    for block in data.values():
        for model in MODELS:
            if len(block.get(model, {})) == N_RUNS:
                complete_before += 1
    print("DESIGN=21 p_HK x 3 models x 20 paired seeds x 200 sweeps")
    print("RESUME_COMPLETE_BLOCKS={}/{}".format(
        complete_before, total_blocks), flush=True)

    num_steps = int(round(int(NETWORK["N"])
                          * float(SIMULATION["main_sweeps"])))
    record_interval = int(round(int(NETWORK["N"])
                                * float(SIMULATION["record_sweeps"])))
    completed_blocks = 0
    for p_hk in P_VALUES:
        block = data.setdefault(p_key(p_hk), {})
        for model in MODELS:
            model_runs = block.setdefault(model, {})
            if len(model_runs) == N_RUNS and all(
                    str(seed) in model_runs for seed in SEEDS):
                completed_blocks += 1
                print("SKIP p_HK={:.2f} model={} ({}/{})".format(
                    p_hk, model, completed_blocks, total_blocks), flush=True)
                continue
            block_started = time.time()
            for seed in SEEDS:
                seed_key = str(seed)
                if seed_key in model_runs:
                    continue
                simulation = SingleRun(
                    N=int(NETWORK["N"]),
                    m=int(NETWORK["m"]),
                    p_hk=float(p_hk),
                    epsilon_0=float(TRUST["epsilon_0_default"]),
                    alpha=float(TRUST["alpha"]),
                    mu=float(DYNAMICS["mu"]),
                    beta=float(DYNAMICS["beta_default"]),
                    model_type=model,
                    num_steps=num_steps,
                    seed=int(seed),
                    update_mode="asynchronous",
                )
                result = simulation.execute(record_interval=record_interval)
                model_runs[seed_key] = compact_result(result)
            write_atomic(payload, DATA_PATH)
            completed_blocks += 1
            print(
                "DONE p_HK={:.2f} model={} ({}/{}) seconds={:.1f}".format(
                    p_hk, model, completed_blocks, total_blocks,
                    time.time() - block_started),
                flush=True,
            )

    runtime = time.time() - started
    metadata = build_metadata(
        experiment="formal_trajectory_reconstruction",
        n_runs=N_RUNS,
        extra={
            "control": control,
            "runtime_seconds_this_invocation": float(runtime),
            "raw_trajectory_metrics": TIME_METRICS,
            "terminal_metrics": FINAL_METRICS,
            "checkpoint_granularity": "one p_HK-by-model block",
        },
    )
    write_atomic(metadata, META_PATH)
    print("STATUS=COMPLETE")
    print("RUNTIME_SECONDS={:.1f}".format(runtime))
    print("WROTE={}".format(DATA_PATH))
    print("WROTE={}".format(META_PATH))


if __name__ == "__main__":
    main()
