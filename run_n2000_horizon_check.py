"""
Horizon-robustness check for the N=2000 size experiment.

We keep the same mechanism parameters as the size-robustness experiment and
compare 120/180/240 Monte Carlo sweeps under the full model. The goal is to
check whether the main macroscopic indicators have already stabilized.
"""

import json
import os
import time

import numpy as np

from config import DATA_DIR, DYNAMICS, TRUST
from dynamics import OpinionDynamics, TrustThreshold
from metrics import Metrics
from network import NetworkGenerator
from statistics_utils import summarize
from provenance import build_metadata


def aggregate(values):
    return summarize(values)


def run_one(seed, sweeps, N, m, p_hk, epsilon_0, alpha, mu, beta):
    rng = np.random.RandomState(seed)
    G = NetworkGenerator.generate_holme_kim(N, m, p_hk, seed=seed)
    triangles_list, node_triangles = NetworkGenerator.extract_triangles(G)
    weights = NetworkGenerator.compute_jaccard_weights(G)
    clustering = NetworkGenerator.compute_clustering(G)
    opinions = {node: float(rng.uniform(0.0, 1.0)) for node in G.nodes()}
    trust = TrustThreshold(epsilon_0, alpha)
    engine = OpinionDynamics(
        G, opinions, weights, triangles_list, node_triangles,
        clustering, trust, mu, beta, "full_model",
    )

    steps = int(N * sweeps)
    terminal_window_steps = min(steps, int(N * 10))
    terminal_dx_acc = 0.0
    updates_before_window = None
    for step in range(1, steps + 1):
        if step == steps - terminal_window_steps + 1:
            updates_before_window = engine.get_stats()['update_count']
        dx = engine.step(rng)
        if step > steps - terminal_window_steps:
            terminal_dx_acc += dx

    ops_dict = engine.get_opinions()
    ops_arr = engine.get_opinions_array()
    return {
        "order_param": float(Metrics.order_parameter(ops_arr)),
        "n_clusters": float(Metrics.count_clusters(ops_arr)),
        "variance": float(Metrics.opinion_variance(ops_arr)),
        "triangle_alignment_fixed": float(
            Metrics.triangle_alignment_fixed(
                ops_dict, triangles_list, epsilon_0
            )
        ),
        "triangle_admissibility_adaptive": float(
            Metrics.triangle_admissibility_adaptive(
                ops_dict, triangles_list, weights, trust
            )
        ),
        "edge_disagreement_penalty": float(
            Metrics.edge_disagreement_penalty(ops_dict, G, beta, rng=rng)),
        "terminal_mean_abs_dx": float(terminal_dx_acc / terminal_window_steps),
        "terminal_accepted_update_rate": float(
            (engine.get_stats()['update_count'] - updates_before_window)
            / terminal_window_steps),
        "terminal_window_sweeps": float(terminal_window_steps / N),
        "steps": steps,
        "sweeps": sweeps,
    }


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    N = 2000
    m = 4
    p_hk = 0.6
    epsilon_0 = TRUST["epsilon_0_default"]
    alpha = TRUST["alpha"]
    mu = DYNAMICS["mu"]
    beta = DYNAMICS["beta_default"]
    sweeps_list = [120, 180, 240]
    num_runs = 10

    results = {
        "meta": {
            "N": N,
            "m": m,
            "p_hk": p_hk,
            "epsilon_0": epsilon_0,
            "alpha": alpha,
            "mu": mu,
            "beta": beta,
            "model_type": "full_model",
            "num_runs": num_runs,
            "sweeps_list": sweeps_list,
            "time_unit": "Monte Carlo sweeps; one sweep = N asynchronous updates",
        },
        "by_sweeps": {},
        "provenance": build_metadata("n2000_horizon_check", num_runs),
    }

    t0 = time.time()
    for sweeps in sweeps_list:
        print(f"\n[sweeps={sweeps}]")
        raw = []
        for run_i in range(num_runs):
            seed = run_i * 997 + 7
            one = run_one(seed, sweeps, N, m, p_hk, epsilon_0, alpha, mu, beta)
            raw.append(one)
            print(
                f"  run {run_i + 1}/{num_runs}: "
                f"R={one['order_param']:.4f}, "
                f"T={one['triangle_alignment_fixed']:.4f}, "
                f"var={one['variance']:.4e}, "
                f"dx={one['terminal_mean_abs_dx']:.3e}",
                flush=True,
            )

        summary = {}
        for metric in [
            "order_param",
            "n_clusters",
            "variance",
            "triangle_alignment_fixed",
            "triangle_admissibility_adaptive",
            "edge_disagreement_penalty",
            "terminal_mean_abs_dx",
            "terminal_accepted_update_rate",
        ]:
            summary[metric] = aggregate([row[metric] for row in raw])
        summary["raw"] = raw
        results["by_sweeps"][str(sweeps)] = summary

        print(
            "  mean:"
            f" R={summary['order_param']['mean']:.4f},"
            f" T={summary['triangle_alignment_fixed']['mean']:.4f},"
            f" var={summary['variance']['mean']:.4e},"
            f" dx={summary['terminal_mean_abs_dx']['mean']:.3e}"
        )

    out_path = os.path.join(DATA_DIR, "exp_n2000_horizon_check.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {out_path}")
    print(f"Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
