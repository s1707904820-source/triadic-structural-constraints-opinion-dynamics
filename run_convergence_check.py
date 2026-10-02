"""
Representative long-time convergence check.

This script compares No Triadic Feedback and Full Model under the same
N=1000 setting, using Monte Carlo sweeps as the time unit. It is intended
to diagnose whether the main reputation-feedback conclusion is caused by
a fixed-step truncation or by genuinely slower updates.
"""

import json
import os
import time

import numpy as np

from config import DATA_DIR, DYNAMICS, NETWORK, TRUST
from dynamics import OpinionDynamics, TrustThreshold
from metrics import Metrics
from network import NetworkGenerator
from statistics_utils import paired_difference, summarize
from provenance import build_metadata


def aggregate(values):
    return summarize(values)


def run_one(model_type, seed, N, m, p_hk, epsilon_0, alpha, mu, beta, checkpoints):
    rng = np.random.RandomState(seed)
    G = NetworkGenerator.generate_holme_kim(N, m, p_hk, seed=seed)
    triangles_list, node_triangles = NetworkGenerator.extract_triangles(G)
    weights = NetworkGenerator.compute_jaccard_weights(G)
    clustering = NetworkGenerator.compute_clustering(G)
    opinions = {node: float(rng.uniform(0, 1)) for node in G.nodes()}
    trust = TrustThreshold(epsilon_0, alpha)
    engine = OpinionDynamics(
        G, opinions, weights, triangles_list, node_triangles,
        clustering, trust, mu, beta, model_type,
    )

    checkpoint_steps = {int(sweep * N): sweep for sweep in checkpoints}
    max_steps = max(checkpoint_steps)
    dx_acc = 0.0
    dx_count = 0
    last_checkpoint = 0
    out = {}

    for step in range(1, max_steps + 1):
        dx = engine.step(rng)
        dx_acc += dx
        dx_count += 1

        if step in checkpoint_steps:
            sweep = checkpoint_steps[step]
            ops_dict = engine.get_opinions()
            ops_arr = engine.get_opinions_array()
            mean_dx = dx_acc / dx_count if dx_count else 0.0
            out[str(sweep)] = {
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
                "mean_dx_since_last": float(mean_dx),
                "steps_since_last": int(step - last_checkpoint),
            }
            dx_acc = 0.0
            dx_count = 0
            last_checkpoint = step
    return out


def main(num_runs=10):
    os.makedirs(DATA_DIR, exist_ok=True)

    N = NETWORK["N"]
    m = NETWORK["m"]
    p_hk = 0.6
    epsilon_0 = TRUST["epsilon_0_default"]
    alpha = TRUST["alpha"]
    mu = DYNAMICS["mu"]
    beta = DYNAMICS["beta_default"]
    checkpoints = [50, 100, 150, 200]
    models = {
        "no_triadic_feedback": "No Triadic Feedback",
        "full_model": "Full Model",
    }

    results = {
        "meta": {
            "N": N,
            "m": m,
            "p_hk": p_hk,
            "epsilon_0": epsilon_0,
            "alpha": alpha,
            "mu": mu,
            "beta": beta,
            "checkpoints_sweeps": checkpoints,
            "num_runs": num_runs,
            "time_unit": "Monte Carlo sweeps; one sweep = N asynchronous updates",
        },
        "models": {},
        "provenance": build_metadata("convergence_check", num_runs),
    }

    t0 = time.time()
    for model_type, label in models.items():
        print(f"\n[{label}]")
        raw = []
        for run_i in range(num_runs):
            seed = run_i * 997 + 7
            run_result = run_one(
                model_type, seed, N, m, p_hk, epsilon_0,
                alpha, mu, beta, checkpoints,
            )
            raw.append(run_result)
            final = run_result[str(checkpoints[-1])]
            print(
                f"  run {run_i + 1}/{num_runs}: "
                f"R={final['order_param']:.4f}, "
                f"T={final['triangle_alignment_fixed']:.4f}, "
                f"dx={final['mean_dx_since_last']:.3e}",
                flush=True,
            )

        model_summary = {"label": label, "raw": raw, "by_sweep": {}}
        for sweep in checkpoints:
            key = str(sweep)
            model_summary["by_sweep"][key] = {
                metric: aggregate([r[key][metric] for r in raw])
                for metric in [
                    "order_param",
                    "n_clusters",
                    "variance",
                    "triangle_alignment_fixed",
                    "triangle_admissibility_adaptive",
                    "mean_dx_since_last",
                ]
            }
        results["models"][model_type] = model_summary

    first = results["models"]["full_model"]["raw"]
    second = results["models"]["no_triadic_feedback"]["raw"]
    results["paired_comparisons"] = {
        "definition": "full_model_minus_no_triadic_feedback; matched network and initial-opinion seed",
        "by_sweep": {
            str(sweep): {
                metric: paired_difference(
                    [row[str(sweep)][metric] for row in first],
                    [row[str(sweep)][metric] for row in second],
                )
                for metric in (
                    "order_param", "n_clusters", "variance",
                    "triangle_alignment_fixed",
                    "triangle_admissibility_adaptive",
                    "mean_dx_since_last",
                )
            }
            for sweep in checkpoints
        },
    }

    out_path = os.path.join(DATA_DIR, "exp_convergence_check.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {out_path}")
    print(f"Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
