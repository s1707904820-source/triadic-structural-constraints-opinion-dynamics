"""Generate formal network diagnostics for the mean-field threshold audit.

The calculation uses the same N, m, p_HK grid and 20 standard seeds as the
formal ablation.  It does not run opinion dynamics.  Its purpose is to verify
the non-triangle-edge Jaccard claim and quantify the Jensen gap produced by
substituting the mean edge weight into the nonlinear threshold map.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import NETWORK, TRUST  # noqa: E402
from dynamics import TrustThreshold  # noqa: E402
from network import NetworkGenerator  # noqa: E402
from provenance import build_metadata, standard_seeds  # noqa: E402
from statistics_utils import summarize  # noqa: E402


FORMAL_ROOT = ROOT / "output" / "formal_revision"
ANALYSIS_DIR = FORMAL_ROOT / "analysis"
FIGURE_DIR = FORMAL_ROOT / "figures"
DATA_PATH = ANALYSIS_DIR / "formal_network_threshold_diagnostics.json"
META_PATH = ANALYSIS_DIR / "formal_network_threshold_diagnostics.meta.json"
SOURCE_PATH = ANALYSIS_DIR / "formal_network_threshold_source_data.csv"
SUMMARY_PATH = ANALYSIS_DIR / "formal_network_threshold_summary.csv"
REPORT_PATH = ANALYSIS_DIR / "FORMAL_NETWORK_THRESHOLD_REPORT.md"
FIGURE_PATH = FIGURE_DIR / "fig_formal_network_threshold_diagnostics.png"

P_VALUES = [round(index * 0.05, 2) for index in range(21)]
N_RUNS = 20
SEEDS = standard_seeds(N_RUNS)
RAW_METRICS = [
    "n_edges",
    "n_triangle_edges",
    "n_nontriangle_edges",
    "fraction_nontriangle_edges",
    "mean_jaccard_all_edges",
    "mean_jaccard_triangle_edges",
    "mean_jaccard_nontriangle_edges",
    "max_abs_jaccard_nontriangle_edges",
    "n_nontriangle_edges_with_positive_jaccard",
    "n_triangle_edges_with_zero_jaccard",
    "mean_effective_threshold",
    "effective_threshold_at_mean_weight",
    "jensen_gap",
]


def triangle_edge_partition(graph):
    """Classify edges by independently enumerated triangle membership.

    This deliberately does not use Jaccard weights to define the classes:
    doing so would make the reported zero-weight check tautological.  The
    identity being checked is that, in a simple undirected graph, an edge is
    in a triangle if and only if its endpoints have at least one common
    neighbour, which is also equivalent to a strictly positive Jaccard
    numerator.
    """
    if graph.is_directed() or graph.is_multigraph():
        raise ValueError("triangle-edge partition requires a simple undirected graph")
    if any(u == v for u, v in graph.edges()):
        raise ValueError("self-loops are not supported in triangle-edge partition")

    triangles, _ = NetworkGenerator.extract_triangles(graph)
    triangle_edges = set()
    for a, b, c in triangles:
        triangle_edges.add(frozenset((a, b)))
        triangle_edges.add(frozenset((a, c)))
        triangle_edges.add(frozenset((b, c)))

    all_edges = set(frozenset((u, v)) for u, v in graph.edges())
    if not triangle_edges.issubset(all_edges):
        raise RuntimeError("triangle enumeration returned a non-edge")
    return triangle_edges, all_edges.difference(triangle_edges)


def write_json_atomic(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name("{}.tmp-{}".format(path.name, os.getpid()))
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temp_path), str(path))
    finally:
        if temp_path.exists():
            temp_path.unlink()


def calculate_one(p_hk, seed, trust):
    graph = NetworkGenerator.generate_holme_kim(
        int(NETWORK["N"]), int(NETWORK["m"]), float(p_hk), seed=int(seed))
    weights = NetworkGenerator.compute_jaccard_weights(graph)
    edge_records = []
    for u, v in graph.edges():
        edge_records.append((
            frozenset((u, v)),
            float(weights.get((u, v), weights.get((v, u), 0.0))),
        ))
    edge_weights = np.asarray([weight for _, weight in edge_records], dtype=float)

    # Structural classes are obtained from explicit triangle enumeration,
    # independently of the Jaccard values being audited.
    triangle_edges, nontriangle_edges = triangle_edge_partition(graph)
    triangle_weights = np.asarray([
        weight for edge, weight in edge_records if edge in triangle_edges
    ], dtype=float)
    nontriangle_weights = np.asarray([
        weight for edge, weight in edge_records if edge in nontriangle_edges
    ], dtype=float)
    thresholds = np.asarray([
        trust.effective_threshold(value) for value in edge_weights
    ], dtype=float)
    mean_weight = float(np.mean(edge_weights))
    threshold_at_mean = float(trust.effective_threshold(mean_weight))
    return {
        "p_hk": float(p_hk),
        "seed": int(seed),
        "n_edges": int(edge_weights.size),
        "n_triangle_edges": int(triangle_weights.size),
        "n_nontriangle_edges": int(nontriangle_weights.size),
        "fraction_nontriangle_edges": (
            float(nontriangle_weights.size / edge_weights.size)
            if edge_weights.size else 0.0),
        "mean_jaccard_all_edges": mean_weight,
        "mean_jaccard_triangle_edges": (
            float(np.mean(triangle_weights)) if triangle_weights.size else 0.0),
        "mean_jaccard_nontriangle_edges": (
            float(np.mean(nontriangle_weights))
            if nontriangle_weights.size else 0.0),
        "max_abs_jaccard_nontriangle_edges": (
            float(np.max(np.abs(nontriangle_weights)))
            if nontriangle_weights.size else 0.0),
        "n_nontriangle_edges_with_positive_jaccard": int(
            np.sum(nontriangle_weights > 0.0)),
        "n_triangle_edges_with_zero_jaccard": int(
            np.sum(triangle_weights <= 0.0)),
        "mean_effective_threshold": float(np.mean(thresholds)),
        "effective_threshold_at_mean_weight": threshold_at_mean,
        "jensen_gap": float(threshold_at_mean - np.mean(thresholds)),
    }


def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    trust = TrustThreshold(
        float(TRUST["epsilon_0_default"]), float(TRUST["alpha"]))
    raw_rows = []
    for p_hk in P_VALUES:
        print("p_HK={:.2f}".format(p_hk), flush=True)
        for seed in SEEDS:
            raw_rows.append(calculate_one(p_hk, seed, trust))

    summary = {}
    for p_hk in P_VALUES:
        rows = [row for row in raw_rows if row["p_hk"] == p_hk]
        summary[str(p_hk)] = {
            metric: summarize([row[metric] for row in rows])
            for metric in RAW_METRICS
        }

    nontriangle_zero = all(
        row["max_abs_jaccard_nontriangle_edges"] == 0.0
        and row["n_nontriangle_edges_with_positive_jaccard"] == 0
        for row in raw_rows)
    triangle_positive = all(
        row["n_triangle_edges_with_zero_jaccard"] == 0
        for row in raw_rows)
    minimum_jensen_gap = min(row["jensen_gap"] for row in raw_rows)
    validation = {
        "status": "PASS" if (
            nontriangle_zero and triangle_positive
            and minimum_jensen_gap >= -1e-15) else "FAIL",
        "nontriangle_jaccard_exactly_zero": bool(nontriangle_zero),
        "all_triangle_edges_have_positive_jaccard": bool(triangle_positive),
        "classification_method": "explicit triangle enumeration, independent of Jaccard weights",
        "minimum_observed_jensen_gap": float(minimum_jensen_gap),
        "expected_records": len(P_VALUES) * N_RUNS,
        "observed_records": len(raw_rows),
    }
    payload = {
        "design": {
            "N": int(NETWORK["N"]),
            "m": int(NETWORK["m"]),
            "p_hk_values": P_VALUES,
            "n_networks_per_p": N_RUNS,
            "seeds": SEEDS,
            "epsilon_0": float(TRUST["epsilon_0_default"]),
            "alpha": float(TRUST["alpha"]),
            "uncertainty": "two-sided 95% Student-t confidence interval",
        },
        "validation": validation,
        "summary": summary,
        "raw": raw_rows,
    }
    write_json_atomic(payload, DATA_PATH)
    write_json_atomic(build_metadata(
        experiment="formal_network_threshold_diagnostics",
        n_runs=N_RUNS,
        extra={
            "p_hk_values": P_VALUES,
            "purpose": (
                "verify non-triangle-edge Jaccard values and quantify the "
                "mean-field Jensen gap without running opinion dynamics"),
        }), META_PATH)

    with SOURCE_PATH.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw_rows[0]))
        writer.writeheader()
        writer.writerows(raw_rows)

    summary_fields = [
        "p_hk", "metric", "n", "mean", "sd", "se", "ci95_low",
        "ci95_high", "min", "max",
    ]
    with SUMMARY_PATH.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for p_hk in P_VALUES:
            for metric in RAW_METRICS:
                item = summary[str(p_hk)][metric]
                writer.writerow({
                    "p_hk": p_hk,
                    "metric": metric,
                    **{field: item[field] for field in summary_fields[2:]},
                })

    plot_metrics = [
        ("fraction_nontriangle_edges", "Fraction of non-triangle edges"),
        ("mean_jaccard_all_edges", "Mean edge Jaccard similarity"),
        ("mean_effective_threshold", "Mean effective threshold"),
        ("jensen_gap", "Jensen gap"),
    ]
    colors = ["#2563EB", "#0D9488", "#DC2626", "#7C3AED"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), sharex=True)
    for ax, (metric, label), color in zip(axes.ravel(), plot_metrics, colors):
        means = np.asarray([
            summary[str(p_hk)][metric]["mean"] for p_hk in P_VALUES])
        lows = np.asarray([
            summary[str(p_hk)][metric]["ci95_low"] for p_hk in P_VALUES])
        highs = np.asarray([
            summary[str(p_hk)][metric]["ci95_high"] for p_hk in P_VALUES])
        ax.plot(P_VALUES, means, marker="o", markersize=3.5,
                linewidth=1.7, color=color)
        ax.fill_between(P_VALUES, lows, highs, color=color, alpha=0.15)
        ax.set_title(label)
        ax.set_xlabel(r"$p_{HK}$")
        ax.set_ylabel("Mean (95% CI)")
        ax.grid(alpha=0.24, linestyle="--")
    fig.suptitle(
        "Formal Holme-Kim network and threshold diagnostics (n=20 networks)",
        fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(str(FIGURE_PATH), dpi=300, bbox_inches="tight")
    plt.close(fig)

    max_nontriangle = max(
        row["max_abs_jaccard_nontriangle_edges"] for row in raw_rows)
    with REPORT_PATH.open("w", encoding="utf-8") as handle:
        handle.write("# 正式网络阈值诊断\n\n")
        handle.write("- 验收状态：{}。\n".format(validation["status"]))
        handle.write(
            "- 设计：N={}，m={}，21个 p_HK 点，每点20个独立网络，"
            "种子与正式消融一致。\n".format(
                NETWORK["N"], NETWORK["m"]))
        handle.write(
            "- 三角边/非三角边由显式三角形枚举独立分类，而不是按 Jaccard 权重是否为零分类。"
            "所有非三角边的最大绝对 Jaccard 权重为 {:.3g}；"
            "所有三角边的 Jaccard 权重均严格为正。\n".format(
                max_nontriangle))
        handle.write(
            "- 在简单无向无自环图中，一条边属于三角形当且仅当两个端点至少有一个共同邻居；"
            "因此非三角边 Jaccard 严格为零，而非经验近似。\n")
        handle.write(
            "- epsilon_eff(w) 对 w 为凹函数，所以 epsilon_eff(E[w]) "
            "不小于 E[epsilon_eff(w)]。所有网络观测到的 Jensen gap "
            "均非负；均场代入平均权重会系统性高估平均有效阈值。\n")
        handle.write(
            "- 图中阴影为跨20个独立网络的双侧95% Student-t置信区间。\n")

    print("VALIDATION={}".format(validation["status"]))
    print("MAX_NONTRIANGLE_JACCARD={:.12g}".format(max_nontriangle))
    print("MIN_JENSEN_GAP={:.12g}".format(minimum_jensen_gap))
    for path in (DATA_PATH, META_PATH, SOURCE_PATH, SUMMARY_PATH,
                 REPORT_PATH, FIGURE_PATH):
        print("WROTE={}".format(path))
    if validation["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
