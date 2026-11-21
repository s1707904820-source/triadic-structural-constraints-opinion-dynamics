"""Validate and summarize R1-1 pilot outputs, then plot paired diagnostics."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from r11_threshold_exposure import EDGE_MEAN, MODELS, PROPOSAL_MEAN, SHUFFLED, STRUCTURED


LABELS = {
    STRUCTURED: "Jaccard assignment",
    SHUFFLED: "Shuffled assignment",
    EDGE_MEAN: "Uniform edge mean",
    PROPOSAL_MEAN: "Uniform proposal mean",
}


def read_tables(directory):
    tables = {}
    for path in directory.glob("*.csv"):
        with path.open(encoding="utf-8", newline="") as handle:
            tables[path.stem] = list(csv.DictReader(handle))
    return tables


def group(rows, *fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in fields)].append(row)
    return groups


def mean_column(rows, name):
    return mean(float(row[name]) for row in rows)


def verify(static_dir, pilot_dir, tables, metadata):
    for name in ("edges", "static_pair", "exposure"):
        if (static_dir / (name + ".csv")).read_bytes() != (pilot_dir / (name + ".csv")).read_bytes():
            raise ValueError("static and pilot graph diagnostics disagree: " + name)
    if {name: len(rows) for name, rows in tables.items()} != metadata["row_counts"]:
        raise ValueError("metadata row counts disagree")
    if sum(int(row["opportunities"]) for row in tables["windows"]) != metadata["elementary_opportunities"]:
        raise ValueError("pilot opportunity count does not match metadata")
    paired = group(tables["pairing"], "p_hk", "seed")
    for rows in paired.values():
        if ({row["model"] for row in rows} != set(MODELS) or len(rows) != len(MODELS)
                or any(len({row[key] for row in rows}) != 1 for key in
                       ("network_sha256", "initial_sha256", "proposal_sequence_sha256"))):
            raise ValueError("network, initial condition or proposal fingerprint mismatch")
    edge_counts = group(tables["edge_windows"], "p_hk", "seed", "start_sweep", "end_sweep", "u", "v")
    for rows in edge_counts.values():
        if len(rows) != len(MODELS) or len({row["proposals"] for row in rows}) != 1:
            raise ValueError("paired proposals do not reconcile")
        for row in rows:
            if not 0 <= int(row["nonzero_moves"]) <= int(row["accepted"]) <= int(row["proposals"]):
                raise ValueError("invalid nested event counts")
    window_fields = ("p_hk", "seed", "model", "start_sweep", "end_sweep")
    grouped_edges = group(tables["edge_windows"], *window_fields)
    for row in tables["windows"]:
        members = grouped_edges[tuple(row[name] for name in window_fields)]
        for metric in ("proposals", "accepted", "nonzero_moves", "abs_movement"):
            if not np.isclose(sum(float(member[metric]) for member in members), float(row[metric]),
                              rtol=1e-12, atol=1e-12):
                raise ValueError("edge and window totals differ: " + metric)
    return {"paired_network_seed_units": len(paired), "paired_edge_window_cells": len(edge_counts)}


def summarize(tables):
    result = []
    for (p_hk,), static in group(tables["static_pair"], "p_hk").items():
        edges = [row for row in tables["edges"] if row["p_hk"] == p_hk]
        exposures = [row for row in tables["exposure"] if row["p_hk"] == p_hk]
        snapshots = [row for row in tables["snapshots"] if row["p_hk"] == p_hk]
        windows = [row for row in tables["windows"] if row["p_hk"] == p_hk]
        row = {
            "p_hk": float(p_hk), "n_paired": len(static),
            "shuffled_minus_structured_exposure": mean_column(static, "shuffled_minus_structured_exposure"),
            "shuffled_minus_structured_initial_pass": mean_column(static, "shuffled_minus_structured"),
            "expected_random_minus_structured_initial_pass": (
                mean_column(static, "expected_shuffled_initial_pass_mass") -
                mean_column(static, "structured_pass_mass")),
            "initial_shuffled_only_mass": mean_column(static, "shuffled_only_mass"),
            "initial_structured_only_mass": mean_column(static, "structured_only_mass"),
            "models": {},
        }
        row["strata"] = {}
        predicates = {
            "nontriangle": lambda edge: edge["triangle_edge"] == "False",
            "triangle": lambda edge: edge["triangle_edge"] == "True",
            "bridge": lambda edge: edge["bridge"] == "True",
            "both_degrees_le_6": lambda edge: max(int(edge["degree_u"]), int(edge["degree_v"])) <= 6,
        }
        graphs = list(group(edges, "seed").values())
        for name, belongs in predicates.items():
            strata = [[edge for edge in graph if belongs(edge)] for graph in graphs]
            q_mass = [sum(float(edge["q"]) for edge in members) for members in strata]
            row["strata"][name] = {
                "edge_fraction": mean(len(members) / len(graph) for members, graph in zip(strata, graphs)),
                "proposal_mass": mean(q_mass),
                "shuffled_minus_structured_exposure_contribution": mean(
                    sum(float(edge["q"]) * (float(edge[SHUFFLED]) - float(edge[STRUCTURED]))
                        for edge in members) for members in strata),
            }
        row["within_graph_q_max_min_ratio_range"] = [
            float(value) for value in np.quantile([
                max(float(edge["q"]) for edge in graph) / min(float(edge["q"]) for edge in graph)
                for graph in graphs
            ], [0, 1])]
        for model in MODELS:
            exposure = [x for x in exposures if x["model"] == model]
            model_snapshots = [x for x in snapshots if x["state_model"] == model]
            model_windows = [x for x in windows if x["model"] == model]
            row["models"][model] = {
                "exposure": {key: mean_column(exposure, key) for key in (
                    "edge_mean", "proposal_mean", "initial_pass_mass", "iid_uniform_initial_pass_expectation",
                    "mass_above_baseline")},
                "snapshots": {sweep: {key: mean_column(rows, key) for key in (
                    "order_parameter", "opinion_variance", "cluster_count", "shuffled_minus_structured",
                    "structured_pass_mass", "shuffled_pass_mass")}
                    for (sweep,), rows in group(model_snapshots, "sweep").items()},
                "windows": {sweep: {key: mean_column(rows, key) for key in (
                    "acceptance_per_proposal", "abs_movement", "proposed_threshold_mean",
                    "accepted_threshold_mean", "movement_weighted_threshold_mean")}
                    for (sweep,), rows in group(model_windows, "start_sweep").items()},
            }
        result.append(row)
    return result


def write_threshold_cdf(edge_rows, path):
    fields = ("p_hk", "seed", "model", "threshold", "edge_cdf", "proposal_cdf")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for (p_hk, seed), edges in group(edge_rows, "p_hk", "seed").items():
            total_q = sum(float(edge["q"]) for edge in edges)
            for model in MODELS:
                ordered = sorted(edges, key=lambda edge: float(edge[model]))
                cumulative_edges = 0
                cumulative_q = 0.0
                for (threshold,), members in group(ordered, model).items():
                    cumulative_edges += len(members)
                    cumulative_q += sum(float(edge["q"]) for edge in members)
                    writer.writerow((p_hk, seed, model, threshold,
                                     cumulative_edges / len(edges), cumulative_q / total_q))


def plot(results, directory):
    fig, axes = plt.subplots(2, 3, figsize=(11.1, 5.8), sharex="row", sharey="row",
                             constrained_layout=True)
    for column, row in enumerate(results):
        for model, color in ((STRUCTURED, "#235789"), (SHUFFLED, "#A04832")):
            arm = row["models"][model]
            windows = arm["windows"]
            axes[0, column].plot(range(len(windows)), [
                100 * value["acceptance_per_proposal"] for value in windows.values()],
                marker="o", color=color, label=LABELS[model])
            snapshots = arm["snapshots"]
            axes[1, column].plot([int(sweep) for sweep in snapshots], [
                100 * value["shuffled_minus_structured"] for value in snapshots.values()],
                marker="o", color=color, label=LABELS[model])
        axes[0, column].set_title(r"$p_{HK}=" + str(row["p_hk"]) + "$")
        axes[0, column].set_xticks([0, 1, 2], ["0–1", "1–5", "5–20"])
        axes[0, column].set_xlabel("Sweep interval")
        axes[1, column].axhline(0, color="#65707A", linewidth=0.8)
        axes[1, column].set_xticks([0, 1, 5, 20])
        axes[1, column].set_xlabel("Snapshot sweep")
        for ax in axes[:, column]:
            ax.grid(axis="y", alpha=0.22)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Observed gate passes (%)")
    axes[1, 0].set_ylabel("Same-state pass difference (pp)\nShuffled minus Jaccard")
    axes[0, 0].legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle("R1-1 pilot  |  3 paired networks per setting", fontsize=14)
    fig.savefig(directory / "pilot_gate_diagnostics.png", dpi=190)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("static_dir", type=Path)
    parser.add_argument("pilot_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    if "formal_revision" in args.output_dir.resolve().parts:
        parser.error("output must be outside formal_revision")
    tables = read_tables(args.pilot_dir)
    metadata = json.loads((args.pilot_dir / "metadata.json").read_text(encoding="utf-8"))
    checks = verify(args.static_dir, args.pilot_dir, tables, metadata)
    results = summarize(tables)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    payload = {"checks": checks, "p_hk_results": results}
    (args.output_dir / "pilot_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_threshold_cdf(tables["edges"], args.output_dir / "threshold_cdf.csv")
    plot(results, args.output_dir)
    print(json.dumps(checks, sort_keys=True))


if __name__ == "__main__":
    main()
