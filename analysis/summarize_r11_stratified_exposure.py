"""Reconcile the stratified pilot and summarize network-level paired contrasts."""

import argparse
from collections import defaultdict
import csv
import json
from math import fsum, isclose
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from r11_stratified_exposure import BANDED, DESIGNS, EXACT, GLOBAL
from summarize_r11_threshold_exposure import group, read_tables


ASSIGNMENT_ID = ("p_hk", "seed", "design", "draw")
LABELS = {GLOBAL: "Global shuffle", BANDED: "12 q bands", EXACT: "Equal q"}
COLORS = {GLOBAL: "#A55234", BANDED: "#598257", EXACT: "#235789"}


def reconcile(static_dir, pilot_dir, tables, metadata, prior_dir):
    for name in ("edges", "assignments", "edge_assignments"):
        if (static_dir / (name + ".csv")).read_bytes() != (pilot_dir / (name + ".csv")).read_bytes():
            raise ValueError("static and pilot data differ for " + name)
    if {key: len(rows) for key, rows in tables.items()} != metadata["row_counts"]:
        raise ValueError("row counts disagree with metadata")
    if sum(int(row["opportunities"]) for row in tables["windows"]) != metadata["elementary_opportunities"]:
        raise ValueError("elementary opportunity count disagrees with metadata")

    paired = group(tables["pairing"], "p_hk", "seed")
    n_arms = 1 + len(DESIGNS) * metadata["draws_per_design_per_graph"]
    for rows in paired.values():
        if (len(rows) != n_arms or len({(r["design"], r["draw"]) for r in rows}) != n_arms
                or any(len({row[key] for row in rows}) != 1 for key in
                       ("network_sha256", "initial_sha256", "proposal_sequence_sha256"))):
            raise ValueError("paired trajectory fingerprints differ")
    for row in tables["windows"]:
        if not (0 <= int(row["nonzero_moves"]) <= int(row["accepted"])
                <= int(row["proposals"]) == int(row["opportunities"])):
            raise ValueError("invalid dyadic event counts")
    for row in tables["snapshots"]:
        diagonal = float(row["gate_diagonal_difference"])
        components = float(row["gate_assignment_component"]) + float(row["gate_state_component"])
        if not isclose(diagonal, components, abs_tol=1e-12, rel_tol=0):
            raise ValueError("gate differences do not reconcile")

    # Independently reconstruct every exact-q CDF from the exported edge assignments.
    originals = {}
    for row in tables["edges"]:
        originals[row["p_hk"], row["seed"], row["edge_index"]] = row
    exact = group([r for r in tables["edge_assignments"] if r["design"] == EXACT], *ASSIGNMENT_ID)
    max_error = 0.0
    for key, rows in exact.items():
        before = defaultdict(list)
        after = defaultdict(list)
        for row in rows:
            edge = originals[key[0], key[1], row["edge_index"]]
            q = float(edge["q"])
            before[edge["structural_threshold"]].append(q)
            after[row["threshold"]].append(q)
        if before.keys() != after.keys():
            raise ValueError("exact assignment changed the threshold support")
        increments = [fsum(after[value]) - fsum(before[value]) for value in sorted(before, key=float)]
        error = max(abs(fsum(increments[:i])) for i in range(1, len(increments) + 1))
        max_error = max(error, max_error)
    if max_error > 1e-12:
        raise ValueError("exact assignment changed the proposal-weighted CDF")

    n_replayed = 0
    if prior_dir is not None:
        with (prior_dir / "snapshots.csv").open(newline="", encoding="utf-8") as handle:
            prior_snapshots = list(csv.DictReader(handle))
        prior = {(r["p_hk"], r["seed"], r["state_model"], r["sweep"]): r for r in prior_snapshots}
        prior_models = {"structure": "threshold_only", GLOBAL: "randomized_threshold_only"}
        for row in tables["snapshots"]:
            if row["design"] in prior_models and row["draw"] == "0" and int(row["sweep"]) <= 20:
                key = (row["p_hk"], row["seed"], prior_models[row["design"]], row["sweep"])
                historical = prior[key]
                for new, old in (("order_parameter", "order_parameter"),
                                 ("variance", "opinion_variance"), ("cluster_count", "cluster_count")):
                    if not isclose(float(row[new]), float(historical[old]), abs_tol=1e-14, rel_tol=0):
                        raise ValueError("20-sweep replay mismatch: " + str(key))
                n_replayed += 1
    return {"network_seed_units": len(paired), "trajectories": len(tables["pairing"]),
            "independently_checked_exact_assignments": len(exact),
            "independent_exact_cdf_max_error": max_error,
            "previous_pilot_snapshots_replayed": n_replayed}


def summarize_networks(rows, fields):
    by_network = group(rows, "seed")
    summary = {}
    for field in fields:
        network_means = {
            seed: mean(float(row[field]) for row in network)
            for (seed,), network in by_network.items()
        }
        values = network_means.values()
        summary[field] = {
            "mean": mean(values),
            "min_network_mean": min(values),
            "max_network_mean": max(values),
            "n_positive_network_means": sum(value > 1e-12 for value in values),
            "network_means": network_means,
        }
    return summary


def summarize(tables):
    assignment_fields = (
        "cdf_sup_distance", "wasserstein_distance", "proposal_mean_difference",
        "changed_proposal_mass", "movable_proposal_mass", "changed_edge_fraction",
        "jaccard_threshold_correlation", "shuffled_minus_structured",
        "shuffled_only_mass", "structured_only_mass",
    )
    snapshot_fields = (
        "order_difference", "variance_difference", "gate_diagonal_difference",
        "gate_assignment_component", "gate_state_component",
        "gate_difference_on_structure_state", "gate_difference_on_arm_state",
    )
    results = []
    exposure = group(tables["realized_exposure"], "p_hk", "design")
    windows = group(tables["windows"], "p_hk", "design", "start_sweep", "end_sweep")
    snapshots = group(tables["snapshots"], "p_hk", "design", "sweep")
    for (p_hk, design), rows in group(tables["assignments"], "p_hk", "design").items():
        results.append({
            "p_hk": float(p_hk), "design": design, "networks": len(group(rows, "seed")),
            "assignments": len(rows),
            "static": summarize_networks(rows, assignment_fields),
            "realized_exposure": summarize_networks(
                exposure[p_hk, design], ("cdf_sup_distance", "proposal_mean_difference")),
            "snapshots": {sweep: summarize_networks(members, snapshot_fields)
                          for (p, d, sweep), members in snapshots.items() if (p, d) == (p_hk, design)},
            "windows": {f"{start}-{end}": summarize_networks(members, ("acceptance_difference",))
                        for (p, d, start, end), members in windows.items() if (p, d) == (p_hk, design)},
        })
    return results


def plot(results, path):
    settings = sorted({row["p_hk"] for row in results})
    fig, axes = plt.subplots(2, len(settings), figsize=(11.3, 5.65), sharey="row",
                             layout="constrained")
    for col, p_hk in enumerate(settings):
        axes[0, col].set_title(r"$p_{HK}=" + str(p_hk) + "$")
        for row in results:
            if row["p_hk"] != p_hk:
                continue
            design = row["design"]
            snapshots = row["snapshots"]
            windows = row["windows"]
            points = list(map(int, snapshots))
            axes[0, col].plot(points, [item["order_difference"]["mean"] for item in snapshots.values()],
                              color=COLORS[design], marker="o", label=LABELS[design])
            axes[1, col].plot(range(len(windows)), [
                100 * item["acceptance_difference"]["mean"] for item in windows.values()],
                color=COLORS[design], marker="o")
        axes[0, col].set_xticks([0, 20, 50, 100])
        axes[1, col].set_xticks(range(len(windows)), list(windows))
        axes[0, col].set_xlabel("Sweep")
        axes[1, col].set_xlabel("Sweep interval")
        for ax in axes[:, col]:
            ax.axhline(0, color="#555555", linewidth=0.8)
            ax.grid(axis="y", alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 0].set_ylabel("Compact-order difference")
    axes[1, 0].set_ylabel("Gate-pass difference (pp)")
    fig.suptitle("R1-1 pilot  |  Reassignment minus Jaccard assignment", fontsize=14)
    fig.savefig(path, dpi=190)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("static_dir", type=Path)
    parser.add_argument("pilot_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--prior-pilot", type=Path)
    args = parser.parse_args()
    if "formal_revision" in args.output_dir.resolve().parts:
        parser.error("output must be outside formal_revision")
    tables = read_tables(args.pilot_dir)
    metadata = json.loads((args.pilot_dir / "metadata.json").read_text(encoding="utf-8"))
    checks = reconcile(args.static_dir, args.pilot_dir, tables, metadata, args.prior_pilot)
    results = summarize(tables)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps({"checks": checks, "results": results}, indent=2) + "\n", encoding="utf-8")
    plot(results, args.output_dir / "pilot_paired_differences.png")
    print(json.dumps(checks, sort_keys=True))


if __name__ == "__main__":
    main()
