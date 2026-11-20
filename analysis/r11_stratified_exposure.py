"""R1-1 small-network experiment with exposure-preserving threshold shuffles."""

import argparse
from collections import defaultdict
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import platform

import numpy as np
import networkx as nx

from r11_threshold_exposure import (
    CODE, DYNAMICS, METRICS, TRUST, M, N, P_VALUES, SEEDS, SHUFFLED, STRUCTURED,
    Metrics, NetworkGenerator, ObservedDyadicDynamics, SingleRun, TrustThreshold,
    edge_proposal_probabilities, fingerprint_graph, gaps_for_edges, gate_contrast,
    initial_opinions, observe_windows, summarize_counts, threshold_fingerprint,
    write_tables,
)


GLOBAL = "global"
BANDED = "q_bands"
EXACT = "equal_q"
DESIGNS = (GLOBAL, BANDED, EXACT)
N_BANDS = 12
DRAWS = 6
SWEEPS = (0, 1, 5, 20, 50, 100)


def proposal_strata(edges, degrees, n_bands):
    """Use exact degree fractions, then group adjacent ranks without splitting ties."""
    tied = defaultdict(list)
    for index, (u, v) in enumerate(edges):
        value = Fraction(1, degrees[u]) + Fraction(1, degrees[v])
        tied[value].append(index)
    exact = [np.asarray(tied[key], dtype=int) for key in sorted(tied)]
    bands = defaultdict(list)
    start = 0
    for members in exact:
        midpoint_band = ((2 * start + len(members)) * n_bands) // (2 * len(edges))
        bands[midpoint_band].extend(members)
        start += len(members)
    return exact, [np.asarray(group, dtype=int) for group in bands.values()]


def permute_within_strata(thresholds, groups, rng):
    shuffled = thresholds.copy()
    for indices in groups:
        shuffled[indices] = rng.permutation(thresholds[indices])
    return shuffled


def weighted_cdf_distances(original, shuffled, q):
    """Return sup distance, 1-Wasserstein distance, and signed weighted mean gap."""
    support, inverse = np.unique(original, return_inverse=True)
    before = np.bincount(inverse, weights=q, minlength=len(support))
    after = np.bincount(np.searchsorted(support, shuffled), weights=q, minlength=len(support))
    delta = np.cumsum(after - before) / q.sum()
    return {
        "cdf_sup_distance": float(np.max(np.abs(delta))),
        "wasserstein_distance": float(np.diff(support) @ np.abs(delta[:-1])),
        "proposal_mean_difference": float(q @ (shuffled - original) / q.sum()),
    }


def weighted_correlation(x, y, q):
    x = x - q @ x / q.sum()
    y = y - q @ y / q.sum()
    scale = float(np.sqrt((q @ x**2) * (q @ y**2)))
    return float(q @ (x * y) / scale) if scale else None


def measure_assignment(original, shuffled, jaccard, groups, gaps, q):
    moved = shuffled != original
    movable_mass = sum(float(q[g].sum()) for g in groups if np.unique(original[g]).size > 1)
    return {
        "strata": len(groups),
        "movable_proposal_mass": movable_mass,
        "changed_edge_fraction": float(moved.mean()),
        "changed_proposal_mass": float(q @ moved),
        "proposal_weighted_absolute_threshold_change": float(q @ np.abs(shuffled - original)),
        "jaccard_threshold_correlation": weighted_correlation(jaccard, shuffled, q),
        **weighted_cdf_distances(original, shuffled, q),
        **gate_contrast(gaps, original, shuffled, q),
    }


def simulate(graph, edges, weights, trust, opinions, rng_state, thresholds):
    # The only intervention is the edge-threshold map passed to the unchanged dyadic rule.
    engine = ObservedDyadicDynamics(
        graph, opinions, weights, [], {}, {}, trust, DYNAMICS["mu"],
        DYNAMICS["beta_default"], STRUCTURED,
        edge_thresholds=dict(zip(edges, thresholds)),
    )
    rng = np.random.RandomState()
    rng.set_state(rng_state)
    states, windows, sequence = observe_windows(engine, rng, edges, SWEEPS)
    return states, windows, sequence


def record_dynamics(tables, identifiers, result, reference, thresholds, original, edges, q):
    states, windows, _ = result
    ref_states, ref_windows, _ = reference
    for sweep, state in states.items():
        current = np.asarray(list(state.values()))
        base = np.asarray(list(ref_states[sweep].values()))
        ref_gate = gate_contrast(gaps_for_edges(ref_states[sweep], edges), original, thresholds, q)
        arm_gate = gate_contrast(gaps_for_edges(state, edges), original, thresholds, q)
        assignment_effect = (
            ref_gate["shuffled_minus_structured"] + arm_gate["shuffled_minus_structured"]) / 2
        state_effect = (
            arm_gate["structured_pass_mass"] - ref_gate["structured_pass_mass"]
            + arm_gate["shuffled_pass_mass"] - ref_gate["shuffled_pass_mass"]) / 2
        diagonal = arm_gate["shuffled_pass_mass"] - ref_gate["structured_pass_mass"]
        if not np.isclose(diagonal, assignment_effect + state_effect, atol=1e-12, rtol=0):
            raise RuntimeError("snapshot gate decomposition failed")
        tables["snapshots"].append({
            **identifiers, "sweep": sweep,
            "order_parameter": Metrics.order_parameter(current),
            "order_difference": Metrics.order_parameter(current) - Metrics.order_parameter(base),
            "variance": Metrics.opinion_variance(current),
            "variance_difference": Metrics.opinion_variance(current) - Metrics.opinion_variance(base),
            "cluster_count": Metrics.count_clusters(current),
            "gate_difference_on_structure_state": ref_gate["shuffled_minus_structured"],
            "gate_difference_on_arm_state": arm_gate["shuffled_minus_structured"],
            "gate_diagonal_difference": diagonal,
            "gate_assignment_component": assignment_effect,
            "gate_state_component": state_effect,
        })
    proposals_by_edge = sum(counts[:, 0] for counts in windows.values())
    tables["realized_exposure"].append({
        **identifiers, "end_sweep": SWEEPS[-1],
        **weighted_cdf_distances(original, thresholds, proposals_by_edge),
    })
    for (start, end), counts in windows.items():
        if not np.array_equal(counts[:, 0], ref_windows[start, end][:, 0]):
            raise RuntimeError("matched edge counts differ")
        summary = summarize_counts(counts, thresholds, (end - start) * len(ref_states[0]))
        ref_accepted = ref_windows[start, end][:, 1].sum()
        tables["windows"].append({
            **identifiers, "start_sweep": start, "end_sweep": end, **summary,
            "acceptance_difference": (summary["accepted"] - ref_accepted) / summary["proposals"],
        })


def run_network(p_hk, seed, tables, static_only):
    graph = NetworkGenerator.generate_holme_kim(N, M, p_hk, seed=seed)
    edges = list(graph.edges())
    q = edge_proposal_probabilities(graph, edges).sum(axis=1)
    if not np.isclose(q.sum(), 1, atol=1e-12, rtol=0):
        raise RuntimeError("this HK pilot requires non-isolated nodes")
    degrees = dict(graph.degree())
    exact, bands = proposal_strata(edges, degrees, N_BANDS)
    groups_by_design = {GLOBAL: [np.arange(len(edges))], BANDED: bands, EXACT: exact}
    weights = NetworkGenerator.compute_jaccard_weights(graph)
    jaccard = np.asarray([weights[edge] for edge in edges])
    trust = TrustThreshold(TRUST["epsilon_0_default"], TRUST["alpha"])
    original = np.asarray([trust.effective_threshold(w) for w in jaccard])
    opinions, rng_state = initial_opinions(graph, seed)
    initial_gaps = gaps_for_edges(opinions, edges)
    ids = {"p_hk": p_hk, "seed": seed}
    graph_hash = fingerprint_graph(graph)
    opinion_hash = threshold_fingerprint(list(opinions.values()))
    for index, (u, v) in enumerate(edges):
        tables["edges"].append({
            **ids, "edge_index": index, "u": u, "v": v, "degree_u": degrees[u],
            "degree_v": degrees[v], "q": q[index], "jaccard": jaccard[index],
            "structural_threshold": original[index], "initial_gap": initial_gaps[index],
        })
    if not static_only:
        reference = simulate(graph, edges, weights, trust, opinions, rng_state, original)
        ref_ids = {**ids, "design": "structure", "draw": 0}
        record_dynamics(tables, ref_ids, reference, reference, original, original, edges, q)
        tables["pairing"].append({
            **ref_ids, "network_sha256": graph_hash, "initial_sha256": opinion_hash,
            "proposal_sequence_sha256": reference[2],
        })

    for design_index, design in enumerate(DESIGNS):
        groups = groups_by_design[design]
        for draw in range(DRAWS):
            key = {**ids, "design": design, "draw": draw}
            if design == GLOBAL and draw == 0:
                # Replay the precise global assignment from the previous 20-sweep pilot.
                thresholds = SingleRun._build_edge_threshold_assignment(
                    graph, weights, trust, SHUFFLED, seed)
                shuffled = np.asarray([thresholds[edge] for edge in edges])
            else:
                shuffle_seed = seed + 12007103 + 65537 * design_index + 1009 * draw
                shuffled = permute_within_strata(original, groups, np.random.RandomState(shuffle_seed))
            if any(not np.array_equal(np.sort(shuffled[g]), np.sort(original[g])) for g in groups):
                raise RuntimeError("threshold multiset changed within a stratum")
            measures = measure_assignment(original, shuffled, jaccard, groups, initial_gaps, q)
            if design == EXACT and max(measures["cdf_sup_distance"],
                                       abs(measures["proposal_mean_difference"])) > 1e-12:
                raise RuntimeError("equal-q assignment changed weighted exposure")
            tables["assignments"].append({
                **key, "threshold_sha256": threshold_fingerprint(shuffled), **measures,
            })
            for index, threshold in enumerate(shuffled):
                tables["edge_assignments"].append({
                    **key, "edge_index": index, "threshold": threshold,
                })
            if not static_only:
                result = simulate(graph, edges, weights, trust, opinions, rng_state, shuffled)
                if result[2] != reference[2]:
                    raise RuntimeError("proposal sequences differ across paired arms")
                record_dynamics(tables, key, result, reference, shuffled, original, edges, q)
                tables["pairing"].append({
                    **key, "network_sha256": graph_hash, "initial_sha256": opinion_hash,
                    "proposal_sequence_sha256": result[2],
                })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if "formal_revision" in output.parts or output.exists():
        parser.error("use a new output directory outside formal_revision")
    tables = {key: [] for key in ("edges", "assignments", "edge_assignments",
                                  "pairing", "snapshots", "windows", "realized_exposure")}
    for p_hk in P_VALUES:
        for seed in SEEDS:
            run_network(p_hk, seed, tables, args.static_only)
    output.mkdir(parents=True, exist_ok=False)
    write_tables(output, tables)
    names = ("analysis/r11_stratified_exposure.py", "analysis/r11_threshold_exposure.py",
             "config.py", "dynamics.py", "metrics.py", "network.py", "simulation.py")
    metadata = {
        "data_class": "static_generated_graph_diagnostic" if args.static_only else
        "small_holme_kim_simulation_pilot",
        "n": N, "m": M, "p_hk": P_VALUES, "seeds": SEEDS, "n_bands_target": N_BANDS,
        "draws_per_design_per_graph": DRAWS, "designs": DESIGNS,
        "shuffle_seed": "seed+12007103+65537*design_index+1009*draw; global draw 0 uses original seed rule",
        "sweep_points": () if args.static_only else SWEEPS,
        "elementary_opportunities": sum(r["opportunities"] for r in tables["windows"]),
        "epsilon_0": TRUST["epsilon_0_default"], "alpha": TRUST["alpha"],
        "mu": DYNAMICS["mu"], "order_window_width": METRICS["cluster_diameter_threshold"],
        "python": platform.python_version(), "numpy": np.__version__, "networkx": nx.__version__,
        "row_counts": {key: len(rows) for key, rows in tables.items()},
        "source_sha256": {name: sha256((CODE / name).read_bytes()).hexdigest() for name in names},
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata["row_counts"], sort_keys=True))


if __name__ == "__main__":
    main()
