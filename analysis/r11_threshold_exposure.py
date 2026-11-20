"""Static and small-pilot diagnostics for the asynchronous R1-1 controls."""

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import platform
import struct
import sys

import networkx as nx
import numpy as np

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from config import DYNAMICS, METRICS, TRUST
from dynamics import OpinionDynamics, TrustThreshold
from metrics import Metrics
from network import NetworkGenerator
from simulation import SingleRun


N = 100
M = 3
P_VALUES = (0.0, 0.5, 1.0)
SEEDS = (7, 1004, 2001)
SWEEP_POINTS = (0, 1, 5, 20)
PERMUTATIONS = 1000
STRUCTURED = "threshold_only"
SHUFFLED = "randomized_threshold_only"
EDGE_MEAN = "mean_threshold_only"
PROPOSAL_MEAN = "proposal_mean_threshold_only"
MODELS = (STRUCTURED, SHUFFLED, EDGE_MEAN, PROPOSAL_MEAN)
PERCENTILES = (5, 25, 50, 75, 95)


def edge_proposal_probabilities(graph, edges):
    """Return the two directed probability masses per undirected edge."""
    if (graph.is_directed() or graph.is_multigraph()
            or nx.number_of_selfloops(graph) or not len(graph)):
        raise ValueError("expected a nonempty simple undirected graph")
    degrees = dict(graph.degree())
    directed = np.asarray(
        [(1 / (len(graph) * degrees[u]), 1 / (len(graph) * degrees[v]))
         for u, v in edges], dtype=float).reshape(-1, 2)
    expected_mass = sum(degree > 0 for degree in degrees.values()) / len(graph)
    if not np.isclose(directed.sum(), expected_mass, atol=1e-12, rtol=0):
        raise ValueError("edges do not account for all proposal probability")
    return directed


def weighted_quantiles(values, weights):
    """Use the left inverse of the weighted empirical CDF."""
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order]) / weights.sum()
    return values[order[np.searchsorted(cumulative,
                                        np.asarray(PERCENTILES) / 100)]]


def gate_contrast(gaps, structured, shuffled, q):
    """Evaluate both assignments on the same state, per update opportunity."""
    passes_structured = gaps <= structured
    passes_shuffled = gaps <= shuffled
    gained = float(q @ (passes_shuffled & ~passes_structured))
    lost = float(q @ (passes_structured & ~passes_shuffled))
    return {
        "structured_pass_mass": float(q @ passes_structured),
        "shuffled_pass_mass": float(q @ passes_shuffled),
        "shuffled_only_mass": gained,
        "structured_only_mass": lost,
        "shuffled_minus_structured": gained - lost,
    }


def initial_opinions(graph, seed):
    rng = np.random.RandomState(seed)
    opinions = {node: float(rng.uniform(0, 1)) for node in graph}
    return opinions, rng.get_state()


def threshold_assignments(graph, edges, weights, trust, seed, q):
    structured = np.asarray(
        [trust.effective_threshold(weights[u, v]) for u, v in edges])
    assignments = {STRUCTURED: structured}
    overrides = {STRUCTURED: None}
    for model in (SHUFFLED, EDGE_MEAN):
        thresholds = SingleRun._build_edge_threshold_assignment(
            graph, weights, trust, model, seed)
        overrides[model] = thresholds
        assignments[model] = np.asarray([thresholds[edge] for edge in edges])
    if not np.array_equal(np.sort(structured), np.sort(assignments[SHUFFLED])):
        raise RuntimeError("shuffling changed the threshold multiset")
    matched_mean = float(q @ structured / q.sum())
    assignments[PROPOSAL_MEAN] = np.full(len(edges), matched_mean)
    overrides[PROPOSAL_MEAN] = dict.fromkeys(edges, matched_mean)
    return assignments, overrides


class ObservedDyadicDynamics(OpinionDynamics):
    """Observe the engine's actual selected edge without an extra RNG draw."""

    def step(self, rng):
        self.proposal = None
        return super().step(rng)

    def _edge_threshold(self, u, v):
        threshold = super()._edge_threshold(u, v)
        if self.proposal is not None:
            raise RuntimeError("expected at most one dyadic proposal per step")
        gap = abs(self.opinions[u] - self.opinions[v])
        self.proposal = (u, v, gap, threshold)
        return threshold


def observe_windows(engine, rng, edges, sweep_points):
    edge_index = {directed: index for index, (u, v) in enumerate(edges)
                  for directed in ((u, v), (v, u))}
    node_index = {node: index for index, node in enumerate(engine.nodes)}
    states = {sweep_points[0]: engine.get_opinions()}
    proposal_hash = sha256()
    windows = {}
    for start, end in zip(sweep_points, sweep_points[1:]):
        # Each row is proposals, passed gates, nonzero moves, absolute movement.
        counts = np.zeros((len(edges), 4))
        for _ in range((end - start) * engine.N):
            dx = engine.step(rng)
            if engine.proposal is None:
                proposal_hash.update(struct.pack("<ii", -1, -1))
                if dx:
                    raise RuntimeError("movement with no dyadic proposal")
                continue
            u, v, gap, threshold = engine.proposal
            proposal_hash.update(struct.pack("<ii", node_index[u], node_index[v]))
            passed = gap <= threshold
            if dx and not passed:
                raise RuntimeError("movement after a rejected gate")
            counts[edge_index[u, v]] += (1, passed, dx > 0, dx)
        windows[start, end] = counts
        states[end] = engine.get_opinions()
    totals = sum(windows.values()).sum(axis=0)
    if (totals[0] != engine.stats["edge_count"]
            or totals[1] != engine.stats["update_count"]
            or totals[2] > totals[1] or engine.stats["tri_count"]):
        raise RuntimeError("observer counters disagree with the dyadic engine")
    return states, windows, proposal_hash.hexdigest()


def summarize_counts(counts, thresholds, opportunities):
    proposals, accepted, moves, displacement = counts.T
    n_proposals = int(proposals.sum())
    n_accepted = int(accepted.sum())
    movement = float(displacement.sum())
    return {
        "opportunities": opportunities,
        "proposals": n_proposals,
        "accepted": n_accepted,
        "nonzero_moves": int(moves.sum()),
        "abs_movement": movement,
        "acceptance_per_proposal": n_accepted / n_proposals if n_proposals else None,
        "acceptance_per_opportunity": n_accepted / opportunities,
        "proposed_threshold_mean": float(thresholds @ proposals / n_proposals)
        if n_proposals else None,
        "accepted_threshold_mean": float(thresholds @ accepted / n_accepted)
        if n_accepted else None,
        "movement_weighted_threshold_mean": float(thresholds @ displacement / movement)
        if movement else None,
    }


def permutation_reference(structured, gaps, q, seed):
    """Sample only static reassignments. These draws share one fixed graph."""
    rng = np.random.RandomState(seed + 8975611)
    means = []
    gate_masses = []
    for _ in range(PERMUTATIONS):
        thresholds = rng.permutation(structured)
        means.append(float(q @ thresholds / q.sum()))
        gate_masses.append(float(q @ (gaps <= thresholds)))
    sorted_thresholds = np.sort(structured)
    expected_pass = 1 - np.searchsorted(sorted_thresholds, gaps, side="left") / len(structured)
    return {
        "expected_shuffled_exposure": float(np.mean(structured)),
        "expected_shuffled_initial_pass_mass": float(q @ expected_pass),
        "permutation_exposure_p05": float(np.quantile(means, 0.05)),
        "permutation_exposure_p95": float(np.quantile(means, 0.95)),
        "permutation_initial_pass_p05": float(np.quantile(gate_masses, 0.05)),
        "permutation_initial_pass_p95": float(np.quantile(gate_masses, 0.95)),
    }


def gaps_for_edges(opinions, edges):
    return np.asarray([abs(opinions[u] - opinions[v]) for u, v in edges])


def fingerprint_graph(graph):
    adjacency = [(node, list(graph.neighbors(node))) for node in graph]
    return sha256(json.dumps(adjacency).encode("utf-8")).hexdigest()


def threshold_fingerprint(values):
    return sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def run_network(p_hk, seed, static_only, tables):
    graph = NetworkGenerator.generate_holme_kim(N, M, p_hk, seed=seed)
    edges = list(graph.edges())
    directed = edge_proposal_probabilities(graph, edges)
    q = directed.sum(axis=1)
    trust = TrustThreshold(TRUST["epsilon_0_default"], TRUST["alpha"])
    weights = NetworkGenerator.compute_jaccard_weights(graph)
    opinions, dynamics_rng_state = initial_opinions(graph, seed)
    initial_gaps = gaps_for_edges(opinions, edges)
    assignments, overrides = threshold_assignments(graph, edges, weights, trust, seed, q)
    structured, shuffled = assignments[STRUCTURED], assignments[SHUFFLED]
    identifiers = {"p_hk": p_hk, "seed": seed}
    bridge_edges = {frozenset(edge) for edge in nx.bridges(graph)}
    degree = dict(graph.degree())
    graph_hash = fingerprint_graph(graph)
    initial_hash = threshold_fingerprint(list(opinions.values()))
    for index, (u, v) in enumerate(edges):
        row = {
            **identifiers, "u": u, "v": v, "degree_u": degree[u], "degree_v": degree[v],
            "jaccard": weights[u, v], "triangle_edge": weights[u, v] > 0,
            "bridge": frozenset((u, v)) in bridge_edges,
            "p_u_from_v": directed[index, 0], "p_v_from_u": directed[index, 1],
            "q": q[index], "initial_gap": initial_gaps[index],
        }
        row.update({model: values[index] for model, values in assignments.items()})
        tables["edges"].append(row)

    tables["static_pair"].append({
        **identifiers, "edges": len(edges), "nonempty_step_mass": float(q.sum()),
        "structure_minus_edge_mean": float(q @ structured / q.sum() - structured.mean()),
        "shuffled_minus_structured_exposure": float(q @ (shuffled - structured) / q.sum()),
        **gate_contrast(initial_gaps, structured, shuffled, q),
        **permutation_reference(structured, initial_gaps, q, seed),
    })
    for model, thresholds in assignments.items():
        tables["exposure"].append({
            **identifiers, "model": model, "network_sha256": graph_hash,
            "initial_sha256": initial_hash,
            "threshold_assignment_sha256": threshold_fingerprint(thresholds),
            "edge_mean": float(thresholds.mean()),
            "proposal_mean": float(q @ thresholds / q.sum()),
            **{"proposal_p{}".format(p): v for p, v in
               zip(PERCENTILES, weighted_quantiles(thresholds, q))},
            "mass_above_baseline": float(q @ (thresholds > trust.epsilon_0)),
            "initial_pass_mass": float(q @ (initial_gaps <= thresholds)),
            "iid_uniform_initial_pass_expectation": float(q @ (2 * thresholds - thresholds**2)),
        })
    if static_only:
        return

    proposal_hashes = set()
    for model, thresholds in assignments.items():
        engine_model = STRUCTURED if model == PROPOSAL_MEAN else model
        engine = ObservedDyadicDynamics(
            graph, opinions, weights, [], {}, {}, trust, DYNAMICS["mu"],
            DYNAMICS["beta_default"], engine_model, edge_thresholds=overrides[model],
        )
        if (fingerprint_graph(engine.G) != graph_hash
                or threshold_fingerprint(engine.get_opinions_array()) != initial_hash):
            raise RuntimeError("pilot initial conditions are not paired")
        rng = np.random.RandomState()
        rng.set_state(dynamics_rng_state)
        states, windows, proposal_hash = observe_windows(engine, rng, edges, SWEEP_POINTS)
        proposal_hashes.add(proposal_hash)
        tables["pairing"].append({
            **identifiers, "model": model, "network_sha256": graph_hash,
            "initial_sha256": initial_hash, "proposal_sequence_sha256": proposal_hash,
        })
        for sweep, state in states.items():
            values = np.asarray(list(state.values()))
            tables["snapshots"].append({
                **identifiers, "state_model": model, "sweep": sweep,
                "order_parameter": Metrics.order_parameter(values),
                "opinion_variance": Metrics.opinion_variance(values),
                "cluster_count": Metrics.count_clusters(values),
                **gate_contrast(gaps_for_edges(state, edges), structured, shuffled, q),
            })
        for (start, end), counts in windows.items():
            window = {**identifiers, "model": model, "start_sweep": start, "end_sweep": end}
            tables["windows"].append({
                **window, **summarize_counts(counts, thresholds, (end - start) * N),
            })
            for (u, v), count in zip(edges, counts):
                tables["edge_windows"].append({
                    **window, "u": u, "v": v, "proposals": int(count[0]),
                    "accepted": int(count[1]), "nonzero_moves": int(count[2]),
                    "abs_movement": float(count[3]),
                })
    if len(proposal_hashes) != 1:
        raise RuntimeError("dyadic proposal sequences differ across paired arms")


def write_tables(output, tables):
    for name, rows in tables.items():
        if rows:
            with (output / (name + ".csv")).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if "formal_revision" in output.parts or output.exists():
        parser.error("use a new output directory outside formal_revision")

    tables = {name: [] for name in (
        "edges", "static_pair", "exposure", "pairing", "snapshots", "windows", "edge_windows",
    )}
    for p_hk in P_VALUES:
        for seed in SEEDS:
            run_network(p_hk, seed, args.static_only, tables)
    output.mkdir(parents=True, exist_ok=False)
    write_tables(output, tables)
    source_files = [Path(__file__).resolve(), CODE / "dynamics.py", CODE / "simulation.py",
                    CODE / "network.py", CODE / "metrics.py", CODE / "config.py"]
    metadata = {
        "data_class": "static_generated_graph_diagnostic" if args.static_only
        else "small_holme_kim_simulation_pilot",
        "n": N, "m": M, "p_hk": P_VALUES, "seeds": SEEDS,
        "static_permutations_per_network": PERMUTATIONS,
        "dynamic_shuffle_seed": "(seed + 32452843) mod 2^32 (baseline assignment)",
        "static_permutation_reference_seed": "seed + 8975611",
        "sweep_points": () if args.static_only else SWEEP_POINTS,
        "elementary_opportunities": 0 if args.static_only else
        N * SWEEP_POINTS[-1] * len(P_VALUES) * len(SEEDS) * len(MODELS),
        "models": MODELS, "epsilon_0": TRUST["epsilon_0_default"], "alpha": TRUST["alpha"],
        "mu": DYNAMICS["mu"], "order_window_width": METRICS["cluster_diameter_threshold"],
        "python": platform.python_version(), "numpy": np.__version__,
        "networkx": nx.__version__, "source_sha256": {
            path.relative_to(CODE).as_posix(): sha256(path.read_bytes()).hexdigest()
            for path in source_files
        },
        "row_counts": {name: len(rows) for name, rows in tables.items()},
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata["row_counts"], sort_keys=True))


if __name__ == "__main__":
    main()
