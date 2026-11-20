"""Regression checks for the R1-1 exposure observer and probability tables."""

from itertools import permutations
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

import r11_threshold_exposure as exposure
from dynamics import OpinionDynamics, TrustThreshold
from network import NetworkGenerator
from simulation import SingleRun


class ExposureTests(unittest.TestCase):
    def setUp(self):
        self.graph = nx.Graph([(0, 1), (0, 2), (1, 2), (2, 3)])
        self.edges = list(self.graph.edges())
        self.weights = NetworkGenerator.compute_jaccard_weights(self.graph)
        self.trust = TrustThreshold(0.25, 3.5)

    def test_triangle_with_tail_has_hand_calculable_probabilities(self):
        directed = exposure.edge_proposal_probabilities(self.graph, self.edges)
        expected = [(1/8, 1/8), (1/8, 1/12), (1/8, 1/12), (1/12, 1/4)]
        np.testing.assert_allclose(directed, expected)
        np.testing.assert_allclose(directed.sum(axis=1), [1/4, 5/24, 5/24, 1/3])
        self.assertAlmostEqual(directed.sum(), 1)

    def test_regular_graph_has_uniform_undirected_probabilities(self):
        graph = nx.cycle_graph(5)
        directed = exposure.edge_proposal_probabilities(graph, list(graph.edges()))
        np.testing.assert_allclose(directed.sum(axis=1), 0.2)

    def test_isolated_nodes_are_null_opportunities(self):
        self.graph.add_node(4)
        q = exposure.edge_proposal_probabilities(self.graph, self.edges)
        self.assertAlmostEqual(q.sum(), 4/5)
        graph = nx.empty_graph(3)
        self.assertEqual(exposure.edge_proposal_probabilities(graph, []).shape, (0, 2))

    def test_invalid_graph_types_rejected(self):
        self.graph.add_edge(3, 3)
        for graph in (self.graph, nx.DiGraph([(1, 2)]), nx.MultiGraph([(1, 2)]), nx.Graph()):
            with self.subTest(graph=graph):
                with self.assertRaises(ValueError):
                    exposure.edge_proposal_probabilities(graph, list(graph.edges()))

    def test_fixed_state_includes_boundary_and_different_pass_masses(self):
        result = exposure.gate_contrast(
            np.array([0.25, 0.4, 0.6]), np.array([0.25, 0.3, 0.7]),
            np.array([0.1, 0.4, 0.7]), np.array([0.2, 0.3, 0.5]),
        )
        self.assertAlmostEqual(result["structured_pass_mass"], 0.7)
        self.assertAlmostEqual(result["shuffled_pass_mass"], 0.8)
        self.assertAlmostEqual(result["shuffled_only_mass"], 0.3)
        self.assertAlmostEqual(result["structured_only_mass"], 0.2)
        self.assertAlmostEqual(result["shuffled_minus_structured"], 0.1)

    def test_weighted_empirical_quantiles_include_cdf_jumps(self):
        result = exposure.weighted_quantiles(np.array([0.6, 0.2, 0.4]), np.array([0.2, 0.5, 0.3]))
        np.testing.assert_array_equal(result, [0.2, 0.2, 0.2, 0.4, 0.6])

    def test_static_permutation_reference_agrees_with_exact_enumeration(self):
        thresholds = np.array([0.25, 0.4, 0.7])
        gaps = np.array([0.25, 0.5, 0.3])
        q = np.array([0.2, 0.3, 0.5])
        expected = np.mean([q @ (gaps <= draw) for draw in permutations(thresholds)])
        with patch.object(exposure, "PERMUTATIONS", 12):
            result = exposure.permutation_reference(thresholds, gaps, q, 7)
        self.assertAlmostEqual(result["expected_shuffled_initial_pass_mass"], expected)
        self.assertAlmostEqual(result["expected_shuffled_exposure"], thresholds.mean())

    def test_assignments_preserve_multiset_and_match_the_intended_means(self):
        q = exposure.edge_proposal_probabilities(self.graph, self.edges).sum(axis=1)
        assignments, overrides = exposure.threshold_assignments(
            self.graph, self.edges, self.weights, self.trust, 7, q)
        structured = assignments[exposure.STRUCTURED]
        shuffled = assignments[exposure.SHUFFLED]
        np.testing.assert_array_equal(np.sort(structured), np.sort(shuffled))
        np.testing.assert_allclose(assignments[exposure.EDGE_MEAN], structured.mean())
        np.testing.assert_allclose(assignments[exposure.PROPOSAL_MEAN], q @ structured)
        self.assertNotAlmostEqual(structured.mean(), q @ structured)
        self.assertIsNone(overrides[exposure.STRUCTURED])

    def test_observation_preserves_states_statistics_displacement_and_rng(self):
        self.graph.add_node(4)
        q = exposure.edge_proposal_probabilities(self.graph, self.edges).sum(axis=1)
        opinions, rng_state = exposure.initial_opinions(self.graph, 7)
        _, overrides = exposure.threshold_assignments(
            self.graph, self.edges, self.weights, self.trust, 7, q)
        for model in exposure.MODELS:
            with self.subTest(model=model):
                engine_model = exposure.STRUCTURED if model == exposure.PROPOSAL_MEAN else model
                kwargs = {"model_type": engine_model, "edge_thresholds": overrides[model]}
                args = self.graph, opinions, self.weights, [], {}, {}, self.trust
                actual = exposure.ObservedDyadicDynamics(*args, **kwargs)
                reference = OpinionDynamics(*args, **kwargs)
                rng_actual = np.random.RandomState()
                rng_reference = np.random.RandomState()
                rng_actual.set_state(rng_state)
                rng_reference.set_state(rng_state)
                states, windows, _ = exposure.observe_windows(actual, rng_actual, self.edges, (0, 1, 3, 20))
                dx = [reference.step(rng_reference) for _ in range(20 * len(self.graph))]
                self.assertEqual(states[20], reference.get_opinions())
                self.assertEqual(actual.get_stats(), reference.get_stats())
                np.testing.assert_equal(rng_actual.get_state(), rng_reference.get_state())
                self.assertAlmostEqual(sum(w[:, 3].sum() for w in windows.values()), sum(dx))

    def test_gate_pass_is_separate_from_nonzero_movement(self):
        graph = nx.path_graph(2)
        engine = exposure.ObservedDyadicDynamics(
            graph, {0: 0.5, 1: 0.5}, {(0, 1): 0}, [], {}, {},
            self.trust, model_type=exposure.STRUCTURED)
        _, windows, _ = exposure.observe_windows(
            engine, np.random.RandomState(7), list(graph.edges()), (0, 1))
        row = exposure.summarize_counts(windows[0, 1], np.array([0.25]), 2)
        self.assertEqual(row["accepted"], 2)
        self.assertEqual(row["nonzero_moves"], 0)
        self.assertIsNone(row["movement_weighted_threshold_mean"])

    def test_paired_hk_observer_matches_normal_single_run_entry_point(self):
        tables = {name: [] for name in (
            "edges", "static_pair", "exposure", "pairing", "snapshots", "windows", "edge_windows",
        )}
        with patch.multiple(exposure, N=18, M=2, SWEEP_POINTS=(0, 1, 2), PERMUTATIONS=20):
            exposure.run_network(0.5, 7, False, tables)
        self.assertEqual(len({row["proposal_sequence_sha256"] for row in tables["pairing"]}), 1)
        self.assertEqual(len(tables["pairing"]), 4)
        initial = [row for row in tables["snapshots"] if row["sweep"] == 0]
        self.assertEqual(len({row["shuffled_minus_structured"] for row in initial}), 1)
        for model in (exposure.STRUCTURED, exposure.SHUFFLED, exposure.EDGE_MEAN):
            with self.subTest(model=model):
                observed = next(row for row in tables["snapshots"]
                                if row["sweep"] == 2 and row["state_model"] == model)
                result = SingleRun(N=18, m=2, p_hk=0.5, seed=7, num_steps=36,
                                   model_type=model).execute(record_interval=18)
                self.assertAlmostEqual(observed["order_parameter"], result["final"]["order_param"])
                self.assertAlmostEqual(observed["opinion_variance"], result["final"]["variance"])
                self.assertEqual(observed["cluster_count"], result["final"]["n_clusters"])


if __name__ == "__main__":
    unittest.main()
