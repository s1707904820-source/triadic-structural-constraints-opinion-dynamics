"""Audit tests for the R1-1 threshold-assignment comparison."""

import unittest

import networkx as nx
import numpy as np

from dynamics import OpinionDynamics, TrustThreshold
from network import NetworkGenerator
from simulation import SingleRun


class RecordingRNG:
    def __init__(self, seed):
        self._rng = np.random.RandomState(seed)
        self.randint_calls = []

    def randint(self, high):
        value = int(self._rng.randint(high))
        self.randint_calls.append((int(high), value))
        return value


class AssignmentStreamAuditTests(unittest.TestCase):
    def test_random_reassignment_preserves_multiset_and_update_stream(self):
        graph = nx.powerlaw_cluster_graph(80, 3, 0.2, seed=7)
        triangles, node_triangles = NetworkGenerator.extract_triangles(graph)
        weights = NetworkGenerator.compute_jaccard_weights(graph)
        clustering = NetworkGenerator.compute_clustering(graph)
        trust = TrustThreshold(0.15, 3.5)
        structural = {
            (u, v): trust.effective_threshold(weights[(u, v)])
            for u, v in graph.edges()}
        structural.update({(v, u): value for (u, v), value in structural.items()})
        randomized = SingleRun._build_edge_threshold_assignment(
            graph, weights, trust, 'randomized_threshold_only', seed=7)
        self.assertEqual(sorted(structural.values()), sorted(randomized.values()))
        initial = {node: float((node + 1) / 100.0) for node in graph.nodes()}
        engines = [OpinionDynamics(
            graph, initial, weights, triangles, node_triangles, clustering,
            trust, 0.5, 3.0, 'threshold_only', edge_thresholds=assignment)
                   for assignment in (structural, randomized)]
        rngs = [RecordingRNG(991), RecordingRNG(991)]
        for _ in range(300):
            engines[0].step(rngs[0])
            engines[1].step(rngs[1])
        self.assertEqual(rngs[0].randint_calls, rngs[1].randint_calls)

    def test_structural_assignment_is_baseline_on_nontriangle_edges(self):
        graph = nx.Graph([(0, 1), (1, 2), (2, 0), (2, 3)])
        weights = NetworkGenerator.compute_jaccard_weights(graph)
        trust = TrustThreshold(0.15, 3.5)
        assignment = {(u, v): trust.effective_threshold(weights[(u, v)])
                      for u, v in graph.edges()}
        assignment.update({(v, u): value for (u, v), value in assignment.items()})
        self.assertEqual(assignment[(2, 3)], trust.effective_threshold(0.0))


if __name__ == '__main__':
    unittest.main()
