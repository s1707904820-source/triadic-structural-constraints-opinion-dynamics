"""Vector-valued extension of the opinion dynamics for robustness checks."""

from __future__ import annotations

import numpy as np


class VectorOpinionDynamics:
    """Asynchronous vector opinions with Euclidean bounded confidence."""

    def __init__(self, G, opinions, weights, triangles_list, node_triangles,
                 clustering, trust_model, mu=0.5, beta=3.0,
                 model_type='full_model'):
        allowed = {'threshold_only', 'no_triadic_feedback', 'full_model'}
        if model_type not in allowed:
            raise ValueError(f"vector model_type must be one of {sorted(allowed)}")
        self.G = G
        self.opinions = {
            node: np.asarray(value, dtype=float).copy()
            for node, value in opinions.items()
        }
        dimensions = {value.shape for value in self.opinions.values()}
        if len(dimensions) != 1 or not dimensions or len(next(iter(dimensions))) != 1:
            raise ValueError("all vector opinions must have the same 1D shape")
        self.dimension = int(next(iter(dimensions))[0])
        if self.dimension <= 0:
            raise ValueError("vector opinions must have at least one dimension")
        if any(not np.all(np.isfinite(value)) for value in self.opinions.values()):
            raise ValueError("vector opinions must contain only finite values")
        if any(np.any((value < 0.0) | (value > 1.0))
               for value in self.opinions.values()):
            raise ValueError("vector opinions must lie in [0, 1]^dimension")
        if set(self.opinions) != set(G.nodes()):
            raise ValueError("vector opinion nodes must exactly match graph nodes")
        self.weights = weights
        self.triangles_list = triangles_list
        self.node_triangles = node_triangles
        self.clustering = clustering
        self.trust = trust_model
        self.mu = float(mu)
        self.beta = float(beta)
        if not np.isfinite(self.mu) or not 0.0 <= self.mu <= 1.0:
            raise ValueError("mu must be finite and lie in [0, 1]")
        if not np.isfinite(self.beta) or self.beta < 0.0:
            raise ValueError("beta must be finite and non-negative")
        self.model_type = model_type
        self.nodes = list(G.nodes())
        self.N = len(self.nodes)
        self.neighbors = {node: list(G.neighbors(node)) for node in self.nodes}
        self.stats = {
            'edge_count': 0, 'tri_count': 0, 'tri_success': 0,
            'tri_fallback': 0, 'update_count': 0,
        }

    def _distance(self, a, b):
        return float(np.linalg.norm(self.opinions[a] - self.opinions[b]))

    def _edge_interaction(self, i, rng):
        self.stats['edge_count'] += 1
        j = self.neighbors[i][rng.randint(len(self.neighbors[i]))]
        threshold = self.trust.effective_threshold(
            self.weights.get((i, j), 0.0))
        if self._distance(i, j) <= threshold:
            self.opinions[i] += self.mu * (
                self.opinions[j] - self.opinions[i])
            self.stats['update_count'] += 1

    def _triangle_interaction(self, i, rng, use_feedback):
        tri_indices = self.node_triangles[i]
        triangle = self.triangles_list[
            tri_indices[rng.randint(len(tri_indices))]]
        nodes = list(triangle)
        pairs = ((nodes[0], nodes[1]), (nodes[1], nodes[2]),
                 (nodes[0], nodes[2]))
        for a, b in pairs:
            threshold = self.trust.effective_threshold(
                self.weights.get((a, b), 0.0))
            if self._distance(a, b) > threshold:
                self.stats['tri_fallback'] += 1
                self._edge_interaction(i, rng)
                return
        self.stats['tri_success'] += 1
        others = [node for node in nodes if node != i]
        rng.shuffle(others)
        j, k = others
        factor = (np.exp(-self.beta * self._distance(i, k))
                  if use_feedback else 1.0)
        self.opinions[i] += self.mu * factor * (
            self.opinions[j] - self.opinions[i])
        self.stats['update_count'] += 1

    def step(self, rng):
        i = self.nodes[rng.randint(self.N)]
        if not self.neighbors[i]:
            return 0.0
        old = self.opinions[i].copy()
        if self.model_type == 'threshold_only':
            self._edge_interaction(i, rng)
        else:
            has_triangles = bool(self.node_triangles.get(i))
            if has_triangles and rng.rand() < self.clustering.get(i, 0.0):
                self.stats['tri_count'] += 1
                self._triangle_interaction(
                    i, rng, use_feedback=self.model_type == 'full_model')
            else:
                self._edge_interaction(i, rng)
        self.opinions[i] = np.clip(self.opinions[i], 0.0, 1.0)
        return float(np.linalg.norm(self.opinions[i] - old))

    def get_opinions_array(self):
        return np.vstack([self.opinions[node] for node in self.nodes])

    def get_opinions(self):
        return {node: value.copy() for node, value in self.opinions.items()}

    def get_stats(self):
        return dict(self.stats)
