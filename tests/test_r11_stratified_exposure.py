"""Targeted checks for the matched threshold-assignment intervention."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import r11_stratified_exposure as stratified


class StratifiedExposureTests(unittest.TestCase):
    def test_mathematical_q_ties_are_not_split_by_degree_pair_or_band(self):
        # (3,6) and (4,4) have the same reciprocal degree sum.
        edges = [(0, 1), (2, 3), (4, 5), (6, 7)]
        degrees = dict(enumerate([3, 6, 4, 4, 10, 10, 1, 2]))
        exact, bands = stratified.proposal_strata(edges, degrees, 4)
        self.assertEqual([x.tolist() for x in exact], [[2], [0, 1], [3]])
        self.assertIn([0, 1], [x.tolist() for x in bands])
        self.assertEqual(sorted(np.concatenate(bands)), list(range(4)))

    def test_shuffle_retains_multisets_in_every_group(self):
        original = np.array([0.1, 0.5, 0.6, 0.2, 0.7])
        groups = [np.array([0, 2, 4]), np.array([1, 3])]
        shuffled = stratified.permute_within_strata(original, groups, np.random.RandomState(7))
        for indices in groups:
            np.testing.assert_array_equal(np.sort(original[indices]), np.sort(shuffled[indices]))

    def test_exact_weighted_exposure_can_mask_a_gate_difference(self):
        original = np.array([0.2, 0.6, 0.9])
        shuffled = np.array([0.6, 0.2, 0.9])
        q = np.array([0.2, 0.2, 0.6])
        gaps = np.array([0.4, 0.8, 0.0])
        metrics = stratified.measure_assignment(
            original, shuffled, original, [np.array([0, 1]), np.array([2])], gaps, q)
        self.assertAlmostEqual(metrics["cdf_sup_distance"], 0)
        self.assertAlmostEqual(metrics["wasserstein_distance"], 0)
        self.assertAlmostEqual(metrics["proposal_mean_difference"], 0)
        self.assertAlmostEqual(metrics["changed_proposal_mass"], 0.4)
        self.assertAlmostEqual(metrics["shuffled_minus_structured"], 0.2)

    def test_distances_with_two_support_points(self):
        metrics = stratified.weighted_cdf_distances(
            np.array([0.2, 0.6]), np.array([0.6, 0.2]), np.array([0.3, 0.7]))
        self.assertAlmostEqual(metrics["cdf_sup_distance"], 0.4)
        self.assertAlmostEqual(metrics["wasserstein_distance"], 0.16)
        self.assertAlmostEqual(metrics["proposal_mean_difference"], -0.16)

    def test_tiny_paired_run_preserves_proposal_stream_and_reconciles_components(self):
        tables = {key: [] for key in ("edges", "assignments", "edge_assignments",
                                      "pairing", "snapshots", "windows", "realized_exposure")}
        with patch.multiple(stratified, N=24, M=2, DRAWS=2, SWEEPS=(0, 1, 3)):
            stratified.run_network(0.5, 7, tables, False)
        self.assertEqual(len(tables["pairing"]), 7)
        for key in ("network_sha256", "initial_sha256", "proposal_sequence_sha256"):
            self.assertEqual(len({row[key] for row in tables["pairing"]}), 1)
        self.assertEqual(sum(row["opportunities"] for row in tables["windows"]), 7 * 24 * 3)
        for row in tables["snapshots"]:
            self.assertAlmostEqual(row["gate_diagonal_difference"],
                                   row["gate_assignment_component"] + row["gate_state_component"])
            if row["sweep"] == 0:
                self.assertEqual(row["order_difference"], 0)
                self.assertEqual(row["gate_state_component"], 0)
        for row in tables["assignments"]:
            if row["design"] == stratified.EXACT:
                self.assertAlmostEqual(row["cdf_sup_distance"], 0)


if __name__ == "__main__":
    unittest.main()
