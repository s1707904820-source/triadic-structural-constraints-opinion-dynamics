"""Regression coverage for failed-triangle fallback and archived-data validation."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np
from scipy import stats

CODE = Path(__file__).resolve().parents[1]
for directory in (CODE, CODE / "analysis"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import analyze_formal_triangle_failure_control as analysis
from dynamics import OpinionDynamics, TrustThreshold
from network import NetworkGenerator
from statistics_utils import summarize as historical_summary


class TriangleFailureAnalysisTests(unittest.TestCase):
    def archived_fixture(self, summarize):
        seeds = [7, 1004, 2001, 2998]
        values = {
            "tri_count": [4, 5, 6, 7],
            "tri_success": [1, 1, 2, 3],
            "tri_fallback": [3, 4, 4, 4],
            "tri_fallback_edge_update": [1, 2, 3, 4],
            "tri_fallback_suppressed": [0, 0, 0, 0],
            "update_count": [2, 4, 6, 8],
        }
        control_values = {**values,
                          "tri_fallback_edge_update": [0, 0, 0, 0],
                          "tri_fallback_suppressed": values["tri_fallback"]}
        full, control = [
            {**{metric: summarize([0.1, 0.2, 0.3, 0.4])
                for metric in analysis.FINAL_METRICS},
             "dynamics_stats": {metric: summarize(vector)
                                for metric, vector in counts.items()}}
            for counts in (values, control_values)
        ]
        pair_key = "full_model_minus_" + analysis.COMPARATOR
        block = {
            "full_model": full, analysis.COMPARATOR: control, "_control": {"seeds": seeds},
            "_paired_comparisons": {
                pair_key: {metric: summarize([0.0] * 4)
                           for metric in analysis.FINAL_METRICS},
                pair_key + "_dynamics": {
                    metric: summarize(np.asarray(values[metric]) - control_values[metric])
                    for metric in analysis.DYNAMICS_METRICS
                },
            },
        }
        return (
            {"control": {"N": 2000, "m": 4, "num_runs": 4,
                         "model_order": list(analysis.MODELS), "main_sweeps": 200},
             "data": {"0.00": block}},
            {"n_runs": 4, "seeds": seeds, "config": {"SIMULATION": {"main_sweeps": 200}}},
        )

    def validate_fixture(self, payload, metadata):
        with patch.multiple(analysis, P_VALUES=[0.0], N_RUNS=4, SEEDS=metadata["seeds"]):
            return analysis.validate(payload, metadata)

    def test_validation_accepts_exact_and_historical_intervals(self):
        for summarize in (analysis.summarize, historical_summary):
            with self.subTest(summarize=summarize.__module__):
                result = self.validate_fixture(*self.archived_fixture(summarize))
                self.assertEqual(result["status"], "PASS", result["errors"])

    def test_new_interval_uses_exact_student_t_quantile(self):
        values = np.asarray([0.1, 0.3, 0.2, 0.17])
        exact = analysis.summarize(values)
        expected = stats.t.interval(0.95, len(values) - 1, loc=np.mean(values),
                                    scale=stats.sem(values))
        np.testing.assert_allclose(
            [exact["ci95_low"], exact["ci95_high"]], expected, rtol=0, atol=1e-15)
        self.assertNotAlmostEqual(exact["ci95_low"],
                                  historical_summary(values)["ci95_low"], places=6)

    def test_malformed_array_is_rejected_and_other_counters_are_still_checked(self):
        payload, metadata = self.archived_fixture(historical_summary)
        block = payload["data"]["0.00"]
        block["full_model"]["order_param"]["values"] = [[0.1] * 4]
        block[analysis.COMPARATOR]["dynamics_stats"]["tri_count"]["values"][0] = 99
        result = self.validate_fixture(payload, metadata)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("order_param" in x for x in result["errors"]))
        self.assertTrue(any("attempts differ" in x for x in result["errors"]))

    def test_invalid_interval_and_fallback_counts_are_reported(self):
        payload, metadata = self.archived_fixture(historical_summary)
        block = payload["data"]["0.00"]
        block["full_model"]["order_param"]["ci95_low"] = None
        block["full_model"]["dynamics_stats"]["tri_fallback_edge_update"]["values"][0] = 4
        block[analysis.COMPARATOR]["dynamics_stats"]["tri_fallback_edge_update"]["values"][0] = 1
        errors = self.validate_fixture(payload, metadata)["errors"]
        for message in ("confidence interval", "exceed failed gates", "contains accepted fallback"):
            self.assertTrue(any(message in error for error in errors), errors)

    def test_invalid_input_cannot_write_analysis_files(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "analysis_output"
            with self.assertRaisesRegex(ValueError, "invalid failed-triangle data"):
                analysis.analyze({}, {}, {"errors": ["bad data"]}, output_dir=output)
            self.assertFalse(output.exists())

    def test_failed_triangle_samples_from_all_neighbors_and_only_updates_focal_node(self):
        graph = nx.Graph([(0, 1), (0, 2), (1, 2), (0, 3)])
        opinions = {0: 0.0, 1: 0.1, 2: 0.3, 3: 0.125}
        triangles, index = NetworkGenerator.extract_triangles(graph)

        class Draws:
            def __init__(self):
                self.draws = iter([0, 0, 2])
                self.bounds = []

            def randint(self, high):
                self.bounds.append(high)
                return next(self.draws)

            def rand(self):
                return 0.0

        for model in analysis.MODELS:
            with self.subTest(model=model):
                engine = OpinionDynamics(
                    graph, opinions.copy(), {}, triangles, index, {0: 1.0},
                    TrustThreshold(0.15, 3.5), mu=0.5, beta=3.0, model_type=model)
                draws = Draws()
                dx = engine.step(draws)
                self.assertEqual(draws.bounds, [4, 1, 3])
                self.assertEqual(engine.get_stats()["tri_fallback"], 1)
                expected = 0.0625 if model == "full_model" else 0.0
                self.assertEqual(dx, expected)
                self.assertEqual(engine.get_opinions(), {**opinions, 0: expected})

    def test_failed_triangle_arms_advance_numpy_rng_identically(self):
        graph = nx.complete_graph(3)
        triangles, index = NetworkGenerator.extract_triangles(graph)
        states = []
        for model in analysis.MODELS:
            engine = OpinionDynamics(
                graph, {0: 0.0, 1: 0.1, 2: 0.3}, {}, triangles, index,
                {node: 1.0 for node in graph}, TrustThreshold(0.15, 3.5),
                mu=0.5, beta=3.0, model_type=model)
            rng = np.random.RandomState(0)
            engine.step(rng)
            self.assertEqual(engine.get_stats()["tri_fallback"], 1)
            states.append(rng.get_state())
        np.testing.assert_equal(states[0], states[1])


if __name__ == "__main__":
    unittest.main()
