import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
ANALYSIS_ROOT = os.path.join(ROOT, 'analysis')
if ANALYSIS_ROOT not in sys.path:
    sys.path.insert(0, ANALYSIS_ROOT)

from config import DYNAMICS, METRICS, MODEL_DISPLAY_ORDER, MODEL_TYPES, NETWORK, SIMULATION, TRUST
import config as config_module
from run_all import configure_output_root

# Keep import-time plotting setup and test artifacts out of the read-only
# revision package directory.
TEST_OUTPUT_ROOT = tempfile.mkdtemp(prefix='opinion-dynamics-tests-')
configure_output_root(TEST_OUTPUT_ROOT)

import experiments as experiments_module
import plotting as plotting_module
from dynamics import OpinionDynamics, TrustThreshold
from experiments import ExperimentManager
from metrics import Metrics
from network import NetworkGenerator
from simulation import SingleRun
from statistics_utils import (
    _t975, paired_difference, summarize, summarize_proportion,
)
from vector_dynamics import VectorOpinionDynamics
from run_all import estimate_elementary_updates
from run_all import build_parser, validate_cli_args
from provenance import (
    _python_source_manifest, execution_source_signature,
)
from run_formal_sbm_hysteresis import (
    FORMAL_OUTPUT_FILENAME,
    estimate_updates as estimate_sbm_hysteresis_updates,
    main as run_formal_sbm_hysteresis_main,
)
from analyze_formal_ablation import (
    TIME_SERIES_MODELS, holm_adjust, paired_t_pvalue, save_figure_formats,
)
from analyze_formal_threshold_transition_prediction import (
    P_VALUES as TRANSITION_ANALYSIS_P_VALUES,
    TRANSITION_ORDER_WINDOW_WIDTH,
    TRANSITION_CUTOFF,
    add_figure_legend,
    major_p_ticks,
    trajectory_diagnostics,
    validate_transition_metric_config, validate as validate_transition_payload,
    build_seed_rows as build_transition_seed_rows,
    build_summaries as build_transition_summaries,
)
from analyze_formal_network_thresholds import triangle_edge_partition
from run_formal_threshold_transition_prediction import (
    FORMAL_OUTPUT_FILENAME as TRANSITION_OUTPUT_FILENAME,
    P_VALUES as TRANSITION_RUNNER_P_VALUES,
    estimate_updates as estimate_transition_updates,
    validate_runner_args as validate_transition_runner_args,
    main as run_formal_transition_prediction_main,
)


class RevisionCoreTests(unittest.TestCase):
    @staticmethod
    def _summary_stub(value):
        return {
            'n': 1, 'mean': float(value), 'sd': 0.0, 'std': 0.0,
            'se': 0.0, 'ci95_low': float(value),
            'ci95_high': float(value), 'values': [float(value)],
        }

    def test_triangle_order_is_deterministic(self):
        graph = nx.complete_graph(5)
        triangles, _ = NetworkGenerator.extract_triangles(graph)
        self.assertEqual(triangles, sorted(triangles))

    def test_triangle_edge_partition_is_independent_of_jaccard_weights(self):
        graph = nx.Graph()
        graph.add_edges_from([
            (0, 1), (1, 2), (2, 0), (2, 3), (3, 4),
        ])
        triangle_edges, nontriangle_edges = triangle_edge_partition(graph)
        expected_triangle_edges = {
            frozenset((0, 1)), frozenset((1, 2)), frozenset((0, 2)),
        }
        expected_nontriangle_edges = {
            frozenset((2, 3)), frozenset((3, 4)),
        }
        self.assertEqual(triangle_edges, expected_triangle_edges)
        self.assertEqual(nontriangle_edges, expected_nontriangle_edges)

        weights = NetworkGenerator.compute_jaccard_weights(graph)
        edge_weight = {
            frozenset((u, v)): weights[(u, v)] for u, v in graph.edges()
        }
        self.assertTrue(all(edge_weight[edge] > 0.0
                            for edge in triangle_edges))
        self.assertTrue(all(edge_weight[edge] == 0.0
                            for edge in nontriangle_edges))

    def test_triangle_edge_partition_rejects_non_simple_undirected_graphs(self):
        with self.assertRaises(ValueError):
            triangle_edge_partition(nx.DiGraph([(0, 1), (1, 2), (2, 0)]))
        multi = nx.MultiGraph()
        multi.add_edges_from([(0, 1), (1, 2), (2, 0)])
        with self.assertRaises(ValueError):
            triangle_edge_partition(multi)

    def test_triangle_metrics_have_distinct_semantics(self):
        graph = nx.complete_graph(3)
        triangles, _ = NetworkGenerator.extract_triangles(graph)
        weights = NetworkGenerator.compute_jaccard_weights(graph)
        trust = TrustThreshold(0.15, 3.5)
        opinions = {0: 0.0, 1: 0.20, 2: 0.20}
        fixed = Metrics.triangle_alignment_fixed(opinions, triangles, 0.15)
        adaptive = Metrics.triangle_admissibility_adaptive(
            opinions, triangles, weights, trust)
        self.assertEqual(fixed, 0.0)
        self.assertEqual(adaptive, 1.0)

    def test_edge_penalty_sampling_is_reproducible(self):
        graph = nx.complete_graph(110)
        opinions = {node: node / 109 for node in graph.nodes()}
        a = Metrics.edge_disagreement_penalty(
            opinions, graph, 3.0, rng=np.random.RandomState(123))
        b = Metrics.edge_disagreement_penalty(
            opinions, graph, 3.0, rng=np.random.RandomState(123))
        self.assertEqual(a, b)

    def test_binary_recovery_probability_uses_wilson_interval(self):
        summary = summarize_proportion([True] * 20)
        self.assertEqual(summary['successes'], 20)
        self.assertGreaterEqual(summary['ci95_low'], 0.0)
        self.assertLess(summary['ci95_low'], 1.0)
        self.assertEqual(summary['ci95_high'], 1.0)
        self.assertEqual(summary['interval'], '95% Wilson score interval')

    def test_summary_uses_sample_sd_and_ci(self):
        out = summarize([1.0, 2.0, 3.0])
        self.assertAlmostEqual(out['sd'], 1.0)
        self.assertEqual(out['std'], out['sd'])
        self.assertLess(out['ci95_low'], out['mean'])
        self.assertGreater(out['ci95_high'], out['mean'])
        paired = paired_difference([2, 4, 6], [1, 1, 1])
        self.assertEqual(paired['definition'], 'first_minus_second')

    def test_t_critical_values_remain_accurate_above_30_degrees(self):
        self.assertAlmostEqual(_t975(49), 2.0095752345, places=8)
        self.assertAlmostEqual(_t975(199), 1.9719565442, places=8)

    def test_paired_inference_guards_degenerate_and_invalid_inputs(self):
        self.assertEqual(paired_t_pvalue([0.0, 0.0, 0.0]), 1.0)
        self.assertEqual(paired_t_pvalue([2.0, 2.0, 2.0]), 0.0)
        self.assertTrue(np.isfinite(paired_t_pvalue([1.0, 2.0, 3.0])))
        near_constant_p = paired_t_pvalue([0.0, 1e-10, 2e-10])
        self.assertGreater(near_constant_p, 0.2)
        self.assertLess(near_constant_p, 0.3)
        self.assertEqual(holm_adjust([]), [])
        with self.assertRaises(ValueError):
            paired_t_pvalue([])
        with self.assertRaises(ValueError):
            paired_t_pvalue([0.0, np.nan])
        with self.assertRaises(ValueError):
            holm_adjust([0.01, np.nan])
        with self.assertRaises(ValueError):
            holm_adjust([-0.01, 0.5])

    def test_network_average_path_length_excludes_self_pairs(self):
        properties = NetworkGenerator.compute_full_properties(
            nx.path_graph(3), seed=7)
        self.assertAlmostEqual(properties['avg_path_length'], 4.0 / 3.0)

    def test_empty_only_argument_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                build_parser().parse_args(['--only'])

    def test_cli_rejects_nonfinite_budgets_invalid_hk_size_and_unknown_dry_run(self):
        invalid_args = (
            ['--main-sweeps', 'nan'],
            ['--main-sweeps', 'inf'],
            ['--sweep-sweeps', 'nan'],
            ['--N', str(NETWORK['m'])],
            ['--dry-run', '--only', 'not_an_experiment'],
        )
        for argv in invalid_args:
            with self.subTest(argv=argv):
                args = build_parser().parse_args(argv)
                with self.assertRaises(SystemExit):
                    validate_cli_args(args)

    def test_cli_rejects_ambiguous_step_budgets(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args([
                '--steps', '100', '--main-sweeps', '2'])

    def test_edge_metric_rng_is_independent_and_paired_across_models(self):
        states = []

        def capture_rng(opinions, graph, beta, rng=None, max_edges=5000):
            states.append(rng.get_state())
            return 0.0

        graph = nx.complete_graph(8)
        initial = {node: 0.2 + node * 0.001 for node in graph.nodes()}
        with patch.object(Metrics, 'edge_disagreement_penalty',
                          side_effect=capture_rng):
            for model_type in ('baseline', 'full_model'):
                SingleRun(
                    N=8, m=2, p_hk=0.5, seed=17,
                    model_type=model_type, graph=graph,
                    initial_opinions=initial, num_steps=64,
                ).execute(record_interval=8)
        self.assertEqual(states[0][2], states[1][2])
        self.assertTrue(np.array_equal(states[0][1], states[1][1]))

    def test_model_parameter_ranges_are_validated(self):
        with self.assertRaises(ValueError):
            TrustThreshold(0.15, alpha=-1.0)
        with self.assertRaises(ValueError):
            TrustThreshold(0.15).effective_threshold(1.1)
        with self.assertRaises(ValueError):
            OpinionDynamics(
                nx.path_graph(2), {0: 0.0, 1: 0.1}, {}, [], {}, {},
                TrustThreshold(0.15), mu=1.01)
        with self.assertRaises(ValueError):
            OpinionDynamics(
                nx.path_graph(2), {0: 0.0, 1: 0.1}, {}, [], {}, {},
                TrustThreshold(0.15), beta=-0.01)

    def test_vector_model_parameter_ranges_are_validated(self):
        args = (
            nx.path_graph(2),
            {0: np.array([0.0]), 1: np.array([0.1])},
            {}, [], {}, {}, TrustThreshold(0.15),
        )
        with self.assertRaises(ValueError):
            VectorOpinionDynamics(*args, mu=1.01)
        with self.assertRaises(ValueError):
            VectorOpinionDynamics(*args, beta=-0.01)

    def test_vector_opinion_domain_and_graph_alignment_are_validated(self):
        graph = nx.path_graph(2)
        args = (
            graph, {0: np.array([0.0, 0.2]), 1: np.array([0.1, 0.3])},
            {}, [], {}, {}, TrustThreshold(0.15),
        )
        for bad_values in (
                {0: np.array([-0.01]), 1: np.array([0.3])},
                {0: np.array([np.nan]), 1: np.array([0.3])},
                {0: np.array([np.inf]), 1: np.array([0.3])},
                {0: np.array([]), 1: np.array([])},
                {0: np.array([0.2])}):
            bad_args = (graph, bad_values) + args[2:]
            with self.assertRaises(ValueError):
                VectorOpinionDynamics(*bad_args)
        with self.assertRaises(ValueError):
            VectorOpinionDynamics(
                graph, {0: np.array([0.2]), 1: np.array([1.01])},
                *args[2:])

    def test_source_manifest_includes_tests_only_when_requested(self):
        test_path = "tests/test_revision_core.py"
        self.assertIn(test_path, _python_source_manifest(include_tests=True))
        self.assertNotIn(test_path, _python_source_manifest(include_tests=False))

    def test_main_batch_uses_sweeps_and_new_metric_names(self):
        old_sweeps = SIMULATION['main_sweeps']
        old_record = SIMULATION['record_sweeps']
        try:
            SIMULATION['main_sweeps'] = 2
            SIMULATION['record_sweeps'] = 1
            result = ExperimentManager()._run_batch(
                {'N': 30, 'm': 2, 'p_hk': 0.5, 'model_type': 'full_model'},
                num_runs=1,
            )[0]
        finally:
            SIMULATION['main_sweeps'] = old_sweeps
            SIMULATION['record_sweeps'] = old_record
        self.assertEqual(result['params']['num_steps'], 60)
        self.assertEqual(result['params']['num_sweeps'], 2.0)
        self.assertEqual(result['time_series']['sweep'], [1.0, 2.0])
        self.assertIn('triangle_alignment_fixed', result['final'])
        self.assertIn('triangle_admissibility_adaptive', result['final'])
        self.assertIn('edge_disagreement_penalty', result['final'])
        self.assertNotIn('tri_consensus', result['final'])
        self.assertNotIn('rep_frag', result['final'])

    def test_main_batch_default_fallback_is_long_horizon(self):
        old_main = SIMULATION.pop('main_sweeps', None)
        old_record = SIMULATION.pop('record_sweeps', None)
        try:
            result = ExperimentManager()._run_batch(
                {'N': 12, 'm': 2, 'p_hk': 0.5, 'model_type': 'full_model'},
                num_runs=1,
            )[0]
        finally:
            if old_main is not None:
                SIMULATION['main_sweeps'] = old_main
            if old_record is not None:
                SIMULATION['record_sweeps'] = old_record
        self.assertEqual(result['params']['num_steps'], 12 * 200)
        self.assertEqual(result['params']['record_interval_steps'], 12 * 5)

    def test_single_run_default_horizon_uses_configured_sweeps(self):
        old_main = SIMULATION['main_sweeps']
        try:
            SIMULATION['main_sweeps'] = 3
            run = SingleRun(N=10, m=2, p_hk=0.5, seed=7)
        finally:
            SIMULATION['main_sweeps'] = old_main
        self.assertEqual(run.num_steps, 30)

    def test_model_names_are_not_overclaimed(self):
        self.assertIn('no_triadic_feedback', MODEL_TYPES)
        self.assertIn('threshold_only', MODEL_TYPES)
        self.assertIn('mean_threshold_only', MODEL_TYPES)
        self.assertIn('randomized_threshold_only', MODEL_TYPES)
        self.assertNotIn('mean_threshold_only', MODEL_DISPLAY_ORDER)
        self.assertNotIn('randomized_threshold_only', MODEL_DISPLAY_ORDER)
        self.assertIn('full_model_no_triangle_fallback', MODEL_TYPES)
        self.assertNotIn('full_model_no_triangle_fallback', MODEL_DISPLAY_ORDER)
        self.assertNotIn('no_structure', MODEL_TYPES)
        self.assertIn('unilateral', MODEL_TYPES['baseline']['label'].lower())

    def test_threshold_assignment_controls_preserve_requested_thresholds(self):
        graph = nx.Graph()
        graph.add_edges_from([
            (0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2), (4, 5),
        ])
        weights = NetworkGenerator.compute_jaccard_weights(graph)
        trust = TrustThreshold(0.15, 3.5)
        undirected_edges = list(graph.edges())
        realized = [trust.effective_threshold(weights[(u, v)])
                    for u, v in undirected_edges]

        mean_assignment = SingleRun._build_edge_threshold_assignment(
            graph, weights, trust, 'mean_threshold_only', seed=19)
        mean_values = [mean_assignment[(u, v)] for u, v in undirected_edges]
        self.assertTrue(all(value == mean_values[0] for value in mean_values))
        self.assertAlmostEqual(mean_values[0], float(np.mean(realized)))

        randomized_a = SingleRun._build_edge_threshold_assignment(
            graph, weights, trust, 'randomized_threshold_only', seed=19)
        randomized_b = SingleRun._build_edge_threshold_assignment(
            graph, weights, trust, 'randomized_threshold_only', seed=19)
        randomized_values = [randomized_a[(u, v)] for u, v in undirected_edges]
        self.assertEqual(randomized_a, randomized_b)
        self.assertEqual(sorted(randomized_values), sorted(realized))
        self.assertTrue(all(
            randomized_a[(u, v)] == randomized_a[(v, u)]
            for u, v in undirected_edges))

    def test_failed_triangle_fallback_control_isolates_dyadic_update(self):
        class FixedRNG:
            def rand(self):
                return 0.0

            def randint(self, high):
                return 0

            def shuffle(self, values):
                return None

        graph = nx.complete_graph(3)
        triangles, node_triangles = NetworkGenerator.extract_triangles(graph)
        opinions = {0: 0.0, 1: 0.1, 2: 0.3}
        weights = {(u, v): 0.0 for u, v in graph.edges()}
        weights.update({(v, u): 0.0 for u, v in graph.edges()})
        trust = TrustThreshold(0.15, 3.5)
        engines = []
        for model in ('full_model', 'full_model_no_triangle_fallback'):
            engines.append(OpinionDynamics(
                graph, opinions, weights, triangles, node_triangles,
                {node: 1.0 for node in graph.nodes()}, trust,
                mu=0.5, beta=3.0, model_type=model))
        dx_with_fallback = engines[0].step(FixedRNG())
        dx_without_fallback = engines[1].step(FixedRNG())
        self.assertAlmostEqual(dx_with_fallback, 0.05)
        self.assertEqual(dx_without_fallback, 0.0)
        self.assertEqual(engines[0].get_stats()['tri_fallback'], 1)
        self.assertEqual(engines[0].get_stats()['tri_fallback_edge_update'], 1)
        self.assertEqual(engines[1].get_stats()['tri_fallback_suppressed'], 1)
        self.assertEqual(engines[1].get_stats()['update_count'], 0)

    def test_transition_metric_config_is_centralized_and_validated(self):
        self.assertAlmostEqual(
            TRANSITION_ORDER_WINDOW_WIDTH,
            METRICS['transition_order_window_width'])
        self.assertAlmostEqual(
            TRANSITION_CUTOFF, METRICS['transition_order_cutoff'])
        valid = {
            'transition_metric': (
                'maximum fraction of opinions in any interval of configured width'),
            'transition_order_window_width': 0.05,
            'transition_cutoff': 0.95,
            'transition_order_parameter_formula': (
                'max_x count_i(x_i in [x, x + window_width]) / N'),
            'transition_criterion': (
                'per-seed first forward-grid epsilon whose final order parameter >= 0.95'),
        }
        self.assertEqual(validate_transition_metric_config(valid), [])
        invalid = dict(valid)
        invalid['transition_cutoff'] = 0.90
        self.assertIn(
            'transition order cutoff mismatch',
            validate_transition_metric_config(invalid))

    def _small_transition_payload(self):
        p_values = [0.5]
        eps_values = [0.0, 0.5, 1.0]
        seeds = [7, 1004]
        control = {
            "N": 10, "m": 2, "num_runs": 2, "seeds": seeds,
            "p_values": p_values, "eps_values": eps_values,
            "sweeps_per_epsilon": 1.0,
            "num_steps_per_epsilon": 10,
            "transition_metric": (
                "maximum fraction of opinions in any interval of configured width"),
            "transition_order_window_width": 0.05,
            "transition_cutoff": 0.95,
            "transition_order_parameter_formula": (
                "max_x count_i(x_i in [x, x + window_width]) / N"),
            "transition_criterion": (
                "per-seed first forward-grid epsilon whose final order parameter >= 0.95"),
            "variants": {
                "uniform_reference": {"alpha": 0.0},
                "structure_only": {"alpha": 3.5},
                "structure_full": {"alpha": 3.5},
            },
        }
        interaction = {
            "edge_count": 8, "tri_count": 2, "tri_success": 1,
            "tri_fallback": 1, "tri_fallback_edge_update": 1,
            "update_count": 10,
        }
        data = {"0.50": {"p_hk": 0.5, "seeds": seeds, "runs": {}}}
        for seed in seeds:
            data["0.50"]["runs"][str(seed)] = {
                "seed": seed, "n_edges": 10, "gamma_G": 0.1,
                "gamma_dyadic_proposal": 0.12,
                "variants": {
                    "uniform_reference": {
                        "order_parameter": [0.1, 0.5, 0.96],
                        "interactions": [dict(interaction) for _ in eps_values],
                    },
                    "structure_only": {
                        "order_parameter": [0.1, 0.7, 0.97],
                        "interactions": [dict(interaction) for _ in eps_values],
                    },
                    "structure_full": {
                        "order_parameter": [0.1, 0.6, 0.96],
                        "interactions": [dict(interaction) for _ in eps_values],
                    },
                },
            }
        metadata = {
            "n_runs": 2, "seeds": seeds,
            "transition_metric_config": {
                "metric": control["transition_metric"],
                "order_window_width": control["transition_order_window_width"],
                "cutoff": control["transition_cutoff"],
                "formula": control["transition_order_parameter_formula"],
            },
            "source_sha256": {},
        }
        return {"control": control, "data": data}, metadata

    def test_transition_analyzer_accepts_self_consistent_pilot_design(self):
        payload, metadata = self._small_transition_payload()
        validation = validate_transition_payload(
            payload, metadata, require_formal_design=False)
        self.assertEqual(validation["status"], "PASS", validation["errors"])
        self.assertEqual(validation["p_points"], 1)
        self.assertEqual(validation["runs_per_p"], 2)
        self.assertEqual(validation["threshold_points"], 3)
        strict_validation = validate_transition_payload(payload, metadata)
        self.assertEqual(strict_validation["status"], "FAIL")

    def test_transition_analyzer_rejects_nonfinite_pilot_horizon_safely(self):
        payload, metadata = self._small_transition_payload()
        payload["control"]["sweeps_per_epsilon"] = float("nan")
        validation = validate_transition_payload(
            payload, metadata, require_formal_design=False)
        self.assertEqual(validation["status"], "FAIL")
        self.assertIn("invalid per-threshold horizon", validation["errors"])

    def test_transition_analyzer_uses_pilot_seed_and_p_grids(self):
        payload, _ = self._small_transition_payload()
        rows, counts, alpha = build_transition_seed_rows(payload)
        summaries, summary_rows = build_transition_summaries(rows, counts)
        self.assertEqual(len(rows), 2)
        self.assertEqual(set(summaries), {"0.50"})
        self.assertEqual(counts["0.50"]["paired_error_n"][
            "structure_only_mean_edge"], 2)
        self.assertEqual(summaries["0.50"]["metrics"][
            "epsilon_c_uniform_reference"]["n"], 2)
        self.assertTrue(summary_rows)
        self.assertAlmostEqual(alpha, 3.5)

    def test_transition_figure_legend_uses_registered_errorbar_labels(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots()
        try:
            axis.errorbar([0.0, 1.0], [0.2, 0.4], label='Series A')
            legend = add_figure_legend(
                figure, axis, loc='upper right', frameon=False)
            self.assertEqual(
                [text.get_text() for text in legend.get_texts()], ['Series A'])
        finally:
            plt.close(figure)

    def test_transition_major_p_ticks_are_sparse_and_grid_aligned(self):
        p_grid = [round(index * 0.05, 2) for index in range(21)]
        self.assertEqual(
            major_p_ticks(p_grid),
            [round(index * 0.10, 2) for index in range(11)])
        self.assertEqual(
            major_p_ticks([0.0, 0.03, 0.06, 0.09, 0.12], major_step=0.06),
            [0.0, 0.06, 0.12])
        with self.assertRaises(ValueError):
            major_p_ticks(p_grid, major_step=0.0)

    def test_transition_trajectory_diagnostics_report_nonmonotonicity(self):
        monotone = trajectory_diagnostics(
            [0.10, 0.40, 0.95, 0.98], cutoff=0.95)
        self.assertEqual(monotone["decrease_count"], 0)
        self.assertFalse(monotone["nonmonotone"])
        self.assertEqual(monotone["first_crossing_index"], 2)
        self.assertEqual(monotone["cutoff_recrossing_count"], 0)

        recrossing = trajectory_diagnostics(
            [0.10, 0.96, 0.93, 0.97, 0.91], cutoff=0.95)
        self.assertEqual(recrossing["decrease_count"], 2)
        self.assertTrue(recrossing["nonmonotone"])
        self.assertEqual(recrossing["first_crossing_index"], 1)
        self.assertEqual(recrossing["cutoff_recrossing_count"], 2)
        self.assertTrue(recrossing["cutoff_recrossed"])

    def test_transition_trajectory_diagnostics_reject_empty_series(self):
        with self.assertRaises(ValueError):
            trajectory_diagnostics([])

    def test_transition_prediction_scan_matches_and_resumes(self):
        old_n = NETWORK['N']
        old_m = NETWORK['m']
        old_sweeps = SIMULATION.get('sweep_sweeps')
        with tempfile.TemporaryDirectory() as output_dir:
            try:
                NETWORK['N'] = 40
                NETWORK['m'] = 3
                SIMULATION['sweep_sweeps'] = 0.5
                configure_output_root(output_dir)
                manager = ExperimentManager()
                observed_widths = []
                original_order_parameter = Metrics.order_parameter

                def capture_order_width(opinions, **kwargs):
                    observed_widths.append(kwargs.get('width'))
                    return original_order_parameter(opinions, **kwargs)

                with patch.object(
                        Metrics, 'order_parameter',
                        side_effect=capture_order_width):
                    result = manager.experiment_threshold_transition_prediction(
                        num_runs=1, p_values=[0.4],
                        eps_values=[0.0, 0.1], sweeps_per_epsilon=0.5)
                self.assertTrue(observed_widths)
                self.assertTrue(all(
                    abs(float(width) - METRICS['transition_order_window_width'])
                    < 1e-12 for width in observed_widths))
                self.assertEqual(
                    result['control']['transition_order_window_width'], 0.05)
                self.assertEqual(result['control']['transition_cutoff'], 0.95)
                meta_path = Path(output_dir) / 'data' / (
                    'exp_formal_threshold_transition_prediction_revision_v3.meta.json')
                with meta_path.open(encoding='utf-8') as handle:
                    metadata = json.load(handle)
                self.assertEqual(
                    metadata['transition_metric_config']['order_window_width'],
                    0.05)
                self.assertEqual(
                    metadata['transition_metric_config']['cutoff'], 0.95)
                p_block = result['data']['0.40']
                run = p_block['runs']['7']
                self.assertEqual(run['seed'], 7)
                self.assertGreater(run['gamma_G'], 0.0)
                self.assertLess(run['gamma_G'], 1.0)
                self.assertGreater(run['gamma_dyadic_proposal'], 0.0)
                self.assertLess(run['gamma_dyadic_proposal'], 1.0)
                for variant in ('uniform_reference', 'structure_only', 'structure_full'):
                    self.assertEqual(
                        len(run['variants'][variant]['order_parameter']), 2)
                    stats = run['variants'][variant]['interactions']
                    for row in stats:
                        self.assertEqual(
                            row['edge_count'] + row['tri_count'], 20)
                        self.assertEqual(
                            row['tri_success'] + row['tri_fallback'],
                            row['tri_count'])
                resumed = manager.experiment_threshold_transition_prediction(
                    num_runs=1, p_values=[0.4],
                    eps_values=[0.0, 0.1], sweeps_per_epsilon=0.5,
                    workers=3, resume=True)
                self.assertEqual(
                    resumed['data']['0.40']['runs']['7']['gamma_G'],
                    run['gamma_G'])
                with meta_path.open(encoding='utf-8') as handle:
                    resumed_metadata = json.load(handle)
                self.assertEqual(resumed_metadata['workers'], 3)
            finally:
                NETWORK['N'] = old_n
                NETWORK['m'] = old_m
                SIMULATION['sweep_sweeps'] = old_sweeps
        configure_output_root(TEST_OUTPUT_ROOT)

    def test_transition_prediction_checkpoint_mismatch_never_overwrites(self):
        old_n = NETWORK['N']
        old_m = NETWORK['m']
        old_sweeps = SIMULATION.get('sweep_sweeps')
        old_data_dir = experiments_module.DATA_DIR
        try:
            with tempfile.TemporaryDirectory() as output_dir:
                NETWORK['N'] = 30
                NETWORK['m'] = 2
                SIMULATION['sweep_sweeps'] = 0.1
                experiments_module.DATA_DIR = output_dir
                manager = ExperimentManager()
                filename = 'transition_checkpoint_guard.json'
                first = manager.experiment_threshold_transition_prediction(
                    num_runs=1, p_values=[0.4], eps_values=[0.1],
                    sweeps_per_epsilon=0.1, output_filename=filename)
                path = Path(output_dir) / filename
                original_bytes = path.read_bytes()
                with self.assertRaises(ValueError):
                    manager.experiment_threshold_transition_prediction(
                        num_runs=1, p_values=[0.4], eps_values=[0.2],
                        sweeps_per_epsilon=0.1, output_filename=filename)
                self.assertEqual(path.read_bytes(), original_bytes)
                with self.assertRaises(FileExistsError):
                    manager.experiment_threshold_transition_prediction(
                        num_runs=1, p_values=[0.4], eps_values=[0.1],
                        sweeps_per_epsilon=0.1, resume=False,
                        output_filename=filename)
                self.assertEqual(path.read_bytes(), original_bytes)
        finally:
            NETWORK['N'] = old_n
            NETWORK['m'] = old_m
            SIMULATION['sweep_sweeps'] = old_sweeps
            experiments_module.DATA_DIR = old_data_dir

    def test_triangle_failure_control_runs_and_pairs_seeds(self):
        old_main = SIMULATION['main_sweeps']
        old_record = SIMULATION['record_sweeps']
        old_n = NETWORK['N']
        old_data_dir = experiments_module.DATA_DIR
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                SIMULATION['main_sweeps'] = 1
                SIMULATION['record_sweeps'] = 1
                NETWORK['N'] = 24
                experiments_module.DATA_DIR = tmpdir
                result = ExperimentManager().experiment_triangle_failure_control(
                    num_runs=2, p_values=[0.5], resume=False)
        finally:
            SIMULATION['main_sweeps'] = old_main
            SIMULATION['record_sweeps'] = old_record
            NETWORK['N'] = old_n
            experiments_module.DATA_DIR = old_data_dir

        block = result['data']['0.50']
        self.assertEqual(block['_control']['seeds'], [7, 1004])
        self.assertEqual(
            block['full_model_no_triangle_fallback']['order_param']['n'], 2)
        self.assertIn(
            'full_model_minus_full_model_no_triangle_fallback',
            block['_paired_comparisons'])
        self.assertIn(
            'tri_fallback_edge_update',
            block['full_model']['dynamics_stats'])
        self.assertIn(
            'tri_fallback_suppressed',
            block['full_model_no_triangle_fallback']['dynamics_stats'])

    def test_triangle_failure_control_workload_estimate(self):
        estimates = estimate_elementary_updates(
            ['triangle_failure_control'], quick=False, runs=2)
        expected = (len(NETWORK['p_values']) * 2 * 2
                    * int(round(NETWORK['N'] * SIMULATION['main_sweeps'])))
        self.assertEqual(estimates['triangle_failure_control'], expected)

    def test_threshold_assignment_experiment_runs_and_pairs_seeds(self):
        old_main = SIMULATION['main_sweeps']
        old_record = SIMULATION['record_sweeps']
        old_data_dir = experiments_module.DATA_DIR
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                SIMULATION['main_sweeps'] = 1
                SIMULATION['record_sweeps'] = 1
                experiments_module.DATA_DIR = tmpdir
                result = ExperimentManager().experiment_threshold_assignment_controls(
                    num_runs=2, p_values=[0.5], resume=False)
        finally:
            SIMULATION['main_sweeps'] = old_main
            SIMULATION['record_sweeps'] = old_record
            experiments_module.DATA_DIR = old_data_dir

        block = result['data']['0.50']
        self.assertEqual(block['_control']['seeds'], [7, 1004])
        self.assertIn('mean_threshold_only', block)
        self.assertIn('randomized_threshold_only', block)
        self.assertEqual(
            block['mean_threshold_only']['order_param']['n'], 2)
        self.assertIn(
            'threshold_only_minus_randomized_threshold_only',
            block['_paired_comparisons'])

    def test_resume_signature_tracks_core_parameters(self):
        signature = ExperimentManager._core_parameter_signature()
        self.assertEqual({key: signature[key] for key in (
                'epsilon_0', 'alpha', 'mu', 'beta')}, {
            'epsilon_0': float(TRUST['epsilon_0_default']),
            'alpha': float(TRUST['alpha']),
            'mu': float(DYNAMICS['mu']),
            'beta': float(DYNAMICS['beta_default']),
        })
        self.assertEqual(len(signature['execution_source_signature']), 64)

    def test_dry_run_documentation_and_lock_files_exist(self):
        root = ROOT
        self.assertTrue(os.path.exists(os.path.join(root, 'requirements-lock.txt')))
        self.assertTrue(os.path.exists(os.path.join(root, 'REPRODUCIBILITY.md')))

    def test_output_root_is_propagated_to_experiment_module(self):
        old = {
            'config': (config_module.OUTPUT_DIR, config_module.DATA_DIR,
                       config_module.FIGURE_DIR),
            'experiments': (experiments_module.OUTPUT_DIR,
                            experiments_module.DATA_DIR,
                            experiments_module.FIGURE_DIR),
        }
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                expected = os.path.abspath(tmpdir)
                self.assertEqual(configure_output_root(tmpdir), expected)
                self.assertEqual(experiments_module.OUTPUT_DIR, expected)
                self.assertEqual(experiments_module.DATA_DIR,
                                 os.path.join(expected, 'data'))
        finally:
            (config_module.OUTPUT_DIR, config_module.DATA_DIR,
             config_module.FIGURE_DIR) = old['config']
            (experiments_module.OUTPUT_DIR, experiments_module.DATA_DIR,
             experiments_module.FIGURE_DIR) = old['experiments']

    def test_result_checkpoint_is_atomic_and_valid_json(self):
        old_data = experiments_module.DATA_DIR
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                experiments_module.DATA_DIR = tmpdir
                with contextlib.redirect_stdout(io.StringIO()):
                    ExperimentManager().save_json(
                        {'value': np.float64(1.25)}, 'atomic_test.json',
                        metadata={'n_runs': 1},
                    )
                result_path = os.path.join(tmpdir, 'atomic_test.json')
                meta_path = os.path.join(tmpdir, 'atomic_test.meta.json')
                with open(result_path, 'r', encoding='utf-8') as handle:
                    self.assertEqual(json.load(handle)['value'], 1.25)
                self.assertTrue(os.path.exists(meta_path))
                self.assertFalse(any('.tmp-' in name
                                     for name in os.listdir(tmpdir)))
        finally:
            experiments_module.DATA_DIR = old_data

    def test_workload_estimate_for_formal_ablation(self):
        estimate = estimate_elementary_updates(
            ['ablation'], quick=False, runs=20)
        expected = len(config_module.NETWORK['p_values']) * 6 * 20 * (
            config_module.NETWORK['N'] * SIMULATION['main_sweeps'])
        self.assertEqual(estimate['ablation'], expected)

    def test_ablation_resume_rejects_incompatible_checkpoint(self):
        old_network = (config_module.NETWORK['N'], config_module.NETWORK['m'])
        old_sim = (SIMULATION['main_sweeps'], SIMULATION['record_sweeps'])
        old_data = experiments_module.DATA_DIR
        try:
            config_module.NETWORK['N'] = 20
            config_module.NETWORK['m'] = 2
            SIMULATION['main_sweeps'] = 1
            SIMULATION['record_sweeps'] = 1
            with tempfile.TemporaryDirectory() as tmpdir:
                experiments_module.DATA_DIR = tmpdir
                stale = {
                    '0.0': {
                        'sentinel': True,
                        '_control': {
                            'N': 20, 'm': 2, 'n_runs': 1,
                            'main_sweeps': 1, 'record_sweeps': 1,
                            'model_order': list(MODEL_TYPES),
                            'core_parameters': {'beta': -999},
                            'seeds': [7],
                        },
                    },
                }
                path = os.path.join(tmpdir, 'resume_test.json')
                with open(path, 'w', encoding='utf-8') as handle:
                    json.dump(stale, handle)
                before = open(path, 'rb').read()
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(FileExistsError):
                        ExperimentManager().experiment_ablation(
                            num_runs=1, p_values=[0.0], resume=True,
                            output_filename='resume_test.json')
                self.assertEqual(open(path, 'rb').read(), before)
        finally:
            config_module.NETWORK['N'], config_module.NETWORK['m'] = old_network
            SIMULATION['main_sweeps'], SIMULATION['record_sweeps'] = old_sim
            experiments_module.DATA_DIR = old_data

    def test_alpha_sweep_resume_skips_completed_point(self):
        old_data = experiments_module.DATA_DIR
        old_alpha_range = list(experiments_module.SENSITIVITY['alpha_range'])
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                experiments_module.DATA_DIR = tmpdir
                experiments_module.SENSITIVITY['alpha_range'] = [1.0]
                manager = ExperimentManager()
                fake = {
                    '1.0': {
                        'order_param': self._summary_stub(0.8),
                        'n_clusters': self._summary_stub(2),
                    }
                }
                with patch.object(
                        manager, '_run_alpha_sweep_for_alpha',
                        return_value=fake) as runner:
                    with contextlib.redirect_stdout(io.StringIO()):
                        manager.experiment_alpha_sweep(
                            num_runs=1, resume=True,
                            output_filename='alpha_resume_test.json')
                        manager.experiment_alpha_sweep(
                            num_runs=1, resume=True,
                            output_filename='alpha_resume_test.json')
                self.assertEqual(runner.call_count, 1)
        finally:
            experiments_module.SENSITIVITY['alpha_range'] = old_alpha_range
            experiments_module.DATA_DIR = old_data

    def test_controlled_resume_rejects_beta_or_sweeps_change(self):
        old_data = experiments_module.DATA_DIR
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                experiments_module.DATA_DIR = tmpdir
                manager = ExperimentManager()
                base = {
                    'experiment': 'alpha_sweep', 'beta': 3.0,
                    'main_sweeps': 200.0,
                }
                expected_control = dict(base)
                expected_control['execution_source_signature'] = (
                    execution_source_signature())
                manager.save_json(
                    {'_control': expected_control, '1.0': {'sentinel': True}},
                    'controlled_resume_test.json')
                same = manager._load_controlled_results(
                    'controlled_resume_test.json', base, resume=True)
                self.assertIn('1.0', same)
                with self.assertRaises(FileExistsError):
                    manager._load_controlled_results(
                        'controlled_resume_test.json',
                        {**base, 'beta': 4.0}, resume=True)
                with self.assertRaises(FileExistsError):
                    manager._load_controlled_results(
                        'controlled_resume_test.json',
                        {**base, 'main_sweeps': 100.0}, resume=True)
        finally:
            experiments_module.DATA_DIR = old_data

    def test_joint_p_beta_keys_preserve_point_zero_five_and_resume(self):
        old_network = (config_module.NETWORK['N'], config_module.NETWORK['m'])
        old_sim = (SIMULATION['main_sweeps'], SIMULATION['record_sweeps'])
        old_data = experiments_module.DATA_DIR
        try:
            config_module.NETWORK['N'] = 20
            config_module.NETWORK['m'] = 2
            SIMULATION['main_sweeps'] = 1
            SIMULATION['record_sweeps'] = 1
            with tempfile.TemporaryDirectory() as tmpdir:
                experiments_module.DATA_DIR = tmpdir
                manager = ExperimentManager()
                with contextlib.redirect_stdout(io.StringIO()):
                    result = manager.experiment_joint_p_beta(
                        num_runs=1, p_values=[0.0, 0.05],
                        beta_values=[0.0], resume=False,
                        output_filename='joint_key_test.json')
                self.assertEqual(set(result['data']), {'0.00_0.00',
                                                       '0.05_0.00'})
                with patch.object(
                        manager, '_run_joint_p_beta_for_p') as runner:
                    with contextlib.redirect_stdout(io.StringIO()):
                        manager.experiment_joint_p_beta(
                            num_runs=1, p_values=[0.0, 0.05],
                            beta_values=[0.0], resume=True,
                            output_filename='joint_key_test.json')
                runner.assert_not_called()
        finally:
            config_module.NETWORK['N'], config_module.NETWORK['m'] = old_network
            SIMULATION['main_sweeps'], SIMULATION['record_sweeps'] = old_sim
            experiments_module.DATA_DIR = old_data

    def test_run_all_propagates_resume_flag(self):
        manager = ExperimentManager()
        method_names = [
            'experiment_network_validation', 'experiment_ablation',
            'experiment_epsilon_sweep', 'experiment_hysteresis_sweep',
            'experiment_perturbation_sweep',
            'experiment_update_mode_control',
            'experiment_sbm_community_conflict',
            'experiment_sbm_hysteresis',
            'experiment_vector_opinions',
            'experiment_empirical_topologies', 'experiment_beta_sweep',
            'experiment_alpha_sweep', 'experiment_mu_sweep',
            'experiment_joint_eps_beta', 'experiment_joint_p_beta',
            'experiment_size_robustness', 'experiment_m_robustness',
            'experiment_init_distribution',
        ]
        resumable = set(method_names)
        mocks = {}
        with contextlib.ExitStack() as stack:
            for name in method_names:
                mocks[name] = stack.enter_context(
                    patch.object(manager, name, return_value={}))
            with contextlib.redirect_stdout(io.StringIO()):
                manager.run_all(quick=False, resume=False, num_runs=7)
        for name in resumable:
            self.assertFalse(mocks[name].call_args[1]['resume'], name)
        for name in set(method_names) - {'experiment_network_validation'}:
            self.assertEqual(mocks[name].call_args[1]['num_runs'], 7, name)

    def test_sensitivity_plots_ignore_control_metadata(self):
        old_data_path = plotting_module.DATA_PATH
        old_out_dir = plotting_module.OUT_DIR
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                data_dir = os.path.join(tmpdir, 'data')
                figure_dir = os.path.join(tmpdir, 'figures')
                os.makedirs(data_dir)
                plotting_module.DATA_PATH = plotting_module.Path(data_dir)
                plotting_module.OUT_DIR = plotting_module.Path(figure_dir)
                plotting_module.OUT_DIR.mkdir(parents=True, exist_ok=True)
                metric_block = {
                    'order_param': self._summary_stub(0.8),
                    'n_clusters': self._summary_stub(2),
                }
                alpha_mu = {'_control': {'test': True}, '1.0': metric_block}
                beta_block = {
                    **metric_block,
                    'triangle_alignment_fixed': self._summary_stub(0.7),
                    'edge_disagreement_penalty': self._summary_stub(0.2),
                }
                beta = {
                    '_control': {'test': True},
                    '0.0': {'1.0': beta_block},
                }
                network = {
                    '_control': {'test': True},
                    '0.0': {
                        'f_triangle': self._summary_stub(0.4),
                        'avg_clustering': self._summary_stub(0.3),
                    },
                }
                hysteresis = {
                    '_control': {'test': True},
                    '0.00': {
                        'hysteresis_area': 0.01,
                        'backward_low_eps_order_param': 0.8,
                        'peak_sensitivity': 0.2,
                        'backward_low_eps_clusters': 2.0,
                    },
                }
                ablation = {
                    '_control': {'test': True},
                    '0.0': {
                        model: {'order_param': self._summary_stub(0.8)}
                        for model in MODEL_TYPES
                    },
                }
                init_distribution = {
                    '_control': {'test': True},
                    'uniform': metric_block,
                    'normal': metric_block,
                    'bimodal': metric_block,
                }
                fixtures = {
                    'exp8_alpha_sweep.json': alpha_mu,
                    'exp8_mu_sweep.json': alpha_mu,
                    'exp8_beta_sweep.json': beta,
                    'exp7_network_validation.json': network,
                    'exp_hysteresis_sweep.json': hysteresis,
                    'exp6_ablation.json': ablation,
                    'exp8_init_distribution.json': init_distribution,
                }
                for filename, payload in fixtures.items():
                    with open(os.path.join(data_dir, filename), 'w',
                              encoding='utf-8') as handle:
                        json.dump(payload, handle)
                plotting_module.plot_alpha_sweep()
                plotting_module.plot_beta_sweep()
                plotting_module.plot_mu_sweep()
                plotting_module.plot_summary_dashboard()
                for filename in (
                        'fig5_alpha_sweep.png', 'fig6_beta_sweep.png',
                        'fig10_mu_sweep.png',
                        'fig13_summary_dashboard.png'):
                    self.assertTrue(os.path.exists(
                        os.path.join(figure_dir, filename)))
        finally:
            plotting_module.DATA_PATH = old_data_path
            plotting_module.OUT_DIR = old_out_dir

    def test_threshold_recovery_branches_each_epsilon_and_tracks_censoring(self):
        manager = ExperimentManager()
        with tempfile.TemporaryDirectory() as temp_dir:
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(
                    experiments_module, 'DATA_DIR', temp_dir))
                stack.enter_context(patch.dict(
                    experiments_module.NETWORK, {'N': 24, 'm': 2}))
                stack.enter_context(patch.dict(
                    SIMULATION, {'sweep_sweeps': 0.5,
                                 'record_sweeps': 0.25}))
                result = manager.experiment_threshold_recovery(
                    num_runs=1, p_values=[0.4], sigma_values=[0.0, 0.1],
                    eps_values=[0.1, 0.2], recovery_order_threshold=1.0,
                    recovery_persistence=2, resume=False,
                    output_filename='threshold_recovery_test.json')
                block = result['data']['0.40']
                self.assertEqual(block['time_sweeps'], [0.0, 0.25, 0.5])
                self.assertEqual(set(block['by_epsilon']), {'0.1', '0.2'})
                for epsilon in ('0.1', '0.2'):
                    shock = block['by_epsilon'][epsilon]['0.100']
                    self.assertEqual(
                        len(shock['trajectory']['opinion_sd']['mean']), 3)
                    recovery = shock['recovery']
                    self.assertEqual(
                        recovery['n_recovered'] + recovery['n_censored'], 1)
                    self.assertEqual(len(recovery['time_by_run']), 1)
                    self.assertGreaterEqual(
                        recovery['probability']['ci95_low'], 0.0)
                    self.assertLessEqual(
                        recovery['probability']['ci95_high'], 1.0)
                self.assertTrue(os.path.exists(os.path.join(
                    temp_dir, 'threshold_recovery_test.meta.json')))

    def test_perturbation_clipping_diagnostics(self):
        opinions = {0: 0.0, 1: 1.0, 2: 0.5}
        perturbed, stats = ExperimentManager._perturb_opinions(
            opinions, np.random.RandomState(7), 1.0, return_stats=True)
        self.assertEqual(stats['n_perturbed'], 3)
        self.assertGreater(stats['n_clipped'], 0)
        self.assertTrue(all(0.0 <= value <= 1.0
                            for value in perturbed.values()))

    def test_standardized_perturbations_share_direction_and_scale(self):
        opinions = {0: 0.25, 1: 0.75}
        z_values = {0: 1.0, 1: -1.0}
        zero, zero_stats = ExperimentManager._apply_standardized_perturbation(
            opinions, z_values, 0.0, return_stats=True)
        low = ExperimentManager._apply_standardized_perturbation(
            opinions, z_values, 0.1)
        high = ExperimentManager._apply_standardized_perturbation(
            opinions, z_values, 0.2)
        self.assertEqual(zero, opinions)
        self.assertEqual(zero_stats['n_perturbed'], 0)
        for node in opinions:
            self.assertAlmostEqual(
                high[node] - opinions[node],
                2.0 * (low[node] - opinions[node]))

    def test_perturbation_gradient_reuses_one_forward_scan_per_seed(self):
        manager = ExperimentManager()
        original_generator = NetworkGenerator.generate_holme_kim
        with patch.dict(experiments_module.NETWORK, {'N': 16, 'm': 2}), \
                patch.dict(SIMULATION, {'sweep_sweeps': 0.125,
                                        'record_sweeps': 0.125}), \
                patch.object(NetworkGenerator, 'generate_holme_kim',
                             wraps=original_generator) as generate:
            blocks = manager._run_perturbation_sweep_for_p_shared(
                0.4, 2, [0.0, 0.1, 0.2], [0.1, 0.2], [0.2, 0.1])
        self.assertEqual(generate.call_count, 2)
        reference = blocks['0.000']['forward']
        for sigma_key in ('0.100', '0.200'):
            self.assertEqual(blocks[sigma_key]['forward'], reference)
            self.assertTrue(blocks[sigma_key]['shared_forward_scan'])
            self.assertTrue(blocks[sigma_key]['common_random_numbers'])
            self.assertIn('backward_tail_mean_abs_dx', blocks[sigma_key])

    def test_perturbation_checkpoint_rejects_control_mismatch(self):
        manager = ExperimentManager()
        with tempfile.TemporaryDirectory() as temp_dir, \
                patch.object(experiments_module, 'DATA_DIR', temp_dir), \
                patch.dict(experiments_module.NETWORK, {'N': 12, 'm': 2}), \
                patch.dict(SIMULATION, {'sweep_sweeps': 0.1,
                                        'record_sweeps': 0.1}):
            filename = 'perturbation_checkpoint_test.json'
            manager.experiment_perturbation_sweep(
                num_runs=1, p_values=[0.0], sigma_values=[0.0],
                eps_values=[0.1], resume=False, output_filename=filename)
            path = os.path.join(temp_dir, filename)
            with open(path, 'r', encoding='utf-8') as handle:
                payload = json.load(handle)
            payload['control']['num_runs'] = 999
            with open(path, 'w', encoding='utf-8') as handle:
                json.dump(payload, handle)
            with patch.object(
                    manager, '_run_perturbation_sweep_for_p_shared',
                    wraps=manager._run_perturbation_sweep_for_p_shared) as run_shared:
                with self.assertRaises(FileExistsError):
                    manager.experiment_perturbation_sweep(
                        num_runs=1, p_values=[0.0], sigma_values=[0.0],
                        eps_values=[0.1], resume=True, output_filename=filename)
            self.assertEqual(run_shared.call_count, 0)

    def test_synchronous_mode_uses_sweep_budget(self):
        result = SingleRun(
            N=30, m=2, p_hk=0.5, epsilon_0=0.15, alpha=3.5,
            mu=0.5, beta=3.0, model_type='full_model',
            num_steps=60, seed=7, update_mode='synchronous',
        ).execute(record_interval=30)
        self.assertEqual(result['params']['update_mode'], 'synchronous')
        self.assertEqual(result['time_series']['sweep'], [1.0, 2.0])
        self.assertEqual(result['convergence']['steps_run'], 60)

    def test_sbm_hysteresis_runs_end_to_end(self):
        with patch.dict(experiments_module.NETWORK, {'N': 16}), \
                patch.dict(SIMULATION, {'sweep_sweeps': 0.1,
                                        'record_sweeps': 0.1}):
            with tempfile.TemporaryDirectory() as tmpdir:
                with patch.object(experiments_module, 'DATA_DIR', tmpdir):
                    with contextlib.redirect_stdout(io.StringIO()):
                        result = ExperimentManager().experiment_sbm_hysteresis(
                            num_runs=1, mixing_values=[0.2],
                            sigma_values=[0.0, 0.1],
                            eps_values=[0.1, 0.2],
                            resume=False,
                            output_filename='sbm_hyst_test.json')
        block = result['data']['0.200']
        self.assertTrue(block['_control']['complete'])
        self.assertIn('0.000', block['backward'])
        self.assertIn('0.100', block['backward'])
        self.assertEqual(
            block['backward']['0.000']['order_param']['0.1']['n'], 1)
        self.assertIn('hysteresis_area_stats', block['backward']['0.000'])
        self.assertFalse(block['perturb_each_step'])
        one_shot = block['perturbation_diagnostics']['0.100']['by_run'][0]
        self.assertEqual(one_shot['n_perturbed'], 16)
        self.assertLessEqual(one_shot['n_clipped'], one_shot['n_perturbed'])
        self.assertAlmostEqual(
            one_shot['clipped_fraction'],
            one_shot['n_clipped'] / one_shot['n_perturbed'])

    def test_sbm_hysteresis_parallel_matches_serial_blocks(self):
        kwargs = dict(
            num_runs=1, mixing_values=[0.1, 0.2],
            sigma_values=[0.0, 0.1], eps_values=[0.1, 0.2],
            resume=False)
        with patch.dict(experiments_module.NETWORK, {'N': 16}), \
                patch.dict(SIMULATION, {'sweep_sweeps': 0.1,
                                        'record_sweeps': 0.1}), \
                tempfile.TemporaryDirectory() as serial_dir, \
                tempfile.TemporaryDirectory() as parallel_dir:
            with patch.object(experiments_module, 'DATA_DIR', serial_dir), \
                    contextlib.redirect_stdout(io.StringIO()):
                serial = ExperimentManager().experiment_sbm_hysteresis(
                    workers=1, output_filename='serial.json', **kwargs)
            with patch.object(experiments_module, 'DATA_DIR', parallel_dir), \
                    contextlib.redirect_stdout(io.StringIO()):
                parallel = ExperimentManager().experiment_sbm_hysteresis(
                    workers=2, output_filename='parallel.json', **kwargs)
        self.assertEqual(serial['data'], parallel['data'])

    def test_sbm_hysteresis_applies_perturbation_at_each_reverse_threshold(self):
        directions = []
        original_perturb = ExperimentManager._apply_standardized_perturbation

        def capture_direction(opinions, z_values, sigma, return_stats=False):
            directions.append(dict(z_values))
            return original_perturb(
                opinions, z_values, sigma, return_stats=return_stats)

        with patch.dict(experiments_module.NETWORK, {'N': 16}), \
                patch.dict(SIMULATION, {'sweep_sweeps': 0.1,
                                        'record_sweeps': 0.1}):
            with tempfile.TemporaryDirectory() as tmpdir:
                with patch.object(experiments_module, 'DATA_DIR', tmpdir), \
                        patch.object(
                            ExperimentManager,
                            '_apply_standardized_perturbation',
                            new=staticmethod(capture_direction)):
                    with contextlib.redirect_stdout(io.StringIO()):
                        result = ExperimentManager().experiment_sbm_hysteresis(
                            num_runs=1, mixing_values=[0.2],
                            sigma_values=[0.1, 0.2], eps_values=[0.1, 0.2],
                            perturb_each_step=True, resume=False,
                            output_filename='sbm_hyst_step_test.json')
        block = result['data']['0.200']
        self.assertTrue(block['perturb_each_step'])
        for sigma_key in ('0.100', '0.200'):
            diag = block['perturbation_diagnostics'][sigma_key]['by_run'][0]
            self.assertEqual(diag['n_perturbed'], 16 * 2)
            self.assertLessEqual(diag['n_clipped'], diag['n_perturbed'])
            self.assertAlmostEqual(
                diag['clipped_fraction'],
                diag['n_clipped'] / diag['n_perturbed'])
        self.assertEqual(len(directions), 4)
        self.assertEqual(directions[:2], directions[2:])

    def test_formal_sbm_hysteresis_runner_dry_run_estimates_and_stops(self):
        self.assertEqual(
            estimate_sbm_hysteresis_updates(
                2, [0.1, 0.2], [0.0, 0.1], [0.1, 0.2, 0.3], 4.0005),
            288036)
        argv = [
            'run_formal_sbm_hysteresis.py', '--runs', '1', '--sweeps', '2',
            '--mixing-values', '0.2', '--sigma-values', '0.1',
            '--eps-values', '0.1', '0.2', '--dry-run',
        ]
        with patch.object(sys, 'argv', argv), \
                patch.object(
                    ExperimentManager, 'experiment_sbm_hysteresis') as run, \
                contextlib.redirect_stdout(io.StringIO()):
            run_formal_sbm_hysteresis_main()
        run.assert_not_called()

    def test_formal_sbm_hysteresis_runner_targets_clean_versioned_output(self):
        argv = [
            'run_formal_sbm_hysteresis.py', '--runs', '1', '--sweeps', '0.1',
            '--mixing-values', '0.2', '--sigma-values', '0.0',
            '--eps-values', '0.1',
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(sys, 'argv', argv), \
                    patch(
                        'run_formal_sbm_hysteresis.FORMAL_DATA_DIR',
                        Path(tmpdir)), \
                    patch.object(
                        ExperimentManager, 'experiment_sbm_hysteresis') as run, \
                    contextlib.redirect_stdout(io.StringIO()):
                run_formal_sbm_hysteresis_main()
        run.assert_called_once()
        self.assertEqual(
            run.call_args[1]['output_filename'], FORMAL_OUTPUT_FILENAME)
        self.assertEqual(
            FORMAL_OUTPUT_FILENAME, 'exp_sbm_hysteresis_revision_v2.json')

    def test_transition_prediction_grid_and_checkpoint_are_versioned(self):
        self.assertEqual(
            TRANSITION_RUNNER_P_VALUES, TRANSITION_ANALYSIS_P_VALUES)
        self.assertEqual(len(TRANSITION_ANALYSIS_P_VALUES), 21)
        self.assertEqual(
            TRANSITION_OUTPUT_FILENAME,
            'exp_formal_threshold_transition_prediction_revision_v3.json')
        argv = [
            'run_formal_threshold_transition_prediction.py', '--runs', '1',
            '--workers', '1', '--sweeps', '0.1', '--eps-values', '0.1', '0.2',
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(sys, 'argv', argv), \
                    patch.object(
                        sys.modules[
                            'run_formal_threshold_transition_prediction'],
                        'FORMAL_DATA_DIR', Path(tmpdir)), \
                    patch.object(
                        ExperimentManager,
                        'experiment_threshold_transition_prediction') as run, \
                    contextlib.redirect_stdout(io.StringIO()):
                run_formal_transition_prediction_main()
        run.assert_called_once()
        self.assertEqual(
            run.call_args[1]['output_filename'], TRANSITION_OUTPUT_FILENAME)

    def test_transition_prediction_runner_validates_before_dry_run(self):
        from argparse import Namespace

        valid = Namespace(
            runs=20, workers=4, sweeps=100.0,
            p_values=list(TRANSITION_RUNNER_P_VALUES),
            eps_values=[round(index * 0.025, 3) for index in range(23)],
            alpha=TRUST["alpha"],
        )
        expected = 3 * 21 * 20 * 23 * int(NETWORK["N"]) * 100
        self.assertEqual(validate_transition_runner_args(valid), expected)
        self.assertEqual(estimate_transition_updates(
            20, valid.p_values, valid.eps_values, 100.0), expected)

        invalid_cases = []
        for field, value in (
                ("runs", 0), ("workers", 0), ("sweeps", float("nan")),
                ("sweeps", 0.0), ("alpha", float("inf")),
                ("alpha", 0.0)):
            candidate = Namespace(**vars(valid))
            setattr(candidate, field, value)
            invalid_cases.append(candidate)

        for field, values in (
                ("p_values", []),
                ("p_values", [0.0, 0.5, 0.5]),
                ("p_values", [0.0, 1.1]),
                ("p_values", [0.0, float("nan")]),
                ("eps_values", [0.1, 0.0]),
                ("eps_values", [0.0, 1.01]),
                ("eps_values", [0.0, float("inf")]),
        ):
            candidate = Namespace(**vars(valid))
            setattr(candidate, field, values)
            invalid_cases.append(candidate)

        for candidate in invalid_cases:
            with self.subTest(candidate=vars(candidate)):
                with self.assertRaises(ValueError):
                    validate_transition_runner_args(candidate)

    def test_transition_prediction_dry_run_rejects_invalid_parameters(self):
        argv = [
            'run_formal_threshold_transition_prediction.py', '--runs', '0',
            '--dry-run',
        ]
        with patch.object(sys, 'argv', argv), \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                run_formal_transition_prediction_main()
        self.assertEqual(raised.exception.code, 2)

    def test_transition_prediction_dry_run_does_not_create_output_directory(self):
        argv = [
            'run_formal_threshold_transition_prediction.py', '--runs', '1',
            '--workers', '1', '--sweeps', '0.1', '--p-values', '0.4',
            '--eps-values', '0.1', '0.2', '--dry-run',
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / 'must_not_be_created'
            with patch.object(sys, 'argv', argv),                     patch.object(
                        sys.modules[
                            'run_formal_threshold_transition_prediction'],
                        'FORMAL_DATA_DIR', target),                     contextlib.redirect_stdout(io.StringIO()):
                run_formal_transition_prediction_main()
            self.assertFalse(target.exists())

    def test_formal_ablation_trajectory_analysis_covers_all_six_models(self):
        self.assertEqual(TIME_SERIES_MODELS, list(MODEL_DISPLAY_ORDER))

    def test_formal_ablation_figure_exports_editable_formats(self):
        import matplotlib
        import matplotlib.pyplot as plt

        self.assertEqual(matplotlib.rcParams['pdf.fonttype'], 42)
        self.assertEqual(matplotlib.rcParams['svg.fonttype'], 'none')
        fig, ax = plt.subplots()
        ax.set_title('editable-label')
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                outputs = save_figure_formats(
                    fig, os.path.join(tmpdir, 'reviewer-figure'))
                self.assertEqual(
                    [os.path.splitext(path)[1] for path in outputs],
                    ['.png', '.pdf', '.svg'])
                for path in outputs:
                    self.assertTrue(os.path.isfile(path))
                with open(outputs[2], 'r', encoding='utf-8') as handle:
                    self.assertIn('editable-label', handle.read())
        finally:
            plt.close(fig)

    def test_two_block_sbm_and_custom_initial_state(self):
        graph, membership, params = NetworkGenerator.generate_two_block_sbm(
            40, target_degree=6.0, mixing=0.2, seed=7)
        self.assertEqual(graph.number_of_nodes(), 40)
        self.assertEqual(set(membership.values()), {0, 1})
        initial = {
            node: 0.25 if membership[node] == 0 else 0.75
            for node in graph.nodes()
        }
        result = SingleRun(
            N=40, m=2, p_hk=0.0, epsilon_0=0.15, alpha=3.5,
            mu=0.5, beta=3.0, model_type='full_model',
            num_steps=40, seed=7, graph=graph,
            initial_opinions=initial, network_type='two_block_sbm',
            network_params=params,
        ).execute(record_interval=40)
        self.assertEqual(result['params']['network_type'], 'two_block_sbm')
        self.assertEqual(result['params']['network_params']['mixing'], 0.2)

    def test_vector_metrics_and_one_step(self):
        graph = nx.complete_graph(6)
        triangles, node_triangles = NetworkGenerator.extract_triangles(graph)
        weights = NetworkGenerator.compute_jaccard_weights(graph)
        clustering = NetworkGenerator.compute_clustering(graph)
        trust = TrustThreshold(0.15, 3.5)
        opinions = {node: np.array([0.2, 0.3]) + node * 0.01
                    for node in graph.nodes()}
        engine = VectorOpinionDynamics(
            graph, opinions, weights, triangles, node_triangles,
            clustering, trust, 0.5, 3.0, 'full_model')
        dx = engine.step(np.random.RandomState(7))
        self.assertGreaterEqual(dx, 0.0)
        arr = engine.get_opinions_array()
        self.assertEqual(arr.shape, (6, 2))
        self.assertGreaterEqual(Metrics.vector_compact_order(arr), 0.0)
        self.assertLessEqual(Metrics.vector_compact_order(arr), 1.0)

    def test_empirical_benchmarks_and_experiment_structure(self):
        loaded = {}
        for name in ('karate', 'davis', 'florentine'):
            graph, metadata = NetworkGenerator.load_empirical_benchmark(name)
            loaded[name] = graph
            self.assertGreater(graph.number_of_nodes(), 0)
            self.assertGreater(graph.number_of_edges(), 0)
            self.assertFalse(graph.is_directed())
            self.assertEqual(nx.number_of_selfloops(graph), 0)
            self.assertEqual(metadata['topology_replicates'], 1)
            self.assertEqual(len(metadata['canonical_edge_sha256']), 64)
            self.assertEqual(list(graph.nodes()),
                             list(range(graph.number_of_nodes())))
            for node in graph.nodes():
                self.assertEqual(list(graph.neighbors(node)),
                                 sorted(graph.neighbors(node)))
        self.assertIsInstance(loaded['davis'], nx.Graph)
        self.assertFalse(loaded['davis'].is_multigraph())

        old_sweeps = SIMULATION['main_sweeps']
        old_record = SIMULATION['record_sweeps']
        try:
            SIMULATION['main_sweeps'] = 2
            SIMULATION['record_sweeps'] = 1
            with tempfile.TemporaryDirectory() as tmpdir:
                with patch.object(experiments_module, 'DATA_DIR', tmpdir):
                    with contextlib.redirect_stdout(io.StringIO()):
                        result = ExperimentManager().experiment_empirical_topologies(
                            num_runs=2,
                            topology_names=['karate', 'davis', 'florentine'],
                            resume=False,
                            output_filename='empirical_test.json',
                        )
        finally:
            SIMULATION['main_sweeps'] = old_sweeps
            SIMULATION['record_sweeps'] = old_record

        self.assertEqual(set(result['data']),
                         {'karate', 'davis', 'florentine'})
        for block in result['data'].values():
            self.assertTrue(block['_control']['complete'])
            for model in ('threshold_only', 'no_triadic_feedback',
                          'full_model'):
                self.assertEqual(len(block[model]['order_param']['values']), 2)
            paired = block['_paired_comparisons'][
                'full_model_minus_threshold_only']['order_param']
            self.assertEqual(paired['n'], 2)


if __name__ == '__main__':
    unittest.main()
