import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import (DYNAMICS, EPSILON_0_GRID, JOINT_SWEEP, MODEL_TYPES,
                    SIMULATION, TRUST, format_eps_key)
import config as config_module
import experiments as experiments_module
import plotting as plotting_module
from dynamics import TrustThreshold
from experiments import ExperimentManager
from metrics import Metrics
from network import NetworkGenerator
from simulation import SingleRun
from statistics_utils import paired_difference, summarize
from vector_dynamics import VectorOpinionDynamics
from run_all import configure_output_root, estimate_elementary_updates, run_selected
from analysis.validate_local_energy_counterexample import (
    global_edge_energy, local_edge_energy)


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

    def test_counterexample_energy_uses_unweighted_manuscript_definition(self):
        graph = nx.complete_graph(3)
        before = {0: 0.4, 1: 0.6, 2: 0.0}
        phi = np.exp(-3.0 * abs(before[0] - before[2]))
        after = dict(before)
        after[0] = before[0] + 0.5 * phi * (before[1] - before[0])

        selected_before = 0.5 * (before[0] - before[1]) ** 2
        selected_after = 0.5 * (after[0] - after[1]) ** 2
        self.assertAlmostEqual(selected_before, 0.020000, places=6)
        self.assertAlmostEqual(selected_after, 0.014430, places=6)
        self.assertAlmostEqual(
            local_edge_energy(graph, before, 0), 0.100000, places=6)
        self.assertAlmostEqual(
            local_edge_energy(graph, after, 0), 0.106931, places=6)
        self.assertAlmostEqual(
            global_edge_energy(graph, after)
            - global_edge_energy(graph, before),
            0.006931,
            places=6,
        )

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

    def test_summary_uses_sample_sd_and_ci(self):
        out = summarize([1.0, 2.0, 3.0])
        self.assertAlmostEqual(out['sd'], 1.0)
        self.assertEqual(out['std'], out['sd'])
        self.assertLess(out['ci95_low'], out['mean'])
        self.assertGreater(out['ci95_high'], out['mean'])
        paired = paired_difference([2, 4, 6], [1, 1, 1])
        self.assertEqual(paired['definition'], 'first_minus_second')

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
        self.assertNotIn('no_structure', MODEL_TYPES)
        self.assertIn('unilateral', MODEL_TYPES['baseline']['label'].lower())

    def test_canonical_epsilon_grid_preserves_quarter_hundredths(self):
        self.assertEqual(len(EPSILON_0_GRID), 19)
        self.assertEqual(EPSILON_0_GRID[1], 0.125)
        self.assertEqual(EPSILON_0_GRID[-1], 0.55)
        self.assertEqual(list(TRUST['epsilon_0_range']), list(EPSILON_0_GRID))
        self.assertEqual(list(TRUST['epsilon_values']), list(EPSILON_0_GRID))
        self.assertEqual(list(JOINT_SWEEP['eps_values']), list(EPSILON_0_GRID))
        self.assertEqual(format_eps_key(0.125), '0.125')
        self.assertEqual(len(set(format_eps_key(value)
                                 for value in EPSILON_0_GRID)), 19)

    def test_run_all_requires_explicit_selection(self):
        run_all_path = os.path.join(ROOT, 'run_all.py')
        with open(run_all_path, 'r', encoding='utf-8') as handle:
            source = handle.read()
        self.assertIn('安全停止', source)
        self.assertIn('--all', source)

    def test_resume_signature_tracks_core_parameters(self):
        signature = ExperimentManager._core_parameter_signature()
        self.assertEqual(signature, {
            'epsilon_0': float(TRUST['epsilon_0_default']),
            'alpha': float(TRUST['alpha']),
            'mu': float(DYNAMICS['mu']),
            'beta': float(DYNAMICS['beta_default']),
        })

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

    def test_perturbation_workload_counts_one_shared_forward_scan(self):
        runs = 20
        estimate = estimate_elementary_updates(
            ['perturbation'], quick=False, runs=runs)
        expected = (
            3
            * (1 + len(SIMULATION['perturb_sigma_values']))
            * len(EPSILON_0_GRID)
            * runs
            * config_module.NETWORK['N']
            * SIMULATION['sweep_sweeps']
        )
        self.assertEqual(estimate['perturbation'], expected)

    def test_quick_mode_honors_explicit_runs_and_canonical_epsilon_subset(self):
        manager = ExperimentManager()
        with patch.object(
                manager, 'experiment_epsilon_sweep', return_value={}) as runner:
            with contextlib.redirect_stdout(io.StringIO()):
                run_selected(
                    manager, ['epsilon'], quick=True, runs=1, resume=False)
        kwargs = runner.call_args[1]
        self.assertEqual(kwargs['num_runs'], 1)
        self.assertEqual(
            kwargs['eps_values'],
            [EPSILON_0_GRID[0], EPSILON_0_GRID[len(EPSILON_0_GRID) // 2],
             EPSILON_0_GRID[-1]],
        )

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
                with contextlib.redirect_stdout(io.StringIO()):
                    result = ExperimentManager().experiment_ablation(
                        num_runs=1, p_values=[0.0], resume=True,
                        output_filename='resume_test.json')
                block = result['0.0']
                self.assertNotIn('sentinel', block)
                self.assertEqual(
                    block['_control']['core_parameters'],
                    ExperimentManager._core_parameter_signature())
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
                manager.save_json(
                    {'_control': base, '1.0': {'sentinel': True}},
                    'controlled_resume_test.json')
                same = manager._load_controlled_results(
                    'controlled_resume_test.json', base, resume=True)
                beta_changed = manager._load_controlled_results(
                    'controlled_resume_test.json',
                    {**base, 'beta': 4.0}, resume=True)
                sweeps_changed = manager._load_controlled_results(
                    'controlled_resume_test.json',
                    {**base, 'main_sweeps': 100.0}, resume=True)
                self.assertIn('1.0', same)
                self.assertEqual(set(beta_changed), {'_control'})
                self.assertEqual(set(sweeps_changed), {'_control'})
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
                manager.experiment_perturbation_sweep(
                    num_runs=1, p_values=[0.0], sigma_values=[0.0],
                    eps_values=[0.1], resume=True, output_filename=filename)
            self.assertEqual(run_shared.call_count, 1)

    def test_synchronous_mode_uses_sweep_budget(self):
        result = SingleRun(
            N=30, m=2, p_hk=0.5, epsilon_0=0.15, alpha=3.5,
            mu=0.5, beta=3.0, model_type='full_model',
            num_steps=60, seed=7, update_mode='synchronous',
        ).execute(record_interval=30)
        self.assertEqual(result['params']['update_mode'], 'synchronous')
        self.assertEqual(result['time_series']['sweep'], [1.0, 2.0])
        self.assertEqual(result['convergence']['steps_run'], 60)

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
