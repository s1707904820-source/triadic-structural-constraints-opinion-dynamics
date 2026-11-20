"""Equivalence checks for safe p-branch parallel execution."""

import contextlib
import io
import tempfile
import unittest
from unittest.mock import patch

import experiments as experiments_module
from config import NETWORK, SIMULATION
from experiments import ExperimentManager


class ParallelFormalBranchTests(unittest.TestCase):
    def _compare(self, method_name, kwargs):
        with patch.dict(NETWORK, {'N': 20, 'm': 2}), \
                patch.dict(SIMULATION, {
                    'main_sweeps': 0.2, 'record_sweeps': 0.1,
                    'sweep_sweeps': 0.2}), \
                tempfile.TemporaryDirectory() as serial_dir, \
                tempfile.TemporaryDirectory() as parallel_dir:
            with patch.object(experiments_module, 'DATA_DIR', serial_dir), \
                    contextlib.redirect_stdout(io.StringIO()):
                serial = getattr(ExperimentManager(), method_name)(
                    workers=1, output_filename='serial.json', **kwargs)
            with patch.object(experiments_module, 'DATA_DIR', parallel_dir), \
                    contextlib.redirect_stdout(io.StringIO()):
                parallel = getattr(ExperimentManager(), method_name)(
                    workers=2, output_filename='parallel.json', **kwargs)
        self.assertEqual(serial['data'], parallel['data'])

    def test_threshold_assignment_parallel_matches_serial(self):
        self._compare(
            'experiment_threshold_assignment_controls',
            dict(num_runs=1, p_values=[0.0, 0.4], resume=False))

    def test_triangle_failure_parallel_matches_serial(self):
        self._compare(
            'experiment_triangle_failure_control',
            dict(num_runs=1, p_values=[0.0, 0.4], resume=False))

    def test_recovery_parallel_matches_serial(self):
        self._compare(
            'experiment_threshold_recovery',
            dict(num_runs=1, p_values=[0.0, 0.4], sigma_values=[0.0],
                 eps_values=[0.1, 0.2], recovery_persistence=2,
                 resume=False))

    def test_perturbation_parallel_matches_serial(self):
        self._compare(
            'experiment_perturbation_sweep',
            dict(num_runs=1, p_values=[0.0, 0.4], sigma_values=[0.0, 0.1],
                 eps_values=[0.1, 0.2], resume=False))


if __name__ == '__main__':
    unittest.main()
