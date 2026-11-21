"""Checkpoint safety tests for independent threshold-level recovery."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import experiments as experiments_module
from experiments import ExperimentManager
from config import NETWORK, SIMULATION


class RecoveryCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.params = dict(
            num_runs=1, p_values=[0.4], sigma_values=[0.0, 0.1],
            eps_values=[0.1, 0.2], recovery_persistence=2,
            output_filename='new_recovery_checkpoint.json')
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(
            experiments_module, 'DATA_DIR', self.tempdir.name))
        self.stack.enter_context(patch.dict(NETWORK, {'N': 24, 'm': 2}))
        self.stack.enter_context(patch.dict(SIMULATION, {
            'sweep_sweeps': 0.5, 'record_sweeps': 0.25}))
        self.manager = ExperimentManager()
        self.path = Path(self.tempdir.name) / self.params['output_filename']

    def _create(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.manager.experiment_threshold_recovery(
                resume=False, **self.params)

    def test_valid_whole_block_resumes_without_simulation(self):
        original = self._create()
        before = self.path.read_bytes()
        with patch.object(self.manager, '_run_threshold_recovery_for_p') as run:
            with contextlib.redirect_stdout(io.StringIO()):
                resumed = self.manager.experiment_threshold_recovery(
                    resume=True, **self.params)
        run.assert_not_called()
        self.assertEqual(resumed['data'], original['data'])
        self.assertEqual(self.path.read_bytes(), before)

    def test_incomplete_sigma_cell_fails_without_overwrite(self):
        self._create()
        payload = json.loads(self.path.read_text())
        del payload['data']['0.40']['by_epsilon']['0.1']['0.100']
        self.path.write_text(json.dumps(payload))
        before = self.path.read_bytes()
        with patch.object(self.manager, '_run_threshold_recovery_for_p') as run:
            with self.assertRaisesRegex(ValueError, 'incomplete recovery'):
                self.manager.experiment_threshold_recovery(
                    resume=True, **self.params)
        run.assert_not_called()
        self.assertEqual(self.path.read_bytes(), before)

    def test_corrupted_event_flags_fail_without_overwrite(self):
        self._create()
        payload = json.loads(self.path.read_text())
        recovery = payload['data']['0.40']['by_epsilon']['0.1']['0.100'][
            'recovery']
        recovery['recovered_by_run'][0] = not recovery['recovered_by_run'][0]
        self.path.write_text(json.dumps(payload))
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'incomplete recovery'):
            self.manager.experiment_threshold_recovery(
                resume=True, **self.params)
        self.assertEqual(self.path.read_bytes(), before)

    def test_source_mismatch_and_no_resume_preserve_checkpoint(self):
        self._create()
        before = self.path.read_bytes()
        with self.assertRaises(FileExistsError):
            self.manager.experiment_threshold_recovery(
                resume=False, **self.params)
        with patch.object(experiments_module, 'execution_source_signature',
                          return_value='changed-code'):
            with self.assertRaisesRegex(ValueError, 'control/source mismatch'):
                self.manager.experiment_threshold_recovery(
                    resume=True, **self.params)
        self.assertEqual(self.path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
