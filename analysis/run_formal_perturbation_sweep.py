"""Run the formal shared-forward perturbation gradient after horizon audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments as experiments_module  # noqa: E402
from config import SIMULATION  # noqa: E402
from experiments import ExperimentManager  # noqa: E402


FORMAL_DATA_DIR = ROOT / "output" / "formal_revision" / "data"
FORMAL_OUTPUT_FILENAME = "exp_formal_perturbation_sweep_revision_v2.json"
P_VALUES = [0.0, 0.4, 0.8]
SIGMA_VALUES = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]
EPS_VALUES = [round(0.10 + index * 0.025, 3) for index in range(19)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--sweeps", type=float, default=100.0)
    parser.add_argument("--workers", type=int, default=3,
                        help="parallel p branches; parent process writes checkpoints")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    old_dir = experiments_module.DATA_DIR
    old_sweeps = SIMULATION.get("sweep_sweeps")
    try:
        experiments_module.DATA_DIR = str(FORMAL_DATA_DIR)
        SIMULATION["sweep_sweeps"] = float(args.sweeps)
        manager = ExperimentManager()
        manager.experiment_perturbation_sweep(
            num_runs=args.runs,
            p_values=P_VALUES,
            sigma_values=SIGMA_VALUES,
            eps_values=EPS_VALUES,
            perturb_each_step=False,
            workers=args.workers,
            resume=args.resume,
            output_filename=FORMAL_OUTPUT_FILENAME,
        )
    finally:
        experiments_module.DATA_DIR = old_dir
        SIMULATION["sweep_sweeps"] = old_sweeps


if __name__ == "__main__":
    main()
