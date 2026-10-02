"""Compute one p_HK block for the formal perturbation sweep.

Workers write separate files. The parent merge step is intentionally separate
so that concurrent processes never modify the same JSON checkpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments as experiments_module  # noqa: E402
from config import EPSILON_0_GRID, SIMULATION  # noqa: E402
from experiments import ExperimentManager  # noqa: E402


FORMAL_DATA_DIR = ROOT / "output" / "formal_revision" / "data" / "perturbation_workers"
SIGMA_VALUES = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]
EPS_VALUES = list(EPSILON_0_GRID)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=float, required=True)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--sweeps", type=float, default=100.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    old_dir = experiments_module.DATA_DIR
    old_sweeps = SIMULATION.get("sweep_sweeps")
    try:
        FORMAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
        experiments_module.DATA_DIR = str(FORMAL_DATA_DIR)
        SIMULATION["sweep_sweeps"] = float(args.sweeps)
        manager = ExperimentManager()
        manager.experiment_perturbation_sweep(
            num_runs=args.runs, p_values=[args.p], sigma_values=SIGMA_VALUES,
            eps_values=EPS_VALUES, perturb_each_step=False,
            resume=args.resume,
            output_filename="exp_formal_perturbation_p_{:.2f}.json".format(args.p),
        )
    finally:
        experiments_module.DATA_DIR = old_dir
        SIMULATION["sweep_sweeps"] = old_sweeps


if __name__ == "__main__":
    main()
