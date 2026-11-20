"""Run the formal threshold-by-threshold perturbation/recovery experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments as experiments_module  # noqa: E402
from config import NETWORK, SIMULATION  # noqa: E402
from experiments import ExperimentManager  # noqa: E402


FORMAL_DATA_DIR = ROOT / "output" / "formal_revision" / "data"
P_VALUES = [0.0, 0.4, 0.8]
SIGMA_VALUES = [0.02]
EPS_VALUES = [round(0.10 + index * 0.025, 3) for index in range(19)]


def estimate_updates(runs, p_values, sigma_values, eps_values, sweeps):
    branches = 1 + len(sigma_values)
    return int(
        len(p_values) * runs * len(eps_values) * branches
        * int(NETWORK["N"]) * float(sweeps)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--sweeps", type=float, default=100.0)
    parser.add_argument("--record-sweeps", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=3,
                        help="parallel p branches; parent process writes checkpoints")
    parser.add_argument("--p-values", nargs="+", type=float, default=P_VALUES)
    parser.add_argument("--sigma-values", nargs="+", type=float,
                        default=SIGMA_VALUES)
    parser.add_argument("--eps-values", nargs="+", type=float,
                        default=EPS_VALUES)
    parser.add_argument("--recovery-persistence", type=int, default=3)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    estimate = estimate_updates(
        args.runs, args.p_values, args.sigma_values, args.eps_values,
        args.sweeps)
    print("Threshold-by-threshold recovery experiment")
    print("  N={}, m={}, p={}, sigma={}, epsilon points={}, runs={}".format(
        NETWORK["N"], NETWORK["m"], args.p_values, args.sigma_values,
        len(args.eps_values), args.runs))
    print("  horizon={} sweeps/epsilon; record interval={} sweep".format(
        args.sweeps, args.record_sweeps))
    print("  estimated elementary updates={:,}".format(estimate))
    print("  workers={}".format(min(args.workers, len(args.p_values))))
    if args.dry_run:
        return

    FORMAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    old_data_dir = experiments_module.DATA_DIR
    old_sweeps = SIMULATION.get("sweep_sweeps")
    old_record_sweeps = SIMULATION.get("record_sweeps")
    try:
        experiments_module.DATA_DIR = str(FORMAL_DATA_DIR)
        SIMULATION["sweep_sweeps"] = float(args.sweeps)
        SIMULATION["record_sweeps"] = float(args.record_sweeps)
        manager = ExperimentManager()
        manager.experiment_threshold_recovery(
            num_runs=args.runs,
            p_values=args.p_values,
            sigma_values=args.sigma_values,
            eps_values=args.eps_values,
            recovery_order_threshold=1.0,
            recovery_persistence=args.recovery_persistence,
            workers=args.workers,
            resume=not args.no_resume,
            output_filename="exp_formal_threshold_recovery.json",
        )
    finally:
        experiments_module.DATA_DIR = old_data_dir
        SIMULATION["sweep_sweeps"] = old_sweeps
        SIMULATION["record_sweeps"] = old_record_sweeps


if __name__ == "__main__":
    main()
