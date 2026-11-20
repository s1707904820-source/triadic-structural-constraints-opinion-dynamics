"""Run the formal paired two-community SBM hysteresis experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments as experiments_module  # noqa: E402
from config import NETWORK, SIMULATION, TRUST  # noqa: E402
from experiments import ExperimentManager  # noqa: E402


FORMAL_DATA_DIR = ROOT / "output" / "formal_revision" / "data"
FORMAL_OUTPUT_FILENAME = "exp_sbm_hysteresis_revision_v2.json"
MIXING_VALUES = [0.02, 0.05, 0.10, 0.20, 0.35]
SIGMA_VALUES = [0.0, 0.05, 0.10, 0.20]
EPS_VALUES = list(TRUST["epsilon_0_range"])


def estimate_updates(runs, mixing_values, sigma_values, eps_values, sweeps):
    """Count node updates across the shared forward and backward branches."""
    branches = 1 + len(sigma_values)
    updates_per_threshold = int(round(
        int(NETWORK["N"]) * float(sweeps)))
    return int(
        branches * len(mixing_values) * int(runs) * len(eps_values)
        * updates_per_threshold
    )


def main():
    parser = argparse.ArgumentParser(
        description="Formal paired forward/backward SBM hysteresis scan")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--sweeps", type=float, default=200.0)
    parser.add_argument("--mixing-values", nargs="+", type=float,
                        default=MIXING_VALUES)
    parser.add_argument("--sigma-values", nargs="+", type=float,
                        default=SIGMA_VALUES)
    parser.add_argument("--eps-values", nargs="+", type=float,
                        default=EPS_VALUES)
    parser.add_argument("--target-degree", type=float, default=8.0)
    parser.add_argument("--workers", type=int, default=5,
                        help="parallel mixing branches; use at most one worker per mixing value")
    parser.add_argument("--perturb-each-step", action="store_true",
                        help="inject a matched shock before every reverse threshold")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.sweeps <= 0.0:
        parser.error("--sweeps must be positive")
    if not args.mixing_values or not args.sigma_values or not args.eps_values:
        parser.error("mixing, sigma, and epsilon grids must be non-empty")
    if any(value < 0.0 or value > 1.0
           for value in args.mixing_values):
        parser.error("mixing values must be between 0 and 1")
    if any(value < 0.0 for value in args.sigma_values):
        parser.error("sigma values must be non-negative")
    if any(value < 0.0 or value > 1.0 for value in args.eps_values):
        parser.error("epsilon values must be between 0 and 1")
    if args.target_degree <= 0.0:
        parser.error("--target-degree must be positive")
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    estimate = estimate_updates(
        args.runs, args.mixing_values, args.sigma_values,
        args.eps_values, args.sweeps)
    print("Formal two-community SBM hysteresis experiment")
    print("  N={}, mixing points={}, sigma branches={}, epsilon points={}, runs={}".format(
        NETWORK["N"], len(args.mixing_values), len(args.sigma_values),
        len(args.eps_values), args.runs))
    print("  mixing={}, sigma={}, epsilon={}".format(
        args.mixing_values, args.sigma_values, args.eps_values))
    print("  target degree={}, horizon={} sweeps/epsilon".format(
        args.target_degree, args.sweeps))
    print("  perturb before every reverse epsilon={}".format(
        args.perturb_each_step))
    print("  workers={}".format(min(args.workers, len(args.mixing_values))))
    print("  output file={}".format(FORMAL_OUTPUT_FILENAME))
    print("  estimated elementary updates={:,}".format(estimate))
    if args.dry_run:
        return

    FORMAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    old_data_dir = experiments_module.DATA_DIR
    old_sweeps = SIMULATION.get("sweep_sweeps")
    try:
        experiments_module.DATA_DIR = str(FORMAL_DATA_DIR)
        SIMULATION["sweep_sweeps"] = float(args.sweeps)
        manager = ExperimentManager()
        manager.experiment_sbm_hysteresis(
            num_runs=args.runs,
            mixing_values=args.mixing_values,
            sigma_values=args.sigma_values,
            eps_values=args.eps_values,
            perturb_each_step=args.perturb_each_step,
            target_degree=args.target_degree,
            workers=args.workers,
            resume=not args.no_resume,
            output_filename=FORMAL_OUTPUT_FILENAME,
        )
    finally:
        experiments_module.DATA_DIR = old_data_dir
        SIMULATION["sweep_sweeps"] = old_sweeps


if __name__ == "__main__":
    main()
