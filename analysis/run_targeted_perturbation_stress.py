"""Run a non-destructive stress scan near a possible perturbation boundary.

This is an extension experiment. It writes to
``output/formal_revision_extension/data`` and never overwrites the original
``output/formal_revision`` formal result.
"""

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

EXTENSION_DATA_DIR = ROOT / "output" / "formal_revision_extension" / "data"
DEFAULT_P = [0.0, 0.4, 0.8]
DEFAULT_SIGMA = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
DEFAULT_EPS = [0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15,
               0.20, 0.30, 0.40, 0.50, 0.55]


def estimate_updates(runs, p_values, sigma_values, eps_values, sweeps):
    branches = 1 + len(sigma_values)
    return int(len(p_values) * int(runs) * branches * len(eps_values)
               * int(NETWORK["N"]) * float(sweeps))


def main():
    parser = argparse.ArgumentParser(
        description="Targeted perturbation stress scan; preserves formal outputs")
    parser.add_argument("--runs", type=int, default=5,
                        help="Pilot default is 5; use 20 after the pilot")
    parser.add_argument("--sweeps", type=float, default=150.0)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--p-values", nargs="+", type=float, default=DEFAULT_P)
    parser.add_argument("--sigma-values", nargs="+", type=float,
                        default=DEFAULT_SIGMA)
    parser.add_argument("--eps-values", nargs="+", type=float,
                        default=DEFAULT_EPS)
    parser.add_argument("--perturb-each-step", action="store_true",
                        help="Inject a matched shock before every reverse threshold")
    parser.add_argument("--output-name", default="exp_targeted_perturbation_stress_v1.json")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.runs < 1 or args.sweeps <= 0 or args.workers < 1:
        parser.error("runs, sweeps, and workers must be positive")
    if any(x < 0.0 or x > 1.0 for x in args.p_values):
        parser.error("p-values must lie in [0, 1]")
    if any(x < 0.0 for x in args.sigma_values):
        parser.error("sigma values must be non-negative")
    if any(x < 0.0 or x > 1.0 for x in args.eps_values):
        parser.error("epsilon values must lie in [0, 1]")
    if Path(args.output_name).name != args.output_name:
        parser.error("output-name must be a filename, not a path")

    updates = estimate_updates(args.runs, args.p_values, args.sigma_values,
                                args.eps_values, args.sweeps)
    print("Targeted perturbation stress scan (new extension output)")
    print(f"  N={NETWORK['N']}, runs={args.runs}, workers={args.workers}")
    print(f"  p={args.p_values}")
    print(f"  sigma={args.sigma_values}")
    print(f"  epsilon={args.eps_values}")
    print(f"  horizon={args.sweeps} sweeps/epsilon")
    print(f"  perturb_each_step={args.perturb_each_step}")
    print(f"  estimated elementary updates={updates:,}")
    print(f"  output={EXTENSION_DATA_DIR / args.output_name}")
    if args.dry_run:
        return

    EXTENSION_DATA_DIR.mkdir(parents=True, exist_ok=True)
    old_data_dir = experiments_module.DATA_DIR
    old_sweeps = SIMULATION.get("sweep_sweeps")
    try:
        experiments_module.DATA_DIR = str(EXTENSION_DATA_DIR)
        SIMULATION["sweep_sweeps"] = float(args.sweeps)
        manager = ExperimentManager()
        manager.experiment_perturbation_sweep(
            num_runs=args.runs, p_values=args.p_values,
            sigma_values=args.sigma_values, eps_values=args.eps_values,
            perturb_each_step=args.perturb_each_step, workers=args.workers,
            resume=not args.no_resume, output_filename=args.output_name)
    finally:
        experiments_module.DATA_DIR = old_data_dir
        SIMULATION["sweep_sweeps"] = old_sweeps


if __name__ == "__main__":
    main()
