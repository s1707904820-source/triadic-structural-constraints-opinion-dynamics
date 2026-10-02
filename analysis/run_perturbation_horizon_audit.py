"""Pilot audit of the per-epsilon horizon for perturbation/hysteresis scans.

This is a design audit, not a formal result.  It runs the shared-forward
perturbation gradient at 50, 100, and 200 sweeps per epsilon point and keeps
each horizon in a separate, resumable checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments as experiments_module  # noqa: E402
from config import EPSILON_0_GRID, NETWORK, SIMULATION  # noqa: E402
from experiments import ExperimentManager  # noqa: E402


AUDIT_DIR = ROOT / "output" / "formal_revision" / "data" / "perturbation_horizon_audit"
P_VALUES = [0.0, 0.4, 0.8]
SIGMA_VALUES = [0.0, 0.05, 0.2]
EPS_VALUES = list(EPSILON_0_GRID)


def estimate_updates(horizon, n_runs):
    # Per seed and p: one shared forward plus one backward branch per sigma.
    branches = 1 + len(SIGMA_VALUES)
    return int(
        len(P_VALUES) * n_runs * len(EPS_VALUES) * branches
        * int(NETWORK["N"]) * float(horizon)
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", nargs="+", type=float,
                        default=[50.0, 100.0, 200.0])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    horizons = [float(value) for value in args.horizons]
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    total = sum(estimate_updates(value, args.runs) for value in horizons)
    print("Perturbation horizon audit")
    print("  status: pilot/design audit, not formal evidence")
    print("  N={}, p={}, sigma={}, epsilon points={}, runs={}".format(
        NETWORK["N"], P_VALUES, SIGMA_VALUES, len(EPS_VALUES), args.runs))
    print("  horizons={}".format(horizons))
    print("  estimated elementary updates={:,}".format(total))
    if args.dry_run:
        return

    old_data_dir = experiments_module.DATA_DIR
    old_sweeps = SIMULATION.get("sweep_sweeps")
    try:
        experiments_module.DATA_DIR = str(AUDIT_DIR)
        manager = ExperimentManager(workers=args.workers)
        timing = {}
        for horizon in horizons:
            SIMULATION["sweep_sweeps"] = float(horizon)
            label = ("{:g}".format(horizon)).replace(".", "p")
            filename = "exp_perturbation_horizon_{}.json".format(label)
            started = time.time()
            manager.experiment_perturbation_sweep(
                num_runs=args.runs,
                p_values=P_VALUES,
                sigma_values=SIGMA_VALUES,
                eps_values=EPS_VALUES,
                perturb_each_step=False,
                resume=not args.no_resume,
                output_filename=filename,
            )
            timing[str(horizon)] = {
                "elapsed_seconds_this_invocation": time.time() - started,
                "estimated_elementary_updates": estimate_updates(
                    horizon, args.runs),
                "checkpoint": str(AUDIT_DIR / filename),
            }
        timing_path = AUDIT_DIR / "horizon_audit_timing.json"
        timing_path.write_text(json.dumps(timing, indent=2), encoding="utf-8")
        print("Timing summary: {}".format(timing_path))
    finally:
        experiments_module.DATA_DIR = old_data_dir
        SIMULATION["sweep_sweeps"] = old_sweeps


if __name__ == "__main__":
    main()
