"""Run the formal paired transition-prediction experiment."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments as experiments_module  # noqa: E402
from config import METRICS, NETWORK, SIMULATION, TRUST  # noqa: E402
from experiments import ExperimentManager  # noqa: E402


FORMAL_DATA_DIR = ROOT / "output" / "formal_revision" / "data"
P_VALUES = [round(index * 0.05, 2) for index in range(21)]
EPS_VALUES = [round(index * 0.025, 3) for index in range(23)]
FORMAL_OUTPUT_FILENAME = (
    "exp_formal_threshold_transition_prediction_revision_v3.json")


def estimate_updates(runs, p_values, eps_values, sweeps):
    variants = 3
    return int(
        variants * len(p_values) * runs * len(eps_values)
        * int(NETWORK["N"]) * float(sweeps)
    )


def validate_runner_args(args):
    """Validate the full scientific design before even a dry-run returns."""
    if isinstance(args.runs, bool) or not isinstance(args.runs, int) or args.runs < 1:
        raise ValueError("--runs must be a positive integer")
    if (isinstance(args.workers, bool) or
            not isinstance(args.workers, int) or args.workers < 1):
        raise ValueError("--workers must be a positive integer")

    sweeps = float(args.sweeps)
    alpha = float(args.alpha)
    if not math.isfinite(sweeps) or sweeps <= 0.0:
        raise ValueError("--sweeps must be finite and positive")
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("--alpha must be finite and positive")

    n = int(NETWORK["N"])
    m = int(NETWORK["m"])
    if n < 2 or m < 1 or m >= n:
        raise ValueError("network configuration requires N >= 2 and 1 <= m < N")
    updates_per_epsilon = float(n) * sweeps
    if (not math.isfinite(updates_per_epsilon) or
            int(round(updates_per_epsilon)) < 1):
        raise ValueError("--sweeps must imply at least one elementary update")

    for label, values in (("--p-values", args.p_values),
                          ("--eps-values", args.eps_values)):
        if not values:
            raise ValueError("{} must contain at least one value".format(label))
        numeric = [float(value) for value in values]
        if any(not math.isfinite(value) for value in numeric):
            raise ValueError("{} must contain only finite values".format(label))
        if any(value < 0.0 or value > 1.0 for value in numeric):
            raise ValueError("{} values must lie in [0, 1]".format(label))
        if any(right <= left for left, right in zip(numeric, numeric[1:])):
            raise ValueError("{} must be strictly increasing without duplicates".format(label))

    width = float(METRICS["transition_order_window_width"])
    cutoff = float(METRICS["transition_order_cutoff"])
    if not math.isfinite(width) or not (0.0 < width <= 1.0):
        raise ValueError("transition order window width must lie in (0, 1]")
    if not math.isfinite(cutoff) or not (0.0 < cutoff <= 1.0):
        raise ValueError("transition order cutoff must lie in (0, 1]")

    return estimate_updates(
        args.runs, args.p_values, args.eps_values, sweeps)


def output_target_status():
    """Describe checkpoint targets without creating or changing any files."""
    data_path = FORMAL_DATA_DIR / FORMAL_OUTPUT_FILENAME
    metadata_path = data_path.with_name(data_path.stem + ".meta.json")
    if not FORMAL_DATA_DIR.exists():
        return "formal data directory absent; dry-run will not create it"
    if data_path.is_symlink() or metadata_path.is_symlink():
        return "WARNING: checkpoint or metadata target is a symlink"
    if data_path.exists() and not data_path.is_file():
        return "WARNING: checkpoint target exists but is not a regular file"
    if metadata_path.exists() and not metadata_path.is_file():
        return "WARNING: metadata target exists but is not a regular file"
    if data_path.exists() and not metadata_path.exists():
        return "WARNING: checkpoint exists without metadata; execution must stop for review"
    if metadata_path.exists() and not data_path.exists():
        return "WARNING: metadata exists without checkpoint; execution must stop for review"
    if data_path.exists():
        return "checkpoint exists; runtime will require an exact control/signature match"
    return "versioned checkpoint target is unused"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sweeps", type=float, default=100.0)
    parser.add_argument("--p-values", nargs="+", type=float, default=P_VALUES)
    parser.add_argument("--eps-values", nargs="+", type=float,
                        default=EPS_VALUES)
    parser.add_argument("--alpha", type=float, default=TRUST["alpha"])
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        estimate = validate_runner_args(args)
    except (TypeError, ValueError, OverflowError) as exc:
        parser.error(str(exc))
    print("Paired transition-prediction experiment")
    print("  N={}, m={}, p={}, epsilon points={}, runs={}".format(
        NETWORK["N"], NETWORK["m"], args.p_values, len(args.eps_values),
        args.runs))
    print("  alpha={}, horizon={} sweeps/epsilon".format(
        args.alpha, args.sweeps))
    print("  workers={}".format(args.workers))
    print("  transition order window width={}".format(
        METRICS['transition_order_window_width']))
    print("  transition order cutoff={}".format(
        METRICS['transition_order_cutoff']))
    print("  estimated elementary updates={:,}".format(estimate))
    print("  output filename={}".format(FORMAL_OUTPUT_FILENAME))
    print("  output target status={}".format(output_target_status()))
    if args.dry_run:
        return

    FORMAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    old_data_dir = experiments_module.DATA_DIR
    old_sweeps = SIMULATION.get("sweep_sweeps")
    try:
        experiments_module.DATA_DIR = str(FORMAL_DATA_DIR)
        SIMULATION["sweep_sweeps"] = float(args.sweeps)
        manager = ExperimentManager()
        manager.experiment_threshold_transition_prediction(
            num_runs=args.runs,
            p_values=args.p_values,
            eps_values=args.eps_values,
            sweeps_per_epsilon=args.sweeps,
            alpha_target=args.alpha,
            workers=args.workers,
            resume=not args.no_resume,
            output_filename=FORMAL_OUTPUT_FILENAME,
        )
    finally:
        experiments_module.DATA_DIR = old_data_dir
        SIMULATION["sweep_sweeps"] = old_sweeps


if __name__ == "__main__":
    main()
