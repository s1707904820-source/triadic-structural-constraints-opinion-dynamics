# Reproducibility checklist

The revised experiments use Monte Carlo sweeps as the time unit: one sweep is
`N` asynchronous unilateral updates.  The default long-horizon setting is 200
sweeps, with 20 paired seeds for the formal ablation.

## Environment

The versions used for the authoritative formal batch are pinned in
`requirements-lock.txt`. The formal results were generated with Python 3.13.0,
NumPy 2.5.3, NetworkX 3.7, Matplotlib 3.11.2 and SciPy 1.18.1. No
machine-specific executable path is required for reproduction.

## Smoke tests

From this directory:

```text
python -m compileall -q .
python -m unittest discover -s tests -v
python run_all.py --quick --N 50 --main-sweeps 2 --sweep-sweeps 2 --workers 2 --only empirical --plot
```

The explicit overrides make this a small two-sweep smoke run. `--quick` is a
smoke-test profile: unless explicitly overridden, it sets `N=200`, uses three
replicates, reduces selected parameter grids, and sets both the main and scan
horizons to two sweeps. Therefore quick output is never scientific evidence.
Quick output is isolated under `output/smoke/` and must not be cited as a
formal result.  Use `--output-dir PATH` when a separate named run directory
is required.

The command-line entry point will not run experiments without an explicit
selection. Use `--only NAME [NAME ...]` for selected experiments. `--all` is
required to start all 17 experiment classes; inspect the same command with
`--dry-run` first.

## Formal run order

Run the six-model ablation first, followed by the perturbation, update-mode,
SBM, vector-opinion and empirical-topology controls.  Keep each experiment's
JSON and `.meta.json` sidecar together.  The sidecar records configuration,
seeds, Python/platform information and SHA-256 hashes of all source files.
Result files are written through an atomic same-directory replacement so an
interrupted checkpoint cannot truncate the previous resumable JSON.
All experiment entry points checkpoint completed parameter blocks and validate
their full control signature before resuming.  A changed network size, sweep
budget, parameter grid, seed set or core dynamical parameter causes the stale
checkpoint to be ignored.  Use `--no-resume` to force a clean recomputation;
the flag is propagated in both selected and full-run modes.

Use `--dry-run` to inspect the effective configuration before starting a long
calculation, for example:

```text
python run_all.py --dry-run --only ablation --runs 20 --main-sweeps 200 --workers 4 --output-dir output/formal_py313
```

## Multiprocessing

`--workers K` enables spawn-safe process parallelism for independent seeds or
independent parameter blocks.  The command-line default is four workers; use
`--workers 1` for serial troubleshooting.  Forward and
backward epsilon paths remain sequential within each replicate, and the parent
process alone writes checkpoints.  Ordered result collection preserves seed
pairing and produces the same scientific JSON for serial and parallel runs.
The worker count is recorded in each `.meta.json` sidecar but is intentionally
not part of the scientific resume signature.
