# Reproducibility checklist

The revised experiments use Monte Carlo sweeps as the time unit: one sweep is
`N` asynchronous unilateral updates.  The default long-horizon setting is 200
sweeps, with 20 paired seeds for the formal ablation.

## Environment

The versions used during the revision audit are pinned in
`requirements-lock.txt`.  The code was executed with Python 3.7.0, NumPy
1.21.6, NetworkX 2.6.3, Matplotlib 2.2.3 and SciPy 1.1.0.

## Smoke tests

From this directory:

```text
python -m compileall -q .
python -m unittest discover -s tests -v
python run_all.py --quick --N 50 --main-sweeps 2 --sweep-sweeps 2 --only empirical --plot
```

The explicit `--N 50 --main-sweeps 2 --sweep-sweeps 2` options make this a
small two-sweep smoke run.  `--quick` alone reduces replicate counts and some
parameter grids; it does not silently change the scientific time budget.
Quick output is isolated under `output/smoke/` and must not be cited as a
formal result.  Use `--output-dir PATH` when a separate named run directory
is required.

## Formal run order

Run the six-model ablation, then the threshold-assignment controls and the
failed-triangle fallback control before the perturbation, update-mode, SBM,
vector-opinion and empirical-topology controls. The two additional controls
can be run independently:

    python run_all.py --only threshold_control --runs 20 --output-dir output/formal_revision
    python run_all.py --only triangle_failure_control --runs 20 --output-dir output/formal_revision
    python -B analysis/analyze_formal_threshold_controls.py
    python -B analysis/analyze_formal_triangle_failure_control.py
    python -B analysis/run_formal_threshold_recovery.py --runs 20 --sweeps 100
    python -B analysis/analyze_formal_threshold_recovery.py

The threshold control compares edge-specific thresholds with their realized
unweighted edge mean and with the same threshold multiset permuted across
undirected edges. The triangle-failure control compares the current dyadic
fallback with no opinion change after a failed triangle gate. Both controls
use the existing matched seed schedule and 200-sweep main horizon by default.
The threshold-recovery control obtains one unperturbed high-to-low
continuation path, then branches independently at every epsilon. Matched
standard-normal shock directions are scaled across sigma levels; each branch
continues at the same epsilon for the selected horizon. It records opinion SD,
range, and the 0.05-diameter order parameter. Recovery is defined as
$R_{0.05} >= 1$ for three consecutive recorded measurements; trajectories
that do not recover within the horizon are right-censored. The 100-sweep
setting follows the perturbation horizon audit.
Keep each experiment's JSON and .meta.json sidecar together. The sidecar
records configuration, seeds, Python/platform information and SHA-256 hashes
of all top-level simulation source files.
Result files are written through an atomic same-directory replacement so an
interrupted checkpoint cannot truncate the previous resumable JSON.
All experiment entry points checkpoint completed parameter blocks and validate
their full control signature before resuming. A changed network size, sweep
budget, parameter grid, seed set or core dynamical parameter causes the stale
checkpoint to be ignored. Use --no-resume to force a clean recomputation;
the flag is propagated in both selected and full-run modes.

Use --dry-run to inspect the effective configuration before starting a long
calculation, for example:

```text
python run_all.py --dry-run --only ablation --runs 20 --main-sweeps 200 --output-dir output/formal_revision
```
