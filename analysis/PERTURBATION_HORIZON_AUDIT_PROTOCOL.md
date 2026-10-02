# Perturbation and hysteresis horizon audit

Status: pilot design audit. Results from this audit must not be described as
formal manuscript evidence.

## Purpose

Determine whether 100 sweeps per epsilon point is sufficient relative to the
current 200-sweep setting. The audit also checks whether 50 sweeps is visibly
under-converged. The decision is based on the quantities used for the
reviewer-facing inference, not on runtime alone.

## Fixed design

- Network size: N = 2000, m = 4.
- Holme-Kim values: p_HK = 0.0, 0.4, 0.8.
- Perturbation amplitudes: sigma = 0.0, 0.05, 0.20.
- Epsilon grid: 0.10 to 0.55 in increments of 0.025, 19 points.
- Five paired seeds per condition.
- One forward scan is shared by all sigma branches for each p_HK and seed.
- The same standard-normal perturbation direction is scaled by sigma.
- Backward dynamics use common random numbers across sigma conditions.
- Perturbation is applied once, at the start of the backward scan.

## Horizon decision rule

Compare 50, 100, and 200 sweeps per epsilon point using paired seed-level:

1. backward order parameter and cluster count at the lowest epsilon;
2. hysteresis area;
3. full backward curves;
4. the mean absolute update in the final five-sweep window at every epsilon;
5. clipping fractions.

Adopt 100 sweeps for the formal experiment only if the 100- and 200-sweep
results are practically stable across these diagnostics. A non-significant
difference alone is not evidence of equivalence. If stability is unclear,
retain 200 sweeps or refine the grid near the unstable boundary.

## Invocation

```powershell
python analysis/run_perturbation_horizon_audit.py
```

The script writes separate resumable checkpoints under
`output/formal_revision/data/perturbation_horizon_audit/`.
