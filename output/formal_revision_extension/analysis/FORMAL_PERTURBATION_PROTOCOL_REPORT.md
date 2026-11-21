# Formal perturbation-protocol comparison

This report compares a single shock at the beginning of the backward scan with repeated shocks before every reverse-threshold step.
The analysis uses the supplied seed-level summary CSV files and does not rerun or overwrite simulations.

- One-shot input: `output\formal_revision_extension\analysis\targeted_perturbation_stress_one_shot_summary.csv`
- Per-threshold input: `output\formal_revision_extension\analysis\targeted_perturbation_stress_each_step_summary.csv`
- One-shot contrast rows: 60
- Per-threshold contrast rows: 60

## Main result

The one-shot protocol is effectively unchanged across the tested sigma grid, whereas the per-threshold protocol produces substantial reductions in low-epsilon order and increases in fragmentation.

The per-threshold protocol should therefore be reported as a separate, stronger stress protocol. It is not equivalent to the original one-shot perturbation design.

## Multiple-comparison summary

Holm correction is applied separately within each protocol and endpoint family across the 15 nonzero sigma-by-p_HK contrasts.

- `backward_auc`: maximum Holm-adjusted p = 2.25e-09
- `hysteresis_area`: maximum Holm-adjusted p = 0.00946
- `low_clusters`: maximum Holm-adjusted p = 2.23e-10
- `low_op`: maximum Holm-adjusted p = 1.87e-07

## Interpretation boundary

These results establish protocol-dependent perturbation sensitivity for the tested graph family, p_HK grid, sigma grid, epsilon grid, clipping rule, and simulation horizon. They do not establish a universal instability threshold for all perturbation types or networks.

Comparison table: `formal_perturbation_protocol_comparison.csv`
Figure: `fig_formal_perturbation_protocol_comparison.png`
