# Formal trajectory reconstruction and analysis protocol

## Purpose

Recover the per-seed trajectories that were summarized, but not retained, in
the completed formal ablation. The reconstruction covers the full 21-point
`p_HK` grid so that the earlier visual separation near `p_HK=0.10` does not
lead to selective parameter-point testing.

## Fixed design

- Network size: `N=2000`, `m=4`.
- `p_HK`: 0.00 to 1.00 in increments of 0.05.
- Models: `threshold_only`, `no_triadic_feedback`, and `full_model`.
- Independent unit: one seeded simulation replicate.
- Replication: 20 seeds (`run_i * 997 + 7`), paired across models within each
  `p_HK` value.
- Horizon: 200 Monte Carlo sweeps.
- Recording interval: 5 sweeps.
- The run must reproduce the saved terminal values from the accepted formal
  six-model ablation before trajectory inference is accepted.

## Prespecified analyses

Primary contrast: Full Model minus No Triadic Feedback.

Secondary contrast: Full Model minus Threshold Only.

Primary path endpoint: trapezoidal time AUC of the order parameter over the
recorded 5-200 sweep interval.

Secondary path endpoints: time AUC of opinion variance, fixed-scale triangle
alignment, mean absolute update, and accepted-update rate.

Multiplicity:

1. For each contrast and endpoint, point-level time-AUC tests across the 21
   `p_HK` values use Holm correction within that endpoint.
2. The global `p_HK`-by-time surface AUCs for the five endpoints use Holm
   correction across endpoints within each contrast.
3. Early (5-50), middle (50-125), and late (125-200) order-parameter surface
   AUCs use Holm correction across the three temporal segments within each
   contrast.
4. Order-parameter paired differences at 50, 100, 150, and 200 sweeps use
   Holm correction across the four landmarks within each `p_HK` and contrast;
   they are supportive rather than primary evidence.

All tests are two-sided paired t tests on per-seed differences. Reports must
include the mean paired difference, two-sided 95% Student-t confidence
interval, Cohen's dz, exact raw p value, adjusted p value, and n. The repeated
time points are not treated as independent observations. Non-significance is
not interpreted as equivalence.

