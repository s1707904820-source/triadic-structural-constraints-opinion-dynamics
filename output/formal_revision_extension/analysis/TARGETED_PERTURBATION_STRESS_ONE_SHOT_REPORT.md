# Targeted perturbation stress scan

This is a new extension analysis and does not overwrite formal output.

- Input: `output\formal_revision_extension\data\exp_targeted_perturbation_stress_v1.json`
- Runs: 5; p_HK: [0.0, 0.4, 0.8]
- Sigma grid: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
- Epsilon grid: [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.3, 0.4, 0.5, 0.55]
- Sweeps per epsilon: 150.0
- Perturbation: standardized Gaussian shock, clipping [0, 1]

## Interpretation

All low-epsilon order-parameter endpoints are at the metric ceiling; this scan does not identify a boundary.

This is a bounded negative result for the tested graph family, perturbation protocol, grid, and simulation horizon. It is not a claim of universal stability.

Summary table: `targeted_perturbation_stress_one_shot_summary.csv`
