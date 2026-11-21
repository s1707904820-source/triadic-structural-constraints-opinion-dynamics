# Formal SBM hysteresis analysis

This report analyzes the existing formal JSON and does not rerun or modify it.

- Network: balanced two-block SBM, N=2000, target degree=8.0
- Replicates: 20 per mixing × sigma condition
- Mixing values: [0.02, 0.05, 0.1, 0.2, 0.35]
- Sigma values: [0.0, 0.05, 0.1, 0.2]
- Epsilon grid: 19 points from 0.1 to 0.55
- Sweeps per epsilon: 200.0

## Main result

At sigma=0, hysteresis area decreases as inter-community mixing increases.
This supports the interpretation that stronger community separation produces a larger forward/backward discrepancy under the tested protocol.

## Statistical caution

The SBM experiment is a numerical multi-community robustness check, not a theorem about all community networks. Sigma comparisons are paired within the same seed and are protocol-specific.

Summary table: `sbm_hysteresis_summary.csv`
Sigma comparisons: `sbm_hysteresis_sigma_comparisons.csv`
