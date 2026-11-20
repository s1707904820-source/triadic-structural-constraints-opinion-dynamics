# Formal threshold-assignment controls

Design: N=2000, m=4, 21 values of p_HK, 20 paired seeds, and 200 Monte Carlo sweeps per run. The three arms use the same network and initial-condition seeds. The mean-threshold arm uses the realized unweighted mean of the proposed edge thresholds; the randomized arm permutes the realized threshold multiset over undirected edges.

Validation: **PASS** (0 errors). Pointwise paired tests use two-sided t-tests with Holm adjustment over the 21 network parameter values within each contrast and metric. Global tests compare per-seed trapezoidal AUCs across p_HK, with Holm adjustment over the 12 contrast-metric tests.

## Global paired AUC contrasts

Differences are edge-specific assignment minus the named control; AUC integrates over p_HK from 0 to 1. Positive order-parameter differences favor edge-specific assignment.

| Control | Outcome | Mean AUC difference | 95% CI | Holm-adjusted p | Paired dz |
|---|---|---:|---:|---:|---:|
| Mean threshold | Order parameter | 0.0869 | [0.0727, 0.1011] | 5.09e-10 | 2.865 |
| Mean threshold | Number of clusters | 1.7075 | [1.5011, 1.9139] | 3.88e-12 | 3.871 |
| Mean threshold | Opinion variance | -0.0104 | [-0.0121, -0.0088] | 2.89e-10 | -3.011 |
| Shuffled thresholds | Order parameter | -0.0233 | [-0.0299, -0.0168] | 1.38e-06 | -1.669 |
| Shuffled thresholds | Number of clusters | 1.2075 | [1.0985, 1.3165] | 2.55e-14 | 5.186 |
| Shuffled thresholds | Opinion variance | 0.0020 | [0.0015, 0.0026] | 1.2e-06 | 1.719 |

These are matched-seed contrasts under the stated simulation protocol. They quantify whether the location of a threshold on the network matters beyond its marginal distribution; they do not establish a causal effect beyond this model and parameter range. A non-significant result is not an equivalence claim.

Source data and pointwise/global contrast tables are saved as CSV files beside this report. The figure shows terminal order parameter and paired order-parameter and cluster-count contrasts.
