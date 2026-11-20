# Formal perturbation-gradient report

Design: paired seeds, shared forward scans, common standardized perturbation directions, and common backward RNG streams. The independent experimental unit is the seed-level simulation replicate.

The six sigma conditions are compared with sigma=0. Two-sided paired t tests use seed-level differences. Holm correction is applied within each endpoint family across the 15 (five nonzero sigma values × three p_HK values) contrasts. A non-significant result is not interpreted as strict equivalence.

| endpoint | sigma | p_HK | mean difference | 95% CI | dz | Holm p |
|---|---:|---:|---:|---|---:|---:|
| backward low-epsilon OP | 0.01 | 0.00 | 0 | [0, 0] | 0.000 | 1 |
| backward low-epsilon OP | 0.01 | 0.40 | 0 | [0, 0] | 0.000 | 1 |
| backward low-epsilon OP | 0.01 | 0.80 | 0 | [0, 0] | 0.000 | 1 |
| backward low-epsilon OP | 0.02 | 0.00 | 0 | [0, 0] | 0.000 | 1 |
| backward low-epsilon OP | 0.02 | 0.40 | 0 | [0, 0] | 0.000 | 1 |
| backward low-epsilon OP | 0.02 | 0.80 | 0 | [0, 0] | 0.000 | 1 |
| backward low-epsilon OP | 0.05 | 0.00 | 0 | [0, 0] | 0.000 | 1 |
| backward low-epsilon OP | 0.05 | 0.40 | 0 | [0, 0] | 0.000 | 1 |
| backward low-epsilon OP | 0.05 | 0.80 | 0 | [0, 0] | 0.000 | 1 |
| backward low-epsilon OP | 0.10 | 0.00 | 0 | [0, 0] | 0.000 | 1 |
| backward low-epsilon OP | 0.10 | 0.40 | 0 | [0, 0] | 0.000 | 1 |
| backward low-epsilon OP | 0.10 | 0.80 | 0 | [0, 0] | 0.000 | 1 |
| backward low-epsilon OP | 0.20 | 0.00 | 0 | [0, 0] | 0.000 | 1 |
| backward low-epsilon OP | 0.20 | 0.40 | 0 | [0, 0] | 0.000 | 1 |
| backward low-epsilon OP | 0.20 | 0.80 | 0 | [0, 0] | 0.000 | 1 |
| backward low-epsilon clusters | 0.01 | 0.00 | 0 | [0, 0] | 0.000 | 1 |
| backward low-epsilon clusters | 0.01 | 0.40 | 0 | [0, 0] | 0.000 | 1 |
| backward low-epsilon clusters | 0.01 | 0.80 | 0 | [0, 0] | 0.000 | 1 |
| backward low-epsilon clusters | 0.02 | 0.00 | 0 | [0, 0] | 0.000 | 1 |
| backward low-epsilon clusters | 0.02 | 0.40 | 0 | [0, 0] | 0.000 | 1 |
| backward low-epsilon clusters | 0.02 | 0.80 | 0 | [0, 0] | 0.000 | 1 |
| backward low-epsilon clusters | 0.05 | 0.00 | 0 | [0, 0] | 0.000 | 1 |
| backward low-epsilon clusters | 0.05 | 0.40 | 0 | [0, 0] | 0.000 | 1 |
| backward low-epsilon clusters | 0.05 | 0.80 | 0 | [0, 0] | 0.000 | 1 |
| backward low-epsilon clusters | 0.10 | 0.00 | 0 | [0, 0] | 0.000 | 1 |
| backward low-epsilon clusters | 0.10 | 0.40 | 0 | [0, 0] | 0.000 | 1 |
| backward low-epsilon clusters | 0.10 | 0.80 | 0 | [0, 0] | 0.000 | 1 |
| backward low-epsilon clusters | 0.20 | 0.00 | 0 | [0, 0] | 0.000 | 1 |
| backward low-epsilon clusters | 0.20 | 0.40 | 0 | [0, 0] | 0.000 | 1 |
| backward low-epsilon clusters | 0.20 | 0.80 | 0 | [0, 0] | 0.000 | 1 |
| hysteresis area | 0.01 | 0.00 | 0 | [0, 0] | 0.000 | 1 |
| hysteresis area | 0.01 | 0.40 | 0 | [0, 0] | 0.000 | 1 |
| hysteresis area | 0.01 | 0.80 | 0 | [0, 0] | 0.000 | 1 |
| hysteresis area | 0.02 | 0.00 | 0 | [0, 0] | 0.000 | 1 |
| hysteresis area | 0.02 | 0.40 | 0 | [0, 0] | 0.000 | 1 |
| hysteresis area | 0.02 | 0.80 | 0 | [0, 0] | 0.000 | 1 |
| hysteresis area | 0.05 | 0.00 | 0 | [0, 0] | 0.000 | 1 |
| hysteresis area | 0.05 | 0.40 | 0 | [0, 0] | 0.000 | 1 |
| hysteresis area | 0.05 | 0.80 | 0 | [0, 0] | 0.000 | 1 |
| hysteresis area | 0.10 | 0.00 | 0 | [0, 0] | 0.000 | 1 |
| hysteresis area | 0.10 | 0.40 | 0 | [0, 0] | 0.000 | 1 |
| hysteresis area | 0.10 | 0.80 | 0 | [0, 0] | 0.000 | 1 |
| hysteresis area | 0.20 | 0.00 | 0 | [0, 0] | 0.000 | 1 |
| hysteresis area | 0.20 | 0.40 | 0 | [0, 0] | 0.000 | 1 |
| hysteresis area | 0.20 | 0.80 | 0 | [0, 0] | 0.000 | 1 |
| forward peak sensitivity | 0.01 | 0.00 | 0 | [0, 0] | 0.000 | 1 |
| forward peak sensitivity | 0.01 | 0.40 | 0 | [0, 0] | 0.000 | 1 |
| forward peak sensitivity | 0.01 | 0.80 | 0 | [0, 0] | 0.000 | 1 |
| forward peak sensitivity | 0.02 | 0.00 | 0 | [0, 0] | 0.000 | 1 |
| forward peak sensitivity | 0.02 | 0.40 | 0 | [0, 0] | 0.000 | 1 |
| forward peak sensitivity | 0.02 | 0.80 | 0 | [0, 0] | 0.000 | 1 |
| forward peak sensitivity | 0.05 | 0.00 | 0 | [0, 0] | 0.000 | 1 |
| forward peak sensitivity | 0.05 | 0.40 | 0 | [0, 0] | 0.000 | 1 |
| forward peak sensitivity | 0.05 | 0.80 | 0 | [0, 0] | 0.000 | 1 |
| forward peak sensitivity | 0.10 | 0.00 | 0 | [0, 0] | 0.000 | 1 |
| forward peak sensitivity | 0.10 | 0.40 | 0 | [0, 0] | 0.000 | 1 |
| forward peak sensitivity | 0.10 | 0.80 | 0 | [0, 0] | 0.000 | 1 |
| forward peak sensitivity | 0.20 | 0.00 | 0 | [0, 0] | 0.000 | 1 |
| forward peak sensitivity | 0.20 | 0.40 | 0 | [0, 0] | 0.000 | 1 |
| forward peak sensitivity | 0.20 | 0.80 | 0 | [0, 0] | 0.000 | 1 |
| backward OP AUC | 0.01 | 0.00 | 0 | [0, 0] | 0.000 | 1 |
| backward OP AUC | 0.01 | 0.40 | 0 | [0, 0] | 0.000 | 1 |
| backward OP AUC | 0.01 | 0.80 | 0 | [0, 0] | 0.000 | 1 |
| backward OP AUC | 0.02 | 0.00 | 0 | [0, 0] | 0.000 | 1 |
| backward OP AUC | 0.02 | 0.40 | 0 | [0, 0] | 0.000 | 1 |
| backward OP AUC | 0.02 | 0.80 | 0 | [0, 0] | 0.000 | 1 |
| backward OP AUC | 0.05 | 0.00 | 0 | [0, 0] | 0.000 | 1 |
| backward OP AUC | 0.05 | 0.40 | 0 | [0, 0] | 0.000 | 1 |
| backward OP AUC | 0.05 | 0.80 | 0 | [0, 0] | 0.000 | 1 |
| backward OP AUC | 0.10 | 0.00 | 0 | [0, 0] | 0.000 | 1 |
| backward OP AUC | 0.10 | 0.40 | 0 | [0, 0] | 0.000 | 1 |
| backward OP AUC | 0.10 | 0.80 | 0 | [0, 0] | 0.000 | 1 |
| backward OP AUC | 0.20 | 0.00 | 0 | [0, 0] | 0.000 | 1 |
| backward OP AUC | 0.20 | 0.40 | 0 | [0, 0] | 0.000 | 1 |
| backward OP AUC | 0.20 | 0.80 | 0 | [0, 0] | 0.000 | 1 |

## Perturbation clipping diagnostics

| sigma | p_HK | mean clipped fraction |
|---:|---:|---:|
| 0.00 | 0.00 | 0 |
| 0.00 | 0.40 | 0 |
| 0.00 | 0.80 | 0 |
| 0.01 | 0.00 | 0 |
| 0.01 | 0.40 | 0 |
| 0.01 | 0.80 | 0 |
| 0.02 | 0.00 | 0 |
| 0.02 | 0.40 | 0 |
| 0.02 | 0.80 | 0 |
| 0.05 | 0.00 | 0 |
| 0.05 | 0.40 | 0 |
| 0.05 | 0.80 | 0 |
| 0.10 | 0.00 | 0 |
| 0.10 | 0.40 | 0 |
| 0.10 | 0.80 | 0 |
| 0.20 | 0.00 | 0.012675 |
| 0.20 | 0.40 | 0.012675 |
| 0.20 | 0.80 | 0.01275 |

## Result boundary

Across this epsilon range, the backward order parameter reached the metric ceiling of 1.0 for every p_HK and sigma condition. Accordingly, this experiment does not identify a near-consensus loss boundary or a nonzero sigma-dependent hysteresis effect within the tested range. The appropriate conclusion is a boundary non-detection under this protocol, not proof of universal stability or equivalence.

## Interpretation guardrails

The report separates update-scale damping from macroscopic path or terminal effects. Any significant result for hysteresis, low-epsilon OP, cluster count, or backward AUC is interpreted as a sigma-dependent perturbation response, not as evidence that a small perturbation establishes a universal stability boundary.
