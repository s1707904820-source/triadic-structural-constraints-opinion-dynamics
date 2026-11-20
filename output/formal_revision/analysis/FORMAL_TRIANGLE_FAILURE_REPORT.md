# Formal failed-triangle fallback control

Design: N=2000, m=4, 21 values of p_HK, 20 paired seeds, and 200 Monte Carlo sweeps. Both arms share the network, initial opinions, threshold rule, successful-triangle rule, and third-party damping. The only changed rule is whether a failed triangle gate is followed by a dyadic update.

Data validation: **PASS** (0 consistency errors; 0 recorded source hashes differ from the current files). Pointwise contrasts use two-sided paired t-tests with Holm adjustment over 21 p_HK values within each outcome. Global terminal-outcome contrasts compare per-seed trapezoidal AUCs with Holm adjustment over six outcomes.

The fallback acceptance fraction reports the share of failed triangle attempts for which the ordinary dyadic confidence test would accept an update in the full model. The no-fallback arm makes no opinion change after every failed triangle attempt.

Source data and contrast tables are saved as CSV files beside this report. Confidence intervals describe across-seed uncertainty; a non-significant contrast is not an equivalence result.
