# Formal forward-transition prediction audit

## Design and operational transition

The paired scan uses N=2000, m=4, p_HK=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0], 20 seeds, epsilon0=[0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55] and 100.0 sweeps at each threshold.

For each seed and model, the operational transition is the first point on the ascending epsilon0 grid at which the final compact order parameter is at least 0.950. The order parameter is the maximum fraction of opinions contained in any interval of width 0.050; thus the cutoff requires at least 95.0% of opinions in one such interval. The smallest adjacent epsilon-grid spacing is 0.025; this is a finite-horizon first-crossing marker, not an asymptotic critical point. No interpolation between grid points is used. Runs that do not cross within the scanned grid are reported as not observed and are excluded from threshold means and prediction-error means; their counts and the paired error sample sizes are disclosed below.

The uniform reference is the asynchronous pairwise BCM with alpha=0. The structure-only model uses the Jaccard edge thresholds without triangle triggering. The full-model scan adds triangle triggering, three-edge gates, fallback, and third-party damping. All variants share each seed's network and initial opinions; each threshold point has a deterministic update stream shared across variants.

## Prediction and edge-selection diagnostic

The mean-field prediction uses the exact unweighted network statistic Gamma_G = mean_edges[1-exp(-alpha*w_ij)] and epsilon0,pred = max(0,(epsilon0,c,uniform-Gamma_G)/(1-Gamma_G)). This treats the edge-average effective threshold as a first-order surrogate for the transition condition.

Uniform node sampling followed by uniform neighbor sampling does not sample undirected edges uniformly. Before confidence gating, the dyadic proposal probability for edge (u,v) is (1/N)(1/k_u+1/k_v). A degree-weighted Gamma using these proposal probabilities is reported as a diagnostic alternative. It corrects dyadic proposal frequency only; it does not account for acceptance conditioning or triangle selection. The full-model comparison exposes the additional effect of triangle attempts and their state-dependent gates.

For each seed s, the mean-edge prediction error is d_s = epsilon_c,s(structure) - max{0, [epsilon_c,s(uniform) - Gamma_G,s] / [1 - Gamma_G,s]}; the degree-proposal diagnostic substitutes Gamma_dyadic,s for Gamma_G,s. Intervals are two-sided 95% Student-t intervals across the available paired seed-level errors. Each table cell reports its own n. These intervals quantify between-seed variability in the discrete first-crossing errors; they do not include the additional uncertainty induced by the epsilon-grid resolution stated above. Positive errors mean the simulated transition occurs at a larger baseline threshold than predicted.

Validation: **PASS**; 21 p_HK values, 20 seeds per value, 49 source hashes checked.

## Results

| p_HK | Gamma edge mean | Gamma dyadic-proposal weighted | uniform-reference transition | mean-edge prediction | dyadic prediction | structure-only simulated | full-model simulated | structure-only error (mean-edge) | structure-only error (dyadic) | full-model error (mean-edge) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.015 [0.015, 0.015], n=20 | 0.012 [0.011, 0.012], n=20 | 0.416 [0.403, 0.430], n=20 | 0.407 [0.394, 0.421], n=20 | 0.409 [0.396, 0.423], n=20 | 0.338 [0.329, 0.346], n=20 | 0.341 [0.333, 0.350], n=20 | -0.070 [-0.086, -0.054], n=20 | -0.072 [-0.088, -0.056], n=20 | -0.066 [-0.080, -0.052], n=20 |

At p_HK=0.00, first crossings were observed for 20/20 uniform references, 20/20 structure-only runs, and 20/20 full-model runs. Paired error n: structure-only/mean-edge=20/20, structure-only/degree-proposal=20/20, full-model/mean-edge=20/20.
Non-monotone order trajectories (at least one downward grid step): uniform=5/20, structure-only=8/20, full=7/20. Post-first-crossing recrossings below the configured cutoff (0.950): uniform=0/20, structure-only=0/20, full=0/20.

| 0.05 | 0.030 [0.029, 0.030], n=20 | 0.028 [0.027, 0.029], n=20 | 0.419 [0.406, 0.432], n=20 | 0.401 [0.387, 0.414], n=20 | 0.402 [0.388, 0.415], n=20 | 0.299 [0.286, 0.312], n=20 | 0.311 [0.294, 0.329], n=20 | -0.102 [-0.121, -0.083], n=20 | -0.103 [-0.122, -0.084], n=20 | -0.090 [-0.111, -0.069], n=20 |

At p_HK=0.05, first crossings were observed for 20/20 uniform references, 20/20 structure-only runs, and 20/20 full-model runs. Paired error n: structure-only/mean-edge=20/20, structure-only/degree-proposal=20/20, full-model/mean-edge=20/20.
Non-monotone order trajectories (at least one downward grid step): uniform=2/20, structure-only=9/20, full=8/20. Post-first-crossing recrossings below the configured cutoff (0.950): uniform=0/20, structure-only=0/20, full=0/20.

| 0.10 | 0.044 [0.043, 0.045], n=20 | 0.044 [0.043, 0.045], n=20 | 0.417 [0.403, 0.432], n=20 | 0.391 [0.376, 0.406], n=20 | 0.391 [0.376, 0.406], n=20 | 0.296 [0.272, 0.320], n=20 | 0.291 [0.264, 0.318], n=20 | -0.094 [-0.120, -0.069], n=20 | -0.094 [-0.120, -0.069], n=20 | -0.099 [-0.126, -0.073], n=20 |

At p_HK=0.10, first crossings were observed for 20/20 uniform references, 20/20 structure-only runs, and 20/20 full-model runs. Paired error n: structure-only/mean-edge=20/20, structure-only/degree-proposal=20/20, full-model/mean-edge=20/20.
Non-monotone order trajectories (at least one downward grid step): uniform=4/20, structure-only=8/20, full=10/20. Post-first-crossing recrossings below the configured cutoff (0.950): uniform=0/20, structure-only=0/20, full=0/20.

| 0.15 | 0.058 [0.058, 0.059], n=20 | 0.060 [0.059, 0.061], n=20 | 0.405 [0.392, 0.418], n=20 | 0.368 [0.354, 0.382], n=20 | 0.367 [0.353, 0.381], n=20 | 0.340 [0.310, 0.370], n=20 | 0.310 [0.276, 0.344], n=20 | -0.028 [-0.061, 0.005], n=20 | -0.027 [-0.060, 0.006], n=20 | -0.058 [-0.093, -0.023], n=20 |

At p_HK=0.15, first crossings were observed for 20/20 uniform references, 20/20 structure-only runs, and 20/20 full-model runs. Paired error n: structure-only/mean-edge=20/20, structure-only/degree-proposal=20/20, full-model/mean-edge=20/20.
Non-monotone order trajectories (at least one downward grid step): uniform=2/20, structure-only=7/20, full=4/20. Post-first-crossing recrossings below the configured cutoff (0.950): uniform=0/20, structure-only=0/20, full=0/20.

| 0.20 | 0.073 [0.072, 0.074], n=20 | 0.077 [0.075, 0.078], n=20 | 0.406 [0.394, 0.419], n=20 | 0.359 [0.346, 0.373], n=20 | 0.357 [0.343, 0.371], n=20 | 0.375 [0.358, 0.392], n=20 | 0.351 [0.324, 0.378], n=20 | 0.016 [-0.007, 0.038], n=20 | 0.018 [-0.004, 0.041], n=20 | -0.008 [-0.042, 0.026], n=20 |

At p_HK=0.20, first crossings were observed for 20/20 uniform references, 20/20 structure-only runs, and 20/20 full-model runs. Paired error n: structure-only/mean-edge=20/20, structure-only/degree-proposal=20/20, full-model/mean-edge=20/20.
Non-monotone order trajectories (at least one downward grid step): uniform=3/20, structure-only=5/20, full=7/20. Post-first-crossing recrossings below the configured cutoff (0.950): uniform=0/20, structure-only=0/20, full=0/20.

| 0.25 | 0.088 [0.087, 0.089], n=20 | 0.094 [0.092, 0.095], n=20 | 0.405 [0.384, 0.426], n=20 | 0.348 [0.324, 0.371], n=20 | 0.344 [0.320, 0.367], n=20 | 0.359 [0.349, 0.368], n=20 | 0.350 [0.333, 0.367], n=20 | 0.011 [-0.017, 0.039], n=20 | 0.015 [-0.013, 0.043], n=20 | 0.002 [-0.025, 0.030], n=20 |

At p_HK=0.25, first crossings were observed for 20/20 uniform references, 20/20 structure-only runs, and 20/20 full-model runs. Paired error n: structure-only/mean-edge=20/20, structure-only/degree-proposal=20/20, full-model/mean-edge=20/20.
Non-monotone order trajectories (at least one downward grid step): uniform=6/20, structure-only=1/20, full=6/20. Post-first-crossing recrossings below the configured cutoff (0.950): uniform=0/20, structure-only=0/20, full=0/20.

| 0.30 | 0.102 [0.101, 0.103], n=20 | 0.110 [0.108, 0.111], n=20 | 0.425 [0.408, 0.442], n=20 | 0.360 [0.341, 0.378], n=20 | 0.354 [0.335, 0.373], n=20 | 0.347 [0.339, 0.356], n=20 | 0.352 [0.344, 0.361], n=20 | -0.012 [-0.029, 0.005], n=20 | -0.007 [-0.024, 0.010], n=20 | -0.007 [-0.027, 0.013], n=20 |

At p_HK=0.30, first crossings were observed for 20/20 uniform references, 20/20 structure-only runs, and 20/20 full-model runs. Paired error n: structure-only/mean-edge=20/20, structure-only/degree-proposal=20/20, full-model/mean-edge=20/20.
Non-monotone order trajectories (at least one downward grid step): uniform=6/20, structure-only=2/20, full=1/20. Post-first-crossing recrossings below the configured cutoff (0.950): uniform=0/20, structure-only=0/20, full=0/20.

| 0.35 | 0.116 [0.115, 0.118], n=20 | 0.126 [0.125, 0.127], n=20 | 0.404 [0.389, 0.418], n=20 | 0.325 [0.309, 0.341], n=20 | 0.318 [0.301, 0.334], n=20 | 0.324 [0.318, 0.330], n=20 | 0.330 [0.323, 0.337], n=20 | -0.001 [-0.019, 0.017], n=20 | 0.006 [-0.012, 0.024], n=20 | 0.005 [-0.015, 0.025], n=20 |

At p_HK=0.35, first crossings were observed for 20/20 uniform references, 20/20 structure-only runs, and 20/20 full-model runs. Paired error n: structure-only/mean-edge=20/20, structure-only/degree-proposal=20/20, full-model/mean-edge=20/20.
Non-monotone order trajectories (at least one downward grid step): uniform=0/20, structure-only=0/20, full=0/20. Post-first-crossing recrossings below the configured cutoff (0.950): uniform=0/20, structure-only=0/20, full=0/20.

| 0.40 | 0.130 [0.129, 0.131], n=20 | 0.141 [0.140, 0.142], n=20 | 0.415 [0.404, 0.426], n=20 | 0.327 [0.314, 0.341], n=20 | 0.319 [0.306, 0.332], n=20 | 0.306 [0.298, 0.315], n=20 | 0.305 [0.297, 0.313], n=20 | -0.021 [-0.031, -0.011], n=20 | -0.013 [-0.023, -0.003], n=20 | -0.022 [-0.037, -0.008], n=20 |

At p_HK=0.40, first crossings were observed for 20/20 uniform references, 20/20 structure-only runs, and 20/20 full-model runs. Paired error n: structure-only/mean-edge=20/20, structure-only/degree-proposal=20/20, full-model/mean-edge=20/20.
Non-monotone order trajectories (at least one downward grid step): uniform=7/20, structure-only=0/20, full=0/20. Post-first-crossing recrossings below the configured cutoff (0.950): uniform=0/20, structure-only=0/20, full=0/20.

| 0.45 | 0.144 [0.143, 0.145], n=20 | 0.157 [0.156, 0.158], n=20 | 0.414 [0.401, 0.427], n=20 | 0.315 [0.300, 0.330], n=20 | 0.304 [0.289, 0.320], n=20 | 0.283 [0.274, 0.291], n=20 | 0.276 [0.268, 0.284], n=20 | -0.033 [-0.051, -0.014], n=20 | -0.022 [-0.040, -0.004], n=20 | -0.039 [-0.056, -0.022], n=20 |

At p_HK=0.45, first crossings were observed for 20/20 uniform references, 20/20 structure-only runs, and 20/20 full-model runs. Paired error n: structure-only/mean-edge=20/20, structure-only/degree-proposal=20/20, full-model/mean-edge=20/20.
Non-monotone order trajectories (at least one downward grid step): uniform=6/20, structure-only=0/20, full=0/20. Post-first-crossing recrossings below the configured cutoff (0.950): uniform=0/20, structure-only=0/20, full=0/20.

| 0.50 | 0.159 [0.158, 0.160], n=20 | 0.174 [0.173, 0.175], n=20 | 0.417 [0.403, 0.432], n=20 | 0.307 [0.290, 0.324], n=20 | 0.295 [0.277, 0.312], n=20 | 0.245 [0.236, 0.254], n=20 | 0.247 [0.239, 0.256], n=20 | -0.062 [-0.081, -0.043], n=20 | -0.050 [-0.069, -0.031], n=20 | -0.060 [-0.076, -0.043], n=20 |

At p_HK=0.50, first crossings were observed for 20/20 uniform references, 20/20 structure-only runs, and 20/20 full-model runs. Paired error n: structure-only/mean-edge=20/20, structure-only/degree-proposal=20/20, full-model/mean-edge=20/20.
Non-monotone order trajectories (at least one downward grid step): uniform=5/20, structure-only=2/20, full=0/20. Post-first-crossing recrossings below the configured cutoff (0.950): uniform=0/20, structure-only=0/20, full=0/20.

| 0.55 | 0.175 [0.173, 0.176], n=20 | 0.191 [0.189, 0.193], n=20 | 0.415 [0.398, 0.432], n=20 | 0.291 [0.270, 0.312], n=20 | 0.277 [0.255, 0.298], n=20 | 0.224 [0.216, 0.232], n=20 | 0.224 [0.216, 0.232], n=20 | -0.067 [-0.086, -0.048], n=20 | -0.053 [-0.073, -0.034], n=20 | -0.067 [-0.087, -0.047], n=20 |

At p_HK=0.55, first crossings were observed for 20/20 uniform references, 20/20 structure-only runs, and 20/20 full-model runs. Paired error n: structure-only/mean-edge=20/20, structure-only/degree-proposal=20/20, full-model/mean-edge=20/20.
Non-monotone order trajectories (at least one downward grid step): uniform=2/20, structure-only=0/20, full=0/20. Post-first-crossing recrossings below the configured cutoff (0.950): uniform=0/20, structure-only=0/20, full=0/20.

| 0.60 | 0.190 [0.188, 0.191], n=20 | 0.208 [0.206, 0.210], n=20 | 0.399 [0.385, 0.413], n=20 | 0.258 [0.241, 0.275], n=20 | 0.241 [0.224, 0.259], n=20 | 0.190 [0.180, 0.200], n=20 | 0.200 [0.190, 0.210], n=20 | -0.068 [-0.089, -0.047], n=20 | -0.051 [-0.072, -0.030], n=20 | -0.058 [-0.078, -0.038], n=20 |

At p_HK=0.60, first crossings were observed for 20/20 uniform references, 20/20 structure-only runs, and 20/20 full-model runs. Paired error n: structure-only/mean-edge=20/20, structure-only/degree-proposal=20/20, full-model/mean-edge=20/20.
Non-monotone order trajectories (at least one downward grid step): uniform=3/20, structure-only=0/20, full=0/20. Post-first-crossing recrossings below the configured cutoff (0.950): uniform=0/20, structure-only=0/20, full=0/20.

| 0.65 | 0.205 [0.204, 0.206], n=20 | 0.225 [0.223, 0.226], n=20 | 0.407 [0.395, 0.420], n=20 | 0.255 [0.239, 0.270], n=20 | 0.236 [0.219, 0.252], n=20 | 0.169 [0.160, 0.177], n=20 | 0.165 [0.155, 0.175], n=20 | -0.086 [-0.103, -0.068], n=20 | -0.067 [-0.085, -0.049], n=20 | -0.090 [-0.110, -0.069], n=20 |

At p_HK=0.65, first crossings were observed for 20/20 uniform references, 20/20 structure-only runs, and 20/20 full-model runs. Paired error n: structure-only/mean-edge=20/20, structure-only/degree-proposal=20/20, full-model/mean-edge=20/20.
Non-monotone order trajectories (at least one downward grid step): uniform=4/20, structure-only=1/20, full=1/20. Post-first-crossing recrossings below the configured cutoff (0.950): uniform=0/20, structure-only=0/20, full=0/20.

| 0.70 | 0.221 [0.219, 0.222], n=20 | 0.242 [0.240, 0.243], n=20 | 0.401 [0.390, 0.412], n=20 | 0.232 [0.217, 0.246], n=20 | 0.210 [0.196, 0.225], n=20 | 0.140 [0.133, 0.147], n=20 | 0.146 [0.134, 0.159], n=20 | -0.092 [-0.108, -0.076], n=20 | -0.070 [-0.087, -0.054], n=20 | -0.086 [-0.105, -0.066], n=20 |

At p_HK=0.70, first crossings were observed for 20/20 uniform references, 20/20 structure-only runs, and 20/20 full-model runs. Paired error n: structure-only/mean-edge=20/20, structure-only/degree-proposal=20/20, full-model/mean-edge=20/20.
Non-monotone order trajectories (at least one downward grid step): uniform=4/20, structure-only=2/20, full=0/20. Post-first-crossing recrossings below the configured cutoff (0.950): uniform=0/20, structure-only=0/20, full=0/20.

| 0.75 | 0.237 [0.234, 0.239], n=20 | 0.259 [0.256, 0.262], n=20 | 0.405 [0.390, 0.420], n=20 | 0.221 [0.201, 0.240], n=20 | 0.197 [0.177, 0.217], n=20 | 0.099 [0.084, 0.113], n=20 | 0.102 [0.088, 0.117], n=20 | -0.122 [-0.151, -0.093], n=20 | -0.098 [-0.128, -0.069], n=20 | -0.118 [-0.143, -0.093], n=20 |

At p_HK=0.75, first crossings were observed for 20/20 uniform references, 20/20 structure-only runs, and 20/20 full-model runs. Paired error n: structure-only/mean-edge=20/20, structure-only/degree-proposal=20/20, full-model/mean-edge=20/20.
Non-monotone order trajectories (at least one downward grid step): uniform=6/20, structure-only=0/20, full=2/20. Post-first-crossing recrossings below the configured cutoff (0.950): uniform=0/20, structure-only=0/20, full=0/20.

| 0.80 | 0.254 [0.253, 0.256], n=20 | 0.278 [0.276, 0.280], n=20 | 0.404 [0.391, 0.416], n=20 | 0.200 [0.184, 0.217], n=20 | 0.174 [0.157, 0.191], n=20 | 0.053 [0.039, 0.066], n=20 | 0.064 [0.049, 0.078], n=20 | -0.148 [-0.171, -0.125], n=20 | -0.122 [-0.145, -0.098], n=20 | -0.137 [-0.162, -0.111], n=20 |

At p_HK=0.80, first crossings were observed for 20/20 uniform references, 20/20 structure-only runs, and 20/20 full-model runs. Paired error n: structure-only/mean-edge=20/20, structure-only/degree-proposal=20/20, full-model/mean-edge=20/20.
Non-monotone order trajectories (at least one downward grid step): uniform=7/20, structure-only=2/20, full=2/20. Post-first-crossing recrossings below the configured cutoff (0.950): uniform=0/20, structure-only=0/20, full=0/20.

| 0.85 | 0.272 [0.270, 0.274], n=20 | 0.297 [0.295, 0.299], n=20 | 0.396 [0.385, 0.407], n=20 | 0.171 [0.156, 0.185], n=20 | 0.141 [0.126, 0.156], n=20 | 0.026 [0.013, 0.040], n=20 | 0.029 [0.016, 0.041], n=20 | -0.144 [-0.162, -0.127], n=20 | -0.115 [-0.133, -0.097], n=20 | -0.142 [-0.160, -0.123], n=20 |

At p_HK=0.85, first crossings were observed for 20/20 uniform references, 20/20 structure-only runs, and 20/20 full-model runs. Paired error n: structure-only/mean-edge=20/20, structure-only/degree-proposal=20/20, full-model/mean-edge=20/20.
Non-monotone order trajectories (at least one downward grid step): uniform=3/20, structure-only=4/20, full=0/20. Post-first-crossing recrossings below the configured cutoff (0.950): uniform=0/20, structure-only=0/20, full=0/20.

| 0.90 | 0.292 [0.290, 0.294], n=20 | 0.319 [0.317, 0.322], n=20 | 0.399 [0.384, 0.413], n=20 | 0.151 [0.129, 0.172], n=20 | 0.118 [0.098, 0.138], n=20 | 0.005 [0.000, 0.010], n=20 | 0.011 [0.004, 0.018], n=20 | -0.146 [-0.167, -0.125], n=20 | -0.113 [-0.133, -0.093], n=20 | -0.139 [-0.158, -0.121], n=20 |

At p_HK=0.90, first crossings were observed for 20/20 uniform references, 20/20 structure-only runs, and 20/20 full-model runs. Paired error n: structure-only/mean-edge=20/20, structure-only/degree-proposal=20/20, full-model/mean-edge=20/20.
Non-monotone order trajectories (at least one downward grid step): uniform=5/20, structure-only=4/20, full=2/20. Post-first-crossing recrossings below the configured cutoff (0.950): uniform=0/20, structure-only=0/20, full=0/20.

| 0.95 | 0.311 [0.309, 0.314], n=20 | 0.340 [0.337, 0.342], n=20 | 0.394 [0.379, 0.409], n=20 | 0.120 [0.098, 0.142], n=20 | 0.083 [0.060, 0.105], n=20 | 0.006 [0.001, 0.011], n=20 | 0.006 [0.001, 0.011], n=20 | -0.113 [-0.135, -0.091], n=20 | -0.076 [-0.099, -0.054], n=20 | -0.113 [-0.136, -0.090], n=20 |

At p_HK=0.95, first crossings were observed for 20/20 uniform references, 20/20 structure-only runs, and 20/20 full-model runs. Paired error n: structure-only/mean-edge=20/20, structure-only/degree-proposal=20/20, full-model/mean-edge=20/20.
Non-monotone order trajectories (at least one downward grid step): uniform=8/20, structure-only=3/20, full=3/20. Post-first-crossing recrossings below the configured cutoff (0.950): uniform=0/20, structure-only=0/20, full=1/20.

| 1.00 | 0.332 [0.330, 0.334], n=20 | 0.362 [0.361, 0.364], n=20 | 0.409 [0.394, 0.424], n=20 | 0.115 [0.093, 0.137], n=20 | 0.074 [0.053, 0.096], n=20 | 0.007 [0.002, 0.013], n=20 | 0.011 [0.005, 0.017], n=20 | -0.107 [-0.129, -0.085], n=20 | -0.067 [-0.089, -0.045], n=20 | -0.103 [-0.124, -0.082], n=20 |

At p_HK=1.00, first crossings were observed for 20/20 uniform references, 20/20 structure-only runs, and 20/20 full-model runs. Paired error n: structure-only/mean-edge=20/20, structure-only/degree-proposal=20/20, full-model/mean-edge=20/20.
Non-monotone order trajectories (at least one downward grid step): uniform=14/20, structure-only=3/20, full=3/20. Post-first-crossing recrossings below the configured cutoff (0.950): uniform=0/20, structure-only=0/20, full=1/20.

## Interpretation limits

These results correspond to the prespecified formal experiment design.

The edge-mean formula is an approximation, not a theorem for the asynchronous network process. The degree-weighted proxy addresses only the unconditioned dyadic proposal law. Acceptance depends on current opinions; the full model also changes which three-edge gates and fallback interactions occur. Prediction errors quantify these finite-horizon deviations for the tested graph ensemble, threshold grid, order criterion, and update rule.

Figure exports: fig_formal_threshold_transition_prediction. Seed-level values and summary tables are provided in the accompanying CSV files.
