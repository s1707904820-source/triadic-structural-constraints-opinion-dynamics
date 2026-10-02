# Source data

This directory contains the preserved source data supporting the manuscript's formal comparisons, continuation scans, and supplementary controls.

## File groups

- `tableS1_terminal_auc_contrasts.csv`: terminal six-endpoint AUC contrasts.
- `tableS2_trajectory_global_contrasts.csv`: global trajectory contrasts.
- `tableS3_trajectory_segments.csv`: early, middle, and late trajectory contrasts.
- `fig_formal_forward_backward_source.csv`: forward and backward continuation branches.
- `fig_hysteresis_area_phk_source.csv`: 21-point finite-time hysteresis-area summary.
- `figureS1_network_threshold_summary.csv`: network and threshold diagnostics.
- `figureS2_update_mode.json`: update-mode control.
- `figureS3_vector_opinions.json`: two-dimensional opinion control.
- `figureS4_empirical_topologies.json`: fixed benchmark-topology control.
- `exp8_*.json`: parameter, network-size, connectivity, and initialization sensitivity outputs.
- `exp_hysteresis_sweep.json`: complete finite-time forward-backward scan.
- `*.meta.json`: configuration, seed, platform, and source-hash metadata.

`DATA_PROVENANCE.md` provides the result-to-file map. CSV files use comma-separated UTF-8 text. JSON files use UTF-8 encoding and preserve the nested condition, replicate-summary, and control metadata structures written by the analysis pipeline.

