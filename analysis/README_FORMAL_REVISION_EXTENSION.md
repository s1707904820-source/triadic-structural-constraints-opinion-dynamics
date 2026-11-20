# Formal revision extension

These are new, non-destructive files. They do not overwrite the original
`output/formal_revision` data or figures.

## 1. Targeted perturbation stress scan

Run a short pilot first:

```powershell
python analysis\run_targeted_perturbation_stress.py --runs 5 --sweeps 150 --workers 3
python analysis\analyze_targeted_perturbation_stress.py
```

The default scan adds larger shocks and lower epsilon values. Its output is
written to `output/formal_revision_extension/`.

If the pilot shows a candidate boundary, rerun the same grid with `--runs 20`
and a fresh output filename:

```powershell
python analysis\run_targeted_perturbation_stress.py --runs 20 --sweeps 200 --workers 3 --output-name exp_targeted_perturbation_stress_v1_formal.json
python analysis\analyze_targeted_perturbation_stress.py --input output\formal_revision_extension\data\exp_targeted_perturbation_stress_v1_formal.json
```

To test a shock before every reverse threshold, add
`--perturb-each-step`. Treat that as a separate protocol, not as a replacement
for the one-shot perturbation scan. Use a different analysis stem so the two
summary files do not overwrite one another:

```powershell
python analysis\analyze_targeted_perturbation_stress.py `
  --input output\formal_revision_extension\data\exp_targeted_perturbation_stress_each_step_formal_v1.json `
  --output-stem targeted_perturbation_stress_each_step
```

## 2. SBM analysis

The existing formal SBM JSON is analyzed without rerunning it:

```powershell
python analysis\analyze_formal_sbm_hysteresis.py
```

The report, CSV tables, and figure are written to
`output/formal_revision_extension/analysis/` and
`output/formal_revision_extension/figures/`.

## 3. Compare the two perturbation protocols

After you have the one-shot and per-threshold CSV summaries, run:

```powershell
python analysis\analyze_formal_perturbation_protocols.py `
  --one-shot output\formal_revision_extension\analysis\targeted_perturbation_stress_summary.csv `
  --each-step output\formal_revision_extension\analysis\targeted_perturbation_stress_summary_each_step.csv
```

This produces a Holm-adjusted comparison table, a formal Markdown report, and
a figure for Reviewer 4 Revision 1, comment 4.
