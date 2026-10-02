# Triadic Structural Constraints in Opinion Dynamics

This repository contains the simulation, analysis, plotting, validation, and source-data files supporting the manuscript *Triadic Structural Constraints in Opinion Dynamics: A Bounded-Confidence Model with Structure-Dependent Thresholds and Third-Party Inconsistency Damping*.

## Contents

- Core model and experiment code: Python files in the repository root.
- Formal analysis and validation scripts: `analysis/`.
- Automated checks: `tests/`.
- Source data underlying the reported tables, continuation scans, and supplementary controls: `source_data/`.
- Environment pins and reproduction instructions: `requirements-lock.txt` and `REPRODUCIBILITY.md`.
- Validated three-node energy counterexample: `output/formal_revision/analysis/`.

## Reproduction

Create an isolated Python environment, install the versions in `requirements-lock.txt`, and run:

```text
python -m pytest -q
python analysis/validate_local_energy_counterexample.py
```

The formal simulations are computationally intensive. Use the dry-run and quick profiles described in `REPRODUCIBILITY.md` before launching a full batch. Quick-profile output is for software verification only and is not scientific evidence.

## Energy definition

The validation code uses the same unweighted disagreement energy as the manuscript:

```text
E_G = (1/2) sum_{(u,v) in E} (x_u - x_v)^2.
```

Jaccard similarity is used to calculate structure-dependent confidence thresholds and triangular gates. It is not used as an energy weight.

## Data provenance

`source_data/DATA_PROVENANCE.md` maps each reported result to the preserved file in this repository. Metadata sidecars record the simulation configuration, seed list, and source hashes.

## Licence

The source code is released under the MIT License. The files under `source_data/` are released under the Creative Commons Attribution 4.0 International License (CC BY 4.0), except where a file states otherwise.


