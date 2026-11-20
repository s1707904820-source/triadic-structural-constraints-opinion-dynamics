"""Summarize the 50/100/200-sweep perturbation horizon pilot."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from statistics_utils import paired_difference


AUDIT_DIR = ROOT / "output" / "formal_revision" / "data" / "perturbation_horizon_audit"
OUT_MD = ROOT / "output" / "formal_revision" / "analysis" / "PERTURBATION_HORIZON_AUDIT_REPORT.md"


def load(horizon):
    path = AUDIT_DIR / "exp_perturbation_horizon_{}.json".format(
        ("{:g}".format(horizon)).replace(".", "p"))
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def values(block, field, epsilon):
    return block[field][str(epsilon)]["values"]


def main():
    horizons = [50.0, 100.0, 200.0]
    payloads = {h: load(h) for h in horizons}
    p_values = payloads[200.0]["control"]["p_values"]
    sigmas = payloads[200.0]["control"]["sigma_values"]
    eps_values = payloads[200.0]["control"]["eps_values"]
    rows = []
    for sigma in sigmas:
        skey = "{:.3f}".format(sigma)
        for p in p_values:
            pkey = "{:.2f}".format(p)
            condition = [payloads[h]["data"][skey][pkey] for h in horizons]
            low_op = {h: values(condition[i], "backward", eps_values[0])
                      for i, h in enumerate(horizons)}
            low_cl = {h: values(condition[i], "backward_clusters", eps_values[0])
                      for i, h in enumerate(horizons)}
            hyst = {h: condition[i]["hysteresis_area_stats"]["values"]
                    for i, h in enumerate(horizons)}
            curve_max = {}
            tail_max = {}
            for h in (50.0, 100.0):
                b100 = condition[1 if h == 100.0 else 0]
                b200 = condition[2]
                curve_max[h] = max(abs(
                    b100["backward"][str(e)]["mean"]
                    - b200["backward"][str(e)]["mean"])
                    for e in eps_values)
                tail_max[h] = max(abs(
                    b100["backward_tail_mean_abs_dx"][str(e)]["mean"]
                    - b200["backward_tail_mean_abs_dx"][str(e)]["mean"])
                    for e in eps_values)
            d_op100 = paired_difference(low_op[100.0], low_op[200.0])
            d_cl100 = paired_difference(low_cl[100.0], low_cl[200.0])
            d_hyst100 = paired_difference(hyst[100.0], hyst[200.0])
            rows.append({
                "sigma": sigma, "p": p,
                "low_op_50": float(np.mean(low_op[50.0])),
                "low_op_100": float(np.mean(low_op[100.0])),
                "low_op_200": float(np.mean(low_op[200.0])),
                "low_clusters_100": float(np.mean(low_cl[100.0])),
                "low_clusters_200": float(np.mean(low_cl[200.0])),
                "delta_low_op_100_minus_200": d_op100,
                "delta_low_clusters_100_minus_200": d_cl100,
                "delta_hysteresis_100_minus_200": d_hyst100,
                "max_curve_abs_diff_50_vs_200": curve_max[50.0],
                "max_curve_abs_diff_100_vs_200": curve_max[100.0],
                "max_tail_dx_abs_diff_50_vs_200": tail_max[50.0],
                "max_tail_dx_abs_diff_100_vs_200": tail_max[100.0],
            })

    def max_row(key):
        return max(row[key] for row in rows)

    # These are predeclared practical diagnostics, not equivalence tests.
    op_tol = 0.01
    curve_tol = 0.01
    hyst_tol = 0.005
    cluster_tol = 0.5
    stable100 = (
        max(abs(row["delta_low_op_100_minus_200"]["mean"]) for row in rows)
        <= op_tol
        and max(row["max_curve_abs_diff_100_vs_200"] for row in rows)
        <= curve_tol
        and max(abs(row["delta_hysteresis_100_minus_200"]["mean"]) for row in rows)
        <= hyst_tol
        and max(abs(row["delta_low_clusters_100_minus_200"]["mean"]) for row in rows)
        <= cluster_tol
    )

    lines = [
        "# Perturbation horizon audit report",
        "",
        "This is a pilot/design audit, not formal manuscript evidence. It uses five paired seeds at p_HK = 0, 0.4, 0.8 and sigma = 0, 0.05, 0.2.",
        "",
        "## Predeclared practical diagnostics",
        "",
        "The 100-sweep setting is considered practically stable only when its paired 100 versus 200 differences satisfy all four diagnostics: low-epsilon backward OP <= 0.01, maximum backward-curve discrepancy <= 0.01, hysteresis-area difference <= 0.005, and low-epsilon cluster difference <= 0.5. These tolerances are sensitivity criteria, not equivalence tests, and do not convert a non-significant result into equivalence.",
        "",
        "| sigma | p_HK | low OP 50 | low OP 100 | low OP 200 | low clusters 100 | low clusters 200 | dOP 100-200 | dHyst 100-200 | max curve 100-200 | max tail-dx 100-200 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append("| {sigma:.2f} | {p:.2f} | {low_op_50:.4f} | {low_op_100:.4f} | {low_op_200:.4f} | {low_clusters_100:.3f} | {low_clusters_200:.3f} | {dop:.4f} | {dh:.5f} | {curve:.5f} | {tail:.3g} |".format(
            sigma=row["sigma"], p=row["p"], low_op_50=row["low_op_50"],
            low_op_100=row["low_op_100"], low_op_200=row["low_op_200"],
            low_clusters_100=row["low_clusters_100"], low_clusters_200=row["low_clusters_200"],
            dop=row["delta_low_op_100_minus_200"]["mean"],
            dh=row["delta_hysteresis_100_minus_200"]["mean"],
            curve=row["max_curve_abs_diff_100_vs_200"],
            tail=row["max_tail_dx_abs_diff_100_vs_200"]))
    lines += [
        "",
        "## Decision",
        "",
        "100-sweep practical-stability criteria: **{}**.".format("PASS" if stable100 else "FAIL"),
        "50-versus-200 maximum curve discrepancy: {:.5f}.".format(max_row("max_curve_abs_diff_50_vs_200")),
        "100-versus-200 maximum curve discrepancy: {:.5f}.".format(max_row("max_curve_abs_diff_100_vs_200")),
        "",
        "If the criteria pass, 100 sweeps may be used for the formal perturbation experiment with the 200-sweep audit retained as a sensitivity check. If they fail, retain 200 sweeps or refine the epsilon grid near the unstable boundary.",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_MD)
    print("stable100=", stable100)


if __name__ == "__main__":
    main()
