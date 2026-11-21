"""Analyze the targeted perturbation stress scan without touching formal output."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "output" / "formal_revision_extension" / "data" / "exp_targeted_perturbation_stress_v1.json"
DEFAULT_ANALYSIS = ROOT / "output" / "formal_revision_extension" / "analysis"
DEFAULT_FIGURES = ROOT / "output" / "formal_revision_extension" / "figures"


def pkey(value):
    return f"{float(value):.2f}"


def skey(value):
    return f"{float(value):.3f}"


def nearest_key(mapping, value):
    return min(mapping, key=lambda item: abs(float(item) - float(value)))


def paired_stats(treatment, baseline):
    diff = np.asarray(treatment, dtype=float) - np.asarray(baseline, dtype=float)
    n = len(diff)
    mean = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    if n > 1 and sd > 0.0:
        tcrit = float(stats.t.ppf(0.975, n - 1))
        low = mean - tcrit * sd / np.sqrt(n)
        high = mean + tcrit * sd / np.sqrt(n)
        p_value = float(stats.ttest_1samp(diff, 0.0).pvalue)
    else:
        low = high = mean
        p_value = 1.0 if mean == 0.0 else 0.0
    return {"n": n, "mean_difference": mean, "sd_difference": sd,
            "ci95_low": float(low), "ci95_high": float(high),
            "p_raw": p_value}


def metric_values(block, metric, eps_values):
    if metric == "low_op":
        ekey = nearest_key(block["backward"], eps_values[0])
        return block["backward"][ekey]["values"]
    if metric == "low_clusters":
        ekey = nearest_key(block["backward_clusters"], eps_values[0])
        return block["backward_clusters"][ekey]["values"]
    if metric == "hysteresis_area":
        return block["hysteresis_area_stats"]["values"]
    if metric == "backward_auc":
        ekeys = [nearest_key(block["backward"], value) for value in eps_values]
        return [float(np.trapz(
            [block["backward"][ekey]["values"][run_i] for ekey in ekeys],
            eps_values)) for run_i in range(block["num_runs"])]
    raise KeyError(metric)


def summary_stat(block, metric, eps_values):
    if metric == "low_op":
        return block["backward"][nearest_key(
            block["backward"], eps_values[0])]
    if metric == "low_clusters":
        return block["backward_clusters"][nearest_key(
            block["backward_clusters"], eps_values[0])]
    if metric == "hysteresis_area":
        return block["hysteresis_area_stats"]
    values = metric_values(block, metric, eps_values)
    mean = float(np.mean(values))
    return {"mean": mean, "ci95_low": mean, "ci95_high": mean}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--output-stem", default="targeted_perturbation_stress",
                        help="Distinct prefix so one-shot and per-threshold outputs do not overwrite")
    args = parser.parse_args()

    with args.input.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    control, data = payload["control"], payload["data"]
    p_values = control["p_values"]
    sigma_values = control["sigma_values"]
    eps_values = control["eps_values"]
    metrics = ["low_op", "low_clusters", "hysteresis_area", "backward_auc"]
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for sigma in sigma_values:
        for p in p_values:
            block = data[skey(sigma)][pkey(p)]
            baseline = data[skey(0.0)][pkey(p)]
            for metric in metrics:
                result = paired_stats(metric_values(block, metric, eps_values),
                                       metric_values(baseline, metric, eps_values))
                result.update({"sigma": float(sigma), "p_hk": float(p),
                               "metric": metric})
                rows.append(result)

    csv_path = args.analysis_dir / f"{args.output_stem}_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    colors = {float(p): color for p, color in zip(
        p_values, plt.cm.viridis(np.linspace(0.1, 0.9, len(p_values))))}
    for p in p_values:
        op_means, op_lows, op_highs = [], [], []
        h_means, h_lows, h_highs = [], [], []
        for sigma in sigma_values:
            block = data[skey(sigma)][pkey(p)]
            op = summary_stat(block, "low_op", eps_values)
            hy = summary_stat(block, "hysteresis_area", eps_values)
            op_means.append(op["mean"]); op_lows.append(op["ci95_low"])
            op_highs.append(op["ci95_high"])
            h_means.append(hy["mean"]); h_lows.append(hy["ci95_low"])
            h_highs.append(hy["ci95_high"])
        color = colors[float(p)]
        axes[0].errorbar(
            sigma_values, op_means,
            yerr=[np.array(op_means) - op_lows,
                  np.array(op_highs) - np.array(op_means)],
            marker="o", linewidth=1.2, capsize=2, color=color,
            label=f"p_HK={p:g}")
        axes[1].errorbar(
            sigma_values, h_means,
            yerr=[np.array(h_means) - h_lows,
                  np.array(h_highs) - np.array(h_means)],
            marker="o", linewidth=1.2, capsize=2, color=color,
            label=f"p_HK={p:g}")
    axes[0].set_title("Backward low-ε order parameter")
    axes[0].set_ylabel("Mean (95% CI)")
    axes[0].set_ylim(-0.02, 1.04)
    axes[1].set_title("Hysteresis area")
    axes[1].set_ylabel("Area (95% CI)")
    for ax in axes:
        ax.set_xlabel("Perturbation strength σ")
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Targeted perturbation stress scan", fontsize=12)
    fig.savefig(args.figure_dir / f"fig_{args.output_stem}.png", dpi=300)
    fig.savefig(args.figure_dir / f"fig_{args.output_stem}.pdf")
    plt.close(fig)

    all_ceiling = True
    for sigma in sigma_values:
        for p in p_values:
            values = metric_values(data[skey(sigma)][pkey(p)], "low_op", eps_values)
            all_ceiling &= min(values) >= 0.999999
    report_path = args.analysis_dir / f"{args.output_stem.upper()}_REPORT.md"
    result_text = (
        "All low-epsilon order-parameter endpoints are at the metric ceiling; "
        "this scan does not identify a boundary."
        if all_ceiling else
        "At least one condition leaves the low-epsilon order-parameter ceiling; "
        "inspect the summary CSV and figure for candidate boundary cells."
    )
    report_path.write_text("\n".join([
        "# Targeted perturbation stress scan", "",
        "This is a new extension analysis and does not overwrite formal output.",
        "", f"- Input: `{args.input}`",
        f"- Runs: {control['num_runs']}; p_HK: {control['p_values']}",
        f"- Sigma grid: {control['sigma_values']}",
        f"- Epsilon grid: {control['eps_values']}",
        f"- Sweeps per epsilon: {control['sweeps_per_epsilon']}",
        f"- Perturbation: standardized Gaussian shock, clipping {control['clipping']}",
        "", "## Interpretation", "", result_text, "",
        "This is a bounded negative result for the tested graph family, "
        "perturbation protocol, grid, and simulation horizon. It is not a "
        "claim of universal stability.", "",
        f"Summary table: `{csv_path.name}`", "",
    ]), encoding="utf-8")
    print(f"Wrote {report_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
