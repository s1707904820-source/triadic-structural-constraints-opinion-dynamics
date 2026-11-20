"""Analyze the existing formal SBM hysteresis JSON without changing it."""

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
DEFAULT_INPUT = ROOT / "output" / "formal_revision" / "data" / "exp_sbm_hysteresis_revision_v2.json"
DEFAULT_ANALYSIS = ROOT / "output" / "formal_revision_extension" / "analysis"
DEFAULT_FIGURES = ROOT / "output" / "formal_revision_extension" / "figures"


def skey(value):
    return f"{float(value):.3f}"


def paired(values, baseline):
    diff = np.asarray(values, dtype=float) - np.asarray(baseline, dtype=float)
    n = len(diff)
    mean = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    if n > 1 and sd > 0.0:
        tcrit = float(stats.t.ppf(0.975, n - 1))
        low = mean - tcrit * sd / np.sqrt(n)
        high = mean + tcrit * sd / np.sqrt(n)
        p = float(stats.ttest_1samp(diff, 0.0).pvalue)
    else:
        low = high = mean
        p = 1.0 if mean == 0.0 else 0.0
    return {"n": n, "mean_difference": mean, "ci95_low": float(low),
            "ci95_high": float(high), "p_raw": p}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURES)
    args = parser.parse_args()

    with args.input.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    control, data = payload["control"], payload["data"]
    mixing_values = [float(x) for x in control["mixing_values"]]
    sigma_values = [float(x) for x in control["sigma_values"]]
    eps_values = [float(x) for x in control["eps_values"]]
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    rows, comparisons = [], []
    for mixing in mixing_values:
        block = data[skey(mixing)]
        cross = block["cross_edge_fraction"]
        for sigma in sigma_values:
            branch = block["backward"][skey(sigma)]
            area = branch["hysteresis_area_stats"]
            low_key = min(branch["order_param"], key=lambda x: float(x))
            low = branch["order_param"][low_key]
            rows.append({
                "mixing": mixing, "sigma": sigma,
                "cross_edge_fraction_mean": cross["mean"],
                "cross_edge_fraction_ci95_low": cross["ci95_low"],
                "cross_edge_fraction_ci95_high": cross["ci95_high"],
                "hysteresis_area_mean": area["mean"],
                "hysteresis_area_ci95_low": area["ci95_low"],
                "hysteresis_area_ci95_high": area["ci95_high"],
                "low_epsilon": float(low_key),
                "low_epsilon_order_mean": low["mean"],
                "low_epsilon_order_ci95_low": low["ci95_low"],
                "low_epsilon_order_ci95_high": low["ci95_high"],
            })
        baseline = block["backward"][skey(0.0)]["hysteresis_area_stats"]["values"]
        for sigma in sigma_values:
            if sigma == 0.0:
                continue
            values = block["backward"][skey(sigma)]["hysteresis_area_stats"]["values"]
            result = paired(values, baseline)
            result.update({"mixing": mixing, "sigma": sigma,
                           "metric": "hysteresis_area"})
            comparisons.append(result)

    summary_path = args.analysis_dir / "sbm_hysteresis_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    comparison_path = args.analysis_dir / "sbm_hysteresis_sigma_comparisons.csv"
    with comparison_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader(); writer.writerows(comparisons)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(mixing_values)))
    for color, mixing in zip(colors, mixing_values):
        block = data[skey(mixing)]
        forward = block["forward"]["order_param"]
        fmean = [forward[str(e)]["mean"] for e in eps_values]
        axes[0, 0].plot(eps_values, fmean, marker="o", markersize=2,
                        linewidth=1.0, color=color, label=f"q={mixing:g}")
        backward = block["backward"][skey(0.0)]["order_param"]
        bmean = [backward[str(e)]["mean"] for e in reversed(eps_values)]
        axes[0, 1].plot(list(reversed(eps_values)), bmean, marker="o",
                        markersize=2, linewidth=1.0, color=color,
                        label=f"q={mixing:g}")

    for ax, title in zip(axes[0], ["Forward scan (σ=0)", "Backward scan (σ=0)"]):
        ax.set_title(title); ax.set_xlabel("ε₀"); ax.set_ylabel("Order parameter")
        ax.grid(alpha=0.25, linestyle="--"); ax.set_ylim(0, 1.03)
    axes[0, 1].legend(frameon=False, fontsize=7)

    area0 = []
    for color, mixing in zip(colors, mixing_values):
        block = data[skey(mixing)]
        means, lows, highs = [], [], []
        for sigma in sigma_values:
            area = block["backward"][skey(sigma)]["hysteresis_area_stats"]
            means.append(area["mean"]); lows.append(area["ci95_low"])
            highs.append(area["ci95_high"])
        area0.append(means[0])
        axes[1, 0].errorbar(
            sigma_values, means,
            yerr=[np.array(means) - lows, np.array(highs) - np.array(means)],
            marker="o", linewidth=1.0, capsize=2, color=color,
            label=f"q={mixing:g}")
    axes[1, 0].set_title("Hysteresis area across perturbation strengths")
    axes[1, 0].set_xlabel("σ"); axes[1, 0].set_ylabel("Area (95% CI)")
    axes[1, 0].grid(alpha=0.25, linestyle="--")
    axes[1, 1].plot(mixing_values, area0, marker="o", linewidth=1.3,
                    color="#7C3AED")
    axes[1, 1].set_title("Hysteresis area vs inter-community mixing")
    axes[1, 1].set_xlabel("SBM mixing parameter"); axes[1, 1].set_ylabel("Area, σ=0")
    axes[1, 1].grid(alpha=0.25, linestyle="--")
    fig.suptitle("Formal two-community SBM hysteresis analysis", fontsize=13)
    fig.savefig(args.figure_dir / "fig_sbm_hysteresis_summary.png", dpi=300)
    fig.savefig(args.figure_dir / "fig_sbm_hysteresis_summary.pdf")
    plt.close(fig)

    direction = "decreases" if area0[-1] < area0[0] else "does not decrease monotonically"
    report_path = args.analysis_dir / "SBM_HYSTERESIS_REPORT.md"
    report_path.write_text("\n".join([
        "# Formal SBM hysteresis analysis", "",
        "This report analyzes the existing formal JSON and does not rerun or modify it.",
        "", f"- Network: balanced two-block SBM, N={control['N']}, target degree={control['target_degree']}",
        f"- Replicates: {control['num_runs']} per mixing × sigma condition",
        f"- Mixing values: {mixing_values}", f"- Sigma values: {sigma_values}",
        f"- Epsilon grid: {len(eps_values)} points from {min(eps_values):g} to {max(eps_values):g}",
        f"- Sweeps per epsilon: {control['sweeps_per_epsilon']}", "",
        "## Main result", "",
        f"At sigma=0, hysteresis area {direction} as inter-community mixing increases.",
        "This supports the interpretation that stronger community separation produces a larger forward/backward discrepancy under the tested protocol.",
        "", "## Statistical caution", "",
        "The SBM experiment is a numerical multi-community robustness check, not a theorem about all community networks. Sigma comparisons are paired within the same seed and are protocol-specific.",
        "", f"Summary table: `{summary_path.name}`",
        f"Sigma comparisons: `{comparison_path.name}`", "",
    ]), encoding="utf-8")
    print(f"Wrote {report_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {comparison_path}")


if __name__ == "__main__":
    main()
