"""Analyze formal perturbation-gradient output with paired seeds and Holm correction."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from statistics_utils import paired_difference  # noqa: E402


DATA_PATH = ROOT / "output" / "formal_revision" / "data" / "exp_formal_perturbation_sweep_revision_v2.json"
ANALYSIS_DIR = ROOT / "output" / "formal_revision" / "analysis"
FIGURE_DIR = ROOT / "output" / "formal_revision" / "figures"


def pvalue(values):
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError("paired differences must be a finite non-empty vector")
    if arr.size < 2 or np.all(arr == arr[0]):
        return 1.0 if np.all(arr == 0.0) else 0.0
    p_value = float(stats.ttest_1samp(arr, 0.0).pvalue)
    return 1.0 if not np.isfinite(p_value) else p_value


def holm(p_values):
    p_values = np.asarray(p_values, dtype=float)
    if p_values.ndim != 1:
        raise ValueError("p-values must be a one-dimensional vector")
    if p_values.size == 0:
        return []
    if (not np.all(np.isfinite(p_values))
            or np.any((p_values < 0.0) | (p_values > 1.0))):
        raise ValueError("p-values must be finite and lie in [0, 1]")
    order = np.argsort(p_values)
    out = np.empty_like(p_values)
    running = 0.0
    m = len(p_values)
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p_values[idx])
        running = max(running, value)
        out[idx] = running
    return out.tolist()


def skey(value):
    return "{:.3f}".format(float(value))


def pkey(value):
    return "{:.2f}".format(float(value))


def qkey(value):
    return str(float(value))


def paired_metric(block, metric, eps_values, low_eps):
    if metric == "low_op":
        return block["backward"][qkey(low_eps)]["values"]
    if metric == "low_clusters":
        return block["backward_clusters"][qkey(low_eps)]["values"]
    if metric == "hysteresis_area":
        return block["hysteresis_area_stats"]["values"]
    if metric == "peak_sensitivity":
        return block["peak_sensitivity_stats"]["values"]
    if metric == "backward_auc":
        curves = []
        for idx in range(block["num_runs"]):
            curves.append(np.trapz(
                [block["backward"][qkey(e)]["values"][idx] for e in eps_values],
                eps_values))
        return curves
    raise KeyError(metric)


def main():
    with DATA_PATH.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    control = payload["control"]
    data = payload["data"]
    p_values = control["p_values"]
    sigma_values = control["sigma_values"]
    eps_values = control["eps_values"]
    low_eps = eps_values[0]
    nonzero_sigmas = [s for s in sigma_values if float(s) > 0.0]
    metrics = ["low_op", "low_clusters", "hysteresis_area",
               "peak_sensitivity", "backward_auc"]

    rows = []
    for metric in metrics:
        raw_p = []
        temp = []
        for sigma in nonzero_sigmas:
            for p in p_values:
                baseline = data[skey(0.0)][pkey(p)]
                treatment = data[skey(sigma)][pkey(p)]
                a = np.asarray(paired_metric(treatment, metric, eps_values, low_eps))
                b = np.asarray(paired_metric(baseline, metric, eps_values, low_eps))
                diff = a - b
                item = paired_difference(a, b)
                item.update({"metric": metric, "sigma": float(sigma),
                             "p_hk": float(p), "p_raw": pvalue(diff)})
                temp.append(item)
                raw_p.append(item["p_raw"])
        adjusted = holm(raw_p)
        for item, adj in zip(temp, adjusted):
            item["p_holm"] = float(adj)
            rows.append(item)

    report_path = ANALYSIS_DIR / "FORMAL_PERTURBATION_REPORT.md"
    lines = [
        "# Formal perturbation-gradient report",
        "",
        "Design: paired seeds, shared forward scans, common standardized perturbation directions, and common backward RNG streams. The independent experimental unit is the seed-level simulation replicate.",
        "",
        "The six sigma conditions are compared with sigma=0. Two-sided paired t tests use seed-level differences. Holm correction is applied within each endpoint family across the 15 (five nonzero sigma values × three p_HK values) contrasts. A non-significant result is not interpreted as strict equivalence.",
        "",
        "| endpoint | sigma | p_HK | mean difference | 95% CI | dz | Holm p |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    labels = {
        "low_op": "backward low-epsilon OP",
        "low_clusters": "backward low-epsilon clusters",
        "hysteresis_area": "hysteresis area",
        "peak_sensitivity": "forward peak sensitivity",
        "backward_auc": "backward OP AUC",
    }
    for row in rows:
        lines.append("| {} | {:.2f} | {:.2f} | {:.6g} | [{:.6g}, {:.6g}] | {:.3f} | {:.4g} |".format(
            labels[row["metric"]], row["sigma"], row["p_hk"], row["mean"],
            row["ci95_low"], row["ci95_high"], row["cohen_dz"] or 0.0,
            row["p_holm"]))

    clip_lines = ["", "## Perturbation clipping diagnostics", "",
                  "| sigma | p_HK | mean clipped fraction |", "|---:|---:|---:|"]
    for sigma in sigma_values:
        for p in p_values:
            block = data[skey(sigma)][pkey(p)]
            clip_lines.append("| {:.2f} | {:.2f} | {:.6g} |".format(
                sigma, p, block["perturbation_diagnostics"]["clipped_fraction"]["mean"]))
    lines.extend(clip_lines)
    lines += [
        "", "## Result boundary", "",
        "Across this epsilon range, the backward order parameter reached the metric ceiling of 1.0 for every p_HK and sigma condition. Accordingly, this experiment does not identify a near-consensus loss boundary or a nonzero sigma-dependent hysteresis effect within the tested range. The appropriate conclusion is a boundary non-detection under this protocol, not proof of universal stability or equivalence.",
        "", "## Interpretation guardrails", "",
        "The report separates update-scale damping from macroscopic path or terminal effects. Any significant result for hysteresis, low-epsilon OP, cluster count, or backward AUC is interpreted as a sigma-dependent perturbation response, not as evidence that a small perturbation establishes a universal stability boundary.",
    ]
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9, "pdf.fonttype": 42, "svg.fonttype": "none",
        "axes.spines.right": False, "axes.spines.top": False,
    })
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    colors = {0.0: "#4C566A", 0.01: "#88C0D0", 0.02: "#5E81AC",
              0.05: "#EBCB8B", 0.1: "#D08770", 0.2: "#BF616A"}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), sharey=True,
                             constrained_layout=True)
    for ax, p in zip(axes, p_values):
        for sigma in sigma_values:
            block = data[skey(sigma)][pkey(p)]
            means = [block["backward"][qkey(e)]["mean"] for e in eps_values]
            lows = [block["backward"][qkey(e)]["ci95_low"] for e in eps_values]
            highs = [block["backward"][qkey(e)]["ci95_high"] for e in eps_values]
            label = r"$σ={}$".format("{:g}".format(sigma))
            ax.plot(eps_values, means, color=colors[float(sigma)], lw=1.2,
                    label=label)
            ax.fill_between(eps_values, lows, highs, color=colors[float(sigma)],
                            alpha=0.10, linewidth=0)
        ax.set_title(r"$p_{{HK}}={:.1f}$".format(p), fontsize=9)
        ax.set_xlabel(r"$ε_0$")
        ax.set_xlim(min(eps_values), max(eps_values))
        ax.grid(axis="y", color="#D8DEE9", lw=0.4, alpha=0.8)
    axes[0].set_ylabel("Backward order parameter")
    axes[-1].legend(frameon=False, fontsize=7, ncol=2, loc="lower right")
    for idx, ax in enumerate(axes):
        ax.text(-0.18, 1.04, chr(ord("a") + idx), transform=ax.transAxes,
                fontweight="bold", va="bottom")
    curve_path = FIGURE_DIR / "fig_formal_perturbation_backward_curves"
    fig.savefig(str(curve_path) + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(str(curve_path) + ".tiff", dpi=600, bbox_inches="tight")
    fig.savefig(str(curve_path) + ".pdf", bbox_inches="tight")
    fig.savefig(str(curve_path) + ".svg", bbox_inches="tight")
    plt.close(fig)

    # A paired sigma-response figure for hysteresis area.
    fig, ax = plt.subplots(figsize=(3.6, 2.6), constrained_layout=True)
    for p in p_values:
        means = []
        lows = []
        highs = []
        for sigma in sigma_values:
            summary = data[skey(sigma)][pkey(p)]["hysteresis_area_stats"]
            means.append(summary["mean"])
            lows.append(summary["ci95_low"])
            highs.append(summary["ci95_high"])
        ax.plot(sigma_values, means, marker="o", ms=3.5, lw=1.2,
                label=r"$p_{{HK}}={:.1f}$".format(p))
        ax.fill_between(sigma_values, lows, highs, alpha=0.12)
    ax.set_xlabel(r"Perturbation amplitude $σ$")
    ax.set_ylabel("Hysteresis area")
    ax.tick_params(axis="both", labelsize=9)
    # Keep the six-condition design in the source data, but show a readable
    # subset of tick labels at the final figure size.
    ax.set_xticks([0.0, 0.05, 0.10, 0.20])
    ax.set_xticklabels(["0", "0.05", "0.10", "0.20"])
    ax.grid(axis="y", color="#D8DEE9", lw=0.4, alpha=0.8)
    ax.legend(frameon=False, fontsize=9)
    hyst_path = FIGURE_DIR / "fig_formal_perturbation_hysteresis"
    fig.savefig(str(hyst_path) + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(str(hyst_path) + ".tiff", dpi=600, bbox_inches="tight")
    fig.savefig(str(hyst_path) + ".pdf", bbox_inches="tight")
    fig.savefig(str(hyst_path) + ".svg", bbox_inches="tight")
    plt.close(fig)
    print(report_path)
    print(curve_path)
    print(hyst_path)


if __name__ == "__main__":
    main()
