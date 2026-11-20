"""Validate and analyze the formal edge-threshold assignment controls."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7.5,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 5.8,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})
import numpy as np
from scipy import stats

from figure_qa import (
    add_panel_label, write_alignment_manifest,
    audit_pdf_text_geometry,
)


ROOT = Path(__file__).resolve().parents[1]
FORMAL_ROOT = ROOT / "output" / "formal_revision"
DATA_FILE = FORMAL_ROOT / "data" / "exp_threshold_assignment_controls.json"
META_FILE = FORMAL_ROOT / "data" / "exp_threshold_assignment_controls.meta.json"
ANALYSIS_DIR = FORMAL_ROOT / "analysis"
FIGURE_DIR = FORMAL_ROOT / "figures"

N_RUNS = 20
SEEDS = [run_i * 997 + 7 for run_i in range(N_RUNS)]
P_VALUES = [round(index * 0.05, 2) for index in range(21)]
MODEL_ORDER = (
    "threshold_only", "mean_threshold_only", "randomized_threshold_only")
COMPARATORS = ("mean_threshold_only", "randomized_threshold_only")
METRICS = (
    "order_param", "n_clusters", "variance",
    "triangle_alignment_fixed", "triangle_admissibility_adaptive",
    "edge_disagreement_penalty",
)
METRIC_LABELS = {
    "order_param": "Order parameter",
    "n_clusters": "Number of clusters",
    "variance": "Opinion variance",
    "triangle_alignment_fixed": "Fixed-scale triangle alignment",
    "triangle_admissibility_adaptive": "Adaptive triangle admissibility",
    "edge_disagreement_penalty": "Mean edge-disagreement penalty",
}
MODEL_LABELS = {
    "threshold_only": "Edge-specific",
    "mean_threshold_only": "Mean threshold",
    "randomized_threshold_only": "Shuffled thresholds",
}
COLORS = {
    "threshold_only": "#7C3AED",
    "mean_threshold_only": "#0891B2",
    "randomized_threshold_only": "#65A30D",
}
T975 = (
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306,
    2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120,
    2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064,
    2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(a, b, atol=1e-12):
    return math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=atol)


def summarize(values):
    values = np.asarray(values, dtype=float)
    n = int(values.size)
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if n > 1 else 0.0
    se = float(sd / math.sqrt(n)) if n else float("nan")
    critical = T975[n - 2] if 2 <= n <= len(T975) + 1 else 1.960
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "se": se,
        "ci95_low": float(mean - critical * se),
        "ci95_high": float(mean + critical * se),
        "cohen_dz": float(mean / sd) if sd > 0 else (
            0.0 if mean == 0 else None),
        "values": values.tolist(),
    }


def paired_t_pvalue(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("paired differences must be a finite non-empty vector")
    if np.all(values == values[0]):
        return 1.0 if values[0] == 0.0 else 0.0
    p_value = float(stats.ttest_1samp(values, 0.0).pvalue)
    return 1.0 if not math.isfinite(p_value) else p_value


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    if p_values.size == 0:
        return []
    if (p_values.ndim != 1 or not np.all(np.isfinite(p_values))
            or np.any((p_values < 0.0) | (p_values > 1.0))):
        raise ValueError("p-values must be finite and lie in [0, 1]")
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate(payload, metadata):
    errors = []
    control = payload.get("control", {})
    blocks = payload.get("data", {})
    actual_p = sorted(float(key) for key in blocks)
    if actual_p != P_VALUES:
        errors.append("p_HK grid mismatch: {}".format(actual_p))
    if control.get("N") != 2000 or control.get("m") != 4:
        errors.append("network configuration is not N=2000, m=4")
    if control.get("num_runs") != N_RUNS:
        errors.append("control num_runs is not 20")
    if control.get("model_order") != list(MODEL_ORDER):
        errors.append("model order mismatch")
    if float(control.get("main_sweeps", -1)) != 200.0:
        errors.append("main_sweeps is not 200")
    if metadata.get("n_runs") != N_RUNS:
        errors.append("metadata n_runs is not 20")
    if metadata.get("seeds") != SEEDS:
        errors.append("metadata seeds mismatch")
    sim_config = metadata.get("config", {}).get("SIMULATION", {})
    if float(sim_config.get("main_sweeps", -1)) != 200.0:
        errors.append("metadata main_sweeps is not 200")
    for filename, recorded in sorted(
            metadata.get("source_sha256", {}).items()):
        path = ROOT / filename
        if not path.exists():
            errors.append("recorded source missing: {}".format(filename))
        elif sha256(path) != recorded:
            errors.append("source changed since run: {}".format(filename))

    for comparator in COMPARATORS:
        for p_key in sorted(blocks, key=float):
            block = blocks[p_key]
            block_control = block.get("_control", {})
            if block_control.get("seeds") != SEEDS:
                errors.append("seed list mismatch at p={}".format(p_key))
            if block_control.get("model_order") != list(MODEL_ORDER):
                errors.append("block model order mismatch at p={}".format(p_key))
            for model in MODEL_ORDER:
                for metric in METRICS:
                    summary = block.get(model, {}).get(metric, {})
                    values = summary.get("values", [])
                    if summary.get("n") != N_RUNS or len(values) != N_RUNS:
                        errors.append(
                            "{} {} incomplete at p={}".format(
                                model, metric, p_key))
                        continue
                    expected = summarize(values)
                    for field in (
                            "mean", "sd", "se", "ci95_low", "ci95_high"):
                        if not close(summary.get(field, float("nan")),
                                     expected[field]):
                            errors.append(
                                "summary mismatch {} {} {} p={}".format(
                                    model, metric, field, p_key))
            for comparator in COMPARATORS:
                paired = block.get("_paired_comparisons", {}).get(
                    "threshold_only_minus_{}".format(comparator), {})
                for metric in METRICS:
                    saved = paired.get(metric, {})
                    expected = (
                        np.asarray(block["threshold_only"][metric]["values"])
                        - np.asarray(block[comparator][metric]["values"]))
                    if (len(saved.get("values", [])) != N_RUNS
                            or not np.allclose(
                                saved.get("values", []), expected,
                                atol=1e-12)):
                        errors.append(
                            "paired values mismatch {} {} p={}".format(
                                comparator, metric, p_key))
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "p_points": len(actual_p),
        "models": list(MODEL_ORDER),
        "n_per_cell": N_RUNS,
        "independent_unit": (
            "seed-level paired simulation replicate; the three arms use "
            "matched network and initial-condition seeds"),
    }


def analyze(payload, metadata, validation):
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    blocks = payload["data"]
    p_keys = sorted(blocks, key=float)
    p_values = [float(key) for key in p_keys]

    source_rows = []
    for p_key, p in zip(p_keys, p_values):
        block = blocks[p_key]
        for model in MODEL_ORDER:
            for index, seed in enumerate(SEEDS):
                row = {"p_hk": p, "model": model, "seed": seed}
                for metric in METRICS:
                    row[metric] = block[model][metric]["values"][index]
                source_rows.append(row)

    point_rows = []
    global_rows = []
    for comparator in COMPARATORS:
        for metric in METRICS:
            raw_p = []
            rows = []
            differences_by_seed = []
            for p_key, p in zip(p_keys, p_values):
                saved = blocks[p_key]["_paired_comparisons"][
                    "threshold_only_minus_{}".format(comparator)][metric]
                differences = np.asarray(saved["values"], dtype=float)
                differences_by_seed.append(differences)
                raw = paired_t_pvalue(differences)
                raw_p.append(raw)
                rows.append({
                    "comparator": comparator,
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "p_hk": p,
                    "n": saved["n"],
                    "mean_difference": saved["mean"],
                    "sd_difference": saved["sd"],
                    "se_difference": saved["se"],
                    "ci95_low": saved["ci95_low"],
                    "ci95_high": saved["ci95_high"],
                    "cohen_dz": saved.get("cohen_dz"),
                    "paired_t_p_raw": raw,
                })
            adjusted = holm_adjust(raw_p)
            for row, value in zip(rows, adjusted):
                row["paired_t_p_holm_21"] = value
                point_rows.append(row)

            per_seed_auc = np.trapz(
                np.asarray(differences_by_seed, dtype=float),
                x=np.asarray(p_values, dtype=float), axis=0)
            summary = summarize(per_seed_auc)
            global_rows.append({
                "comparator": comparator,
                "metric": metric,
                "metric_label": METRIC_LABELS[metric],
                "n": summary["n"],
                "mean_auc_difference": summary["mean"],
                "sd_auc_difference": summary["sd"],
                "se_auc_difference": summary["se"],
                "ci95_low": summary["ci95_low"],
                "ci95_high": summary["ci95_high"],
                "cohen_dz": summary["cohen_dz"],
                "paired_t_p_raw": paired_t_pvalue(per_seed_auc),
                "values": summary["values"],
            })
    adjusted_global = holm_adjust(
        [row["paired_t_p_raw"] for row in global_rows])
    for row, adjusted in zip(global_rows, adjusted_global):
        row["paired_t_p_holm_12"] = adjusted

    write_csv(source_rows, str(
        ANALYSIS_DIR / "formal_threshold_controls_source_data.csv"))
    write_csv(point_rows, str(
        ANALYSIS_DIR / "formal_threshold_controls_pointwise.csv"))
    write_csv([{
        key: value for key, value in row.items() if key != "values"
    } for row in global_rows], str(
        ANALYSIS_DIR / "formal_threshold_controls_global_auc.csv"))

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.55), sharex=True)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.67,
                        bottom=0.17, wspace=0.43)
    for ax, label in zip(axes, ("a", "b", "c")):
        ax.set_gid(label)
        add_panel_label(ax, label)

    # (a) Overall response across the network-structure parameter.
    for model in MODEL_ORDER:
        means = [blocks[key][model]["order_param"]["mean"] for key in p_keys]
        lows = [blocks[key][model]["order_param"]["ci95_low"] for key in p_keys]
        highs = [blocks[key][model]["order_param"]["ci95_high"] for key in p_keys]
        axes[0].plot(
            p_values, means, color=COLORS[model], marker="o",
            markersize=2.2, linewidth=1.35, label=MODEL_LABELS[model])
        axes[0].fill_between(
            p_values, lows, highs, color=COLORS[model], alpha=0.12)
    axes[0].set_title("Terminal order parameter", pad=7)
    axes[0].set_ylabel("Mean (95% CI)")
    axes[0].set_ylim(0, 1.03)

    # (b) Primary paired test of structural assignment.
    for comparator in COMPARATORS:
        rows = [row for row in point_rows
                if row["comparator"] == comparator
                and row["metric"] == "order_param"]
        axes[1].plot(
            p_values, [row["mean_difference"] for row in rows],
            marker="o", markersize=2.2, linewidth=1.35,
            color=COLORS[comparator],
            label=MODEL_LABELS[comparator])
        axes[1].fill_between(
            p_values, [row["ci95_low"] for row in rows],
            [row["ci95_high"] for row in rows],
            color=COLORS[comparator], alpha=0.13)
    axes[1].axhline(0, color="#111827", linestyle="--", linewidth=0.8)
    axes[1].set_title("Order-parameter contrast", pad=7)
    axes[1].set_ylabel("Structured minus control (95% CI)")

    # (c) Supporting endpoint: fragmentation.
    for comparator in COMPARATORS:
        rows = [row for row in point_rows
                if row["comparator"] == comparator
                and row["metric"] == "n_clusters"]
        axes[2].plot(
            p_values, [row["mean_difference"] for row in rows],
            marker="o", markersize=2.2, linewidth=1.35,
            color=COLORS[comparator],
            label=MODEL_LABELS[comparator])
        axes[2].fill_between(
            p_values, [row["ci95_low"] for row in rows],
            [row["ci95_high"] for row in rows],
            color=COLORS[comparator], alpha=0.13)
    axes[2].axhline(0, color="#111827", linestyle="--", linewidth=0.8)
    axes[2].set_title("Cluster-count contrast", pad=7)
    axes[2].set_ylabel("Structured minus control (95% CI)")

    for ax in axes:
        ax.set_xlabel("p_HK")
        ax.set_xticks(np.arange(0.0, 1.01, 0.2))
        ax.set_xlim(min(p_values), max(p_values))
        ax.grid(alpha=0.2, linestyle="--", linewidth=0.5)
        ax.margins(x=0.02)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, legend_labels, loc="upper center",
        bbox_to_anchor=(0.5, 0.90), ncol=3, fontsize=6,
        frameon=False, handlelength=1.8, columnspacing=1.2)
    fig.suptitle(
        "Does structural assignment of edge thresholds matter?",
        fontsize=8.5, fontweight="bold", y=0.985)

    base = FIGURE_DIR / "fig_formal_threshold_assignment_controls"
    write_alignment_manifest(
        fig, axes, ("a", "b", "c"),
        base.with_suffix(".alignment-layout.json"),
        row_groups=(("a", "b", "c"),),
        column_groups=(),
    )
    fig.savefig(str(base.with_suffix(".svg")))
    pdf_path = base.with_suffix(".pdf")
    fig.savefig(str(pdf_path))
    audit_pdf_text_geometry(
        pdf_path, base.with_suffix(".pdf-text-geometry-audit.json"))
    fig.savefig(str(base.with_suffix(".tiff")), dpi=600)
    fig.savefig(str(base.with_suffix(".png")), dpi=300)
    plt.close(fig)

    result = {
        "validation": validation,
        "design": payload.get("control"),
        "metadata_created_utc": metadata.get("created_utc"),
        "input_sha256": sha256(DATA_FILE),
        "metadata_sha256": sha256(META_FILE),
        "pointwise_paired_contrasts": point_rows,
        "global_across_p_auc_contrasts": global_rows,
        "multiplicity": {
            "pointwise": "Holm correction across 21 p_HK points within each model contrast and metric",
            "global": "Holm correction across 12 global AUC tests",
        },
        "interpretation_boundary": (
            "Contrasts quantify the effect of assigning the realized "
            "edge-specific threshold distribution to its structural edges. "
            "They do not establish a causal mechanism outside this simulated "
            "network and parameter protocol."),
    }
    json_path = ANALYSIS_DIR / "formal_threshold_controls_analysis.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)

    report_path = ANALYSIS_DIR / "FORMAL_THRESHOLD_CONTROLS_REPORT.md"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("# Formal threshold-assignment controls\n\n")
        handle.write(
            "Design: N=2000, m=4, 21 values of p_HK, 20 paired seeds, "
            "and 200 Monte Carlo sweeps per run. The three arms use the same "
            "network and initial-condition seeds. The mean-threshold arm uses "
            "the realized unweighted mean of the proposed edge thresholds; "
            "the randomized arm permutes the realized threshold multiset over "
            "undirected edges.\n\n")
        handle.write(
            "Validation: **{}** ({} errors). Pointwise paired tests use "
            "two-sided t-tests with Holm adjustment over the 21 network "
            "parameter values within each contrast and metric. Global tests "
            "compare per-seed trapezoidal AUCs across p_HK, with Holm "
            "adjustment over the 12 contrast-metric tests.\n\n".format(
                validation["status"], len(validation["errors"])))
        handle.write("## Global paired AUC contrasts\n\n")
        handle.write(
            "Differences are edge-specific assignment minus the named control; "
            "AUC integrates over p_HK from 0 to 1. Positive order-parameter "
            "differences favor edge-specific assignment.\n\n")
        handle.write(
            "| Control | Outcome | Mean AUC difference | 95% CI | "
            "Holm-adjusted p | Paired dz |\n"
            "|---|---|---:|---:|---:|---:|\n")
        for row in global_rows:
            if row["metric"] not in (
                    "order_param", "n_clusters", "variance"):
                continue
            handle.write(
                "| {} | {} | {:.4f} | [{:.4f}, {:.4f}] | {:.3g} | {} |\n"
                .format(
                    MODEL_LABELS[row["comparator"]],
                    METRIC_LABELS[row["metric"]],
                    row["mean_auc_difference"],
                    row["ci95_low"], row["ci95_high"],
                    row["paired_t_p_holm_12"],
                    "NA" if row["cohen_dz"] is None
                    else "{:.3f}".format(row["cohen_dz"])))
        handle.write(
            "\nThese are matched-seed contrasts under the stated simulation "
            "protocol. They quantify whether the location of a threshold on "
            "the network matters beyond its marginal distribution; they do "
            "not establish a causal effect beyond this model and parameter "
            "range. A non-significant result is not an equivalence claim.\n\n")
        handle.write(
            "Source data and pointwise/global contrast tables are saved as "
            "CSV files beside this report. The figure shows terminal order "
            "parameter and paired order-parameter and cluster-count contrasts.\n")

    print("Validation:", validation["status"])
    if validation["errors"]:
        for error in validation["errors"][:30]:
            print(" -", error)
        raise SystemExit(1)
    print("Wrote", json_path)
    print("Wrote", base.with_suffix(".pdf"))
    print("Wrote", report_path)


def main():
    with DATA_FILE.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    with META_FILE.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    validation = validate(payload, metadata)
    analyze(payload, metadata, validation)


if __name__ == "__main__":
    main()
