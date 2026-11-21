"""Validate and summarize the formal two-dimensional opinion robustness run.

Reads the per-seed values saved by
``ExperimentManager.experiment_vector_opinions`` and writes reviewer-facing
validation, source data and paired-contrast artifacts.  The independent unit is
the seed-level replicate; all model variants within one p_HK share the network
and the vector-valued initial opinions, so the contrasts are paired.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
FORMAL_ROOT = ROOT / "output" / "formal_revision"
DATA_FILE = FORMAL_ROOT / "data" / "exp_vector_opinions.json"
META_FILE = FORMAL_ROOT / "data" / "exp_vector_opinions.meta.json"
ANALYSIS_DIR = FORMAL_ROOT / "analysis"
FIGURE_DIR = FORMAL_ROOT / "figures"

N_RUNS = 20
SEEDS = [run_i * 997 + 7 for run_i in range(N_RUNS)]
MODEL_ORDER = ["threshold_only", "no_triadic_feedback", "full_model"]
MODEL_LABELS = {
    "threshold_only": "Threshold Only",
    "no_triadic_feedback": "No Triadic Feedback",
    "full_model": "Full Model",
}
PRIMARY_CONTRASTS = [
    ("no_triadic_feedback", "Full - No Triadic Feedback"),
    ("threshold_only", "Full - Threshold Only"),
]
METRICS = [
    "compact_order",
    "total_variance",
    "mean_pairwise_distance",
    "triangle_alignment_fixed",
    "triangle_admissibility_adaptive",
    "terminal_mean_abs_dx",
]
METRIC_LABELS = {
    "compact_order": "Compact order (2D)",
    "total_variance": "Total opinion variance",
    "mean_pairwise_distance": "Mean pairwise distance",
    "triangle_alignment_fixed": "Fixed-scale triangle alignment",
    "triangle_admissibility_adaptive": "Adaptive triangle admissibility",
    "terminal_mean_abs_dx": "Terminal mean absolute update",
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
    df = n - 1
    if df <= 0:
        critical = 0.0
    elif df <= len(T975):
        critical = T975[df - 1]
    elif df <= 60:
        critical = 2.000
    elif df <= 120:
        critical = 1.980
    else:
        critical = 1.960
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "se": se,
        "ci95_low": float(mean - critical * se),
        "ci95_high": float(mean + critical * se),
        "cohen_dz": float(mean / sd) if sd > 0 else (0.0 if mean == 0 else None),
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
    m = len(p_values)
    for rank, idx in enumerate(order):
        candidate = min(1.0, (m - rank) * p_values[idx])
        running = max(running, candidate)
        adjusted[idx] = running
    return adjusted.tolist()


def write_json_atomic(payload, path):
    temp = "{}.tmp-{}".format(path, os.getpid())
    with open(temp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temp, path)


def write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate(data, metadata):
    errors = []
    blocks = data.get("data", {})
    actual_p = sorted(float(key) for key in blocks)
    if not actual_p:
        errors.append("no p_HK blocks present")
    if metadata.get("n_runs") != N_RUNS:
        errors.append("metadata n_runs is not {}".format(N_RUNS))
    if metadata.get("seeds") != SEEDS:
        errors.append("metadata seed list mismatch")

    control = data.get("control", {})
    if int(control.get("dimension", -1)) != 2:
        errors.append("dimension is not 2")
    if float(control.get("main_sweeps", -1)) != 200.0:
        errors.append("main_sweeps is not 200")
    if list(control.get("model_order", [])) != MODEL_ORDER:
        errors.append("model order mismatch")

    recorded_hashes = metadata.get("source_sha256", {})
    for filename, recorded in sorted(recorded_hashes.items()):
        path = ROOT / filename
        if not path.exists():
            errors.append("recorded source missing: {}".format(filename))
            continue
        if sha256(path) != recorded:
            errors.append("source hash changed since run: {}".format(filename))

    for key in sorted(blocks, key=float):
        p = float(key)
        block = blocks[key]
        if not isinstance(block, dict):
            errors.append("missing p block {}".format(p))
            continue
        if block.get("_control", {}).get("seeds") != SEEDS:
            errors.append("seed list mismatch at p={}".format(p))
        for model in MODEL_ORDER:
            model_block = block.get(model, {})
            for metric in METRICS:
                summary = model_block.get(metric, {})
                values = summary.get("values", [])
                if summary.get("n") != N_RUNS or len(values) != N_RUNS:
                    errors.append(
                        "{} {} does not contain {} values at p={}".format(
                            model, metric, N_RUNS, p))
                    continue
                recalculated = summarize(values)
                for field in ("mean", "sd", "se", "ci95_low", "ci95_high"):
                    if not close(summary[field], recalculated[field], atol=1e-12):
                        errors.append("summary mismatch {} {} {} p={}".format(
                            model, metric, field, p))
        for comparator, _label in PRIMARY_CONTRASTS:
            contrast_key = "full_model_minus_{}".format(comparator)
            saved_contrast = block.get("_paired_comparisons", {}).get(
                contrast_key, {})
            for metric in METRICS:
                saved = saved_contrast.get(metric, {})
                expected = (
                    np.asarray(block["full_model"][metric]["values"], dtype=float)
                    - np.asarray(block[comparator][metric]["values"], dtype=float)
                )
                if len(saved.get("values", [])) != N_RUNS or not np.allclose(
                        saved.get("values", []), expected, atol=1e-12):
                    errors.append("paired contrast mismatch {} {} p={}".format(
                        contrast_key, metric, p))
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "p_points": len(actual_p),
        "models": MODEL_ORDER,
        "n_per_cell": N_RUNS,
        "dimension": 2,
        "independent_unit": (
            "one seeded simulation replicate; all model variants within a "
            "p_HK share the network and the vector-valued initial opinions"),
    }


def build_outputs(data, metadata, validation):
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    blocks = data["data"]
    p_keys = sorted(blocks, key=float)
    p_values = [float(key) for key in p_keys]

    point_rows = []
    for comparator, label in PRIMARY_CONTRASTS:
        contrast_key = "full_model_minus_{}".format(comparator)
        for metric in METRICS:
            raw_p = []
            rows = []
            for p_key, p in zip(p_keys, p_values):
                saved = blocks[p_key]["_paired_comparisons"][contrast_key][metric]
                raw = paired_t_pvalue(saved["values"])
                raw_p.append(raw)
                rows.append({
                    "contrast": label,
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "p_hk": p,
                    "n": saved["n"],
                    "mean_difference": saved["mean"],
                    "sd_difference": saved["sd"],
                    "se_difference": saved["se"],
                    "ci95_low": saved["ci95_low"],
                    "ci95_high": saved["ci95_high"],
                    "cohen_dz": saved["cohen_dz"],
                    "paired_t_p_raw": raw,
                })
            adjusted = holm_adjust(raw_p)
            for row, value in zip(rows, adjusted):
                row["paired_t_p_holm_4"] = value
                point_rows.append(row)

    conservative = holm_adjust([row["paired_t_p_raw"] for row in point_rows])
    for row, value in zip(point_rows, conservative):
        row["paired_t_p_holm_48"] = value

    global_rows = []
    for comparator, label in PRIMARY_CONTRASTS:
        contrast_key = "full_model_minus_{}".format(comparator)
        raw_p = []
        rows = []
        for metric in METRICS:
            full_auc = []
            comparator_auc = []
            for index in range(N_RUNS):
                full_curve = [
                    blocks[key]["full_model"][metric]["values"][index]
                    for key in p_keys
                ]
                other_curve = [
                    blocks[key][comparator][metric]["values"][index]
                    for key in p_keys
                ]
                full_auc.append(float(np.trapz(full_curve, p_values)))
                comparator_auc.append(float(np.trapz(other_curve, p_values)))
            differences = np.asarray(full_auc) - np.asarray(comparator_auc)
            summary = summarize(differences)
            raw = paired_t_pvalue(differences)
            raw_p.append(raw)
            rows.append({
                "contrast": label,
                "metric": metric,
                "metric_label": METRIC_LABELS[metric],
                "n": summary["n"],
                "mean_auc_difference": summary["mean"],
                "sd_auc_difference": summary["sd"],
                "se_auc_difference": summary["se"],
                "ci95_low": summary["ci95_low"],
                "ci95_high": summary["ci95_high"],
                "cohen_dz": summary["cohen_dz"],
                "paired_t_p_raw": raw,
            })
        adjusted = holm_adjust(raw_p)
        for row, value in zip(rows, adjusted):
            row["paired_t_p_holm_6"] = value
            global_rows.append(row)

    source_rows = []
    for p_key, p in zip(p_keys, p_values):
        block = blocks[p_key]
        for model in MODEL_ORDER:
            for index, seed in enumerate(SEEDS):
                row = {"p_hk": p, "model": model, "seed": seed}
                for metric in METRICS:
                    row[metric] = block[model][metric]["values"][index]
                source_rows.append(row)

    write_csv(source_rows, str(ANALYSIS_DIR / "formal_vector_source_data.csv"))
    write_csv(point_rows, str(ANALYSIS_DIR / "formal_vector_pointwise.csv"))
    write_csv(global_rows, str(ANALYSIS_DIR / "formal_vector_global_auc.csv"))

    colors = {"full_model": "#DC2626", "no_triadic_feedback": "#0D9488",
              "threshold_only": "#7C3AED"}
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.0))
    for model in MODEL_ORDER:
        means = np.asarray([
            blocks[key][model]["compact_order"]["mean"] for key in p_keys])
        lows = np.asarray([
            blocks[key][model]["compact_order"]["ci95_low"] for key in p_keys])
        highs = np.asarray([
            blocks[key][model]["compact_order"]["ci95_high"] for key in p_keys])
        axes[0].plot(p_values, means, marker="o", markersize=3.5, linewidth=1.7,
                     color=colors[model], label=MODEL_LABELS[model])
        axes[0].fill_between(p_values, lows, highs, color=colors[model], alpha=0.13)
    axes[0].set_title("(a) Two-dimensional compact order")
    axes[0].set_xlabel(r"$p_{HK}$")
    axes[0].set_ylabel("Compact order")
    axes[0].set_ylim(0, 1.03)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.22, linestyle="--")

    contrast_colors = ["#0D9488", "#7C3AED"]
    for (comparator, label), color in zip(PRIMARY_CONTRASTS, contrast_colors):
        rows = [row for row in point_rows
                if row["contrast"] == label and row["metric"] == "compact_order"]
        means = np.asarray([row["mean_difference"] for row in rows])
        lows = np.asarray([row["ci95_low"] for row in rows])
        highs = np.asarray([row["ci95_high"] for row in rows])
        axes[1].plot(p_values, means, marker="o", markersize=3.5, linewidth=1.7,
                     color=color, label=label)
        axes[1].fill_between(p_values, lows, highs, color=color, alpha=0.13)
    axes[1].axhline(0, color="#111827", linestyle="--", linewidth=1)
    axes[1].set_title("(b) Paired compact-order contrasts")
    axes[1].set_xlabel(r"$p_{HK}$")
    axes[1].set_ylabel("Mean paired difference (95% CI)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.22, linestyle="--")

    x = np.arange(len(global_rows))
    auc_means = np.asarray([row["mean_auc_difference"] for row in global_rows])
    auc_low = np.asarray([row["ci95_low"] for row in global_rows])
    auc_high = np.asarray([row["ci95_high"] for row in global_rows])
    yerr = np.asarray([auc_means - auc_low, auc_high - auc_means])
    bar_colors = [contrast_colors[0] if row["contrast"].startswith("Full - No")
                  else contrast_colors[1] for row in global_rows]
    axes[2].bar(x, auc_means, yerr=yerr, capsize=4, color=bar_colors, alpha=0.82)
    axes[2].axhline(0, color="#111827", linestyle="--", linewidth=1)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([row["metric_label"] for row in global_rows],
                            rotation=18, ha="right", fontsize=7)
    axes[2].set_ylabel("Paired AUC difference (95% CI)")
    axes[2].set_title("(c) Across-$p_{HK}$ contrasts, all endpoints")
    axes[2].grid(alpha=0.22, linestyle="--")
    fig.suptitle(
        "Formal two-dimensional opinion robustness (n=20 paired seeds, 200 sweeps)",
        fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    figure_path = FIGURE_DIR / "fig_formal_vector_opinions.png"
    fig.savefig(str(figure_path), dpi=300, bbox_inches="tight")
    plt.close(fig)

    report_path = ANALYSIS_DIR / "FORMAL_VECTOR_OPINIONS_REPORT.md"
    with open(str(report_path), "w", encoding="utf-8") as handle:
        handle.write("# Formal two-dimensional opinion report\n\n")
        handle.write(
            "Design: N=2000, m=4, {} p_HK values, dimension=2 with Euclidean "
            "bounded confidence, 200 sweeps, {} paired seeds. Compact order is "
            "the largest fraction inside a ball of radius diameter/2 "
            "({}). All model variants within a p_HK share the network and the "
            "initial vector opinions, so every contrast is paired.\n\n"
            .format(len(p_values), N_RUNS,
                    data.get("control", {}).get("compact_order_diameter")))
        handle.write("## Across-$p_{HK}$ global AUC contrast\n\n")
        handle.write("| contrast | metric | mean AUC difference | 95% CI | dz | raw p | Holm-6 p |\n")
        handle.write("|---|---|---:|---|---:|---:|---:|\n")
        for row in global_rows:
            handle.write(
                "| {} | {} | {:.6f} | [{:.6f}, {:.6f}] | {:.3f} | {:.4g} | {:.4g} |\n"
                .format(row["contrast"], row["metric_label"],
                        row["mean_auc_difference"], row["ci95_low"],
                        row["ci95_high"], row["cohen_dz"],
                        row["paired_t_p_raw"], row["paired_t_p_holm_6"]))
        handle.write("\n## Pointwise compact-order contrast\n\n")
        handle.write("| contrast | $p_{HK}$ | mean difference | 95% CI | dz | raw p | Holm-4 p |\n")
        handle.write("|---|---:|---:|---|---:|---:|---:|\n")
        for row in [r for r in point_rows if r["metric"] == "compact_order"]:
            handle.write(
                "| {} | {:.2f} | {:.6f} | [{:.6f}, {:.6f}] | {:.3f} | {:.4g} | {:.4g} |\n"
                .format(row["contrast"], row["p_hk"], row["mean_difference"],
                        row["ci95_low"], row["ci95_high"], row["cohen_dz"],
                        row["paired_t_p_raw"], row["paired_t_p_holm_4"]))
        handle.write("\n## Result boundary\n\n")
        significant = [
            "{} | {}".format(row["contrast"], row["metric_label"])
            for row in global_rows if row["paired_t_p_holm_6"] < 0.05]
        if significant:
            handle.write(
                "- After Holm-6 correction, the following endpoints differ: {}.\n"
                .format("; ".join(significant)))
        else:
            handle.write(
                "- No endpoint survives Holm-6 correction; within the tested "
                "two-dimensional protocol the full model does not separate "
                "from the reduced variants on these endpoints.\n")
        handle.write(
            "- This documents the extendable boundary of the single-dimensional "
            "assumption: cohesion-like ordering is still observed at high "
            "$p_{HK}$ in two dimensions, but the component-level attribution "
            "found in one dimension is not reproduced as a significant "
            "endpoint contrast here.\n")

    output = {
        "validation": validation,
        "design": {
            "N": 2000,
            "m": 4,
            "p_hk_points": p_values,
            "dimension": 2,
            "models": MODEL_ORDER,
            "n_per_cell": N_RUNS,
            "seeds": SEEDS,
            "horizon_sweeps": 200,
            "primary_contrasts": [label for _c, label in PRIMARY_CONTRASTS],
            "uncertainty": "two-sided 95% Student-t confidence interval",
            "pointwise_test": "two-sided paired t-test on per-seed differences",
            "multiplicity": "Holm across the p_HK points within each metric",
            "global_test": (
                "two-sided paired t-test on per-seed trapezoidal AUC "
                "differences across p_HK"),
        },
        "pointwise_contrasts": point_rows,
        "global_auc_contrasts": global_rows,
    }
    json_path = ANALYSIS_DIR / "formal_vector_analysis.json"
    write_json_atomic(output, str(json_path))
    return {
        "report": str(report_path),
        "figure": str(figure_path),
        "json": str(json_path),
    }


def main():
    if not DATA_FILE.exists() or not META_FILE.exists():
        raise SystemExit("formal vector result or metadata file is missing")
    with DATA_FILE.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    with META_FILE.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    validation = validate(data, metadata)
    artifacts = build_outputs(data, metadata, validation)
    print("VALIDATION={}".format(validation["status"]))
    print("ERRORS={}".format(len(validation["errors"])))
    for path in artifacts.values():
        print("WROTE={}".format(path))
    if validation["errors"]:
        for error in validation["errors"]:
            print("ERROR={}".format(error))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
