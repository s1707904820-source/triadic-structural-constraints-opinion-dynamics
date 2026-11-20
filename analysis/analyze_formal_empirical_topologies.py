"""Validate and summarize the fixed empirical-topology stress test.

Reads the per-seed values saved by
``ExperimentManager.experiment_empirical_topologies``.  Each benchmark supplies
exactly one fixed topology, so the uncertainty reported here reflects initial
opinions and update sequences only, not variation across independent empirical
networks.  This is a topology stress test, not a validation against observed
opinion behaviour.
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
DATA_FILE = FORMAL_ROOT / "data" / "exp_empirical_topologies.json"
META_FILE = FORMAL_ROOT / "data" / "exp_empirical_topologies.meta.json"
ANALYSIS_DIR = FORMAL_ROOT / "analysis"
FIGURE_DIR = FORMAL_ROOT / "figures"

N_RUNS = 20
SEEDS = [run_i * 997 + 7 for run_i in range(N_RUNS)]
TOPOLOGIES = ["karate", "davis", "florentine"]
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
    "order_param",
    "n_clusters",
    "variance",
    "triangle_alignment_fixed",
    "triangle_admissibility_adaptive",
    "edge_disagreement_penalty",
]
METRIC_LABELS = {
    "order_param": "Order parameter",
    "n_clusters": "Number of clusters",
    "variance": "Opinion variance",
    "triangle_alignment_fixed": "Fixed-scale triangle alignment",
    "triangle_admissibility_adaptive": "Adaptive triangle admissibility",
    "edge_disagreement_penalty": "Mean edge-disagreement penalty",
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


def fmt(value, spec=".3f"):
    """Format a possibly-None numeric value without raising."""
    if value is None:
        return "NA"
    return format(float(value), spec)


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
    control = data.get("control", {})
    blocks = data.get("data", {})
    actual = sorted(blocks)
    if actual != sorted(TOPOLOGIES):
        errors.append("topology set mismatch: {}".format(actual))
    if int(control.get("num_runs_per_fixed_topology", -1)) != N_RUNS:
        errors.append("num_runs_per_fixed_topology is not {}".format(N_RUNS))
    if int(control.get("topology_replicates_per_benchmark", -1)) != 1:
        errors.append("topology_replicates_per_benchmark is not 1")
    if float(control.get("main_sweeps", -1)) != 200.0:
        errors.append("main_sweeps is not 200")
    if list(control.get("model_order", [])) != MODEL_ORDER:
        errors.append("model order mismatch")
    if metadata.get("n_runs") != N_RUNS:
        errors.append("metadata n_runs is not {}".format(N_RUNS))
    if metadata.get("seeds") != SEEDS:
        errors.append("metadata seed list mismatch")

    recorded_hashes = metadata.get("source_sha256", {})
    for filename, recorded in sorted(recorded_hashes.items()):
        path = ROOT / filename
        if not path.exists():
            errors.append("recorded source missing: {}".format(filename))
            continue
        if sha256(path) != recorded:
            errors.append("source hash changed since run: {}".format(filename))

    for key in sorted(blocks):
        block = blocks[key]
        network = block.get("network", {})
        if int(network.get("num_nodes", 0)) <= 0:
            errors.append("missing node count for {}".format(key))
        if int(network.get("topology_replicates", 0)) != 1:
            errors.append("topology_replicates is not 1 for {}".format(key))
        if block.get("_control", {}).get("seeds") != SEEDS:
            errors.append("seed list mismatch for {}".format(key))
        for model in MODEL_ORDER:
            model_block = block.get(model, {})
            for metric in METRICS:
                summary = model_block.get(metric, {})
                values = summary.get("values", [])
                if summary.get("n") != N_RUNS or len(values) != N_RUNS:
                    errors.append("{} {} {} does not contain {} values".format(
                        key, model, metric, N_RUNS))
                    continue
                recalculated = summarize(values)
                for field in ("mean", "sd", "se", "ci95_low", "ci95_high"):
                    if not close(summary[field], recalculated[field], atol=1e-12):
                        errors.append("summary mismatch {} {} {} {}".format(
                            key, model, metric, field))
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
                    errors.append("paired contrast mismatch {} {} {}".format(
                        contrast_key, metric, key))
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "topologies": actual,
        "n_per_cell": N_RUNS,
        "topology_replicates_per_benchmark": 1,
        "independent_unit": (
            "one seeded simulation replicate on one fixed topology; the "
            "uncertainty reflects initial opinions and update sequences, not "
            "variation across independent empirical networks"),
    }


def build_outputs(data, metadata, validation):
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    blocks = data["data"]
    names = [name for name in TOPOLOGIES if name in blocks]

    network_rows = []
    for name in names:
        network = blocks[name]["network"]
        network_rows.append({
            "topology": name,
            "citation_label": network.get("citation_label"),
            "num_nodes": network.get("num_nodes"),
            "num_edges": network.get("num_edges"),
            "avg_degree": network.get("avg_degree"),
            "avg_clustering": network.get("avg_clustering"),
            "global_clustering": network.get("global_clustering"),
            "num_triangles": network.get("num_triangles"),
            "f_triangle": network.get("f_triangle"),
            "modularity": network.get("modularity"),
            "canonical_edge_sha256": network.get("canonical_edge_sha256"),
        })

    model_rows = []
    for name in names:
        block = blocks[name]
        for model in MODEL_ORDER:
            for metric in METRICS:
                summary = block[model][metric]
                model_rows.append({
                    "topology": name,
                    "model": model,
                    "metric": metric,
                    "n": summary["n"],
                    "mean": summary["mean"],
                    "sd": summary["sd"],
                    "se": summary["se"],
                    "ci95_low": summary["ci95_low"],
                    "ci95_high": summary["ci95_high"],
                })

    point_rows = []
    for comparator, label in PRIMARY_CONTRASTS:
        contrast_key = "full_model_minus_{}".format(comparator)
        for metric in METRICS:
            raw_p = []
            rows = []
            for name in names:
                saved = blocks[name]["_paired_comparisons"][contrast_key][metric]
                raw = paired_t_pvalue(saved["values"])
                raw_p.append(raw)
                rows.append({
                    "contrast": label,
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "topology": name,
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
                row["paired_t_p_holm_3"] = value
                point_rows.append(row)

    source_rows = []
    for name in names:
        block = blocks[name]
        for model in MODEL_ORDER:
            for index, seed in enumerate(SEEDS):
                row = {"topology": name, "model": model, "seed": seed}
                for metric in METRICS:
                    row[metric] = block[model][metric]["values"][index]
                source_rows.append(row)

    write_csv(network_rows, str(ANALYSIS_DIR / "formal_empirical_network_summary.csv"))
    write_csv(model_rows, str(ANALYSIS_DIR / "formal_empirical_model_summary.csv"))
    write_csv(point_rows, str(ANALYSIS_DIR / "formal_empirical_pointwise.csv"))
    write_csv(source_rows, str(ANALYSIS_DIR / "formal_empirical_source_data.csv"))

    colors = {"full_model": "#DC2626", "no_triadic_feedback": "#0D9488",
              "threshold_only": "#7C3AED"}
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.0))
    x = np.arange(len(names))
    width = 0.26
    for offset, model in enumerate(MODEL_ORDER):
        means = [blocks[name][model]["order_param"]["mean"] for name in names]
        lows = [blocks[name][model]["order_param"]["ci95_low"] for name in names]
        highs = [blocks[name][model]["order_param"]["ci95_high"] for name in names]
        yerr = np.asarray([
            np.asarray(means) - np.asarray(lows),
            np.asarray(highs) - np.asarray(means),
        ])
        axes[0].bar(x + (offset - 1) * width, means, width, yerr=yerr,
                    capsize=4, color=colors[model], alpha=0.85,
                    label=MODEL_LABELS[model])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names)
    axes[0].set_ylabel("Order parameter (95% CI)")
    axes[0].set_title("(a) Terminal order parameter on fixed topologies")
    axes[0].set_ylim(0, 1.0)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.22, linestyle="--")

    for offset, model in enumerate(MODEL_ORDER):
        means = [blocks[name][model]["triangle_alignment_fixed"]["mean"]
                 for name in names]
        lows = [blocks[name][model]["triangle_alignment_fixed"]["ci95_low"]
                for name in names]
        highs = [blocks[name][model]["triangle_alignment_fixed"]["ci95_high"]
                 for name in names]
        yerr = np.asarray([
            np.asarray(means) - np.asarray(lows),
            np.asarray(highs) - np.asarray(means),
        ])
        axes[1].bar(x + (offset - 1) * width, means, width, yerr=yerr,
                    capsize=4, color=colors[model], alpha=0.85,
                    label=MODEL_LABELS[model])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names)
    axes[1].set_ylabel("Fixed-scale triangle alignment (95% CI)")
    axes[1].set_title("(b) Triangle alignment on fixed topologies")
    axes[1].set_ylim(0, 1.0)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.22, linestyle="--")

    rows = [row for row in point_rows if row["metric"] == "order_param"]
    labels = ["{}\n{}".format(row["topology"], row["contrast"])
              for row in rows]
    means = np.asarray([row["mean_difference"] for row in rows])
    lows = np.asarray([row["ci95_low"] for row in rows])
    highs = np.asarray([row["ci95_high"] for row in rows])
    yerr = np.asarray([means - lows, highs - means])
    bar_colors = ["#0D9488" if row["contrast"].startswith("Full - No")
                  else "#7C3AED" for row in rows]
    axes[2].bar(np.arange(len(rows)), means, yerr=yerr, capsize=4,
                color=bar_colors, alpha=0.85)
    axes[2].axhline(0, color="#111827", linestyle="--", linewidth=1)
    axes[2].set_xticks(np.arange(len(rows)))
    axes[2].set_xticklabels(labels, fontsize=7)
    axes[2].set_ylabel("Mean paired difference (95% CI)")
    axes[2].set_title("(c) Paired order-parameter contrasts")
    axes[2].grid(alpha=0.22, linestyle="--")
    fig.suptitle(
        "Fixed empirical-topology stress test (n=20 seeds per topology, 200 sweeps)",
        fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    figure_path = FIGURE_DIR / "fig_formal_empirical_topologies.png"
    fig.savefig(str(figure_path), dpi=300, bbox_inches="tight")
    plt.close(fig)

    report_path = ANALYSIS_DIR / "FORMAL_EMPIRICAL_TOPOLOGIES_REPORT.md"
    with open(str(report_path), "w", encoding="utf-8") as handle:
        handle.write("# Formal empirical-topology stress test report\n\n")
        handle.write(
            "Design: {} fixed public topologies (one network each), {} seeds "
            "per topology, 200 sweeps, full model plus two reduced variants. "
            "Opinions are still simulated: the topologies are empirical, the "
            "opinion dynamics are not.\n\n".format(len(names), N_RUNS))
        handle.write(
            "**Scope limitation.** Each benchmark contributes exactly one "
            "network (`topology_replicates_per_benchmark = 1`). The reported "
            "uncertainty therefore reflects initial opinions and update "
            "sequences only, not variation across independent empirical "
            "networks. This is a topology stress test, not a validation "
            "against observed opinion behaviour, and it does not substitute "
            "for a real-world social-network dataset.\n\n")
        handle.write("## Network properties\n\n")
        handle.write("| topology | nodes | edges | avg degree | avg clustering | triangles | f_triangle |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in network_rows:
            handle.write(
                "| {} | {} | {} | {:.3f} | {:.4f} | {} | {:.4f} |\n".format(
                    row["topology"], row["num_nodes"], row["num_edges"],
                    row["avg_degree"], row["avg_clustering"],
                    row["num_triangles"], row["f_triangle"]))
        handle.write("\n## Terminal order parameter (mean, 95% CI)\n\n")
        handle.write("| topology | Threshold Only | No Triadic Feedback | Full Model |\n")
        handle.write("|---|---:|---:|---:|\n")
        for name in names:
            cells = []
            for model in MODEL_ORDER:
                summary = blocks[name][model]["order_param"]
                cells.append("{:.4f} [{:.4f}, {:.4f}]".format(
                    summary["mean"], summary["ci95_low"], summary["ci95_high"]))
            handle.write("| {} | {} |\n".format(name, " | ".join(cells)))
        handle.write(
            "\n## Paired contrasts (Holm correction across the {} "
            "topologies within each metric)\n\n".format(len(names)))
        handle.write("| contrast | metric | topology | mean difference | 95% CI | dz | raw p | Holm-3 p |\n")
        handle.write("|---|---|---|---:|---|---:|---:|---:|\n")
        for row in point_rows:
            handle.write(
                "| {} | {} | {} | {:.6f} | [{:.6f}, {:.6f}] | {} | {:.4g} | {:.4g} |\n"
                .format(row["contrast"], row["metric_label"], row["topology"],
                        row["mean_difference"], row["ci95_low"],
                        row["ci95_high"], fmt(row["cohen_dz"]),
                        row["paired_t_p_raw"], row["paired_t_p_holm_3"]))
        handle.write("\n## Result boundary\n\n")
        significant = [
            "{} | {} | {}".format(row["contrast"], row["metric_label"],
                                  row["topology"])
            for row in point_rows if row["paired_t_p_holm_3"] < 0.05]
        if significant:
            handle.write(
                "- After Holm-3 correction, the following contrasts differ: {}.\n"
                .format("; ".join(significant)))
        else:
            handle.write(
                "- No contrast survives Holm-3 correction on these small fixed "
                "topologies; the mechanism attribution cannot be established "
                "from this stress test alone.\n")
        handle.write(
            "- Do not describe this experiment as real-world opinion "
            "validation. It only shows that the dynamics can be executed on "
            "empirical topologies without breaking the implementation.\n")

    output = {
        "validation": validation,
        "design": {
            "topologies": names,
            "networks_per_topology": 1,
            "n_runs_per_topology": N_RUNS,
            "seeds": SEEDS,
            "horizon_sweeps": 200,
            "models": MODEL_ORDER,
            "primary_contrasts": [label for _c, label in PRIMARY_CONTRASTS],
            "uncertainty": "two-sided 95% Student-t confidence interval",
            "pointwise_test": "two-sided paired t-test on per-seed differences",
            "multiplicity": "Holm across the topologies within each metric",
            "scope": (
                "topology stress test only; not a real-world opinion "
                "validation, and uncertainty excludes network-to-sample "
                "variation"),
        },
        "network_summary": network_rows,
        "pointwise_contrasts": point_rows,
    }
    json_path = ANALYSIS_DIR / "formal_empirical_analysis.json"
    write_json_atomic(output, str(json_path))
    return {
        "report": str(report_path),
        "figure": str(figure_path),
        "json": str(json_path),
    }


def main():
    if not DATA_FILE.exists() or not META_FILE.exists():
        raise SystemExit("formal empirical result or metadata file is missing")
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
