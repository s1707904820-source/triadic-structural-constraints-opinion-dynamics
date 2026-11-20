"""Validate and summarize the formal asynchronous/synchronous update control.

Reads the per-seed paired differences saved by
``ExperimentManager.experiment_update_mode_control`` and writes reviewer-facing
validation, source data and paired-contrast artifacts.  The independent unit is
the seed-level simulation replicate; the two update modes share the same
network and initial-opinion seed, so the contrast is paired.
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
DATA_FILE = FORMAL_ROOT / "data" / "exp_update_mode_control.json"
META_FILE = FORMAL_ROOT / "data" / "exp_update_mode_control.meta.json"
ANALYSIS_DIR = FORMAL_ROOT / "analysis"
FIGURE_DIR = FORMAL_ROOT / "figures"

N_RUNS = 20
SEEDS = [run_i * 997 + 7 for run_i in range(N_RUNS)]
P_VALUES = [0.0, 0.4, 0.8, 1.0]
MODES = ("asynchronous", "synchronous")
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
    actual_p = sorted(float(key) for key in data.get("data", {}))
    if actual_p != sorted(P_VALUES):
        errors.append("p_HK grid mismatch: {}".format(actual_p))

    control = metadata.get("config", {}).get("SIMULATION", {})
    if metadata.get("n_runs") != N_RUNS:
        errors.append("metadata n_runs is not {}".format(N_RUNS))
    if metadata.get("seeds") != SEEDS:
        errors.append("metadata seed list mismatch")
    if float(control.get("main_sweeps", -1)) != 200.0:
        errors.append("main_sweeps is not 200")
    if float(control.get("record_sweeps", -1)) != 5.0:
        errors.append("record_sweeps is not 5")

    recorded_hashes = metadata.get("source_sha256", {})
    for filename, recorded in sorted(recorded_hashes.items()):
        path = ROOT / filename
        if not path.exists():
            errors.append("recorded source missing: {}".format(filename))
            continue
        if sha256(path) != recorded:
            errors.append("source hash changed since run: {}".format(filename))

    data_blocks = data.get("data", {})
    for key in sorted(data_blocks, key=float):
        p = float(key)
        block = data_blocks[key]
        if not isinstance(block, dict):
            errors.append("missing p block {}".format(p))
            continue
        block_control = block.get("_control", {})
        if block_control.get("seeds") != SEEDS:
            errors.append("seed list mismatch at p={}".format(p))
        for mode in MODES:
            mode_block = block.get(mode, {})
            for metric in METRICS:
                summary = mode_block.get(metric, {})
                values = summary.get("values", [])
                if summary.get("n") != N_RUNS or len(values) != N_RUNS:
                    errors.append(
                        "{} {} does not contain {} values at p={}".format(
                            mode, metric, N_RUNS, p))
                    continue
                recalculated = summarize(values)
                for field in ("mean", "sd", "se", "ci95_low", "ci95_high"):
                    if not close(summary[field], recalculated[field], atol=1e-12):
                        errors.append(
                            "summary mismatch {} {} {} p={}".format(
                                mode, metric, field, p))
        paired = block.get("paired_synchronous_minus_asynchronous", {})
        for metric in METRICS:
            saved = paired.get(metric, {})
            expected = (
                np.asarray(block["synchronous"][metric]["values"], dtype=float)
                - np.asarray(block["asynchronous"][metric]["values"], dtype=float)
            )
            if len(saved.get("values", [])) != N_RUNS or not np.allclose(
                    saved.get("values", []), expected, atol=1e-12):
                errors.append(
                    "paired contrast mismatch {} p={}".format(metric, p))

    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "p_points": len(actual_p),
        "modes": list(MODES),
        "n_per_cell": N_RUNS,
        "independent_unit": (
            "one seeded simulation replicate; the asynchronous and synchronous "
            "arms share the network and initial-opinion seed"),
    }


def build_outputs(data, metadata, validation):
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    blocks = data["data"]
    p_keys = sorted(blocks, key=float)
    p_values = [float(key) for key in p_keys]

    # ---- pointwise paired table (Holm across the 4 p_HK points per metric) ----
    point_rows = []
    for metric in METRICS:
        raw_p = []
        rows = []
        for p_key, p in zip(p_keys, p_values):
            saved = blocks[p_key]["paired_synchronous_minus_asynchronous"][metric]
            raw = paired_t_pvalue(saved["values"])
            raw_p.append(raw)
            rows.append({
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
        row["paired_t_p_holm_24"] = value

    # ---- across-p_HK global AUC contrast, Holm across the six metrics ----
    global_rows = []
    global_p = []
    for metric in METRICS:
        sync_auc = []
        async_auc = []
        for index in range(N_RUNS):
            sync_curve = [
                blocks[key]["synchronous"][metric]["values"][index]
                for key in p_keys
            ]
            async_curve = [
                blocks[key]["asynchronous"][metric]["values"][index]
                for key in p_keys
            ]
            sync_auc.append(float(np.trapz(sync_curve, p_values)))
            async_auc.append(float(np.trapz(async_curve, p_values)))
        differences = np.asarray(sync_auc) - np.asarray(async_auc)
        summary = summarize(differences)
        raw = paired_t_pvalue(differences)
        global_p.append(raw)
        global_rows.append({
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
    adjusted_6 = holm_adjust(global_p)
    for row, value in zip(global_rows, adjusted_6):
        row["paired_t_p_holm_6"] = value

    source_rows = []
    for p_key, p in zip(p_keys, p_values):
        block = blocks[p_key]
        for mode in MODES:
            for index, seed in enumerate(SEEDS):
                row = {"p_hk": p, "mode": mode, "seed": seed}
                for metric in METRICS:
                    row[metric] = block[mode][metric]["values"][index]
                source_rows.append(row)

    write_csv(source_rows, str(ANALYSIS_DIR / "formal_update_mode_source_data.csv"))
    write_csv(point_rows, str(ANALYSIS_DIR / "formal_update_mode_pointwise.csv"))
    write_csv(global_rows, str(ANALYSIS_DIR / "formal_update_mode_global_auc.csv"))

    # ---- figure ----
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    for metric in ("order_param", "n_clusters", "variance"):
        rows = [row for row in point_rows if row["metric"] == metric]
        means = np.asarray([row["mean_difference"] for row in rows])
        lows = np.asarray([row["ci95_low"] for row in rows])
        highs = np.asarray([row["ci95_high"] for row in rows])
        axes[0].plot(p_values, means, marker="o", markersize=4,
                     linewidth=1.6, label=METRIC_LABELS[metric])
        axes[0].fill_between(p_values, lows, highs, alpha=0.12)
    axes[0].axhline(0, color="#111827", linestyle="--", linewidth=1)
    axes[0].set_xlabel(r"$p_{HK}$")
    axes[0].set_ylabel("Mean paired difference (95% CI)")
    axes[0].set_title("(a) Synchronous - asynchronous, per $p_{HK}$")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.22, linestyle="--")

    x = np.arange(len(global_rows))
    auc_means = np.asarray([row["mean_auc_difference"] for row in global_rows])
    auc_low = np.asarray([row["ci95_low"] for row in global_rows])
    auc_high = np.asarray([row["ci95_high"] for row in global_rows])
    yerr = np.asarray([auc_means - auc_low, auc_high - auc_means])
    axes[1].bar(x, auc_means, yerr=yerr, capsize=5, color="#2563EB", alpha=0.82)
    axes[1].axhline(0, color="#111827", linestyle="--", linewidth=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        [row["metric_label"] for row in global_rows],
        rotation=18, ha="right", fontsize=8)
    axes[1].set_ylabel("Paired AUC difference (95% CI)")
    axes[1].set_title("(b) Across-$p_{HK}$ update-mode contrast")
    axes[1].grid(alpha=0.22, linestyle="--")
    fig.suptitle(
        "Formal update-mode control (n=20 paired seeds, 200 sweeps)",
        fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    figure_path = FIGURE_DIR / "fig_formal_update_mode_control.png"
    fig.savefig(str(figure_path), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- report ----
    report_path = ANALYSIS_DIR / "FORMAL_UPDATE_MODE_REPORT.md"
    with open(str(report_path), "w", encoding="utf-8") as handle:
        handle.write("# Formal update-mode control report\n\n")
        handle.write(
            "Design: N=2000, m=4, {} p_HK values, full model, 200 sweeps, "
            "{} paired seeds. The two arms share the network and "
            "initial-opinion seed, so every cell is a paired contrast. "
            "The independent experimental unit is the seed-level replicate.\n\n"
            .format(len(p_values), N_RUNS))
        handle.write(
            "Synchronous definition: every node proposes once per sweep from "
            "the same frozen start-of-sweep snapshot, and proposals commit "
            "together. A non-significant result is not interpreted as strict "
            "equivalence.\n\n")
        handle.write("## Across-$p_{HK}$ global AUC contrast\n\n")
        handle.write(
            "Each per-seed $p_{HK}$ curve is integrated with the trapezoidal "
            "rule before the paired test; Holm correction is applied across "
            "the six endpoints.\n\n")
        handle.write("| metric | mean AUC difference | 95% CI | dz | raw p | Holm-6 p |\n")
        handle.write("|---|---:|---|---:|---:|---:|\n")
        for row in global_rows:
            handle.write(
                "| {} | {:.6f} | [{:.6f}, {:.6f}] | {:.3f} | {:.4g} | {:.4g} |\n"
                .format(row["metric_label"], row["mean_auc_difference"],
                        row["ci95_low"], row["ci95_high"], row["cohen_dz"],
                        row["paired_t_p_raw"], row["paired_t_p_holm_6"]))
        handle.write("\n## Pointwise order-parameter contrast\n\n")
        handle.write("| $p_{HK}$ | mean difference | 95% CI | dz | raw p | Holm-4 p |\n")
        handle.write("|---:|---:|---|---:|---:|---:|\n")
        for row in [r for r in point_rows if r["metric"] == "order_param"]:
            handle.write(
                "| {:.2f} | {:.6f} | [{:.6f}, {:.6f}] | {:.3f} | {:.4g} | {:.4g} |\n"
                .format(row["p_hk"], row["mean_difference"], row["ci95_low"],
                        row["ci95_high"], row["cohen_dz"],
                        row["paired_t_p_raw"], row["paired_t_p_holm_4"]))
        handle.write("\n## Result boundary\n\n")
        significant_global = [
            row["metric_label"] for row in global_rows
            if row["paired_t_p_holm_6"] < 0.05]
        if significant_global:
            handle.write(
                "- After Holm-6 correction, the update mode changes: {}.\n"
                .format(", ".join(significant_global)))
        else:
            handle.write(
                "- No endpoint survives Holm-6 correction; the tested "
                "synchronous scheme does not produce a detectable "
                "macroscopic difference from the asynchronous scheme.\n")
        handle.write(
            "- This supports \"no qualitative change under synchronous "
            "updating\" within the tested protocol; it is a non-detection, "
            "not a proof of equivalence.\n")

    output = {
        "validation": validation,
        "design": {
            "N": 2000,
            "m": 4,
            "p_hk_points": p_values,
            "model_type": data.get("control", {}).get("model_type"),
            "n_per_cell": N_RUNS,
            "seeds": SEEDS,
            "horizon_sweeps": 200,
            "record_sweeps": 5,
            "modes": list(MODES),
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
    json_path = ANALYSIS_DIR / "formal_update_mode_analysis.json"
    write_json_atomic(output, str(json_path))
    return {
        "report": str(report_path),
        "figure": str(figure_path),
        "json": str(json_path),
    }


def main():
    if not DATA_FILE.exists() or not META_FILE.exists():
        raise SystemExit("formal update-mode result or metadata file is missing")
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
