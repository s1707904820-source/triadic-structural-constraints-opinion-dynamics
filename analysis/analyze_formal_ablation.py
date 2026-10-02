"""Validate and summarize the completed formal six-model ablation.

This script is intentionally separate from the simulation code.  It reads the
raw per-seed values saved by ``experiment_ablation`` and writes reviewer-facing
validation, source-data and primary-contrast artifacts.
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

# NumPy 2.0 introduced ``trapezoid`` and recent releases removed ``trapz``.
# Keep the analysis runnable with both the original NumPy 1.x environment and
# the Python 3.13 / NumPy 2.x environment used for the revised simulations.
_trapezoid = getattr(np, "trapezoid", None)
if _trapezoid is None:
    _trapezoid = np.trapz
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
FORMAL_ROOT = ROOT / "output" / "formal_revision"
DATA_FILE = FORMAL_ROOT / "data" / "exp6_ablation.json"
META_FILE = FORMAL_ROOT / "data" / "exp6_ablation.meta.json"
ANALYSIS_DIR = FORMAL_ROOT / "analysis"
FIGURE_DIR = FORMAL_ROOT / "figures"

MODEL_ORDER = [
    "baseline",
    "weighted_bcm",
    "threshold_only",
    "bcm_triangle",
    "no_triadic_feedback",
    "full_model",
]
MODEL_LABELS = {
    "baseline": "Asynchronous unilateral BCM",
    "weighted_bcm": "Weighted BCM",
    "threshold_only": "Threshold Only",
    "bcm_triangle": "BCM + Triangle",
    "no_triadic_feedback": "No Triadic Feedback",
    "full_model": "Full Model",
}
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
    "mean_abs_dx": "Mean absolute update",
    "accepted_update_rate": "Accepted-update rate",
}
TIME_SERIES_METRICS = [
    "order_param",
    "variance",
    "triangle_alignment_fixed",
    "triangle_admissibility_adaptive",
    "mean_abs_dx",
    "accepted_update_rate",
]
TIME_SERIES_MODELS = [
    "threshold_only",
    "no_triadic_feedback",
    "full_model",
]
REPRESENTATIVE_P = [0.0, 0.1, 0.2, 0.6, 1.0]
PRIMARY_CONTRASTS = [
    ("no_triadic_feedback", "Full - No Triadic Feedback"),
    ("threshold_only", "Full - Threshold Only"),
]
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
    if np.allclose(values, values[0]):
        return 1.0 if close(values[0], 0.0) else 0.0
    return float(stats.ttest_1samp(values, 0.0).pvalue)


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    m = len(p_values)
    for rank, idx in enumerate(order):
        candidate = min(1.0, (m - rank) * p_values[idx])
        running = max(running, candidate)
        adjusted[idx] = running
    return adjusted.tolist()


def validate(data, metadata):
    errors = []
    warnings = []
    expected_p = [round(index * 0.05, 2) for index in range(21)]
    actual_p = sorted(float(key) for key in data)
    if actual_p != expected_p:
        errors.append("p grid mismatch: {}".format(actual_p))

    recorded_hashes = metadata.get("source_sha256", {})
    hash_status = {}
    for filename, recorded in sorted(recorded_hashes.items()):
        path = ROOT / filename
        if not path.exists():
            hash_status[filename] = "missing"
            errors.append("recorded source missing: {}".format(filename))
            continue
        current = sha256(path)
        hash_status[filename] = "match" if current == recorded else "changed"
        if current != recorded:
            errors.append("source hash changed since run: {}".format(filename))

    expected_seeds = [run_i * 997 + 7 for run_i in range(20)]
    if metadata.get("n_runs") != 20:
        errors.append("metadata n_runs is not 20")
    if metadata.get("seeds") != expected_seeds:
        errors.append("metadata seed list mismatch")
    if float(metadata.get("main_sweeps", -1)) != 200.0:
        errors.append("metadata main_sweeps is not 200")

    reference_order = None
    for p in expected_p:
        key = str(p)
        block = data.get(key)
        if not isinstance(block, dict):
            errors.append("missing p block {}".format(key))
            continue
        control = block.get("_control", {})
        model_order = control.get("model_order", [])
        if reference_order is None:
            reference_order = model_order
        if model_order != reference_order or set(model_order) != set(MODEL_ORDER):
            errors.append("model order mismatch at p={}".format(key))
        expected_control = {
            "N": 2000,
            "m": 4,
            "n_runs": 20,
            "main_sweeps": 200.0,
            "record_sweeps": 5.0,
            "seeds": expected_seeds,
        }
        for field, expected in expected_control.items():
            if control.get(field) != expected:
                errors.append("control {} mismatch at p={}".format(field, key))

        for model in MODEL_ORDER:
            model_block = block.get(model)
            if not isinstance(model_block, dict):
                errors.append("missing model {} at p={}".format(model, key))
                continue
            for metric in METRICS:
                summary = model_block.get(metric, {})
                values = summary.get("values", [])
                if summary.get("n") != 20 or len(values) != 20:
                    errors.append("{} {} does not contain 20 values at p={}".format(
                        model, metric, key))
                    continue
                if not np.all(np.isfinite(np.asarray(values, dtype=float))):
                    errors.append("non-finite values in {} {} p={}".format(
                        model, metric, key))
                recalculated = summarize(values)
                for field in ("mean", "sd", "se", "ci95_low", "ci95_high"):
                    if not close(summary[field], recalculated[field], atol=1e-12):
                        errors.append("summary mismatch {} {} {} p={}".format(
                            model, metric, field, key))

            series = model_block.get("time_series_summary", {})
            if series.get("n_runs") != 20:
                errors.append("time-series n_runs mismatch for {} p={}".format(
                    model, key))
            sweeps = series.get("sweep", [])
            if not sweeps or not close(sweeps[-1], 200.0):
                errors.append("time series does not end at 200 sweeps for {} p={}".format(
                    model, key))

        comparisons = block.get("_paired_comparisons", {})
        for comparator in MODEL_ORDER:
            if comparator == "full_model":
                continue
            contrast_key = "full_model_minus_{}".format(comparator)
            contrast = comparisons.get(contrast_key, {})
            for metric in METRICS:
                saved = contrast.get(metric, {})
                full_values = np.asarray(
                    block["full_model"][metric]["values"], dtype=float)
                other_values = np.asarray(
                    block[comparator][metric]["values"], dtype=float)
                expected_values = full_values - other_values
                if len(saved.get("values", [])) != 20 or not np.allclose(
                        saved.get("values", []), expected_values, atol=1e-12):
                    errors.append("paired contrast mismatch {} {} p={}".format(
                        contrast_key, metric, key))

    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "warnings": warnings,
        "source_hash_status": hash_status,
        "completed_p_points": len(actual_p),
        "expected_p_points": 21,
        "models_per_point": 6,
        "independent_replicates_per_model_point": 20,
        "independent_unit": (
            "one seeded simulation replicate (matched network and initial "
            "opinions across model variants within each p_HK)"),
    }


def build_outputs(data, metadata, validation):
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    p_values = sorted(float(key) for key in data)
    seeds = metadata["seeds"]

    raw_path = ANALYSIS_DIR / "formal_ablation_source_data.csv"
    with raw_path.open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["p_hk", "seed", "model"] + METRICS
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for p in p_values:
            block = data[str(p)]
            for model in MODEL_ORDER:
                for index, seed in enumerate(seeds):
                    row = {"p_hk": p, "seed": seed, "model": model}
                    for metric in METRICS:
                        row[metric] = block[model][metric]["values"][index]
                    writer.writerow(row)

    point_rows = []
    for comparator, label in PRIMARY_CONTRASTS:
        raw_p = []
        contrast_rows = []
        key = "full_model_minus_{}".format(comparator)
        for p in p_values:
            saved = data[str(p)]["_paired_comparisons"][key]["order_param"]
            values = saved["values"]
            raw = paired_t_pvalue(values)
            raw_p.append(raw)
            contrast_rows.append({
                "p_hk": p,
                "contrast": label,
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
        for row, value in zip(contrast_rows, adjusted):
            row["paired_t_p_holm_21"] = value
            point_rows.append(row)

    point_path = ANALYSIS_DIR / "formal_ablation_primary_contrasts.csv"
    with point_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(point_rows[0]))
        writer.writeheader()
        writer.writerows(point_rows)

    # Reviewer-facing endpoint analysis.  Each metric is evaluated across the
    # full p_HK curve using paired per-seed AUCs.  Holm correction across the
    # six endpoints is performed separately for each prespecified contrast.
    all_metric_auc_rows = []
    all_metric_auc_results = {}
    all_metric_point_rows = []
    for comparator, label in PRIMARY_CONTRASTS:
        contrast_auc_rows = []
        contrast_auc_p = []
        contrast_point_rows = []
        for metric in METRICS:
            full_auc = []
            comparator_auc = []
            for index in range(len(seeds)):
                full_curve = [
                    data[str(p)]["full_model"][metric]["values"][index]
                    for p in p_values
                ]
                comparator_curve = [
                    data[str(p)][comparator][metric]["values"][index]
                    for p in p_values
                ]
                full_auc.append(float(_trapezoid(full_curve, p_values)))
                comparator_auc.append(float(_trapezoid(
                    comparator_curve, p_values)))
            differences = np.asarray(full_auc) - np.asarray(comparator_auc)
            summary = summarize(differences)
            raw_p = paired_t_pvalue(differences)
            auc_row = {
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
                "paired_t_p_raw": raw_p,
            }
            contrast_auc_rows.append(auc_row)
            contrast_auc_p.append(raw_p)

            metric_point_rows = []
            metric_point_p = []
            key = "full_model_minus_{}".format(comparator)
            for p in p_values:
                saved = data[str(p)]["_paired_comparisons"][key][metric]
                point_p = paired_t_pvalue(saved["values"])
                metric_point_p.append(point_p)
                metric_point_rows.append({
                    "p_hk": p,
                    "contrast": label,
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "n": saved["n"],
                    "mean_difference": saved["mean"],
                    "sd_difference": saved["sd"],
                    "se_difference": saved["se"],
                    "ci95_low": saved["ci95_low"],
                    "ci95_high": saved["ci95_high"],
                    "cohen_dz": saved["cohen_dz"],
                    "paired_t_p_raw": point_p,
                })
            adjusted_21 = holm_adjust(metric_point_p)
            for row, adjusted in zip(metric_point_rows, adjusted_21):
                row["paired_t_p_holm_21_within_metric"] = adjusted
                contrast_point_rows.append(row)

        adjusted_6 = holm_adjust(contrast_auc_p)
        for row, adjusted in zip(contrast_auc_rows, adjusted_6):
            row["paired_t_p_holm_6_endpoints"] = adjusted
            all_metric_auc_rows.append(row)
            all_metric_auc_results[
                "{} | {}".format(row["contrast"], row["metric"])
            ] = dict(row)

        # A second, deliberately conservative column controls all 126
        # exploratory point-level tests within one contrast.
        adjusted_126 = holm_adjust([
            row["paired_t_p_raw"] for row in contrast_point_rows
        ])
        for row, adjusted in zip(contrast_point_rows, adjusted_126):
            row["paired_t_p_holm_126_within_contrast"] = adjusted
            all_metric_point_rows.append(row)

    all_metric_auc_path = (
        ANALYSIS_DIR / "formal_ablation_all_metric_auc_contrasts.csv")
    with all_metric_auc_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(all_metric_auc_rows[0]))
        writer.writeheader()
        writer.writerows(all_metric_auc_rows)

    all_metric_point_path = (
        ANALYSIS_DIR / "formal_ablation_all_metric_pointwise.csv")
    with all_metric_point_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(all_metric_point_rows[0]))
        writer.writeheader()
        writer.writerows(all_metric_point_rows)

    summary_path = ANALYSIS_DIR / "formal_ablation_model_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["p_hk", "model", "metric", "n", "mean", "sd", "se",
                  "ci95_low", "ci95_high"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for p in p_values:
            for model in MODEL_ORDER:
                for metric in METRICS:
                    block = data[str(p)][model][metric]
                    writer.writerow({field: value for field, value in {
                        "p_hk": p, "model": model, "metric": metric,
                        "n": block["n"], "mean": block["mean"],
                        "sd": block["sd"], "se": block["se"],
                        "ci95_low": block["ci95_low"],
                        "ci95_high": block["ci95_high"],
                    }.items()})

    # Export every aggregate trajectory exactly as stored in the formal raw
    # result.  These curves are repeated-run summaries, not per-seed raw
    # trajectories; inferential paired path statistics are therefore not
    # computed from this file.  They are available from the subsequently run
    # dedicated seed-level trajectory experiment and its separate report.
    time_series_path = (
        ANALYSIS_DIR / "formal_ablation_time_series_source_data.csv")
    time_series_fields = [
        "p_hk", "model", "metric", "sweep", "n", "mean", "sd", "se",
        "ci95_low", "ci95_high",
    ]
    with time_series_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=time_series_fields)
        writer.writeheader()
        for p in p_values:
            for model in MODEL_ORDER:
                series = data[str(p)][model]["time_series_summary"]
                for metric in TIME_SERIES_METRICS:
                    metric_summary = series[metric]
                    for index, sweep in enumerate(series["sweep"]):
                        writer.writerow({
                            "p_hk": p,
                            "model": model,
                            "metric": metric,
                            "sweep": sweep,
                            "n": series["n_runs"],
                            "mean": metric_summary["mean"][index],
                            "sd": metric_summary["sd"][index],
                            "se": metric_summary["se"][index],
                            "ci95_low": metric_summary["ci95_low"][index],
                            "ci95_high": metric_summary["ci95_high"][index],
                        })

    landmark_path = ANALYSIS_DIR / "formal_ablation_time_landmarks.csv"
    landmark_sweeps = [50.0, 100.0, 150.0, 200.0]
    landmark_fields = [
        "p_hk", "model", "metric", "sweep", "n", "mean", "sd", "se",
        "ci95_low", "ci95_high",
    ]
    with landmark_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=landmark_fields)
        writer.writeheader()
        for p in REPRESENTATIVE_P:
            for model in TIME_SERIES_MODELS:
                series = data[str(p)][model]["time_series_summary"]
                sweep_to_index = {
                    float(sweep): index
                    for index, sweep in enumerate(series["sweep"])
                }
                for metric in TIME_SERIES_METRICS:
                    metric_summary = series[metric]
                    for sweep in landmark_sweeps:
                        index = sweep_to_index[sweep]
                        writer.writerow({
                            "p_hk": p,
                            "model": model,
                            "metric": metric,
                            "sweep": sweep,
                            "n": series["n_runs"],
                            "mean": metric_summary["mean"][index],
                            "sd": metric_summary["sd"][index],
                            "se": metric_summary["se"][index],
                            "ci95_low": metric_summary["ci95_low"][index],
                            "ci95_high": metric_summary["ci95_high"][index],
                        })

    global_results = {}
    model_auc_summaries = {}
    model_auc_values = {}
    for model in MODEL_ORDER:
        auc_values = []
        for index in range(len(seeds)):
            curve = [data[str(p)][model]["order_param"]["values"][index]
                     for p in p_values]
            auc_values.append(float(_trapezoid(curve, p_values)))
        model_auc_values[model] = auc_values
        model_auc_summaries[model] = summarize(auc_values)

    for comparator, label in PRIMARY_CONTRASTS:
        differences = (np.asarray(model_auc_values["full_model"])
                       - np.asarray(model_auc_values[comparator]))
        auc_summary = summarize(differences)
        auc_summary["paired_t_p_raw"] = paired_t_pvalue(differences)
        auc_summary["definition"] = "trapezoidal AUC(Full) - AUC(comparator)"
        global_results[label] = auc_summary
    global_adjusted = holm_adjust([
        global_results[label]["paired_t_p_raw"]
        for _, label in PRIMARY_CONTRASTS
    ])
    for (_, label), adjusted in zip(PRIMARY_CONTRASTS, global_adjusted):
        global_results[label]["paired_t_p_holm_2"] = adjusted

    output_json = {
        "validation": validation,
        "design": {
            "N": 2000,
            "m": 4,
            "p_hk_points": p_values,
            "models": MODEL_ORDER,
            "n_per_model_point": 20,
            "seeds": seeds,
            "horizon_sweeps": 200,
            "record_sweeps": 5,
            "uncertainty": "two-sided 95% Student-t confidence interval",
            "pointwise_test": "two-sided paired t-test on per-seed differences",
            "multiplicity": (
                "Holm correction across 21 p_HK points, separately within "
                "each primary contrast"),
            "global_test": (
                "two-sided paired t-test on per-seed trapezoidal AUC "
                "differences across p_HK"),
        },
        "global_auc_primary_contrasts": global_results,
        "all_metric_auc_primary_contrasts": all_metric_auc_results,
        "model_auc_summaries": model_auc_summaries,
        "pointwise_primary_contrasts": point_rows,
        "all_metric_pointwise_contrasts": all_metric_point_rows,
        "time_series_scope": {
            "representative_p_hk": REPRESENTATIVE_P,
            "models": TIME_SERIES_MODELS,
            "metrics": TIME_SERIES_METRICS,
            "landmark_sweeps": landmark_sweeps,
            "inference_boundary": (
                "saved formal trajectories are across-replicate summaries; "
                "figures and landmark tables from this file are descriptive; "
                "paired path inference is reported separately in "
                "FORMAL_TRAJECTORY_REPORT.md using the dedicated seed-level "
                "trajectory data"
            ),
        },
    }
    json_path = ANALYSIS_DIR / "formal_ablation_analysis.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(output_json, handle, ensure_ascii=False, indent=2)

    colors = {"full_model": "#DC2626", "no_triadic_feedback": "#0D9488",
              "threshold_only": "#7C3AED"}
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    for model in ("threshold_only", "no_triadic_feedback", "full_model"):
        means = np.asarray([data[str(p)][model]["order_param"]["mean"]
                            for p in p_values])
        lows = np.asarray([data[str(p)][model]["order_param"]["ci95_low"]
                           for p in p_values])
        highs = np.asarray([data[str(p)][model]["order_param"]["ci95_high"]
                            for p in p_values])
        axes[0].plot(p_values, means, marker="o", markersize=3.5,
                     linewidth=1.7, color=colors[model], label=MODEL_LABELS[model])
        axes[0].fill_between(p_values, lows, highs, color=colors[model], alpha=0.13)
    axes[0].set_title("(a) Long-horizon order parameter")
    axes[0].set_xlabel(r"$p_{HK}$")
    axes[0].set_ylabel("Order parameter")
    axes[0].set_ylim(0, 1.03)
    axes[0].legend(fontsize=8)

    contrast_colors = ["#0D9488", "#7C3AED"]
    for (comparator, label), color in zip(PRIMARY_CONTRASTS, contrast_colors):
        key = "full_model_minus_{}".format(comparator)
        means = np.asarray([data[str(p)]["_paired_comparisons"][key][
            "order_param"]["mean"] for p in p_values])
        lows = np.asarray([data[str(p)]["_paired_comparisons"][key][
            "order_param"]["ci95_low"] for p in p_values])
        highs = np.asarray([data[str(p)]["_paired_comparisons"][key][
            "order_param"]["ci95_high"] for p in p_values])
        axes[1].plot(p_values, means, marker="o", markersize=3.5,
                     linewidth=1.7, color=color, label=label)
        axes[1].fill_between(p_values, lows, highs, color=color, alpha=0.13)
    axes[1].axhline(0, color="#111827", linestyle="--", linewidth=1)
    axes[1].set_title("(b) Paired order-parameter contrasts")
    axes[1].set_xlabel(r"$p_{HK}$")
    axes[1].set_ylabel("Mean paired difference (95% CI)")
    axes[1].legend(fontsize=8)

    x = np.arange(len(PRIMARY_CONTRASTS))
    auc_means = [global_results[label]["mean"] for _, label in PRIMARY_CONTRASTS]
    auc_low = [global_results[label]["ci95_low"] for _, label in PRIMARY_CONTRASTS]
    auc_high = [global_results[label]["ci95_high"] for _, label in PRIMARY_CONTRASTS]
    yerr = np.asarray([np.asarray(auc_means) - np.asarray(auc_low),
                       np.asarray(auc_high) - np.asarray(auc_means)])
    axes[2].bar(x, auc_means, yerr=yerr, capsize=5,
                color=contrast_colors, alpha=0.82)
    axes[2].axhline(0, color="#111827", linestyle="--", linewidth=1)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(["Full - No feedback", "Full - Threshold"],
                            rotation=8)
    axes[2].set_ylabel("Paired AUC difference (95% CI)")
    axes[2].set_title("(c) Across-$p_{HK}$ primary contrasts")
    for ax in axes:
        ax.grid(alpha=0.25, linestyle="--")
    fig.suptitle("Formal six-model ablation: primary long-horizon contrasts",
                 fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    figure_path = FIGURE_DIR / "fig_formal_ablation_primary_contrasts.png"
    fig.savefig(str(figure_path), dpi=300, bbox_inches="tight")
    plt.close(fig)

    trajectory_metrics = [
        "order_param", "variance", "triangle_alignment_fixed"]
    fig, axes = plt.subplots(
        len(REPRESENTATIVE_P), len(trajectory_metrics),
        figsize=(15.5, 3.45 * len(REPRESENTATIVE_P)), sharex=True)
    for row_index, p in enumerate(REPRESENTATIVE_P):
        for column_index, metric in enumerate(trajectory_metrics):
            ax = axes[row_index, column_index]
            for model in TIME_SERIES_MODELS:
                series = data[str(p)][model]["time_series_summary"]
                x_values = np.asarray(series["sweep"], dtype=float)
                metric_summary = series[metric]
                means = np.asarray(metric_summary["mean"], dtype=float)
                lows = np.asarray(metric_summary["ci95_low"], dtype=float)
                highs = np.asarray(metric_summary["ci95_high"], dtype=float)
                ax.plot(
                    x_values, means, color=colors[model], linewidth=1.6,
                    label=MODEL_LABELS[model])
                ax.fill_between(
                    x_values, lows, highs, color=colors[model], alpha=0.12)
            ax.grid(alpha=0.22, linestyle="--")
            ax.set_title(
                r"$p_{{HK}}={:.1f}$: {}".format(
                    p, METRIC_LABELS[metric]), fontsize=10)
            if column_index == 0:
                ax.set_ylabel("Mean (95% CI)")
            if row_index == len(REPRESENTATIVE_P) - 1:
                ax.set_xlabel("Monte Carlo sweeps")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, 0.992))
    fig.suptitle(
        "Formal mechanism-focused trajectories (n=20 paired seeds)",
        fontweight="bold", y=0.999)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    trajectory_figure_path = (
        FIGURE_DIR / "fig_formal_ablation_time_trajectories.png")
    fig.savefig(str(trajectory_figure_path), dpi=300, bbox_inches="tight")
    plt.close(fig)

    convergence_metrics = ["mean_abs_dx", "accepted_update_rate"]
    fig, axes = plt.subplots(
        len(REPRESENTATIVE_P), len(convergence_metrics),
        figsize=(11.5, 3.45 * len(REPRESENTATIVE_P)), sharex=True)
    for row_index, p in enumerate(REPRESENTATIVE_P):
        for column_index, metric in enumerate(convergence_metrics):
            ax = axes[row_index, column_index]
            for model in TIME_SERIES_MODELS:
                series = data[str(p)][model]["time_series_summary"]
                x_values = np.asarray(series["sweep"], dtype=float)
                metric_summary = series[metric]
                means = np.asarray(metric_summary["mean"], dtype=float)
                lows = np.asarray(metric_summary["ci95_low"], dtype=float)
                highs = np.asarray(metric_summary["ci95_high"], dtype=float)
                ax.plot(
                    x_values, means, color=colors[model], linewidth=1.6,
                    label=MODEL_LABELS[model])
                ax.fill_between(
                    x_values, lows, highs, color=colors[model], alpha=0.12)
            ax.grid(alpha=0.22, linestyle="--")
            ax.set_title(
                r"$p_{{HK}}={:.1f}$: {}".format(
                    p, METRIC_LABELS[metric]), fontsize=10)
            if column_index == 0:
                ax.set_ylabel("Mean (95% CI)")
            if row_index == len(REPRESENTATIVE_P) - 1:
                ax.set_xlabel("Monte Carlo sweeps")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, 0.992))
    fig.suptitle(
        "Formal convergence diagnostics (n=20 paired seeds)",
        fontweight="bold", y=0.999)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    convergence_figure_path = (
        FIGURE_DIR / "fig_formal_ablation_convergence_diagnostics.png")
    fig.savefig(str(convergence_figure_path), dpi=300, bbox_inches="tight")
    plt.close(fig)

    report_path = ANALYSIS_DIR / "FORMAL_ABLATION_REPORT.md"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("# 正式六模型消融：统计验收与结果摘要\n\n")
        handle.write("## 完整性验收\n\n")
        handle.write("- 状态：{}\n".format(validation["status"]))
        handle.write("- 设计：21 个 p_HK 点 × 6 个模型 × 20 个配对种子，200 sweeps。\n")
        handle.write("- 独立单位：每个种子对应一个独立仿真重复；同一 p_HK 下各模型共享网络和初始意见，因而模型比较为配对设计。\n")
        handle.write("- 误差线：均值的双侧 95% Student-t 置信区间。\n")
        handle.write("- 点水平推断：双侧配对 t 检验；每个主对比内对 21 个 p_HK 点采用 Holm 校正。\n")
        handle.write("- 总体推断：先对每个种子的 p_HK 曲线计算梯形 AUC，再对 AUC 配对差进行双侧配对 t 检验。\n\n")
        handle.write("## 跨 p_HK 总体主对比\n\n")
        for _, label in PRIMARY_CONTRASTS:
            item = global_results[label]
            handle.write("- {}：AUC 配对差 {:.6f}，95% CI [{:.6f}, {:.6f}]，Cohen's dz = {:.3f}，原始 p = {:.4g}，两个总体主对比 Holm 校正 p = {:.4g}。\n".format(
                label, item["mean"], item["ci95_low"], item["ci95_high"],
                item["cohen_dz"], item["paired_t_p_raw"],
                item["paired_t_p_holm_2"]))
        handle.write("\n## 点水平结论边界\n\n")
        for _, label in PRIMARY_CONTRASTS:
            rows = [row for row in point_rows if row["contrast"] == label]
            unadjusted = sum(row["paired_t_p_raw"] < 0.05 for row in rows)
            adjusted = sum(row["paired_t_p_holm_21"] < 0.05 for row in rows)
            excludes_zero = sum((row["ci95_low"] > 0 or row["ci95_high"] < 0)
                                for row in rows)
            handle.write("- {}：{}/21 个未校正 p<0.05，{}/21 个 Holm 校正后 p<0.05；{}/21 个逐点95% CI不跨0。\n".format(
                label, unadjusted, adjusted, excludes_zero))
        handle.write("\n## 六个终态指标的跨 p_HK 配对 AUC 对比\n\n")
        handle.write("以下检验以每个种子的整条 p_HK 曲线 AUC 为单位；同一主对比内对六个终态指标采用 Holm 校正。\n\n")
        for row in all_metric_auc_rows:
            handle.write(
                "- {}，{}：AUC 配对差 {:.6f}，95% CI "
                "[{:.6f}, {:.6f}]，dz={:.3f}，原始 p={:.4g}，"
                "Holm-6 p={:.4g}。\n".format(
                    row["contrast"], row["metric_label"],
                    row["mean_auc_difference"], row["ci95_low"],
                    row["ci95_high"], row["cohen_dz"],
                    row["paired_t_p_raw"],
                    row["paired_t_p_holm_6_endpoints"]))
        handle.write("\n## 时间轨迹证据边界\n\n")
        handle.write("- 正式数据包含每5 sweeps记录的跨20次重复均值、SD、SE和95% t区间，可用于绘制描述性轨迹和检查200 sweeps终点。\n")
        handle.write("- 当前正式文件未保留每个种子的完整原始轨迹，因此不能从这些汇总曲线计算配对轨迹AUC、配对首达时间或其p值。若描述性曲线显示稳定路径差异，再安排保存逐种子轨迹的专门运行。\n")
        handle.write("\n## 报告原则\n\n")
        handle.write("- 不以误差带是否重叠代替配对比较。\n")
        handle.write("- 逐点结果属于同一比较族；任何显著性措辞使用 Holm 校正结果。\n")
        handle.write("- 主要结论优先依据效应量、95% CI和跨 p_HK 的配对 AUC，而非挑选单个参数点。\n")
        handle.write("- 该实验支持终态宏观结论；若要声称路径或速度效应，仍需单独分析时间序列或收敛时间。\n")

    return {
        "validation": validation,
        "global_results": global_results,
        "point_rows": point_rows,
        "artifacts": [
            str(raw_path), str(point_path), str(all_metric_auc_path),
            str(all_metric_point_path), str(summary_path),
            str(time_series_path), str(landmark_path), str(json_path),
            str(report_path), str(figure_path),
            str(trajectory_figure_path), str(convergence_figure_path),
        ],
    }


def main():
    if not DATA_FILE.exists() or not META_FILE.exists():
        raise SystemExit("formal ablation result or metadata file is missing")
    with DATA_FILE.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    with META_FILE.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    validation = validate(data, metadata)
    output = build_outputs(data, metadata, validation)
    print("VALIDATION={}".format(validation["status"]))
    print("ERRORS={}".format(len(validation["errors"])))
    print("P_POINTS={}".format(validation["completed_p_points"]))
    for label, result in output["global_results"].items():
        print("{} mean={:.8f} CI=[{:.8f},{:.8f}] dz={} p={:.8g}".format(
            label, result["mean"], result["ci95_low"],
            result["ci95_high"], result["cohen_dz"],
            result["paired_t_p_raw"]))
    for path in output["artifacts"]:
        print("WROTE={}".format(path))
    if validation["errors"]:
        for error in validation["errors"]:
            print("ERROR={}".format(error))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
