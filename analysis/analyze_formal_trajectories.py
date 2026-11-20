"""Validate and analyse reconstructed per-seed formal trajectories."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
FORMAL_ROOT = ROOT / "output" / "formal_revision"
RAW_PATH = FORMAL_ROOT / "data" / "exp_formal_trajectory_raw.json"
RAW_META_PATH = FORMAL_ROOT / "data" / "exp_formal_trajectory_raw.meta.json"
ABLATION_PATH = FORMAL_ROOT / "data" / "exp6_ablation.json"
ANALYSIS_DIR = FORMAL_ROOT / "analysis"
FIGURE_DIR = FORMAL_ROOT / "figures"

P_VALUES = [round(index * 0.05, 2) for index in range(21)]
MODELS = ["threshold_only", "no_triadic_feedback", "full_model"]
MODEL_LABELS = {
    "threshold_only": "Threshold Only",
    "no_triadic_feedback": "No Triadic Feedback",
    "full_model": "Full Model",
}
CONTRASTS = [
    ("no_triadic_feedback", "Full - No Triadic Feedback"),
    ("threshold_only", "Full - Threshold Only"),
]
PATH_METRICS = [
    "order_param",
    "variance",
    "triangle_alignment_fixed",
    "mean_abs_dx",
    "accepted_update_rate",
]
METRIC_LABELS = {
    "order_param": "Order parameter",
    "variance": "Opinion variance",
    "triangle_alignment_fixed": "Fixed-scale triangle alignment",
    "mean_abs_dx": "Mean absolute update",
    "accepted_update_rate": "Accepted-update rate",
}
FINAL_METRICS = [
    "order_param",
    "n_clusters",
    "variance",
    "triangle_alignment_fixed",
    "triangle_admissibility_adaptive",
    "edge_disagreement_penalty",
]
SEGMENTS = [
    ("early", 5.0, 50.0),
    ("middle", 50.0, 125.0),
    ("late", 125.0, 200.0),
]
LANDMARKS = [50.0, 100.0, 150.0, 200.0]
SELECTED_P = [0.0, 0.1, 0.2, 0.6, 1.0]


def p_key(value):
    return "{:.2f}".format(float(value))


def ablation_p_key(value):
    return str(float(value))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(values):
    values = np.asarray(values, dtype=float)
    n = int(values.size)
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if n > 1 else 0.0
    se = float(sd / math.sqrt(n)) if n else float("nan")
    critical = float(stats.t.ppf(0.975, n - 1)) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "se": se,
        "ci95_low": float(mean - critical * se),
        "ci95_high": float(mean + critical * se),
        "cohen_dz": (
            float(mean / sd) if sd > 0
            else (0.0 if mean == 0 else None)),
        "values": values.tolist(),
    }


def paired_p(values):
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
    total = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def time_mean_auc(sweeps, values, start=None, end=None):
    sweeps = np.asarray(sweeps, dtype=float)
    values = np.asarray(values, dtype=float)
    if start is None:
        start = float(sweeps[0])
    if end is None:
        end = float(sweeps[-1])
    mask = np.logical_and(sweeps >= float(start), sweeps <= float(end))
    x_values = sweeps[mask]
    y_values = values[mask]
    if x_values.size < 2 or not math.isclose(float(x_values[0]), float(start)) \
            or not math.isclose(float(x_values[-1]), float(end)):
        raise ValueError("segment boundaries must occur in the saved sweep grid")
    return float(np.trapz(y_values, x_values) / (float(end) - float(start)))


def seed_run(raw, p_hk, model, seed):
    return raw["data"][p_key(p_hk)][model][str(seed)]


def validate(raw, raw_meta, ablation):
    errors = []
    expected_seeds = [run_i * 997 + 7 for run_i in range(20)]
    control = raw.get("control", {})
    expected_control = {
        "N": 2000,
        "m": 4,
        "p_hk_values": P_VALUES,
        "models": MODELS,
        "n_runs": 20,
        "seeds": expected_seeds,
        "main_sweeps": 200.0,
        "record_sweeps": 5.0,
    }
    for field, expected in expected_control.items():
        if control.get(field) != expected:
            errors.append("control mismatch for {}".format(field))

    hash_status = {}
    for filename, recorded in sorted(raw_meta.get("source_sha256", {}).items()):
        path = ROOT / filename
        if not path.exists():
            hash_status[filename] = "missing"
            errors.append("recorded source missing: {}".format(filename))
            continue
        current = sha256(path)
        hash_status[filename] = "match" if current == recorded else "changed"
        if current != recorded:
            errors.append("source changed since trajectory run: {}".format(
                filename))

    terminal_checks = 0
    maximum_terminal_error = 0.0
    for p_hk in P_VALUES:
        block = raw.get("data", {}).get(p_key(p_hk), {})
        for model in MODELS:
            model_runs = block.get(model, {})
            if sorted(int(seed) for seed in model_runs) != expected_seeds:
                errors.append("seed set mismatch p={} model={}".format(
                    p_hk, model))
                continue
            for seed_index, seed in enumerate(expected_seeds):
                run = model_runs[str(seed)]
                series = run.get("time_series", {})
                sweeps = series.get("sweep", [])
                if len(sweeps) != 40 or not math.isclose(sweeps[0], 5.0) \
                        or not math.isclose(sweeps[-1], 200.0):
                    errors.append("time grid mismatch p={} model={} seed={}".format(
                        p_hk, model, seed))
                for metric in PATH_METRICS:
                    values = np.asarray(series.get(metric, []), dtype=float)
                    if len(values) != 40 or not np.all(np.isfinite(values)):
                        errors.append(
                            "invalid series p={} model={} seed={} metric={}".format(
                                p_hk, model, seed, metric))
                for metric in FINAL_METRICS:
                    reconstructed = float(run["final"][metric])
                    formal = float(ablation[ablation_p_key(p_hk)][model][metric][
                        "values"][seed_index])
                    error = abs(reconstructed - formal)
                    maximum_terminal_error = max(maximum_terminal_error, error)
                    terminal_checks += 1
                    if error > 1e-12:
                        errors.append(
                            "terminal mismatch p={} model={} seed={} metric={} "
                            "error={}".format(
                                p_hk, model, seed, metric, error))
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "source_hash_status": hash_status,
        "terminal_values_checked": terminal_checks,
        "maximum_absolute_terminal_error": maximum_terminal_error,
        "expected_runs": 21 * 3 * 20,
        "expected_time_points_per_run": 40,
    }


def write_rows(path, rows):
    if not rows:
        raise ValueError("cannot write an empty table")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_outputs(raw, validation):
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    seeds = raw["control"]["seeds"]

    raw_rows = []
    for p_hk in P_VALUES:
        for model in MODELS:
            for seed in seeds:
                series = seed_run(raw, p_hk, model, seed)["time_series"]
                for index, sweep in enumerate(series["sweep"]):
                    row = {
                        "p_hk": p_hk,
                        "model": model,
                        "seed": seed,
                        "sweep": sweep,
                    }
                    for metric in PATH_METRICS:
                        row[metric] = series[metric][index]
                    raw_rows.append(row)
    raw_source_path = ANALYSIS_DIR / "formal_trajectory_source_data.csv"
    write_rows(raw_source_path, raw_rows)

    point_auc_rows = []
    global_auc_rows = []
    global_auc_results = {}
    for comparator, contrast_label in CONTRASTS:
        contrast_global_rows = []
        contrast_global_p = []
        for metric in PATH_METRICS:
            metric_rows = []
            metric_p = []
            per_seed_by_p = {seed: {} for seed in seeds}
            for p_hk in P_VALUES:
                differences = []
                for seed in seeds:
                    full_series = seed_run(
                        raw, p_hk, "full_model", seed)["time_series"]
                    comparator_series = seed_run(
                        raw, p_hk, comparator, seed)["time_series"]
                    full_auc = time_mean_auc(
                        full_series["sweep"], full_series[metric])
                    comparator_auc = time_mean_auc(
                        comparator_series["sweep"],
                        comparator_series[metric])
                    difference = full_auc - comparator_auc
                    differences.append(difference)
                    per_seed_by_p[seed][p_hk] = difference
                summary = summarize(differences)
                raw_p = paired_p(differences)
                metric_p.append(raw_p)
                metric_rows.append({
                    "contrast": contrast_label,
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "p_hk": p_hk,
                    "n": summary["n"],
                    "mean_difference": summary["mean"],
                    "sd_difference": summary["sd"],
                    "se_difference": summary["se"],
                    "ci95_low": summary["ci95_low"],
                    "ci95_high": summary["ci95_high"],
                    "cohen_dz": summary["cohen_dz"],
                    "paired_t_p_raw": raw_p,
                })
            for row, adjusted in zip(metric_rows, holm_adjust(metric_p)):
                row["paired_t_p_holm_21_within_metric"] = adjusted
                point_auc_rows.append(row)

            surface_values = []
            for seed in seeds:
                curve = [per_seed_by_p[seed][p_hk] for p_hk in P_VALUES]
                surface_values.append(float(np.trapz(curve, P_VALUES)))
            surface_summary = summarize(surface_values)
            surface_p = paired_p(surface_values)
            global_row = {
                "contrast": contrast_label,
                "metric": metric,
                "metric_label": METRIC_LABELS[metric],
                "n": surface_summary["n"],
                "mean_surface_difference": surface_summary["mean"],
                "sd_surface_difference": surface_summary["sd"],
                "se_surface_difference": surface_summary["se"],
                "ci95_low": surface_summary["ci95_low"],
                "ci95_high": surface_summary["ci95_high"],
                "cohen_dz": surface_summary["cohen_dz"],
                "paired_t_p_raw": surface_p,
            }
            contrast_global_rows.append(global_row)
            contrast_global_p.append(surface_p)
        for row, adjusted in zip(
                contrast_global_rows, holm_adjust(contrast_global_p)):
            row["paired_t_p_holm_5_endpoints"] = adjusted
            global_auc_rows.append(row)
            global_auc_results[
                "{} | {}".format(row["contrast"], row["metric"])
            ] = dict(row)

    point_auc_path = ANALYSIS_DIR / "formal_trajectory_point_auc_contrasts.csv"
    global_auc_path = ANALYSIS_DIR / "formal_trajectory_global_auc_contrasts.csv"
    write_rows(point_auc_path, point_auc_rows)
    write_rows(global_auc_path, global_auc_rows)

    segment_rows = []
    for comparator, contrast_label in CONTRASTS:
        temporary_rows = []
        raw_p_values = []
        for segment_label, start, end in SEGMENTS:
            differences = []
            for seed in seeds:
                p_curve = []
                for p_hk in P_VALUES:
                    full_series = seed_run(
                        raw, p_hk, "full_model", seed)["time_series"]
                    comparator_series = seed_run(
                        raw, p_hk, comparator, seed)["time_series"]
                    p_curve.append(
                        time_mean_auc(
                            full_series["sweep"],
                            full_series["order_param"], start, end)
                        - time_mean_auc(
                            comparator_series["sweep"],
                            comparator_series["order_param"], start, end))
                differences.append(float(np.trapz(p_curve, P_VALUES)))
            summary = summarize(differences)
            raw_p = paired_p(differences)
            raw_p_values.append(raw_p)
            temporary_rows.append({
                "contrast": contrast_label,
                "segment": segment_label,
                "start_sweep": start,
                "end_sweep": end,
                "n": summary["n"],
                "mean_surface_difference": summary["mean"],
                "sd_surface_difference": summary["sd"],
                "se_surface_difference": summary["se"],
                "ci95_low": summary["ci95_low"],
                "ci95_high": summary["ci95_high"],
                "cohen_dz": summary["cohen_dz"],
                "paired_t_p_raw": raw_p,
            })
        for row, adjusted in zip(temporary_rows, holm_adjust(raw_p_values)):
            row["paired_t_p_holm_3_segments"] = adjusted
            segment_rows.append(row)
    segment_path = ANALYSIS_DIR / "formal_trajectory_segment_contrasts.csv"
    write_rows(segment_path, segment_rows)

    landmark_rows = []
    global_landmark_rows = []
    for comparator, contrast_label in CONTRASTS:
        global_temporary = []
        global_p_values = []
        for p_hk in P_VALUES:
            point_temporary = []
            point_p_values = []
            for landmark in LANDMARKS:
                differences = []
                for seed in seeds:
                    full_series = seed_run(
                        raw, p_hk, "full_model", seed)["time_series"]
                    comparator_series = seed_run(
                        raw, p_hk, comparator, seed)["time_series"]
                    index = full_series["sweep"].index(landmark)
                    differences.append(
                        full_series["order_param"][index]
                        - comparator_series["order_param"][index])
                summary = summarize(differences)
                raw_p = paired_p(differences)
                point_p_values.append(raw_p)
                point_temporary.append({
                    "contrast": contrast_label,
                    "p_hk": p_hk,
                    "sweep": landmark,
                    "n": summary["n"],
                    "mean_difference": summary["mean"],
                    "sd_difference": summary["sd"],
                    "se_difference": summary["se"],
                    "ci95_low": summary["ci95_low"],
                    "ci95_high": summary["ci95_high"],
                    "cohen_dz": summary["cohen_dz"],
                    "paired_t_p_raw": raw_p,
                })
            for row, adjusted in zip(
                    point_temporary, holm_adjust(point_p_values)):
                row["paired_t_p_holm_4_landmarks_within_p"] = adjusted
                landmark_rows.append(row)

        for landmark in LANDMARKS:
            differences = []
            for seed in seeds:
                curve = []
                for p_hk in P_VALUES:
                    full_series = seed_run(
                        raw, p_hk, "full_model", seed)["time_series"]
                    comparator_series = seed_run(
                        raw, p_hk, comparator, seed)["time_series"]
                    index = full_series["sweep"].index(landmark)
                    curve.append(
                        full_series["order_param"][index]
                        - comparator_series["order_param"][index])
                differences.append(float(np.trapz(curve, P_VALUES)))
            summary = summarize(differences)
            raw_p = paired_p(differences)
            global_p_values.append(raw_p)
            global_temporary.append({
                "contrast": contrast_label,
                "sweep": landmark,
                "n": summary["n"],
                "mean_p_integrated_difference": summary["mean"],
                "sd_difference": summary["sd"],
                "se_difference": summary["se"],
                "ci95_low": summary["ci95_low"],
                "ci95_high": summary["ci95_high"],
                "cohen_dz": summary["cohen_dz"],
                "paired_t_p_raw": raw_p,
            })
        for row, adjusted in zip(
                global_temporary, holm_adjust(global_p_values)):
            row["paired_t_p_holm_4_landmarks"] = adjusted
            global_landmark_rows.append(row)

    landmark_path = ANALYSIS_DIR / "formal_trajectory_landmark_contrasts.csv"
    global_landmark_path = (
        ANALYSIS_DIR / "formal_trajectory_global_landmark_contrasts.csv")
    write_rows(landmark_path, landmark_rows)
    write_rows(global_landmark_path, global_landmark_rows)

    curve_rows = []
    curve_lookup = {}
    for comparator, contrast_label in CONTRASTS:
        for p_hk in P_VALUES:
            reference = seed_run(raw, p_hk, "full_model", seeds[0])[
                "time_series"]
            for metric in PATH_METRICS:
                for index, sweep in enumerate(reference["sweep"]):
                    differences = [
                        seed_run(raw, p_hk, "full_model", seed)[
                            "time_series"][metric][index]
                        - seed_run(raw, p_hk, comparator, seed)[
                            "time_series"][metric][index]
                        for seed in seeds
                    ]
                    summary = summarize(differences)
                    row = {
                        "contrast": contrast_label,
                        "metric": metric,
                        "p_hk": p_hk,
                        "sweep": sweep,
                        "n": summary["n"],
                        "mean_difference": summary["mean"],
                        "sd_difference": summary["sd"],
                        "se_difference": summary["se"],
                        "ci95_low": summary["ci95_low"],
                        "ci95_high": summary["ci95_high"],
                    }
                    curve_rows.append(row)
                    curve_lookup[(contrast_label, metric, p_hk, sweep)] = row
    curve_path = ANALYSIS_DIR / "formal_trajectory_paired_difference_curves.csv"
    write_rows(curve_path, curve_rows)

    colors = {
        "Full - No Triadic Feedback": "#0D9488",
        "Full - Threshold Only": "#7C3AED",
    }
    figure_metrics = [
        "order_param", "variance", "triangle_alignment_fixed"]
    fig, axes = plt.subplots(
        len(SELECTED_P), len(figure_metrics),
        figsize=(15.5, 3.45 * len(SELECTED_P)), sharex=True)
    for row_index, p_hk in enumerate(SELECTED_P):
        for column_index, metric in enumerate(figure_metrics):
            ax = axes[row_index, column_index]
            for _, contrast_label in CONTRASTS:
                rows = [
                    curve_lookup[(contrast_label, metric, p_hk, sweep)]
                    for sweep in np.arange(5.0, 205.0, 5.0)
                ]
                x_values = np.asarray([row["sweep"] for row in rows])
                means = np.asarray([row["mean_difference"] for row in rows])
                lows = np.asarray([row["ci95_low"] for row in rows])
                highs = np.asarray([row["ci95_high"] for row in rows])
                ax.plot(
                    x_values, means, linewidth=1.6,
                    color=colors[contrast_label], label=contrast_label)
                ax.fill_between(
                    x_values, lows, highs, color=colors[contrast_label],
                    alpha=0.13)
            ax.axhline(0.0, color="#111827", linestyle="--", linewidth=0.9)
            ax.grid(alpha=0.22, linestyle="--")
            ax.set_title(
                r"$p_{{HK}}={:.1f}$: {}".format(
                    p_hk, METRIC_LABELS[metric]), fontsize=10)
            if column_index == 0:
                ax.set_ylabel("Paired difference (95% CI)")
            if row_index == len(SELECTED_P) - 1:
                ax.set_xlabel("Monte Carlo sweeps")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=2, frameon=False,
        bbox_to_anchor=(0.5, 0.992))
    fig.suptitle(
        "Paired formal trajectory contrasts (n=20 seeded simulations)",
        fontweight="bold", y=0.999)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    figure_path = FIGURE_DIR / "fig_formal_trajectory_paired_differences.png"
    fig.savefig(str(figure_path), dpi=300, bbox_inches="tight")
    plt.close(fig)

    auc_figure_metrics = [
        "order_param", "mean_abs_dx", "accepted_update_rate"]
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.5), sharex=True)
    for row_index, (_, contrast_label) in enumerate(CONTRASTS):
        for column_index, metric in enumerate(auc_figure_metrics):
            ax = axes[row_index, column_index]
            rows = [
                row for row in point_auc_rows
                if row["contrast"] == contrast_label
                and row["metric"] == metric
            ]
            rows.sort(key=lambda row: float(row["p_hk"]))
            x_values = np.asarray([row["p_hk"] for row in rows])
            means = np.asarray([row["mean_difference"] for row in rows])
            lows = np.asarray([row["ci95_low"] for row in rows])
            highs = np.asarray([row["ci95_high"] for row in rows])
            color = colors[contrast_label]
            ax.plot(
                x_values, means, marker="o", markersize=3.5,
                linewidth=1.6, color=color)
            ax.fill_between(
                x_values, lows, highs, color=color, alpha=0.15)
            ax.axhline(0.0, color="#111827", linestyle="--", linewidth=0.9)
            ax.grid(alpha=0.22, linestyle="--")
            ax.set_title(METRIC_LABELS[metric], fontsize=10)
            if column_index == 0:
                ax.set_ylabel(
                    "{}\nTime-AUC paired difference\n(95% CI)".format(
                        contrast_label))
            if row_index == 1:
                ax.set_xlabel(r"$p_{HK}$")
    fig.suptitle(
        "Paired time-AUC contrasts across network structure",
        fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    auc_figure_path = FIGURE_DIR / "fig_formal_trajectory_auc_contrasts.png"
    fig.savefig(str(auc_figure_path), dpi=300, bbox_inches="tight")
    plt.close(fig)

    output = {
        "validation": validation,
        "design": raw["control"],
        "analysis_plan": {
            "primary_path_endpoint": (
                "time-averaged trapezoidal AUC of order parameter from "
                "5 to 200 sweeps"),
            "secondary_path_endpoints": PATH_METRICS[1:],
            "point_auc_multiplicity": (
                "Holm correction across 21 p_HK values within each endpoint "
                "and contrast"),
            "global_auc_multiplicity": (
                "Holm correction across five endpoints within each contrast"),
            "segment_multiplicity": (
                "Holm correction across early, middle, and late segments "
                "within each contrast"),
            "landmark_multiplicity": (
                "Holm correction across 50, 100, 150, and 200 sweeps"),
            "test": "two-sided paired t test on per-seed differences",
        },
        "global_auc_contrasts": global_auc_results,
        "segment_contrasts": segment_rows,
        "global_landmark_contrasts": global_landmark_rows,
    }
    json_path = ANALYSIS_DIR / "formal_trajectory_analysis.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)

    report_path = ANALYSIS_DIR / "FORMAL_TRAJECTORY_REPORT.md"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("# 正式逐种子时间轨迹分析\n\n")
        handle.write("## 验收\n\n")
        handle.write("- 状态：{}。\n".format(validation["status"]))
        handle.write("- 重建：21个 p_HK × 3个模型 × 20个配对种子，"
                     "每条轨迹40个记录点，终点为200 sweeps。\n")
        handle.write("- 已逐值核对 {} 个终态指标值；最大绝对误差为 {:.3g}。\n".format(
            validation["terminal_values_checked"],
            validation["maximum_absolute_terminal_error"]))
        handle.write("- 独立单位为每个种子对应的仿真重复；时间点不是独立样本。\n\n")
        handle.write("## 跨 p_HK 与时间的总体路径对比\n\n")
        for row in global_auc_rows:
            handle.write(
                "- {}，{}：平均曲面差 {:.6g}，95% CI [{:.6g}, {:.6g}]，"
                "dz={:.3f}，原始 p={:.4g}，Holm-5 p={:.4g}。\n".format(
                    row["contrast"], row["metric_label"],
                    row["mean_surface_difference"], row["ci95_low"],
                    row["ci95_high"], row["cohen_dz"],
                    row["paired_t_p_raw"],
                    row["paired_t_p_holm_5_endpoints"]))
        handle.write("\n## 序参数的早中晚分段\n\n")
        for row in segment_rows:
            handle.write(
                "- {}，{}（{}–{} sweeps）：平均曲面差 {:.6g}，95% CI "
                "[{:.6g}, {:.6g}]，dz={:.3f}，Holm-3 p={:.4g}。\n".format(
                    row["contrast"], row["segment"],
                    int(row["start_sweep"]), int(row["end_sweep"]),
                    row["mean_surface_difference"], row["ci95_low"],
                    row["ci95_high"], row["cohen_dz"],
                    row["paired_t_p_holm_3_segments"]))
        handle.write("\n## 解释边界\n\n")
        handle.write("- 主结论依据整条曲线和跨 p_HK 的总体效应，不根据单个时间点或误差带重叠作判断。\n")
        handle.write("- 未观察到差异不等同于严格等效；所有结论以效应量和95% CI限定。\n")

    return {
        "validation": validation,
        "global_auc_rows": global_auc_rows,
        "segment_rows": segment_rows,
        "artifacts": [
            raw_source_path, point_auc_path, global_auc_path, segment_path,
            landmark_path, global_landmark_path, curve_path, json_path,
            report_path, figure_path, auc_figure_path,
        ],
    }


def main():
    for path in (RAW_PATH, RAW_META_PATH, ABLATION_PATH):
        if not path.exists():
            raise SystemExit("missing required input: {}".format(path))
    with RAW_PATH.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    with RAW_META_PATH.open("r", encoding="utf-8") as handle:
        raw_meta = json.load(handle)
    with ABLATION_PATH.open("r", encoding="utf-8") as handle:
        ablation = json.load(handle)
    validation = validate(raw, raw_meta, ablation)
    if validation["errors"]:
        print("VALIDATION=FAIL")
        for error in validation["errors"][:50]:
            print("ERROR={}".format(error))
        raise SystemExit(1)
    results = build_outputs(raw, validation)
    print("VALIDATION={}".format(validation["status"]))
    print("TERMINAL_VALUES_CHECKED={}".format(
        validation["terminal_values_checked"]))
    print("MAX_TERMINAL_ERROR={:.12g}".format(
        validation["maximum_absolute_terminal_error"]))
    for row in results["global_auc_rows"]:
        print(
            "GLOBAL {} {} mean={:.8g} CI=[{:.8g},{:.8g}] dz={} p_holm={:.8g}".format(
                row["contrast"], row["metric"],
                row["mean_surface_difference"], row["ci95_low"],
                row["ci95_high"], row["cohen_dz"],
                row["paired_t_p_holm_5_endpoints"]))
    for path in results["artifacts"]:
        print("WROTE={}".format(path))


if __name__ == "__main__":
    main()
