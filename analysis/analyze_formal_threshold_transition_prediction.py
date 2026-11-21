"""Validate, compare, and report the forward transition-prediction experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from figure_qa import (  # noqa: E402
    add_panel_label, audit_pdf_text_geometry,
    require_matplotlib_panel_alignment,
)
from statistics_utils import summarize  # noqa: E402
from config import METRICS  # noqa: E402


FORMAL_ROOT = ROOT / "output" / "formal_revision"
DATA_PATH = FORMAL_ROOT / "data" / "exp_formal_threshold_transition_prediction_revision_v3.json"
META_PATH = FORMAL_ROOT / "data" / "exp_formal_threshold_transition_prediction_revision_v3.meta.json"
ANALYSIS_DIR = FORMAL_ROOT / "analysis"
FIGURE_DIR = FORMAL_ROOT / "figures"
REPORT_PATH = ANALYSIS_DIR / "FORMAL_THRESHOLD_TRANSITION_PREDICTION_REPORT.md"
SUMMARY_PATH = ANALYSIS_DIR / "formal_threshold_transition_prediction_summary.json"
SUMMARY_CSV_PATH = ANALYSIS_DIR / "formal_threshold_transition_prediction_summary.csv"
SEED_CSV_PATH = ANALYSIS_DIR / "formal_threshold_transition_prediction_seed_level.csv"
RASTER_DPI = 600
FIGURE_STEM = "fig_formal_threshold_transition_prediction"

P_VALUES = [round(index * 0.05, 2) for index in range(21)]
EPS_VALUES = [round(index * 0.025, 3) for index in range(23)]
SEEDS = [run_i * 997 + 7 for run_i in range(20)]
TRANSITION_ORDER_WINDOW_WIDTH = float(
    METRICS["transition_order_window_width"])
TRANSITION_CUTOFF = float(METRICS["transition_order_cutoff"])
COLORS = {
    "uniform": "#596273",
    "meanfield": "#167D9A",
    "degree": "#5B8DB8",
    "structure": "#378A5B",
    "full": "#B5473A",
}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "font.size": 7.5,
    "axes.labelsize": 8.2,
    "axes.titlesize": 7.5,
    "xtick.labelsize": 6.2,
    "ytick.labelsize": 6.2,
    "legend.fontsize": 7.0,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_transition_metric_config(control):
    """Return validation errors for the prespecified transition criterion."""
    errors = []
    expected_metric = (
        "maximum fraction of opinions in any interval of configured width")
    if control.get("transition_metric") != expected_metric:
        errors.append("transition metric mismatch")
    try:
        width = float(control.get("transition_order_window_width"))
    except (TypeError, ValueError):
        width = float("nan")
    if (not np.isfinite(width)
            or abs(width - TRANSITION_ORDER_WINDOW_WIDTH) > 1e-12):
        errors.append("transition order-window width mismatch")
    try:
        cutoff = float(control.get("transition_cutoff"))
    except (TypeError, ValueError):
        cutoff = float("nan")
    if (not np.isfinite(cutoff)
            or abs(cutoff - TRANSITION_CUTOFF) > 1e-12):
        errors.append("transition order cutoff mismatch")
    expected_formula = "max_x count_i(x_i in [x, x + window_width]) / N"
    if control.get("transition_order_parameter_formula") != expected_formula:
        errors.append("transition order-parameter formula mismatch")
    criterion = control.get("transition_criterion", "")
    if "first forward-grid epsilon" not in criterion:
        errors.append("transition criterion description mismatch")
    if ">= {:.6g}".format(TRANSITION_CUTOFF) not in criterion:
        errors.append("transition criterion cutoff description mismatch")
    return errors


def validate(payload, metadata, require_formal_design=True):
    """Validate either the prespecified formal design or a self-consistent pilot.

    Non-formal designs are accepted only when explicitly requested by the CLI;
    the default remains strict validation against the manuscript's formal grid.
    """
    errors = []
    control = payload.get("control", {})
    try:
        n_nodes = int(control.get("N", 0))
        m_value = int(control.get("m", 0))
        n_runs = int(control.get("num_runs", 0))
        p_values = [float(value) for value in control.get("p_values", [])]
        eps_values = [float(value) for value in control.get("eps_values", [])]
        seeds = [int(value) for value in control.get("seeds", [])]
        sweeps = float(control.get("sweeps_per_epsilon", float("nan")))
        num_steps = int(control.get("num_steps_per_epsilon", -1))
    except (TypeError, ValueError):
        n_nodes, m_value, n_runs = 0, 0, 0
        p_values, eps_values, seeds = [], [], []
        sweeps, num_steps = float("nan"), -1
        errors.append("invalid or missing design fields")

    if n_nodes < 2 or m_value < 1 or m_value >= n_nodes:
        errors.append("invalid network configuration")
    if n_runs < 1 or metadata.get("n_runs") != n_runs:
        errors.append("run count mismatch")
    if len(seeds) != n_runs or len(set(seeds)) != len(seeds):
        errors.append("seed schedule length/uniqueness mismatch")
    if metadata.get("seeds") not in (None, seeds):
        errors.append("metadata seed schedule mismatch")
    if (not p_values or any(not np.isfinite(value) or value < 0.0 or value > 1.0
                            for value in p_values)
            or p_values != sorted(set(p_values))):
        errors.append("invalid p_HK grid")
    if (not eps_values or any(not np.isfinite(value) or value < 0.0 or value > 1.0
                              for value in eps_values)
            or eps_values != sorted(set(eps_values))):
        errors.append("invalid epsilon grid")
    if not np.isfinite(sweeps) or sweeps <= 0.0:
        errors.append("invalid per-threshold horizon")
        expected_steps = None
    else:
        expected_steps = int(round(n_nodes * sweeps))
    if expected_steps is None or num_steps != expected_steps:
        errors.append("elementary update count mismatch")

    if require_formal_design:
        if n_nodes != 2000 or m_value != 4:
            errors.append("formal network configuration mismatch")
        if n_runs != 20:
            errors.append("formal run count mismatch")
        if p_values != P_VALUES:
            errors.append("formal p_HK grid mismatch")
        if eps_values != EPS_VALUES:
            errors.append("formal epsilon grid mismatch")
        if abs(sweeps - 100.0) > 1e-12:
            errors.append("formal per-threshold horizon mismatch")
        if seeds != SEEDS:
            errors.append("formal seed schedule mismatch")
    metric_errors = validate_transition_metric_config(control)
    errors.extend(metric_errors)
    metadata_metric = metadata.get("transition_metric_config", {})
    expected_metadata_metric = {
        "metric": control.get("transition_metric"),
        "order_window_width": control.get("transition_order_window_width"),
        "cutoff": control.get("transition_cutoff"),
        "formula": control.get("transition_order_parameter_formula"),
    }
    if metadata_metric != expected_metadata_metric:
        errors.append("transition metric metadata mismatch")
    variants = control.get("variants", {})
    if set(variants) != {
            "uniform_reference", "structure_only", "structure_full"}:
        errors.append("model-variant set mismatch")
    if float(variants.get("uniform_reference", {}).get("alpha", -1)) != 0.0:
        errors.append("uniform reference must use alpha=0")

    checked_hashes = 0
    for filename, recorded_hash in sorted(
            metadata.get("source_sha256", {}).items()):
        source_path = ROOT / filename
        if not source_path.exists() or sha256(source_path) != recorded_hash:
            errors.append("source hash mismatch: {}".format(filename))
        checked_hashes += 1

    data = payload.get("data", {})
    expected_p_keys = ["{:.2f}".format(value) for value in p_values]
    if sorted(data, key=float) != expected_p_keys:
        errors.append("p_HK data blocks mismatch")
    n_steps = int(control.get("num_steps_per_epsilon", 0))
    for p_key in expected_p_keys:
        block = data.get(p_key, {})
        if block.get("seeds") != seeds:
            errors.append("seed schedule mismatch at p={}".format(p_key))
        runs = block.get("runs", {})
        if sorted((int(key) for key in runs)) != sorted(seeds):
            errors.append("incomplete seed runs at p={}".format(p_key))
            continue
        for seed_key, run in runs.items():
            if int(run.get("seed", -1)) != int(seed_key):
                errors.append("seed key mismatch at p={}, seed={}".format(
                    p_key, seed_key))
            gamma = float(run.get("gamma_G", -1))
            gamma_dyadic = float(run.get("gamma_dyadic_proposal", -1))
            if not (0.0 <= gamma < 1.0 and 0.0 <= gamma_dyadic < 1.0):
                errors.append("invalid structural expansion at p={}, seed={}".format(
                    p_key, seed_key))
            variant_data = run.get("variants", {})
            for variant in ("uniform_reference", "structure_only",
                            "structure_full"):
                item = variant_data.get(variant, {})
                orders = item.get("order_parameter", [])
                interactions = item.get("interactions", [])
                if len(orders) != len(eps_values) \
                        or len(interactions) != len(eps_values):
                    errors.append("incomplete epsilon scan {} at p={}, seed={}".format(
                        variant, p_key, seed_key))
                    continue
                if any(not (0.0 <= float(value) <= 1.0) for value in orders):
                    errors.append("invalid compact order {} at p={}, seed={}".format(
                        variant, p_key, seed_key))
                for row in interactions:
                    if (row.get("edge_count", -1) + row.get("tri_count", -1)
                            != n_steps):
                        errors.append("interaction opportunity count mismatch")
                    if (row.get("tri_success", -1)
                            + row.get("tri_fallback", -1)
                            != row.get("tri_count", -2)):
                        errors.append("triangle outcome count mismatch")
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors[:100],
        "p_points": len(data),
        "runs_per_p": len(seeds),
        "threshold_points": len(eps_values),
        "source_hashes_checked": checked_hashes,
    }


def first_crossing(values, eps_values, cutoff=TRANSITION_CUTOFF):
    """Return the first grid threshold meeting the high-order criterion."""
    for epsilon, value in zip(eps_values, values):
        if float(value) >= float(cutoff):
            return float(epsilon)
    return None


def crossing_index(values, cutoff=TRANSITION_CUTOFF):
    for index, value in enumerate(values):
        if float(value) >= float(cutoff):
            return index
    return None


def major_p_ticks(p_values, major_step=0.10, tolerance=1e-9):
    """Select sparse, grid-aligned major ticks without changing plotted data."""
    step = float(major_step)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("major tick step must be finite and positive")
    ticks = []
    for value in p_values:
        p_value = float(value)
        multiple = round(p_value / step)
        if abs(p_value - multiple * step) <= float(tolerance):
            ticks.append(p_value)
    if not ticks:
        raise ValueError("p grid contains no values aligned to major tick step")
    return ticks


def add_figure_legend(fig, ax, **kwargs):
    """Add a figure-level legend using the axes' registered artist labels."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles or not labels:
        raise ValueError("cannot create figure legend: axes have no labeled artists")
    return fig.legend(handles, labels, **kwargs)


def trajectory_diagnostics(values, cutoff=TRANSITION_CUTOFF, tolerance=1e-12):
    """Describe non-monotonicity and post-crossing recrossings on an epsilon scan.

    A first-crossing threshold remains the prespecified operational marker.
    These diagnostics prevent it from being mistaken for a monotone or
    absorbing transition when the measured order trajectory fluctuates.
    """
    values = [float(value) for value in values]
    if not values:
        raise ValueError("trajectory diagnostics require at least one value")
    decreases = sum(
        1 for left, right in zip(values[:-1], values[1:])
        if right < left - float(tolerance))
    above = [value >= float(cutoff) for value in values]
    first = next((index for index, flag in enumerate(above) if flag), None)
    recrossings = 0
    if first is not None:
        recrossings = sum(
            1 for index in range(first, len(above) - 1)
            if above[index] and not above[index + 1])
    return {
        "decrease_count": int(decreases),
        "nonmonotone": bool(decreases > 0),
        "first_crossing_index": first,
        "cutoff_recrossing_count": int(recrossings),
        "cutoff_recrossed": bool(recrossings > 0),
    }


def build_seed_rows(payload):
    control = payload["control"]
    eps_values = [float(value) for value in control["eps_values"]]
    alpha = float(control["variants"]["structure_only"]["alpha"])
    transition_cutoff = float(control["transition_cutoff"])
    rows = []
    transition_counts = {}
    for p in control["p_values"]:
        p_key = "{:.2f}".format(float(p))
        block = payload["data"][p_key]
        p_rows = []
        for seed in control["seeds"]:
            run = block["runs"][str(seed)]
            gamma = float(run["gamma_G"])
            gamma_dyadic = float(run["gamma_dyadic_proposal"])
            base_orders = run["variants"]["uniform_reference"]["order_parameter"]
            struct_orders = run["variants"]["structure_only"]["order_parameter"]
            full_orders = run["variants"]["structure_full"]["order_parameter"]
            base_c = first_crossing(
                base_orders, eps_values, cutoff=transition_cutoff)
            struct_c = first_crossing(
                struct_orders, eps_values, cutoff=transition_cutoff)
            full_c = first_crossing(
                full_orders, eps_values, cutoff=transition_cutoff)
            base_index = crossing_index(base_orders, cutoff=transition_cutoff)
            struct_index = crossing_index(
                struct_orders, cutoff=transition_cutoff)
            full_index = crossing_index(full_orders, cutoff=transition_cutoff)
            trajectory_checks = {
                "uniform_reference": trajectory_diagnostics(
                    base_orders, cutoff=transition_cutoff),
                "structure_only": trajectory_diagnostics(
                    struct_orders, cutoff=transition_cutoff),
                "full_model": trajectory_diagnostics(
                    full_orders, cutoff=transition_cutoff),
            }

            predicted_mean = (
                max(0.0, (base_c - gamma) / (1.0 - gamma))
                if base_c is not None else None)
            predicted_degree = (
                max(0.0, (base_c - gamma_dyadic) / (1.0 - gamma_dyadic))
                if base_c is not None else None)
            structure_error_mean = (
                struct_c - predicted_mean
                if struct_c is not None and predicted_mean is not None else None)
            structure_error_degree = (
                struct_c - predicted_degree
                if struct_c is not None and predicted_degree is not None else None)
            full_error_mean = (
                full_c - predicted_mean
                if full_c is not None and predicted_mean is not None else None)

            full_interaction = (
                run["variants"]["structure_full"]["interactions"][full_index]
                if full_index is not None else {})
            tri_count = int(full_interaction.get("tri_count", 0))
            n_steps = int(control["num_steps_per_epsilon"])
            row = {
                "p_hk": float(p),
                "seed": int(seed),
                "n_edges": int(run["n_edges"]),
                "gamma_G": gamma,
                "gamma_dyadic_proposal": gamma_dyadic,
                "epsilon_c_uniform_reference": base_c,
                "epsilon_pred_mean_edge": predicted_mean,
                "epsilon_pred_dyadic_proposal": predicted_degree,
                "epsilon_c_structure_only": struct_c,
                "epsilon_c_full_model": full_c,
                "error_structure_only_mean_edge": structure_error_mean,
                "error_structure_only_dyadic": structure_error_degree,
                "error_full_model_mean_edge": full_error_mean,
                "uniform_reference_grid_index": base_index,
                "structure_only_grid_index": struct_index,
                "full_model_grid_index": full_index,
                "uniform_reference_decrease_count": (
                    trajectory_checks["uniform_reference"]["decrease_count"]),
                "structure_only_decrease_count": (
                    trajectory_checks["structure_only"]["decrease_count"]),
                "full_model_decrease_count": (
                    trajectory_checks["full_model"]["decrease_count"]),
                "uniform_reference_cutoff_recrossing_count": (
                    trajectory_checks["uniform_reference"]["cutoff_recrossing_count"]),
                "structure_only_cutoff_recrossing_count": (
                    trajectory_checks["structure_only"]["cutoff_recrossing_count"]),
                "full_model_cutoff_recrossing_count": (
                    trajectory_checks["full_model"]["cutoff_recrossing_count"]),
                "full_triangle_attempt_rate_at_transition": (
                    tri_count / float(n_steps) if full_index is not None else None),
                "full_triangle_success_fraction_at_transition": (
                    float(full_interaction["tri_success"]) / tri_count
                    if tri_count else None),
                "full_triangle_fallback_fraction_at_transition": (
                    float(full_interaction["tri_fallback"]) / tri_count
                    if tri_count else None),
            }
            rows.append(row)
            p_rows.append(row)
        transition_counts[p_key] = {
            "n_runs": len(p_rows),
            "uniform_reference": sum(
                row["epsilon_c_uniform_reference"] is not None for row in p_rows),
            "structure_only": sum(
                row["epsilon_c_structure_only"] is not None for row in p_rows),
            "full_model": sum(
                row["epsilon_c_full_model"] is not None for row in p_rows),
            "paired_error_n": {
                "structure_only_mean_edge": sum(
                    row["error_structure_only_mean_edge"] is not None
                    for row in p_rows),
                "structure_only_dyadic": sum(
                    row["error_structure_only_dyadic"] is not None
                    for row in p_rows),
                "full_model_mean_edge": sum(
                    row["error_full_model_mean_edge"] is not None
                    for row in p_rows),
            },
            "nonmonotone_runs": {
                "uniform_reference": sum(
                    row["uniform_reference_decrease_count"] > 0
                    for row in p_rows),
                "structure_only": sum(
                    row["structure_only_decrease_count"] > 0
                    for row in p_rows),
                "full_model": sum(
                    row["full_model_decrease_count"] > 0
                    for row in p_rows),
            },
            "cutoff_recrossing_runs": {
                "uniform_reference": sum(
                    row["uniform_reference_cutoff_recrossing_count"] > 0
                    for row in p_rows),
                "structure_only": sum(
                    row["structure_only_cutoff_recrossing_count"] > 0
                    for row in p_rows),
                "full_model": sum(
                    row["full_model_cutoff_recrossing_count"] > 0
                    for row in p_rows),
            },
        }
    return rows, transition_counts, alpha


def build_summaries(rows, transition_counts):
    metrics = [
        "gamma_G",
        "gamma_dyadic_proposal",
        "epsilon_c_uniform_reference",
        "epsilon_pred_mean_edge",
        "epsilon_pred_dyadic_proposal",
        "epsilon_c_structure_only",
        "epsilon_c_full_model",
        "error_structure_only_mean_edge",
        "error_structure_only_dyadic",
        "error_full_model_mean_edge",
        "full_triangle_attempt_rate_at_transition",
        "full_triangle_success_fraction_at_transition",
        "full_triangle_fallback_fraction_at_transition",
    ]
    summaries = {}
    summary_rows = []
    p_values = sorted(float(key) for key in transition_counts)
    for p in p_values:
        p_key = "{:.2f}".format(p)
        subset = [row for row in rows if row["p_hk"] == p]
        summaries[p_key] = {
            "transition_counts": transition_counts[p_key],
            "metrics": {},
        }
        for metric in metrics:
            values = [
                row[metric] for row in subset if row[metric] is not None
            ]
            if not values:
                continue
            item = summarize(values)
            summaries[p_key]["metrics"][metric] = item
            summary_rows.append({
                "p_hk": p,
                "metric": metric,
                **{key: item[key] for key in (
                    "n", "mean", "sd", "se", "ci95_low", "ci95_high",
                    "min", "max")},
            })
    return summaries, summary_rows


def write_seed_csv(rows):
    if not rows:
        raise ValueError("no seed-level rows")
    with SEED_CSV_PATH.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(rows):
    fields = [
        "p_hk", "metric", "n", "mean", "sd", "se",
        "ci95_low", "ci95_high", "min", "max",
    ]
    with SUMMARY_CSV_PATH.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(summaries):
    p_values = sorted(float(key) for key in summaries)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), sharex=True)
    fig.subplots_adjust(
        left=0.10, right=0.985, top=0.68, bottom=0.22, wspace=0.30)
    panel_labels = ("a", "b")
    for ax, label in zip(axes, panel_labels):
        add_panel_label(ax, label)

    panel_a_series = [
        ("Uniform reference", "epsilon_c_uniform_reference",
         COLORS["uniform"], "o", "-"),
        ("Mean-edge prediction", "epsilon_pred_mean_edge",
         COLORS["meanfield"], "s", "--"),
        ("Structure-only simulated", "epsilon_c_structure_only",
         COLORS["structure"], "^", "-"),
        ("Full model simulated", "epsilon_c_full_model",
         COLORS["full"], "D", "-"),
    ]
    offsets_a = [-0.018, -0.006, 0.006, 0.018]
    for (label, metric, color, marker, linestyle), offset in zip(
            panel_a_series, offsets_a):
        means, lows, highs = [], [], []
        for p in p_values:
            item = summaries["{:.2f}".format(p)]["metrics"].get(metric)
            if item is None:
                means.append(np.nan)
                lows.append(np.nan)
                highs.append(np.nan)
            else:
                means.append(item["mean"])
                lows.append(item["ci95_low"])
                highs.append(item["ci95_high"])
        axes[0].errorbar(
            np.asarray(p_values) + offset, means,
            yerr=[np.asarray(means) - np.asarray(lows),
                  np.asarray(highs) - np.asarray(means)],
            color=color, marker=marker, linestyle=linestyle,
            linewidth=0.95, markersize=3.0, capsize=1.8,
            label=label)

    panel_b_series = [
        ("Structure-only: mean-edge", "error_structure_only_mean_edge",
         COLORS["meanfield"], "o"),
        ("Structure-only: degree-weighted", "error_structure_only_dyadic",
         COLORS["degree"], "s"),
        ("Full model: mean-edge", "error_full_model_mean_edge",
         COLORS["full"], "^"),
    ]
    offsets_b = [-0.014, 0.0, 0.014]
    for (label, metric, color, marker), offset in zip(
            panel_b_series, offsets_b):
        means, lows, highs = [], [], []
        for p in p_values:
            item = summaries["{:.2f}".format(p)]["metrics"].get(metric)
            if item is None:
                means.append(np.nan)
                lows.append(np.nan)
                highs.append(np.nan)
            else:
                means.append(item["mean"])
                lows.append(item["ci95_low"])
                highs.append(item["ci95_high"])
        axes[1].errorbar(
            np.asarray(p_values) + offset, means,
            yerr=[np.asarray(means) - np.asarray(lows),
                  np.asarray(highs) - np.asarray(means)],
            color=color, marker=marker, linestyle="-",
            linewidth=0.9, markersize=3.0, capsize=1.8,
            label=label)
    axes[1].axhline(0.0, color="#606060", linewidth=0.65, linestyle=":")
    axes[0].set_ylabel("Transition threshold epsilon0")
    axes[1].set_ylabel("Observed - predicted epsilon0")
    transition_upper_bounds = [
        summaries[key]["metrics"][metric]["ci95_high"]
        for key in summaries
        for metric in (
            "epsilon_c_uniform_reference", "epsilon_pred_mean_edge",
            "epsilon_c_structure_only", "epsilon_c_full_model")
        if summaries[key]["metrics"].get(metric) is not None
    ]
    y_upper = max([0.58] + transition_upper_bounds) + 0.025
    axes[0].set_ylim(0.0, y_upper)
    transition_metrics = (
        "epsilon_c_uniform_reference", "epsilon_pred_mean_edge",
        "epsilon_c_structure_only", "epsilon_c_full_model")
    error_metrics = (
        "error_structure_only_mean_edge", "error_structure_only_dyadic",
        "error_full_model_mean_edge")
    if not any(
            summaries[key]["metrics"].get(metric)
            for key in summaries for metric in transition_metrics):
        axes[0].text(
            0.5, 0.52, "No transition crossings observed in this scan",
            transform=axes[0].transAxes, ha="center", va="center",
            fontsize=7.2, color="#4B5563")
    if not any(
            summaries[key]["metrics"].get(metric)
            for key in summaries for metric in error_metrics):
        axes[1].text(
            0.5, 0.76, "Prediction errors unavailable: no paired crossings",
            transform=axes[1].transAxes, ha="center", va="center",
            fontsize=7.2, color="#4B5563")
    # Keep all p_HK observations, but label only major ticks. Labeling
    # every 0.05 on a two-panel, single-column-width figure causes adjacent
    # tick-label bounding boxes to overlap and fails the PDF geometry audit.
    p_major_ticks = major_p_ticks(p_values, major_step=0.10)
    axes[0].set_xticks(p_major_ticks)
    axes[1].set_xticks(p_major_ticks)
    axes[0].set_xlabel("pHK")
    axes[1].set_xlabel("pHK")
    for ax in axes:
        ax.grid(axis="y", color="#D1D5DB", linewidth=0.45, alpha=0.75)

    add_figure_legend(
        fig, axes[0], loc="upper center", bbox_to_anchor=(0.52, 0.985),
        ncol=4, frameon=False, fontsize=6.4)
    add_figure_legend(
        fig, axes[1], loc="upper center", bbox_to_anchor=(0.52, 0.855),
        ncol=3, frameon=False, fontsize=6.2)

    base = FIGURE_DIR / FIGURE_STEM
    require_matplotlib_panel_alignment(
        fig, axes, panel_labels,
        base.with_suffix(".alignment-layout.json"),
        row_groups=(panel_labels,), column_groups=(),
        grid_cells={"a": (0, 1, 0, 1), "b": (0, 1, 1, 2)},
    )
    for suffix, options in (
            (".svg", {}), (".pdf", {}),
            (".tiff", {"dpi": RASTER_DPI}),
            (".png", {"dpi": 300})):
        fig.savefig(str(base.with_suffix(suffix)), **options)
    plt.close(fig)
    audit_pdf_text_geometry(
        base.with_suffix(".pdf"),
        base.with_suffix(".pdf-text-geometry-audit.json"))
    return base


def write_report(validation, control, summaries, transition_counts, figure_base):
    lines = [
        "# {} forward-transition prediction audit".format(
            validation.get("design_type", "Formal")),
        "",
        "## Design and operational transition",
        "",
        "The paired scan uses N={}, m={}, p_HK={}, {} seeds, epsilon0={} and {} sweeps at each threshold.".format(
            control["N"], control["m"], control["p_values"],
            control["num_runs"], control["eps_values"],
            control["sweeps_per_epsilon"]),
        "",
        "For each seed and model, the operational transition is the first point on the ascending epsilon0 grid at which the final compact order parameter is at least {:.3f}. The order parameter is the maximum fraction of opinions contained in any interval of width {:.3f}; thus the cutoff requires at least {:.1f}% of opinions in one such interval. The smallest adjacent epsilon-grid spacing is {:.3f}; this is a finite-horizon first-crossing marker, not an asymptotic critical point. No interpolation between grid points is used. Runs that do not cross within the scanned grid are reported as not observed and are excluded from threshold means and prediction-error means; their counts and the paired error sample sizes are disclosed below.".format(
            control["transition_cutoff"],
            control["transition_order_window_width"],
            100.0 * control["transition_cutoff"],
            min((right - left for left, right in zip(
                control["eps_values"][:-1], control["eps_values"][1:])),
                default=float("nan"))),
        "",
        "The uniform reference is the asynchronous pairwise BCM with alpha=0. The structure-only model uses the Jaccard edge thresholds without triangle triggering. The full-model scan adds triangle triggering, three-edge gates, fallback, and third-party damping. All variants share each seed's network and initial opinions; each threshold point has a deterministic update stream shared across variants.",
        "",
        "## Prediction and edge-selection diagnostic",
        "",
        "The mean-field prediction uses the exact unweighted network statistic Gamma_G = mean_edges[1-exp(-alpha*w_ij)] and epsilon0,pred = max(0,(epsilon0,c,uniform-Gamma_G)/(1-Gamma_G)). This treats the edge-average effective threshold as a first-order surrogate for the transition condition.",
        "",
        "Uniform node sampling followed by uniform neighbor sampling does not sample undirected edges uniformly. Before confidence gating, the dyadic proposal probability for edge (u,v) is (1/N)(1/k_u+1/k_v). A degree-weighted Gamma using these proposal probabilities is reported as a diagnostic alternative. It corrects dyadic proposal frequency only; it does not account for acceptance conditioning or triangle selection. The full-model comparison exposes the additional effect of triangle attempts and their state-dependent gates.",
        "",
        "For each seed s, the mean-edge prediction error is d_s = epsilon_c,s(structure) - max{0, [epsilon_c,s(uniform) - Gamma_G,s] / [1 - Gamma_G,s]}; the degree-proposal diagnostic substitutes Gamma_dyadic,s for Gamma_G,s. Intervals are two-sided 95% Student-t intervals across the available paired seed-level errors. Each table cell reports its own n. These intervals quantify between-seed variability in the discrete first-crossing errors; they do not include the additional uncertainty induced by the epsilon-grid resolution stated above. Positive errors mean the simulated transition occurs at a larger baseline threshold than predicted.",
        "",
        "Validation: **{}**; {} p_HK values, {} seeds per value, {} source hashes checked.".format(
            validation["status"], validation["p_points"],
            validation["runs_per_p"], validation["source_hashes_checked"]),
        "",
        "## Results",
        "",
        "| p_HK | Gamma edge mean | Gamma dyadic-proposal weighted | uniform-reference transition | mean-edge prediction | dyadic prediction | structure-only simulated | full-model simulated | structure-only error (mean-edge) | structure-only error (dyadic) | full-model error (mean-edge) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    metric_names = {
        "gamma_G": "gamma_G",
        "gamma_dyadic_proposal": "gamma_dyadic_proposal",
        "epsilon_c_uniform_reference": "epsilon_c_uniform_reference",
        "epsilon_pred_mean_edge": "epsilon_pred_mean_edge",
        "epsilon_pred_dyadic_proposal": "epsilon_pred_dyadic_proposal",
        "epsilon_c_structure_only": "epsilon_c_structure_only",
        "epsilon_c_full_model": "epsilon_c_full_model",
        "error_structure_only_mean_edge": "error_structure_only_mean_edge",
        "error_structure_only_dyadic": "error_structure_only_dyadic",
        "error_full_model_mean_edge": "error_full_model_mean_edge",
    }
    p_values = [float(value) for value in control["p_values"]]
    for p in p_values:
        p_key = "{:.2f}".format(p)
        metrics = summaries[p_key]["metrics"]
        values = []
        for key in metric_names.values():
            item = metrics.get(key)
            if item is None:
                values.append("NA")
            else:
                values.append("{:.3f} [{:.3f}, {:.3f}], n={}".format(
                    item["mean"], item["ci95_low"], item["ci95_high"],
                    item["n"]))
        lines.append("| {:.2f} | {} |".format(p, " | ".join(values)))
        counts = transition_counts[p_key]
        lines.append("")
        paired_n = counts["paired_error_n"]
        nonmonotone = counts["nonmonotone_runs"]
        recross = counts["cutoff_recrossing_runs"]
        lines.append(
            "At p_HK={:.2f}, first crossings were observed for {}/{} uniform references, {}/{} structure-only runs, and {}/{} full-model runs. Paired error n: structure-only/mean-edge={}/{}, structure-only/degree-proposal={}/{}, full-model/mean-edge={}/{}.".format(
                p, counts["uniform_reference"], control["num_runs"],
                counts["structure_only"], control["num_runs"],
                counts["full_model"], control["num_runs"],
                paired_n["structure_only_mean_edge"], control["num_runs"],
                paired_n["structure_only_dyadic"], control["num_runs"],
                paired_n["full_model_mean_edge"], control["num_runs"]))
        lines.append(
            "Non-monotone order trajectories (at least one downward grid step): uniform={}/{}, structure-only={}/{}, full={}/{}. Post-first-crossing recrossings below the configured cutoff ({:.3f}): uniform={}/{}, structure-only={}/{}, full={}/{}.".format(
                nonmonotone["uniform_reference"], control["num_runs"],
                nonmonotone["structure_only"], control["num_runs"],
                nonmonotone["full_model"], control["num_runs"],
                control["transition_cutoff"],
                recross["uniform_reference"], control["num_runs"],
                recross["structure_only"], control["num_runs"],
                recross["full_model"], control["num_runs"]))
        lines.append("")

    lines += [
        "## Interpretation limits",
        "",
        ("This is a small pilot used only to verify the real-network simulation-to-analysis pipeline. Its numerical values are not formal evidence and must not be used in the manuscript." if validation.get("design_type") == "Pilot (software validation only)" else "These results correspond to the prespecified formal experiment design."),
        "",
        "The edge-mean formula is an approximation, not a theorem for the asynchronous network process. The degree-weighted proxy addresses only the unconditioned dyadic proposal law. Acceptance depends on current opinions; the full model also changes which three-edge gates and fallback interactions occur. Prediction errors quantify these finite-horizon deviations for the tested graph ensemble, threshold grid, order criterion, and update rule.",
        "",
        "Figure exports: {}. Seed-level values and summary tables are provided in the accompanying CSV files.".format(
            figure_base.name),
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    global DATA_PATH, META_PATH, ANALYSIS_DIR, FIGURE_DIR
    global REPORT_PATH, SUMMARY_PATH, SUMMARY_CSV_PATH, SEED_CSV_PATH
    global FIGURE_STEM
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze the paired forward-transition prediction scan.")
    parser.add_argument("--input-json", type=Path, default=DATA_PATH)
    parser.add_argument("--metadata-json", type=Path, default=META_PATH)
    parser.add_argument("--output-dir", type=Path, default=ANALYSIS_DIR)
    parser.add_argument(
        "--allow-nonformal-design", action="store_true",
        help="Allow a self-consistent pilot design; never use its values as formal results.")
    args = parser.parse_args(argv)

    DATA_PATH = args.input_json
    META_PATH = args.metadata_json
    ANALYSIS_DIR = args.output_dir
    FIGURE_DIR = ANALYSIS_DIR.parent / "figures"
    FIGURE_STEM = ("fig_formal_threshold_transition_prediction"
                   if not args.allow_nonformal_design
                   else "fig_transition_prediction_pilot")
    REPORT_PATH = ANALYSIS_DIR / "THRESHOLD_TRANSITION_PREDICTION_REPORT.md"
    SUMMARY_PATH = ANALYSIS_DIR / "threshold_transition_prediction_summary.json"
    SUMMARY_CSV_PATH = ANALYSIS_DIR / "threshold_transition_prediction_summary.csv"
    SEED_CSV_PATH = ANALYSIS_DIR / "threshold_transition_prediction_seed_level.csv"

    with DATA_PATH.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    with META_PATH.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    validation = validate(
        payload, metadata,
        require_formal_design=not args.allow_nonformal_design)
    validation["design_type"] = (
        "Pilot (software validation only)" if args.allow_nonformal_design
        else "Formal")
    print("Validation:", validation["status"])
    if validation["errors"]:
        for error in validation["errors"][:50]:
            print(" -", error)
        raise SystemExit(1)

    rows, transition_counts, alpha = build_seed_rows(payload)
    summaries, summary_rows = build_summaries(rows, transition_counts)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    write_seed_csv(rows)
    write_summary_csv(summary_rows)
    figure_base = save_figure(summaries)
    write_report(
        validation, payload["control"], summaries, transition_counts,
        figure_base)

    result = {
        "validation": validation,
        "control": payload["control"],
        "alpha_target": alpha,
        "transition_counts": transition_counts,
        "summaries": summaries,
        "figure": str(figure_base),
        "seed_level_csv": str(SEED_CSV_PATH),
        "summary_csv": str(SUMMARY_CSV_PATH),
    }
    SUMMARY_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(REPORT_PATH)
    print(SUMMARY_PATH)
    print(SEED_CSV_PATH)
    print(SUMMARY_CSV_PATH)
    print(figure_base.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
