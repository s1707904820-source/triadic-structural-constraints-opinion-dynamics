"""Validate and summarize formal threshold-level perturbation recovery data."""

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

FORMAL_ROOT = ROOT / "output" / "formal_revision"
DATA_PATH = FORMAL_ROOT / "data" / "exp_formal_threshold_recovery.json"
META_PATH = FORMAL_ROOT / "data" / "exp_formal_threshold_recovery.meta.json"
ANALYSIS_DIR = FORMAL_ROOT / "analysis"
FIGURE_DIR = FORMAL_ROOT / "figures"
REPORT_PATH = ANALYSIS_DIR / "FORMAL_THRESHOLD_RECOVERY_REPORT.md"
SUMMARY_PATH = ANALYSIS_DIR / "formal_threshold_recovery_summary.json"
CSV_PATH = ANALYSIS_DIR / "formal_threshold_recovery_summary.csv"
TRAJECTORY_CSV_PATH = ANALYSIS_DIR / "formal_threshold_recovery_trajectories.csv"
RASTER_DPI = 600

P_VALUES = [0.0, 0.4, 0.8]
SIGMA_VALUES = [0.02]
EPS_VALUES = [round(0.10 + index * 0.025, 3) for index in range(19)]
SEEDS = [run_i * 997 + 7 for run_i in range(20)]
COLORS = {0.0: "#596273", 0.4: "#167D9A", 0.8: "#B5473A"}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "font.size": 7.5,
    "axes.labelsize": 8.2,
    "axes.titlesize": 7.5,
    "xtick.labelsize": 6.2,
    "ytick.labelsize": 6.2,
    "legend.fontsize": 8.5,
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


def validate(payload, metadata):
    errors = []
    control = payload.get("control", {})
    if control.get("N") != 2000 or control.get("m") != 4:
        errors.append("network configuration mismatch")
    if control.get("num_runs") != 20 or metadata.get("n_runs") != 20:
        errors.append("run count mismatch")
    if control.get("p_values") != P_VALUES:
        errors.append("p_HK grid mismatch")
    if control.get("sigma_values") != SIGMA_VALUES:
        errors.append("sigma grid mismatch")
    if control.get("eps_values") != EPS_VALUES:
        errors.append("epsilon grid mismatch")
    if float(control.get("sweeps_per_epsilon", -1)) != 100.0:
        errors.append("per-threshold horizon mismatch")
    if float(control.get("record_sweeps", -1)) != 1.0:
        errors.append("record interval mismatch")
    if float(control.get("recovery_order_parameter_threshold", -1)) != 1.0:
        errors.append("recovery criterion mismatch")
    if control.get("recovery_persistence_measurements") != 3:
        errors.append("recovery persistence mismatch")
    if metadata.get("seeds") != SEEDS:
        errors.append("seed schedule mismatch")
    checked_hashes = 0
    for filename, recorded_hash in sorted(
            metadata.get("source_sha256", {}).items()):
        source_path = ROOT / filename
        if not source_path.exists() or sha256(source_path) != recorded_hash:
            errors.append("source hash mismatch: {}".format(filename))
        checked_hashes += 1

    data = payload.get("data", {})
    expected_p_keys = ["{:.2f}".format(value) for value in P_VALUES]
    expected_eps_keys = [str(value) for value in EPS_VALUES]
    expected_sigma_keys = ["{:.3f}".format(value) for value in SIGMA_VALUES]
    if sorted(data, key=float) != expected_p_keys:
        errors.append("p_HK data blocks mismatch")
    for p_key in expected_p_keys:
        block = data.get(p_key, {})
        if block.get("num_runs") != 20 or block.get("seeds") != SEEDS:
            errors.append("replication protocol mismatch at p={}".format(p_key))
        if sorted(block.get("by_epsilon", {}), key=float) != expected_eps_keys:
            errors.append("epsilon block mismatch at p={}".format(p_key))
            continue
        time_points = block.get("time_sweeps", [])
        if len(time_points) != 101 or time_points[0] != 0.0:
            errors.append("time grid mismatch at p={}".format(p_key))
        for eps_key in expected_eps_keys:
            cells = block["by_epsilon"][eps_key]
            if sorted(cells) != expected_sigma_keys:
                errors.append("sigma cell mismatch at p={}, eps={}".format(
                    p_key, eps_key))
                continue
            for sigma_key in expected_sigma_keys:
                item = cells[sigma_key]
                recovery = item.get("recovery", {})
                probability = recovery.get("probability", {})
                if probability.get("n") != 20 \
                        or len(probability.get("values", [])) != 20:
                    errors.append("incomplete recovery replicates at p={}, eps={}".format(
                        p_key, eps_key))
                if recovery.get("n_recovered", -1) \
                        + recovery.get("n_censored", -1) != 20:
                    errors.append("event/censor count mismatch at p={}, eps={}".format(
                        p_key, eps_key))
                if not (0.0 <= float(probability.get("ci95_low", -1))
                        <= float(probability.get("ci95_high", 2)) <= 1.0):
                    errors.append("invalid Wilson interval at p={}, eps={}".format(
                        p_key, eps_key))
                if len(recovery.get("time_by_run", [])) != 20 \
                        or len(recovery.get("recovered_by_run", [])) != 20:
                    errors.append("recovery times missing at p={}, eps={}".format(
                        p_key, eps_key))
                for metric in ("opinion_sd", "opinion_range", "order_parameter"):
                    series = item.get("trajectory", {}).get(metric, {})
                    if (len(series.get("mean", [])) != len(time_points)
                            or len(series.get("ci95_low", [])) != len(time_points)
                            or len(series.get("ci95_high", [])) != len(time_points)):
                        errors.append("trajectory mismatch {} at p={}, eps={}".format(
                            metric, p_key, eps_key))
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "p_points": len(data),
        "epsilon_points_per_p": len(EPS_VALUES),
        "runs_per_cell": 20,
        "cells": len(data) * len(EPS_VALUES) * len(SIGMA_VALUES),
        "source_hashes_checked": checked_hashes,
    }


def kaplan_meier_rmst(time_by_run, recovered_by_run, horizon):
    """Integrate the Kaplan-Meier survival curve to the fixed horizon."""
    durations = [
        float(value) if event else float(horizon)
        for value, event in zip(time_by_run, recovered_by_run)
    ]
    event_times = sorted(set(
        float(value) for value, event in zip(time_by_run, recovered_by_run)
        if event and value is not None
    ))
    survival = 1.0
    previous = 0.0
    area = 0.0
    for time in event_times:
        area += survival * (time - previous)
        at_risk = sum(duration >= time for duration in durations)
        events = sum(
            bool(event) and float(value) == time
            for value, event in zip(time_by_run, recovered_by_run)
        )
        if at_risk:
            survival *= 1.0 - float(events) / at_risk
        previous = time
    area += survival * (float(horizon) - previous)
    return float(area)


def bootstrap_rmst(time_by_run, recovered_by_run, horizon, seed):
    n = len(time_by_run)
    rng = np.random.RandomState(seed)
    point = kaplan_meier_rmst(time_by_run, recovered_by_run, horizon)
    values = []
    for _ in range(2000):
        indices = rng.randint(0, n, size=n)
        values.append(kaplan_meier_rmst(
            [time_by_run[index] for index in indices],
            [recovered_by_run[index] for index in indices],
            horizon))
    low, high = np.percentile(values, [2.5, 97.5])
    return point, float(low), float(high)


def cell(payload, p, epsilon, sigma):
    return payload["data"]["{:.2f}".format(float(p))]["by_epsilon"][
        str(float(epsilon))]["{:.3f}".format(float(sigma))]


def build_rows(payload):
    horizon = float(payload["control"]["sweeps_per_epsilon"])
    rows = []
    for p in payload["control"]["p_values"]:
        for epsilon in payload["control"]["eps_values"]:
            for sigma in payload["control"]["sigma_values"]:
                item = cell(payload, p, epsilon, sigma)
                recovery = item["recovery"]
                rmst, rmst_low, rmst_high = bootstrap_rmst(
                    recovery["time_by_run"], recovery["recovered_by_run"],
                    horizon, seed=20261111 + int(round(p * 1000))
                    + int(round(epsilon * 10000)))
                rows.append({
                    "p_hk": float(p),
                    "epsilon": float(epsilon),
                    "sigma": float(sigma),
                    "n": int(recovery["probability"]["n"]),
                    "n_recovered": int(recovery["n_recovered"]),
                    "n_censored": int(recovery["n_censored"]),
                    "recovery_probability": float(
                        recovery["probability"]["mean"]),
                    "recovery_probability_ci95_low": float(
                        recovery["probability"]["ci95_low"]),
                    "recovery_probability_ci95_high": float(
                        recovery["probability"]["ci95_high"]),
                    "kaplan_meier_rmst_sweeps": rmst,
                    "kaplan_meier_rmst_ci95_low": rmst_low,
                    "kaplan_meier_rmst_ci95_high": rmst_high,
                    "recovered_only_mean_time_sweeps": (
                        recovery["time_sweeps_recovered_only"]["mean"]
                        if recovery["time_sweeps_recovered_only"] else None),
                    "initial_opinion_sd": float(
                        item["initial"]["opinion_sd"]["mean"]),
                    "initial_opinion_range": float(
                        item["initial"]["opinion_range"]["mean"]),
                    "final_opinion_sd": float(
                        item["endpoint"]["opinion_sd"]["mean"]),
                    "final_opinion_range": float(
                        item["endpoint"]["opinion_range"]["mean"]),
                    "final_order_parameter": float(
                        item["endpoint"]["order_parameter"]["mean"]),
                })
    return rows


def write_csv(rows):
    with CSV_PATH.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)



def trajectory_matrices(payload):
    """Return threshold-by-time mean opinion-SD matrices for each p_HK."""
    p_values = [float(value) for value in payload["control"]["p_values"]]
    eps_values = [float(value) for value in payload["control"]["eps_values"]]
    sigma = float(payload["control"]["sigma_values"][0])
    matrices = []
    time_points = None
    for p in p_values:
        block = payload["data"]["{:.2f}".format(p)]
        if time_points is None:
            time_points = [float(value) for value in block["time_sweeps"]]
        elif [float(value) for value in block["time_sweeps"]] != time_points:
            raise ValueError("p_HK blocks do not share a common time grid")
        rows = []
        for epsilon in eps_values:
            item = block["by_epsilon"][str(float(epsilon))][
                "{:.3f}".format(sigma)]
            series = item["trajectory"]["opinion_sd"]["mean"]
            if len(series) != len(time_points):
                raise ValueError(
                    "opinion-SD trajectory length mismatch at p={}, epsilon={}"
                    .format(p, epsilon))
            rows.append([float(value) for value in series])
        matrices.append(np.asarray(rows, dtype=float))
    return p_values, eps_values, time_points, matrices


def write_trajectory_csv(payload):
    """Export all mean and 95% interval values used by the dynamics figure."""
    p_values, eps_values, time_points, _matrices = trajectory_matrices(payload)
    with TRAJECTORY_CSV_PATH.open(
            "w", newline="", encoding="utf-8-sig") as handle:
        fields = [
            "p_hk", "epsilon", "sigma", "sweep",
            "opinion_sd_mean", "opinion_sd_ci95_low",
            "opinion_sd_ci95_high", "n_runs",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        sigma = float(payload["control"]["sigma_values"][0])
        for p in p_values:
            block = payload["data"]["{:.2f}".format(p)]
            for epsilon in eps_values:
                item = block["by_epsilon"][str(float(epsilon))][
                    "{:.3f}".format(sigma)]
                metric = item["trajectory"]["opinion_sd"]
                for index, sweep in enumerate(time_points):
                    writer.writerow({
                        "p_hk": p,
                        "epsilon": epsilon,
                        "sigma": sigma,
                        "sweep": sweep,
                        "opinion_sd_mean": metric["mean"][index],
                        "opinion_sd_ci95_low": metric["ci95_low"][index],
                        "opinion_sd_ci95_high": metric["ci95_high"][index],
                        "n_runs": 20,
                    })
    return TRAJECTORY_CSV_PATH


def save_trajectory_figure(payload):
    """Plot the threshold-by-time opinion-spread trajectories by p_HK."""
    p_values, eps_values, time_points, matrices = trajectory_matrices(payload)
    all_values = np.concatenate([matrix.ravel() for matrix in matrices])
    if not np.all(np.isfinite(all_values)) or np.any(all_values <= 0.0):
        raise ValueError("log-scaled opinion-SD heatmap requires positive data")
    norm = mpl.colors.LogNorm(
        vmin=1e-10, vmax=float(np.max(all_values)))
    fig, axes = plt.subplots(
        1, len(p_values), figsize=(7.2, 3.25), sharey=True, squeeze=False)
    axes = axes[0]
    fig.subplots_adjust(
        left=0.12, right=0.90, top=0.82, bottom=0.22, wspace=0.12)
    panel_labels = tuple("abc"[:len(axes)])
    time_edges = np.arange(-0.5, len(time_points) + 0.5, 1.0)
    epsilon_array = np.asarray(eps_values, dtype=float)
    midpoints = (epsilon_array[:-1] + epsilon_array[1:]) / 2.0
    epsilon_edges = np.concatenate((
        [epsilon_array[0] - (epsilon_array[1] - epsilon_array[0]) / 2.0],
        midpoints,
        [epsilon_array[-1] + (epsilon_array[-1] - epsilon_array[-2]) / 2.0],
    ))
    image = None
    for index, (ax, p, matrix, label) in enumerate(
            zip(axes, p_values, matrices, panel_labels)):
        add_panel_label(ax, label)
        image = ax.pcolormesh(
            time_edges, epsilon_edges, matrix,
            cmap="magma", norm=norm, shading="flat")
        ax.set_title("pHK = {:.1f}".format(p), pad=5)
        ax.set_xlim(time_edges[0], time_edges[-1])
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Time after shock (sweeps)")
        ax.set_yticks(eps_values[::2])
        ax.set_yticklabels(["{:.2f}".format(value)
                            for value in eps_values[::2]])
        ax.grid(False)
        if index == 0:
            ax.set_ylabel("Threshold epsilon")
    colorbar = fig.colorbar(
        image, ax=list(axes), location="right", fraction=0.035, pad=0.025)
    colorbar.set_label("Mean opinion SD", fontsize=7.0)
    colorbar.set_ticks([1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1])
    colorbar.set_ticklabels(
        ["1e-10", "1e-8", "1e-6", "1e-4", "1e-2", "1e-1"])
    colorbar.ax.tick_params(labelsize=6.0, length=2.0, width=0.6)

    base = FIGURE_DIR / "fig_formal_threshold_recovery_dynamics"
    require_matplotlib_panel_alignment(
        fig, axes, panel_labels,
        base.with_suffix(".alignment-layout.json"),
        row_groups=(panel_labels,),
        column_groups=(),
        grid_cells={
            label: (0, 1, index, index + 1)
            for index, label in enumerate(panel_labels)
        },
    )
    for suffix, options in (
            (".svg", {}), (".pdf", {}),
            (".tiff", {"dpi": RASTER_DPI}), (".png", {"dpi": 300})):
        fig.savefig(str(base.with_suffix(suffix)), **options)
    plt.close(fig)
    audit_pdf_text_geometry(
        base.with_suffix(".pdf"),
        base.with_suffix(".pdf-text-geometry-audit.json"))
    return base


def save_figure(payload, rows):
    p_values = payload["control"]["p_values"]
    eps_values = payload["control"]["eps_values"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), sharex=True)
    fig.subplots_adjust(left=0.11, right=0.985, top=0.70, bottom=0.22,
                        wspace=0.30)
    add_panel_label(axes[0], "a")
    add_panel_label(axes[1], "b")
    row_map = {
        (row["p_hk"], row["epsilon"]): row
        for row in rows
    }
    for p in p_values:
        series = [row_map[(float(p), float(eps))] for eps in eps_values]
        color = COLORS[float(p)]
        probability = [row["recovery_probability"] for row in series]
        p_low = [row["recovery_probability_ci95_low"] for row in series]
        p_high = [row["recovery_probability_ci95_high"] for row in series]
        axes[0].plot(eps_values, probability, color=color, marker="o",
                     markersize=2.2, linewidth=1.1,
                     label="pHK = {:.1f}".format(p))
        axes[0].fill_between(eps_values, p_low, p_high, color=color,
                             alpha=0.13, linewidth=0)
        times = [row["kaplan_meier_rmst_sweeps"] for row in series]
        t_low = [row["kaplan_meier_rmst_ci95_low"] for row in series]
        t_high = [row["kaplan_meier_rmst_ci95_high"] for row in series]
        axes[1].plot(eps_values, times, color=color, marker="o",
                     markersize=2.2, linewidth=1.1,
                     label="pHK = {:.1f}".format(p))
        axes[1].fill_between(eps_values, t_low, t_high,
                             color=color, alpha=0.13, linewidth=0)
    axes[0].set_ylim(-0.04, 1.04)
    axes[0].set_yticks([0.0, 0.5, 1.0])
    axes[0].set_ylabel("Recovery probability")
    axes[1].set_ylim(0, float(payload["control"]["sweeps_per_epsilon"]) * 1.04)
    axes[1].set_ylabel("Kaplan-Meier RMST (sweeps)")
    for ax in axes:
        ax.set_xlim(min(eps_values), max(eps_values))
        ax.set_xlabel("Threshold epsilon")
        ax.grid(axis="y", color="#D1D5DB", linewidth=0.45, alpha=0.75)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.55, 0.87),
               ncol=3, frameon=False, fontsize=8.5)
    base = FIGURE_DIR / "fig_formal_threshold_recovery"
    require_matplotlib_panel_alignment(
        fig, axes, ("a", "b"),
        base.with_suffix(".alignment-layout.json"),
        row_groups=(("a", "b"),),
        column_groups=(),
        grid_cells={"a": (0, 1, 0, 1), "b": (0, 1, 1, 2)},
    )
    for suffix, options in (
            (".svg", {}), (".pdf", {}),
            (".tiff", {"dpi": RASTER_DPI}), (".png", {"dpi": 300})):
        fig.savefig(str(base.with_suffix(suffix)), **options)
    plt.close(fig)
    audit_pdf_text_geometry(
        base.with_suffix(".pdf"),
        base.with_suffix(".pdf-text-geometry-audit.json"))
    return base


def main():
    with DATA_PATH.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    with META_PATH.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    validation = validate(payload, metadata)
    print("Validation:", validation["status"], validation["cells"], "cells")
    if validation["errors"]:
        for error in validation["errors"][:40]:
            print(" -", error)
        raise SystemExit(1)

    rows = build_rows(payload)
    rows.sort(key=lambda row: (row["p_hk"], row["epsilon"]))
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(rows)
    figure_base = save_figure(payload, rows)
    trajectory_csv_path = write_trajectory_csv(payload)
    trajectory_figure_base = save_trajectory_figure(payload)

    control = payload["control"]
    lines = [
        "# Formal threshold-by-threshold recovery report",
        "",
        "Each threshold branch starts from the state entering that threshold on the unperturbed high-to-low continuation path. A fresh Gaussian shock is applied independently at each epsilon, then the branch evolves at that same epsilon. The clean and shocked branches share network seeds and threshold-specific update random streams.",
        "",
        "Recovery is defined as the 0.05-diameter order parameter reaching 1.0 for three consecutive measurements, recorded once per sweep. The probability interval is Wilson 95%. Non-recovered runs are right-censored at the 100-sweep horizon. The recovery-time summary is the Kaplan-Meier restricted mean survival time (RMST) integrated to that horizon; uncertainty intervals use a 2,000-resample seed-level bootstrap.",
        "",
        "Design: N={}, m={}, 20 seeds, p_HK={}, sigma={}, 19 thresholds from {:.3f} to {:.3f}, 100 sweeps per threshold.".format(
            control["N"], control["m"], control["p_values"],
            control["sigma_values"], min(control["eps_values"]),
            max(control["eps_values"])),
        "",
        "Validation: **{}**; {} p_HK blocks, {} threshold points per block, {} paired runs per cell, {} source hashes checked.".format(
            validation["status"], validation["p_points"],
            validation["epsilon_points_per_p"], validation["runs_per_cell"],
            validation["source_hashes_checked"]),
        "",
        "| p_HK | epsilon | sigma | recovered / n | recovery probability (Wilson 95% CI) | Kaplan-Meier RMST (bootstrap 95% CI, sweeps) | initial SD | initial range | final SD | final range | final R_0.05 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {p_hk:.2f} | {epsilon:.3f} | {sigma:.3f} | "
            "{n_recovered}/{n} | {prob:.3f} [{plo:.3f}, {phi:.3f}] | "
            "{rmst:.2f} [{rlo:.2f}, {rhi:.2f}] | {initial_sd:.5f} | "
            "{initial_range:.5f} | {final_sd:.5f} | {final_range:.5f} | "
            "{final_order:.3f} |".format(
                p_hk=row["p_hk"], epsilon=row["epsilon"], sigma=row["sigma"],
                n=row["n"], n_recovered=row["n_recovered"],
                prob=row["recovery_probability"],
                plo=row["recovery_probability_ci95_low"],
                phi=row["recovery_probability_ci95_high"],
                rmst=row["kaplan_meier_rmst_sweeps"],
                rlo=row["kaplan_meier_rmst_ci95_low"],
                rhi=row["kaplan_meier_rmst_ci95_high"],
                initial_sd=row["initial_opinion_sd"],
                initial_range=row["initial_opinion_range"],
                final_sd=row["final_opinion_sd"],
                final_range=row["final_opinion_range"],
                final_order=row["final_order_parameter"]))
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "Results apply to this network family, sigma=0.02 shock, clipping rule, threshold grid, and 100-sweep horizon. They do not establish a general stability boundary or behavior outside the tested range. Recovery probability, censoring count, opinion spread, and RMST are reported separately.",
        "",
        "The companion dynamics heatmap shows the mean opinion SD across 20 runs at each threshold and sweep on one shared logarithmic color scale across p_HK conditions. The trajectory CSV includes the corresponding mean and reported 95% interval at every recorded time point.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    SUMMARY_PATH.write_text(json.dumps({
        "validation": validation,
        "control": control,
        "rows": rows,
        "figure": str(figure_base),
        "trajectory_figure": str(trajectory_figure_base),
        "source_data_csv": str(CSV_PATH),
        "trajectory_source_data_csv": str(trajectory_csv_path),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(REPORT_PATH)
    print(SUMMARY_PATH)
    print(CSV_PATH)
    print(figure_base.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
