"""Validate and analyze the paired failed-triangle fallback experiment."""

from __future__ import annotations

import argparse
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
    "legend.fontsize": 6,
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
DATA_FILE = FORMAL_ROOT / "data" / "exp_triangle_failure_control.json"
META_FILE = FORMAL_ROOT / "data" / "exp_triangle_failure_control.meta.json"
ANALYSIS_DIR = FORMAL_ROOT / "analysis"
FIGURE_DIR = FORMAL_ROOT / "figures"

N_RUNS = 20
SEEDS = [run_i * 997 + 7 for run_i in range(N_RUNS)]
P_VALUES = [round(index * 0.05, 2) for index in range(21)]
MODELS = ("full_model", "full_model_no_triangle_fallback")
COMPARATOR = "full_model_no_triangle_fallback"
FINAL_METRICS = (
    "order_param", "n_clusters", "variance",
    "triangle_alignment_fixed", "triangle_admissibility_adaptive",
    "edge_disagreement_penalty",
)
DYNAMICS_METRICS = (
    "tri_count", "tri_success", "tri_fallback",
    "tri_fallback_edge_update", "tri_fallback_suppressed", "update_count",
)
METRICS = FINAL_METRICS + DYNAMICS_METRICS
LABELS = {
    "order_param": "Order parameter",
    "n_clusters": "Number of clusters",
    "variance": "Opinion variance",
    "triangle_alignment_fixed": "Fixed-scale triangle alignment",
    "triangle_admissibility_adaptive": "Adaptive triangle admissibility",
    "edge_disagreement_penalty": "Mean edge-disagreement penalty",
    "tri_count": "Triangle attempts",
    "tri_success": "Successful triangle gates",
    "tri_fallback": "Failed triangle gates",
    "tri_fallback_edge_update": "Accepted dyadic fallback updates",
    "tri_fallback_suppressed": "Suppressed fallback events",
    "update_count": "All accepted updates",
}
COLORS = {"full_model": "#DC2626", COMPARATOR: "#7C3AED"}
# Retained to validate summaries written with the historical lookup table.
LEGACY_T975 = (
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
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=atol)
    except (TypeError, ValueError):
        return False


def summarize(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n == 0:
        return {
            "n": 0, "mean": None, "sd": None, "se": None,
            "ci95_low": None, "ci95_high": None, "cohen_dz": None,
            "values": [],
        }
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if n > 1 else 0.0
    se = float(sd / math.sqrt(n))
    critical = float(stats.t.ppf(0.975, n - 1)) if n > 1 else 0.0
    return {
        "n": n, "mean": mean, "sd": sd, "se": se,
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
        running = max(running, min(1.0, (count - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def complete_values(summary):
    """Read one complete, finite seed-level vector."""
    if not isinstance(summary, dict):
        return None
    try:
        values = np.asarray(summary.get("values", []), dtype=float)
    except (TypeError, ValueError):
        return None
    if (summary.get("n") != N_RUNS or values.shape != (N_RUNS,)
            or not np.all(np.isfinite(values))):
        return None
    return values


def matches_confidence_interval(summary, expected):
    """Allow either the exact interval or the archived rounded interval."""
    n = expected["n"]
    critical = 0.0
    if n > 1:
        critical = LEGACY_T975[n - 2] if n <= len(LEGACY_T975) + 1 else 1.960
    half_width = critical * expected["se"]
    stored_low = summary.get("ci95_low")
    stored_high = summary.get("ci95_high")
    return (
        (close(stored_low, expected["ci95_low"])
         and close(stored_high, expected["ci95_high"]))
        or
        (close(stored_low, expected["mean"] - half_width)
         and close(stored_high, expected["mean"] + half_width))
    )


def counter_errors(values, allow_fallback):
    """Check event accounting within one model arm."""
    attempts = values["tri_count"]
    success = values["tri_success"]
    failures = values["tri_fallback"]
    accepted_fallback = values["tri_fallback_edge_update"]
    suppressed = values["tri_fallback_suppressed"]
    updates = values["update_count"]
    checks = (
        (attempts == success + failures, "triangle attempts differ from successes plus failures"),
        (accepted_fallback <= failures, "fallback updates exceed failed gates"),
        (updates >= success + accepted_fallback, "accepted updates omit successful triangle or fallback events"),
    )
    if allow_fallback:
        checks += ((suppressed == 0, "full model contains suppressed fallbacks"),)
    else:
        checks += (
            (accepted_fallback == 0, "no-fallback control contains accepted fallback updates"),
            (suppressed == failures, "suppressed fallbacks differ from failed gates"),
        )
    return [message for passed, message in checks if not np.all(passed)]


def validate(payload, metadata):
    errors = []
    source_hash_drift = []
    control = payload.get("control", {})
    blocks = payload.get("data", {})
    actual_p = sorted(float(key) for key in blocks)
    if actual_p != P_VALUES:
        errors.append("p_HK grid mismatch: {}".format(actual_p))
    if control.get("N") != 2000 or control.get("m") != 4:
        errors.append("network configuration is not N=2000, m=4")
    if control.get("num_runs") != N_RUNS:
        errors.append("control num_runs is not 20")
    if control.get("model_order") != list(MODELS):
        errors.append("model order mismatch")
    if float(control.get("main_sweeps", -1)) != 200.0:
        errors.append("main_sweeps is not 200")
    if metadata.get("n_runs") != N_RUNS or metadata.get("seeds") != SEEDS:
        errors.append("metadata replication protocol mismatch")
    sim_config = metadata.get("config", {}).get("SIMULATION", {})
    if float(sim_config.get("main_sweeps", -1)) != 200.0:
        errors.append("metadata main_sweeps is not 200")
    for filename, recorded in sorted(
            metadata.get("source_sha256", {}).items()):
        path = ROOT / filename
        if not path.exists():
            errors.append("recorded source missing: {}".format(filename))
        elif sha256(path) != recorded:
            source_hash_drift.append(filename)

    pair_key = "full_model_minus_{}".format(COMPARATOR)
    dynamics_pair_key = pair_key + "_dynamics"
    for p_key, block in blocks.items():
        if block.get("_control", {}).get("seeds") != SEEDS:
            errors.append("seed list mismatch at p={}".format(p_key))
        arm_values = {}
        for model in MODELS:
            arm = block.get(model, {})
            counters = arm.get("dynamics_stats", {})
            values_by_metric = arm_values[model] = {}
            for metric in FINAL_METRICS:
                summary = arm.get(metric, {})
                values = complete_values(summary)
                label = "{} {} at p={}".format(model, metric, p_key)
                if values is None:
                    errors.append("incomplete or invalid " + label)
                    continue
                values_by_metric[metric] = values
                expected = summarize(values)
                for field in ("mean", "sd", "se"):
                    if not close(summary.get(field), expected[field]):
                        errors.append("summary mismatch {} {}".format(label, field))
                if not matches_confidence_interval(summary, expected):
                    errors.append("summary mismatch {} confidence interval".format(label))
            for metric in DYNAMICS_METRICS:
                values = complete_values(counters.get(metric, {}))
                if (values is None or np.any(values < 0)
                        or np.any(values != np.floor(values))):
                    errors.append("invalid count {} {} at p={}".format(model, metric, p_key))
                    continue
                values_by_metric[metric] = values
            if all(metric in values_by_metric for metric in DYNAMICS_METRICS):
                errors.extend(
                    "{} {} at p={}".format(model, problem, p_key)
                    for problem in counter_errors(values_by_metric, model == "full_model")
                )

        comparisons = block.get("_paired_comparisons", {})
        full = arm_values["full_model"]
        no_fallback = arm_values[COMPARATOR]
        for key, metrics in ((pair_key, FINAL_METRICS), (dynamics_pair_key, DYNAMICS_METRICS)):
            for metric in metrics:
                if metric not in full or metric not in no_fallback:
                    continue
                saved = complete_values(comparisons.get(key, {}).get(metric, {}))
                expected = full[metric] - no_fallback[metric]
                if (saved is None or not np.allclose(
                        saved, expected, atol=1e-12, rtol=1e-10)):
                    errors.append("paired values mismatch {} at p={}".format(metric, p_key))
    return {
        "status": (
            "FAIL" if errors else
            ("PASS_WITH_SOURCE_DRIFT" if source_hash_drift else "PASS")),
        "errors": errors,
        "source_hash_drift": source_hash_drift,
        "p_points": len(actual_p),
        "models": list(MODELS),
        "n_per_cell": N_RUNS,
        "independent_unit": (
            "seed-level paired simulation replicate; both arms share the "
            "network and initial-condition seed"),
    }


def analyze(payload, metadata, validation, *, output_dir=None):
    if validation["errors"]:
        raise ValueError("cannot analyze invalid failed-triangle data")
    analysis_dir = ANALYSIS_DIR if output_dir is None else output_dir / "analysis"
    figure_dir = FIGURE_DIR if output_dir is None else output_dir / "figures"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    blocks = payload["data"]
    p_keys = sorted(blocks, key=float)
    p_values = [float(key) for key in p_keys]
    pair_key = "full_model_minus_{}".format(COMPARATOR)
    dynamics_pair_key = pair_key + "_dynamics"

    source_rows = []
    contrast_rows = []
    global_rows = []
    fallback_fraction_rows = []
    for p_key, p in zip(p_keys, p_values):
        block = blocks[p_key]
        for model in MODELS:
            for index, seed in enumerate(SEEDS):
                row = {"p_hk": p, "model": model, "seed": seed}
                for metric in FINAL_METRICS:
                    row[metric] = block[model][metric]["values"][index]
                for metric in DYNAMICS_METRICS:
                    row[metric] = (
                        block[model]["dynamics_stats"][metric]["values"][index])
                source_rows.append(row)
        full = block["full_model"]["dynamics_stats"]
        fallbacks = np.asarray(full["tri_fallback"]["values"], dtype=float)
        accepted = np.asarray(
            full["tri_fallback_edge_update"]["values"], dtype=float)
        fractions = np.divide(
            accepted, fallbacks, out=np.full_like(accepted, np.nan),
            where=fallbacks > 0)
        fraction_summary = summarize(fractions)
        fallback_fraction_rows.append({
            "p_hk": p, **{key: value for key, value in
                          fraction_summary.items() if key != "values"},
            "values": fraction_summary["values"],
        })

        for metric in METRICS:
            if metric in FINAL_METRICS:
                saved = block["_paired_comparisons"][pair_key][metric]
            else:
                saved = block["_paired_comparisons"][
                    dynamics_pair_key][metric]
            exact = summarize(saved["values"])
            contrast_rows.append({
                "p_hk": p,
                "metric": metric,
                "metric_label": LABELS[metric],
                "n": exact["n"],
                "mean_full_minus_no_fallback": exact["mean"],
                "sd_difference": exact["sd"],
                "se_difference": exact["se"],
                "ci95_low": exact["ci95_low"],
                "ci95_high": exact["ci95_high"],
                "cohen_dz": exact["cohen_dz"] if exact["sd"] > 0 else None,
                "paired_t_p_raw": paired_t_pvalue(saved["values"]),
            })

    # Holm correction across the 21 p_HK values, within each outcome.
    for metric in METRICS:
        rows = [row for row in contrast_rows if row["metric"] == metric]
        adjusted = holm_adjust([row["paired_t_p_raw"] for row in rows])
        for row, value in zip(rows, adjusted):
            row["paired_t_p_holm_21"] = value

    # Global paired AUC contrasts across p_HK for terminal outcomes.
    for metric in FINAL_METRICS:
        differences = []
        for index in range(N_RUNS):
            curve = []
            for p_key in p_keys:
                curve.append(blocks[p_key]["_paired_comparisons"][
                    pair_key][metric]["values"][index])
            differences.append(float(np.trapz(curve, x=p_values)))
        summary = summarize(differences)
        global_rows.append({
            "metric": metric,
            "metric_label": LABELS[metric],
            "n": summary["n"],
            "mean_auc_full_minus_no_fallback": summary["mean"],
            "sd_difference": summary["sd"],
            "se_difference": summary["se"],
            "ci95_low": summary["ci95_low"],
            "ci95_high": summary["ci95_high"],
            "cohen_dz": summary["cohen_dz"],
            "paired_t_p_raw": paired_t_pvalue(differences),
            "values": summary["values"],
        })
    adjusted = holm_adjust([row["paired_t_p_raw"] for row in global_rows])
    for row, value in zip(global_rows, adjusted):
        row["paired_t_p_holm_6"] = value

    write_csv(source_rows, str(
        analysis_dir / "formal_triangle_failure_source_data.csv"))
    write_csv(contrast_rows, str(
        analysis_dir / "formal_triangle_failure_pointwise.csv"))
    write_csv([{
        key: value for key, value in row.items() if key != "values"
    } for row in global_rows], str(
        analysis_dir / "formal_triangle_failure_global_auc.csv"))
    write_csv([{
        key: value for key, value in row.items() if key != "values"
    } for row in fallback_fraction_rows], str(
        analysis_dir / "formal_triangle_failure_fallback_rates.csv"))

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.15), sharex=True)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.90, bottom=0.09,
                        wspace=0.34, hspace=0.34)
    flat_axes = list(axes.flat)
    labels = ("a", "b", "c", "d")
    for ax, label in zip(flat_axes, labels):
        ax.set_gid(label)
        add_panel_label(ax, label)

    # (a) Overall terminal response under matched seeds.
    for model in MODELS:
        summaries = [summarize(blocks[key][model]["order_param"]["values"])
                     for key in p_keys]
        means = [row["mean"] for row in summaries]
        lows = [row["ci95_low"] for row in summaries]
        highs = [row["ci95_high"] for row in summaries]
        axes[0, 0].plot(
            p_values, means, color=COLORS[model], marker="o",
            markersize=2.2, linewidth=1.35, label=model)
        axes[0, 0].fill_between(
            p_values, lows, highs, color=COLORS[model], alpha=0.12)
    axes[0, 0].set_title("Terminal order parameter", pad=7)
    axes[0, 0].set_ylabel("Mean (95% CI)")
    axes[0, 0].set_ylim(0, 1.03)
    axes[0, 0].legend(fontsize=5.8, loc="lower right", frameon=False)

    # (b,c) Paired outcomes; zero marks no mean change from the control.
    for metric, ax, title, ylabel in (
            ("order_param", axes[0, 1], "Order-parameter contrast",
             "Full minus no-fallback (95% CI)"),
            ("n_clusters", axes[1, 0], "Cluster-count contrast",
             "Full minus no-fallback (95% CI)")):
        rows = [row for row in contrast_rows if row["metric"] == metric]
        ax.plot(
            p_values, [row["mean_full_minus_no_fallback"] for row in rows],
            color="#2563EB", marker="o", markersize=2.2, linewidth=1.35)
        ax.fill_between(
            p_values, [row["ci95_low"] for row in rows],
            [row["ci95_high"] for row in rows], color="#2563EB", alpha=0.14)
        ax.axhline(0, color="#111827", linestyle="--", linewidth=0.8)
        ax.set_title(title, pad=7)
        ax.set_ylabel(ylabel)

    # (d) Mechanistic diagnostic: an undefined rate stays missing when there
    # were no failed triangle gates, rather than being plotted as zero.
    rows = fallback_fraction_rows
    rate_mean = [np.nan if row["mean"] is None else row["mean"]
                 for row in rows]
    rate_low = [np.nan if row["ci95_low"] is None else row["ci95_low"]
                for row in rows]
    rate_high = [np.nan if row["ci95_high"] is None else row["ci95_high"]
                 for row in rows]
    axes[1, 1].plot(
        p_values, rate_mean, color=COLORS["full_model"],
        marker="o", markersize=2.2, linewidth=1.35,
        label="Accepted dyadic fallback")
    axes[1, 1].fill_between(
        p_values, rate_low, rate_high,
        color=COLORS["full_model"], alpha=0.14)
    axes[1, 1].set_ylim(-0.03, 1.03)
    axes[1, 1].set_title("Accepted updates after failed gates", pad=7)
    axes[1, 1].set_ylabel("Fraction of failed gates (95% CI)")
    axes[1, 1].legend(fontsize=5.6, loc="best", frameon=False)

    for ax in flat_axes:
        ax.set_xlabel("p_HK")
        ax.set_xticks(np.arange(0.0, 1.01, 0.2))
        ax.set_xlim(min(p_values), max(p_values))
        ax.grid(alpha=0.2, linestyle="--", linewidth=0.5)
        ax.margins(x=0.02)
    fig.suptitle(
        "What changes when a failed triangle does not fall back to a dyad?",
        fontsize=8.5, fontweight="bold", y=0.985)

    base = figure_dir / "fig_formal_triangle_failure_control"
    write_alignment_manifest(
        fig, flat_axes, labels,
        base.with_suffix(".alignment-layout.json"),
        row_groups=(("a", "b"), ("c", "d")),
        column_groups=(("a", "c"), ("b", "d")),
        grid_cells={
            "a": (0, 1, 0, 1), "b": (0, 1, 1, 2),
            "c": (1, 2, 0, 1), "d": (1, 2, 1, 2),
        },
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
        "pointwise_paired_contrasts": contrast_rows,
        "global_auc_contrasts": global_rows,
        "fallback_acceptance_fraction": fallback_fraction_rows,
        "multiplicity": {
            "pointwise": "Holm correction across 21 p_HK values within each outcome",
            "global": "Holm correction across six terminal-outcome AUC tests",
        },
        "interpretation_boundary": (
            "This comparison isolates the failed-triangle dyadic fallback "
            "under the specified static-network simulation protocol."),
    }
    analysis_path = analysis_dir / "formal_triangle_failure_analysis.json"
    with analysis_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)

    report_path = analysis_dir / "FORMAL_TRIANGLE_FAILURE_REPORT.md"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("# Formal failed-triangle fallback control\n\n")
        handle.write(
            "Design: N=2000, m=4, 21 values of p_HK, 20 paired seeds, "
            "and 200 Monte Carlo sweeps. Both arms share the network, "
            "initial opinions, threshold rule, successful-triangle rule, "
            "and third-party damping. The only changed rule is whether a "
            "failed triangle gate is followed by a dyadic update.\n\n")
        handle.write(
            "Data validation: **{}** ({} consistency errors; {} recorded "
            "source hashes differ from the current files). Pointwise contrasts use "
            "two-sided paired t-tests with Holm adjustment over 21 p_HK "
            "values within each outcome. Global terminal-outcome contrasts "
            "compare per-seed trapezoidal AUCs with Holm adjustment over "
            "six outcomes.\n\n".format(
                validation["status"], len(validation["errors"]),
                len(validation["source_hash_drift"])))
        if validation["source_hash_drift"]:
            handle.write(
                "Source files with recorded hash differences: {}.\n\n".format(
                    ", ".join(validation["source_hash_drift"])))
        handle.write(
            "The fallback acceptance fraction reports the share of failed "
            "triangle attempts for which the ordinary dyadic confidence test "
            "would accept an update in the full model. The no-fallback arm "
            "makes no opinion change after every failed triangle attempt.\n\n")
        handle.write(
            "Source data and contrast tables are saved as CSV files beside "
            "this report. Confidence intervals describe across-seed "
            "uncertainty; a non-significant contrast is not an equivalence "
            "result.\n")

    print("Wrote", analysis_path)
    print("Wrote", base.with_suffix(".pdf"))
    print("Wrote", report_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true",
                        help="Check archived inputs without writing analysis files.")
    parser.add_argument("--output-dir", type=Path,
                        help="Write analysis and figures under this directory.")
    args = parser.parse_args(argv)
    with DATA_FILE.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    with META_FILE.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    validation = validate(payload, metadata)
    print("Validation:", validation["status"])
    for name in validation["source_hash_drift"]:
        print(" - source hash differs:", name)
    if validation["errors"]:
        for error in validation["errors"][:30]:
            print(" -", error)
        raise SystemExit(1)
    if not args.validate_only:
        analyze(payload, metadata, validation, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
