#!/usr/bin/env python3
"""
统一绘图入口。

运行方式：
    python plotting.py
    python plot_results.py
    python plotting_fix.py

输出目录：config.FIGURE_DIR
数据目录：config.DATA_DIR
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

import matplotlib
if str(matplotlib.get_backend()).lower() != "agg":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

try:  # scipy 只用于平滑曲线；缺失时自动退回折线
    from scipy.interpolate import make_interp_spline
except Exception:  # pragma: no cover
    make_interp_spline = None

try:
    from .config import DATA_DIR, FIGURE_DIR, MODEL_TYPES, MODEL_DISPLAY_ORDER
except ImportError:  # pragma: no cover
    from config import DATA_DIR, FIGURE_DIR, MODEL_TYPES, MODEL_DISPLAY_ORDER

warnings.filterwarnings("ignore")

OUT_DIR = Path(FIGURE_DIR)
DATA_PATH = Path(DATA_DIR)
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.30,
    "grid.linestyle": "--",
})

COLORS = {
    "blue": "#2563EB",
    "red": "#DC2626",
    "green": "#16A34A",
    "orange": "#EA580C",
    "purple": "#7C3AED",
    "teal": "#0D9488",
    "pink": "#DB2777",
    "gray": "#6B7280",
    "black": "#111827",
}

PHK_COLORS = {
    "0.0": COLORS["blue"],
    "0.4": COLORS["orange"],
    "0.8": COLORS["green"],
    "1.0": COLORS["purple"],
}

LEGACY_MODEL_KEYS = {
    "weighted": "weighted_bcm",
    "trust": "threshold_only",
    "triangle": "no_triadic_feedback",
    "full": "full_model",
    "baseline_dw": "baseline",
    "weighted_dw": "weighted_bcm",
    "trust_only": "threshold_only",
    "triangle_only": "no_triadic_feedback",
    "no_structure": "no_triadic_feedback",
}

LEGACY_METRIC_KEYS = {
    "triangle_alignment_fixed": "tri_consensus",
    "edge_disagreement_penalty": "rep_frag",
}


def _is_bad_number(x) -> bool:
    try:
        return math.isnan(float(x)) or math.isinf(float(x))
    except Exception:
        return True


def _as_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def _clamp01(x):
    v = _as_float(x, default=np.nan)
    if np.isnan(v):
        return x
    return float(np.clip(v, 0.0, 1.0))


def _clamp_order_param_tree(obj, in_order_param=False):
    """只在 order_param 子树中裁剪到 [0,1]，避免历史数据或平滑插值越界。"""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "order_param":
                out[k] = _clamp_order_param_tree(v, True)
            else:
                out[k] = _clamp_order_param_tree(v, in_order_param)
        if in_order_param:
            if "mean" in out:
                out["mean"] = _clamp01(out["mean"])
            if isinstance(out.get("values"), list):
                out["values"] = [_clamp_order_param_tree(v, True) for v in out["values"]]
        return out
    if isinstance(obj, list):
        return [_clamp_order_param_tree(v, in_order_param) for v in obj]
    return _clamp01(obj) if in_order_param else obj


def load_json(filename: str):
    """从当前路径或 DATA_DIR 加载 JSON。"""
    candidates = [Path(filename), DATA_PATH / filename]
    for path in candidates:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return _clamp_order_param_tree(json.load(f))
    raise FileNotFoundError(f"数据文件未找到: {filename}")


def get_p_key(data: dict, p: float) -> Optional[str]:
    """兼容 '0.4'、'0.40'、'0.400' 等 p 键。"""
    candidates = [str(p), f"{p:.1f}", f"{p:.2f}", f"{p:.3f}"]
    for key in candidates:
        if key in data:
            return key
    # 数值近似匹配
    for key in data.keys():
        try:
            if abs(float(key) - float(p)) < 1e-9:
                return key
        except Exception:
            continue
    return None


def get_combo_key(data: dict, p: float, beta: float) -> Optional[str]:
    """兼容联合扫描中不同小数位的 key。"""
    candidates = [
        f"{p:.2f}_{beta:.2f}",
        f"{p:.2f}_{beta:.1f}",
        f"{p:.1f}_{beta:.2f}",
        f"{p:.1f}_{beta:.1f}",
        f"{p}_{beta}",
    ]
    for key in candidates:
        if key in data:
            return key
    for key in data.keys():
        try:
            a, b = key.split("_")
            if abs(float(a) - p) < 1e-9 and abs(float(b) - beta) < 1e-9:
                return key
        except Exception:
            continue
    return None


def get_stat(block: dict, metric: str, stat: str = "mean", default=np.nan) -> float:
    try:
        key = metric if metric in block else LEGACY_METRIC_KEYS.get(metric, metric)
        value = block[key]
        if isinstance(value, dict):
            return _as_float(value.get(stat), default)
        return _as_float(value, default)
    except Exception:
        return default


def ci95_half_width(stat_block: dict, default=0.0) -> float:
    """Use a 95% CI half-width; fall back to historical SD-only files."""
    try:
        low = _as_float(stat_block.get("ci95_low"), np.nan)
        high = _as_float(stat_block.get("ci95_high"), np.nan)
        if not np.isnan(low) and not np.isnan(high):
            return max(0.0, (high - low) / 2.0)
        return _as_float(stat_block.get("std"), default)
    except Exception:
        return default


def get_ci95_half_width(block: dict, metric: str, default=0.0) -> float:
    try:
        key = metric if metric in block else LEGACY_METRIC_KEYS.get(metric, metric)
        value = block[key]
        return ci95_half_width(value, default) if isinstance(value, dict) else default
    except Exception:
        return default


def normalize_model_key(model_key: str) -> str:
    return model_key if model_key in MODEL_TYPES else LEGACY_MODEL_KEYS.get(model_key, model_key)


def smooth_curve(x, y, num_points=220, k=3):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) == 0:
        return x, y
    sort_idx = np.argsort(x)
    x, y = x[sort_idx], y[sort_idx]
    x, unique_idx = np.unique(x, return_index=True)
    y = y[unique_idx]
    if make_interp_spline is None or len(x) < 4:
        return x, y
    kk = min(k, len(x) - 1)
    try:
        x_new = np.linspace(float(np.min(x)), float(np.max(x)), num_points)
        spline = make_interp_spline(x, y, k=kk)
        return x_new, spline(x_new)
    except Exception:
        return x, y


def smooth_fill(ax, x, y_mean, y_std, color, label=None, marker="o",
                smooth=True, alpha_fill=0.15, clip_y=None, **kwargs):
    x = np.asarray(x, dtype=float)
    y_mean = np.asarray(y_mean, dtype=float)
    y_std = np.asarray(y_std, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y_mean))
    if not np.any(mask):
        return
    x, y_mean, y_std = x[mask], y_mean[mask], y_std[mask]
    y_std = np.nan_to_num(y_std, nan=0.0, posinf=0.0, neginf=0.0)

    lower = y_mean - y_std
    upper = y_mean + y_std
    if clip_y is not None:
        lo, hi = clip_y
        y_mean = np.clip(y_mean, lo, hi)
        lower = np.clip(lower, lo, hi)
        upper = np.clip(upper, lo, hi)

    ax.scatter(x, y_mean, color=color, marker=marker, s=30, zorder=5,
               edgecolors="white", linewidth=0.5)

    if smooth and len(x) >= 4:
        xs, ys = smooth_curve(x, y_mean)
        _, ys_up = smooth_curve(x, upper)
        _, ys_dn = smooth_curve(x, lower)
        if clip_y is not None:
            lo, hi = clip_y
            ys = np.clip(ys, lo, hi)
            ys_up = np.clip(ys_up, lo, hi)
            ys_dn = np.clip(ys_dn, lo, hi)
        ax.plot(xs, ys, color=color, linewidth=2, label=label, zorder=4, **kwargs)
        ax.fill_between(xs, ys_dn, ys_up, alpha=alpha_fill, color=color, zorder=2)
    else:
        ax.plot(x, y_mean, color=color, linewidth=2, label=label, marker=marker,
                markersize=5, zorder=4, **kwargs)
        ax.fill_between(x, lower, upper, alpha=alpha_fill, color=color, zorder=2)


def save_fig(fig, filename: str):
    path = OUT_DIR / filename
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] {filename}")


# ============================================================
# Figure 1: 代表性 epsilon 前/后向扫描
# ============================================================
def plot_epsilon_sweep():
    data = load_json("exp_epsilon_sweep.json")
    selected = [0.0, 0.4, 0.8]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    all_y, all_s = [], []
    for ax, p in zip(axes, selected):
        p_key = get_p_key(data, p)
        if p_key is None:
            ax.set_visible(False)
            continue
        d = data[p_key]
        eps_f = sorted(d.get("forward", {}).keys(), key=float)
        eps_b = sorted(d.get("backward", {}).keys(), key=float)
        x_f = [float(e) for e in eps_f]
        y_f = [get_stat(d["forward"][e], "", default=d["forward"][e].get("mean", np.nan)) if isinstance(d["forward"][e], dict) else np.nan for e in eps_f]
        s_f = [ci95_half_width(d["forward"][e]) for e in eps_f]
        x_b = [float(e) for e in eps_b]
        y_b = [d["backward"][e].get("mean", np.nan) for e in eps_b]
        s_b = [ci95_half_width(d["backward"][e]) for e in eps_b]

        all_y.extend(y_f + y_b)
        all_s.extend(s_f + s_b)
        color = PHK_COLORS.get(f"{p:.1f}", COLORS["blue"])
        smooth_fill(ax, x_f, y_f, s_f, color, "Forward", marker="o", clip_y=(0, 1))
        smooth_fill(ax, x_b, y_b, s_b, COLORS["red"], "Backward", marker="s",
                    linestyle="--", clip_y=(0, 1))
        area = _as_float(d.get("hysteresis_area"), 0.0)
        sigma = _as_float(d.get("backward_perturb_sigma"), np.nan)
        label = f"A = {area:.4f}"
        if not np.isnan(sigma):
            label += f"\nσ = {sigma:.3f}"
        ax.text(0.95, 0.05, label, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                                      edgecolor="gray", alpha=0.85))
        ax.set_xlabel(r"$\epsilon_0$ (baseline tolerance)")
        ax.set_title(rf"$p_{{HK}}={p_key}$", fontweight="bold")
        ax.legend(loc="upper left", framealpha=0.9)

    if all_y:
        y = np.asarray(all_y, dtype=float)
        s = np.asarray(all_s, dtype=float)
        lower = max(0.0, float(np.nanmin(y - s)) - 0.03)
        upper = min(1.005, float(np.nanmax(y + s)) + 0.03)
        axes[0].set_ylim(lower, upper)
    axes[0].set_ylabel("Order Parameter")
    fig.suptitle(r"Figure 1: Representative Forward/Backward $\epsilon_0$ Scans",
                 fontweight="bold", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig1_epsilon_sweep.png")


# ============================================================
# Figure 2: 完整 21 点 p_HK 磁滞面积扫描
# ============================================================
def plot_hysteresis_sweep():
    data = load_json("exp_hysteresis_sweep.json")
    p_keys = sorted((key for key in data
                     if not key.startswith('_')
                     and isinstance(data.get(key), dict)), key=float)
    x = [float(k) for k in p_keys]
    h = [_as_float(data[k].get("hysteresis_area"), np.nan) for k in p_keys]
    h_ci = [ci95_half_width(data[k].get("hysteresis_area_stats", {}))
            for k in p_keys]
    sens = [_as_float(data[k].get("peak_sensitivity"), np.nan) for k in p_keys]
    sens_ci = [ci95_half_width(data[k].get("peak_sensitivity_stats", {}))
               for k in p_keys]
    bop = [_as_float(data[k].get("backward_low_eps_order_param"), np.nan) for k in p_keys]
    bcl = [_as_float(data[k].get("backward_low_eps_clusters"), np.nan) for k in p_keys]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    ax1, ax2, ax3, ax4 = axes.flatten()

    smooth_fill(ax1, x, h, h_ci, COLORS["blue"], "Hysteresis area", marker="o")
    ax1.fill_between(x, 0, h, color=COLORS["blue"], alpha=0.08)
    ax1.set_xlabel(r"$p_{HK}$")
    ax1.set_ylabel("Hysteresis Area")
    ax1.set_title("(a) Hysteresis area", fontweight="bold")

    smooth_fill(ax2, x, sens, sens_ci, COLORS["red"], "Peak sensitivity", marker="s")
    ax2.set_xlabel(r"$p_{HK}$")
    ax2.set_ylabel("Peak Sensitivity")
    ax2.set_title("(b) Forward peak sensitivity", fontweight="bold")

    smooth_fill(ax3, x, bop, np.zeros(len(x)), COLORS["green"], r"Backward low-$\epsilon_0$ OP",
                marker="D", clip_y=(0, 1))
    ax3.axhline(0.9, color=COLORS["gray"], linestyle=":", linewidth=1.5,
                label="Consensus threshold")
    ax3.set_xlabel(r"$p_{HK}$")
    ax3.set_ylabel(r"Backward Low-$\epsilon_0$ OP")
    ax3.set_ylim(0, 1.02)
    ax3.set_title("(c) Backward low-tolerance resilience", fontweight="bold")
    ax3.legend(framealpha=0.9)

    smooth_fill(ax4, x, bcl, np.zeros(len(x)), COLORS["purple"], "Backward clusters", marker="^", smooth=False)
    ax4.set_xlabel(r"$p_{HK}$")
    ax4.set_ylabel(r"Backward Low-$\epsilon_0$ Clusters")
    ax4.set_title("(d) Backward fragmentation after perturbation", fontweight="bold")

    sigma_vals = [data[k].get("backward_perturb_sigma") for k in p_keys if "backward_perturb_sigma" in data[k]]
    sigma_text = ""
    if sigma_vals:
        sigma_text = rf"; perturbation $\sigma={_as_float(sigma_vals[0], 0):.3f}$"
    fig.suptitle("Figure 2: Full $p_{HK}$ Hysteresis Scan" + sigma_text,
                 fontweight="bold", fontsize=14, y=1.01)
    fig.tight_layout()
    save_fig(fig, "fig2_hysteresis_sweep.png")


def plot_perturbation_sweep():
    """Plot one-shot perturbation strength, hysteresis, resilience, and clipping."""
    payload = load_json("exp_perturbation_sweep.json")
    data = payload.get("data", {})
    control = payload.get("control", {})
    sigma_keys = sorted(data, key=float)
    p_values = control.get("p_values", [])
    if not p_values and sigma_keys:
        p_values = sorted({float(p) for block in data.values()
                           for p in block}, key=float)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for idx, p in enumerate(p_values):
        p_key = f"{float(p):.2f}"
        available = [(key, data[key][p_key]) for key in sigma_keys
                     if p_key in data[key]]
        if not available:
            continue
        x = [float(key) for key, _ in available]
        blocks = [block for _, block in available]
        color = list(COLORS.values())[idx % len(COLORS)]
        label = rf"$p_{{HK}}={float(p):.2f}$"

        y_area = [block["hysteresis_area_stats"]["mean"]
                  for block in blocks]
        e_area = [ci95_half_width(block["hysteresis_area_stats"])
                  for block in blocks]
        smooth_fill(axes[0], x, y_area, e_area, color, label,
                    marker="o", smooth=False)

        y_back = [block["backward"][str(block["eps_values"][0])]["mean"]
                  for block in blocks]
        e_back = [ci95_half_width(
            block["backward"][str(block["eps_values"][0])])
            for block in blocks]
        smooth_fill(axes[1], x, y_back, e_back, color, label,
                    marker="s", smooth=False, clip_y=(0, 1))

        y_clip = [block["perturbation_diagnostics"]["clipped_fraction"]["mean"]
                  for block in blocks]
        e_clip = [ci95_half_width(
            block["perturbation_diagnostics"]["clipped_fraction"])
            for block in blocks]
        smooth_fill(axes[2], x, y_clip, e_clip, color, label,
                    marker="D", smooth=False, clip_y=(0, 1))

    axes[0].set_title("(a) Hysteresis area", fontweight="bold")
    axes[0].set_ylabel("Area")
    axes[1].set_title(r"(b) Backward low-$\epsilon_0$ order", fontweight="bold")
    axes[1].set_ylabel("Order parameter")
    axes[2].set_title("(c) Boundary clipping", fontweight="bold")
    axes[2].set_ylabel("Fraction clipped to [0,1]")
    for ax in axes:
        ax.set_xlabel(r"Perturbation strength $\sigma$")
        ax.legend(framealpha=0.9, fontsize=8)
    mode = "repeated at every backward point" if control.get(
        "perturb_each_step") else "one shot at backward-scan onset"
    fig.suptitle(f"Perturbation-gradient check ({mode})",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig_perturbation_sweep.png")


def plot_update_mode_control():
    """Compare asynchronous and frozen-snapshot synchronous dynamics."""
    payload = load_json("exp_update_mode_control.json")
    data = payload.get("data", {})
    p_keys = sorted(data, key=float)
    x = [float(key) for key in p_keys]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    styles = {
        'asynchronous': (COLORS['blue'], 'Asynchronous', 'o'),
        'synchronous': (COLORS['orange'], 'Synchronous', 's'),
    }
    for mode, (color, label, marker) in styles.items():
        y = [data[key][mode]['order_param']['mean'] for key in p_keys]
        err = [ci95_half_width(data[key][mode]['order_param'])
               for key in p_keys]
        smooth_fill(axes[0], x, y, err, color, label, marker=marker,
                    smooth=False, clip_y=(0, 1))

        y_var = [data[key][mode]['variance']['mean'] for key in p_keys]
        e_var = [ci95_half_width(data[key][mode]['variance'])
                 for key in p_keys]
        smooth_fill(axes[1], x, y_var, e_var, color, label,
                    marker=marker, smooth=False)

    paired = [data[key]['paired_synchronous_minus_asynchronous'][
        'order_param'] for key in p_keys]
    y_pair = [item['mean'] for item in paired]
    e_pair = [ci95_half_width(item) for item in paired]
    axes[2].errorbar(x, y_pair, yerr=e_pair, color=COLORS['black'],
                     marker='D', capsize=4, linewidth=1.6)
    axes[2].axhline(0.0, color=COLORS['gray'], linestyle='--', linewidth=1)

    axes[0].set_title('(a) Order parameter', fontweight='bold')
    axes[0].set_ylabel('Order parameter')
    axes[1].set_title('(b) Opinion variance', fontweight='bold')
    axes[1].set_ylabel('Variance')
    axes[2].set_title('(c) Paired update-mode contrast', fontweight='bold')
    axes[2].set_ylabel('Synchronous - asynchronous OP')
    for ax in axes:
        ax.set_xlabel(r'$p_{HK}$')
    axes[0].legend(framealpha=0.9)
    axes[1].legend(framealpha=0.9)
    fig.suptitle('Update-mode robustness at matched computational budget',
                 fontweight='bold', y=1.02)
    fig.tight_layout()
    save_fig(fig, 'fig_update_mode_control.png')


def plot_sbm_community_conflict():
    """Plot cohesion and residual community separation on two-block SBMs."""
    payload = load_json('exp_sbm_community_conflict.json')
    data = payload.get('data', {})
    mixing_keys = sorted(data, key=float)
    x = [float(key) for key in mixing_keys]
    model_order = payload.get('control', {}).get('model_order', [
        'threshold_only', 'no_triadic_feedback', 'full_model'])
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    for model_key in model_order:
        style = MODEL_TYPES.get(model_key, {})
        color = style.get('color', COLORS['gray'])
        marker = style.get('marker', 'o')
        label = style.get('label', model_key)
        blocks = [data[key][model_key] for key in mixing_keys]
        for ax, metric, clip in (
                (axes[0], 'order_param', (0, 1)),
                (axes[1], 'community_mean_gap', (0, 1))):
            y = [block[metric]['mean'] for block in blocks]
            err = [ci95_half_width(block[metric]) for block in blocks]
            smooth_fill(ax, x, y, err, color, label, marker=marker,
                        smooth=False, clip_y=clip)

    paired = []
    for key in mixing_keys:
        comparisons = data[key].get('_paired_comparisons', {})
        paired.append(comparisons.get(
            'full_model_minus_threshold_only', {}).get(
                'community_mean_gap', {}))
    y_pair = [_as_float(item.get('mean'), np.nan) for item in paired]
    e_pair = [ci95_half_width(item) for item in paired]
    axes[2].errorbar(x, y_pair, yerr=e_pair, color=COLORS['black'],
                     marker='D', capsize=4, linewidth=1.6)
    axes[2].axhline(0.0, color=COLORS['gray'], linestyle='--', linewidth=1)

    axes[0].set_title('(a) Global compact order', fontweight='bold')
    axes[0].set_ylabel('Order parameter')
    axes[1].set_title('(b) Residual community separation', fontweight='bold')
    axes[1].set_ylabel('Absolute community-mean gap')
    axes[2].set_title('(c) Feedback contribution to separation',
                      fontweight='bold')
    axes[2].set_ylabel('Full - Threshold Only gap')
    for ax in axes:
        ax.set_xlabel('Expected cross-community edge fraction')
    axes[0].legend(framealpha=0.9, fontsize=8)
    axes[1].legend(framealpha=0.9, fontsize=8)
    fig.suptitle('Two-community SBM with adversarial initial opinions',
                 fontweight='bold', y=1.02)
    fig.tight_layout()
    save_fig(fig, 'fig_sbm_community_conflict.png')


def plot_empirical_topologies():
    """Compare mechanisms on fixed, bundled empirical social networks."""
    payload = load_json('exp_empirical_topologies.json')
    data = payload.get('data', {})
    requested = payload.get('control', {}).get(
        'topology_names', ['karate', 'davis', 'florentine'])
    topology_keys = [str(name).lower() for name in requested
                     if str(name).lower() in data]
    if not topology_keys:
        topology_keys = list(data)
    if not topology_keys:
        raise ValueError('empirical-topology result contains no data')

    display_names = {
        'karate': 'Karate club',
        'davis': 'Southern Women\nprojection',
        'florentine': 'Florentine\nfamilies',
    }
    labels = [display_names.get(key, key.replace('_', ' ').title())
              for key in topology_keys]
    x = np.arange(len(topology_keys), dtype=float)
    model_order = payload.get('control', {}).get('model_order', [
        'threshold_only', 'no_triadic_feedback', 'full_model'])
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    offsets = np.linspace(-0.18, 0.18, max(len(model_order), 1))

    for offset, model_key in zip(offsets, model_order):
        style = MODEL_TYPES.get(model_key, {})
        color = style.get('color', COLORS['gray'])
        marker = style.get('marker', 'o')
        label = style.get('label', model_key)
        blocks = [data[key][model_key] for key in topology_keys]
        for ax, metric in (
                (axes[0], 'order_param'),
                (axes[1], 'triangle_alignment_fixed')):
            values = [block[metric]['mean'] for block in blocks]
            errors = [ci95_half_width(block[metric]) for block in blocks]
            ax.errorbar(
                x + offset, values, yerr=errors, color=color,
                marker=marker, linestyle='none', capsize=4,
                markersize=7, label=label,
            )

    paired = [data[key].get('_paired_comparisons', {}).get(
        'full_model_minus_threshold_only', {}).get('order_param', {})
        for key in topology_keys]
    paired_mean = [_as_float(item.get('mean'), np.nan) for item in paired]
    paired_error = [ci95_half_width(item) for item in paired]
    axes[2].errorbar(
        x, paired_mean, yerr=paired_error, color=COLORS['black'],
        marker='D', linestyle='none', capsize=4, markersize=7,
    )
    axes[2].axhline(0.0, color=COLORS['gray'], linestyle='--', linewidth=1)

    axes[0].set_title('(a) Final order parameter', fontweight='bold')
    axes[0].set_ylabel('Order parameter')
    axes[0].set_ylim(-0.02, 1.02)
    axes[1].set_title('(b) Fixed-scale triangle alignment', fontweight='bold')
    axes[1].set_ylabel('Alignment rate')
    axes[1].set_ylim(-0.02, 1.02)
    axes[2].set_title('(c) Paired feedback contribution', fontweight='bold')
    axes[2].set_ylabel('Full - Threshold Only order parameter')
    for ax in axes:
        # Keep compatibility with Matplotlib 2.x, where set_xticks does not
        # accept labels as its second positional argument.
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel('Fixed empirical topology')
    axes[0].legend(framealpha=0.9, fontsize=8)
    fig.suptitle(
        'Robustness across fixed empirical social-network topologies',
        fontweight='bold', y=1.02,
    )
    fig.tight_layout()
    save_fig(fig, 'fig_empirical_topologies.png')


def plot_vector_opinions():
    """Plot the two-dimensional opinion robustness experiment."""
    payload = load_json('exp_vector_opinions.json')
    data = payload.get('data', {})
    p_keys = sorted(data, key=float)
    x = [float(key) for key in p_keys]
    model_order = payload.get('control', {}).get('model_order', [
        'threshold_only', 'no_triadic_feedback', 'full_model'])
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for model_key in model_order:
        style = MODEL_TYPES.get(model_key, {})
        color = style.get('color', COLORS['gray'])
        marker = style.get('marker', 'o')
        label = style.get('label', model_key)
        blocks = [data[key][model_key] for key in p_keys]
        for ax, metric, clip in (
                (axes[0], 'compact_order', (0, 1)),
                (axes[1], 'total_variance', None),
                (axes[2], 'triangle_alignment_fixed', (0, 1))):
            y = [block[metric]['mean'] for block in blocks]
            err = [ci95_half_width(block[metric]) for block in blocks]
            smooth_fill(ax, x, y, err, color, label, marker=marker,
                        smooth=False, clip_y=clip)
    axes[0].set_title('(a) 2D compact order', fontweight='bold')
    axes[0].set_ylabel('Largest observed-centred ball fraction')
    axes[1].set_title('(b) Total coordinate variance', fontweight='bold')
    axes[1].set_ylabel('Variance trace')
    axes[2].set_title('(c) Fixed-scale triangle alignment',
                      fontweight='bold')
    axes[2].set_ylabel('Alignment rate')
    for ax in axes:
        ax.set_xlabel(r'$p_{HK}$')
    axes[0].legend(framealpha=0.9, fontsize=8)
    axes[1].legend(framealpha=0.9, fontsize=8)
    fig.suptitle('Two-dimensional opinion robustness', fontweight='bold', y=1.02)
    fig.tight_layout()
    save_fig(fig, 'fig_vector_opinions.png')


# ============================================================
# Figure 3: Ablation
# ============================================================
def plot_ablation():
    data = load_json("exp6_ablation.json")
    metrics = ["order_param", "triangle_alignment_fixed",
               "edge_disagreement_penalty", "variance"]

    labels = {
        "order_param": "Order Parameter",
        "triangle_alignment_fixed": "Fixed-scale Triangle Alignment",
        "edge_disagreement_penalty": "Mean Edge-disagreement Penalty",
        "variance": "Opinion Variance",
    }

    # 只改图例显示名，不改 JSON / config 里的内部 key
    model_labels = {
        "baseline": "Asynchronous unilateral BCM",
        "weighted_bcm": "Weighted BCM",
        "threshold_only": "Threshold Only",
        "bcm_triangle": "BCM + Triangle",
        "no_triadic_feedback": "No Triadic Feedback",
        "full_model": "Full Model",
    }

    p_keys = sorted((key for key in data if not key.startswith('_')),
                    key=float)
    x = [float(p) for p in p_keys]

    # 建议图例顺序：从简单机制到完整机制
    model_order = [
        "baseline",
        "weighted_bcm",
        "threshold_only",
        "bcm_triangle",
        "no_triadic_feedback",
        "full_model",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)

    for ax, metric in zip(axes.flatten(), metrics):
        for model_key in model_order:
            if model_key not in MODEL_TYPES:
                continue

            style = MODEL_TYPES[model_key]
            means, stds = [], []

            for p in p_keys:
                block = data.get(p, {}).get(model_key)
                if block is None and model_key == "no_triadic_feedback":
                    block = data.get(p, {}).get("no_structure")
                means.append(get_stat(block, metric) if block else np.nan)
                stds.append(get_ci95_half_width(block, metric) if block else 0.0)

            smooth_fill(
                ax,
                x,
                means,
                stds,
                style.get("color", COLORS["gray"]),
                model_labels.get(model_key, style.get("label", model_key)),
                marker=style.get("marker", "o"),
                smooth=False,
                clip_y=(0, 1) if metric in {
                    "order_param", "triangle_alignment_fixed",
                    "edge_disagreement_penalty"
                } else None,
            )

        ax.set_title(labels[metric], fontweight="bold")
        ax.set_ylabel(labels[metric])
        ax.legend(framealpha=0.9, fontsize=8)

    axes[1, 0].set_xlabel(r"$p_{HK}$")
    axes[1, 1].set_xlabel(r"$p_{HK}$")

    fig.suptitle(
        "Figure 3: Ablation Study Across Triadic Closure Probability",
        fontweight="bold",
        fontsize=14,
        y=1.02,
    )

    fig.tight_layout()
    save_fig(fig, "fig3_ablation.png")


# ============================================================
# Figure 4: Network validation
# ============================================================
def plot_network_validation():
    data = load_json("exp7_network_validation.json")
    p_keys = sorted((key for key in data if not key.startswith('_')),
                    key=float)
    x = [float(p) for p in p_keys]

    panels = [
        ("f_triangle", "Triangle Fraction $f_\\Delta$", COLORS["blue"]),
        ("avg_clustering", "Average Clustering", COLORS["red"]),
        ("global_clustering", "Global Clustering", COLORS["green"]),
        ("modularity", "Modularity Q", COLORS["purple"]),
        ("avg_path_length", "Average Path Length", COLORS["orange"]),
        ("num_triangles", "Number of Triangles", COLORS["teal"]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for ax, (metric, ylabel, color) in zip(axes.flatten(), panels):
        means = [get_stat(data[p], metric) for p in p_keys]
        stds = [get_ci95_half_width(data[p], metric) for p in p_keys]
        smooth_fill(ax, x, means, stds, color, None, marker="o")
        ax.set_xlabel(r"$p_{HK}$")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, fontsize=11, fontweight="bold")
    fig.suptitle("Figure 4: Network Structural Validation", fontweight="bold", fontsize=14, y=1.01)
    fig.tight_layout()
    save_fig(fig, "fig4_network_validation.png")


# ============================================================
# Figure 5: alpha sweep
# ============================================================
def plot_alpha_sweep():
    data = load_json("exp8_alpha_sweep.json")
    keys = sorted((key for key in data if not key.startswith('_')), key=float)
    x = [float(k) for k in keys]
    y_op = [get_stat(data[k], "order_param") for k in keys]
    s_op = [get_ci95_half_width(data[k], "order_param") for k in keys]
    y_nc = [get_stat(data[k], "n_clusters") for k in keys]
    s_nc = [get_ci95_half_width(data[k], "n_clusters") for k in keys]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    smooth_fill(ax1, x, y_op, s_op, COLORS["blue"], "Order Parameter", marker="o", clip_y=(0, 1))
    ax1.set_xlabel(r"$\alpha$ (edge-threshold sensitivity)")
    ax1.set_ylabel("Order Parameter")
    ax1.set_title(r"(a) Consensus vs $\alpha$", fontweight="bold")
    smooth_fill(ax2, x, y_nc, s_nc, COLORS["red"], "Number of Clusters", marker="s")
    ax2.set_xlabel(r"$\alpha$ (edge-threshold sensitivity)")
    ax2.set_ylabel("Number of Clusters")
    ax2.set_title(r"(b) Fragmentation vs $\alpha$", fontweight="bold")
    fig.suptitle(r"Figure 5: Sensitivity to the Edge-Threshold Parameter $\alpha$", fontweight="bold", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig5_alpha_sweep.png")


# ============================================================
# Figure 6: beta sweep
# ============================================================
def plot_beta_sweep():
    data = load_json("exp8_beta_sweep.json")
    p_keys = sorted((key for key in data if not key.startswith('_')), key=float)
    fig, axes = plt.subplots(1, max(1, len(p_keys)), figsize=(5.2 * max(1, len(p_keys)), 4.7), squeeze=False)
    for ax, p in zip(axes.flatten(), p_keys):
        betas = sorted(data[p].keys(), key=float)
        x = [float(b) for b in betas]
        tc = [get_stat(data[p][b], "triangle_alignment_fixed") for b in betas]
        tc_s = [get_ci95_half_width(data[p][b], "triangle_alignment_fixed") for b in betas]
        rf = [get_stat(data[p][b], "edge_disagreement_penalty") for b in betas]
        rf_s = [get_ci95_half_width(data[p][b], "edge_disagreement_penalty") for b in betas]
        smooth_fill(ax, x, tc, tc_s, COLORS["blue"], "Fixed-scale Triangle Alignment", marker="o", clip_y=(0, 1))
        smooth_fill(ax, x, rf, rf_s, COLORS["red"], "Mean Edge-disagreement Penalty", marker="s", clip_y=(0, 1))
        ax.set_xlabel(r"$\beta$ (third-party disagreement attenuation)")
        ax.set_title(rf"$p_{{HK}}={p}$", fontweight="bold")
        ax.legend(framealpha=0.9)
        ax.set_ylim(-0.03, 1.03)
    axes.flatten()[0].set_ylabel("Metric Value")
    fig.suptitle(r"Figure 6: Effect of Third-party Disagreement Attenuation $\beta$", fontweight="bold", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig6_beta_sweep.png")


# ============================================================
# Figure 7: joint eps-beta
# ============================================================
def plot_joint_eps_beta():
    data = load_json("exp8_joint_eps_beta.json")
    eps_vals = np.asarray(data.get("eps_values", []), dtype=float)
    beta_vals = np.asarray(data.get("beta_values", []), dtype=float)
    if eps_vals.size == 0 or beta_vals.size == 0:
        raise ValueError("exp8_joint_eps_beta.json 缺少 eps_values 或 beta_values")
    tc = np.full((len(beta_vals), len(eps_vals)), np.nan)
    rf = np.full_like(tc, np.nan)
    for i, beta in enumerate(beta_vals):
        for j, eps in enumerate(eps_vals):
            sub = data.get("data", {})
            candidates = [f"{eps:.2f}_{beta:.1f}", f"{eps:.2f}_{beta:.2f}", f"{eps:.1f}_{beta:.1f}"]
            key = next((k for k in candidates if k in sub), None)
            if key:
                tc[i, j] = get_stat(sub[key], "triangle_alignment_fixed")
                rf[i, j] = get_stat(sub[key], "edge_disagreement_penalty")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    cmap_tc = LinearSegmentedColormap.from_list("consensus", ["#F0F9FF", "#2563EB", "#1E1B4B"])
    cmap_rf = LinearSegmentedColormap.from_list("fragmentation", ["#FFFBEB", "#EA580C", "#7C2D12"])
    for ax, mat, title, cmap, cbar_label in [
        (ax1, tc, "(a) Fixed-scale Triangle Alignment", cmap_tc, "Alignment Rate"),
        (ax2, rf, "(b) Mean Edge-disagreement Penalty", cmap_rf, "Mean Penalty"),
    ]:
        im = ax.imshow(mat, aspect="auto", origin="lower", cmap=cmap, interpolation="nearest", vmin=0, vmax=1)
        ax.set_xticks(range(len(eps_vals)))
        ax.set_xticklabels([f"{v:.2f}" for v in eps_vals], rotation=45)
        ax.set_yticks(range(len(beta_vals)))
        ax.set_yticklabels([f"{v:.2g}" for v in beta_vals])
        ax.set_xlabel(r"$\epsilon_0$")
        ax.set_ylabel(r"$\beta$")
        ax.set_title(title, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.82, label=cbar_label)
        if mat.size <= 400:
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    if not np.isnan(mat[i, j]):
                        ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7,
                                color="white" if mat[i, j] > 0.55 else "black", fontweight="bold")
    fig.suptitle(r"Figure 7: Joint $\epsilon_0 \times \beta$ Phase Diagram", fontweight="bold", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig7_joint_eps_beta.png")


# ============================================================
# Figure 8: initial distribution
# ============================================================
def plot_init_distribution():
    data = load_json("exp8_init_distribution.json")
    dists = [d for d in ["uniform", "normal", "bimodal"] if d in data]
    labels = {"uniform": "Uniform [0,1]", "normal": "Normal", "bimodal": "Bimodal"}
    colors = [COLORS["blue"], COLORS["red"], COLORS["green"]]
    metrics = ["order_param", "n_clusters", "variance",
               "triangle_alignment_fixed"]
    metric_labels = ["Order Parameter", "Number of Clusters", "Opinion Variance", "Fixed-scale Triangle Alignment"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, metric, ylabel in zip(axes.flatten(), metrics, metric_labels):
        x = np.arange(len(dists))
        means = [get_stat(data[d], metric) for d in dists]
        stds = [get_ci95_half_width(data[d], metric) for d in dists]
        bars = ax.bar(x, means, 0.55, yerr=stds, capsize=5, color=colors[:len(dists)],
                      alpha=0.86, edgecolor="white", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([labels[d] for d in dists])
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, fontweight="bold")
        for bar, mean in zip(bars, means):
            if not np.isnan(mean):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{mean:.3f}",
                        ha="center", va="bottom", fontsize=9)
    fig.suptitle("Figure 8: Sensitivity to Initial Opinion Distribution", fontweight="bold", fontsize=14, y=1.01)
    fig.tight_layout()
    save_fig(fig, "fig8_init_distribution.png")


# ============================================================
# Figure 9: robustness
# ============================================================
def plot_robustness():
    size_data = load_json("exp8_size_robustness.json")
    m_data = load_json("exp8_m_robustness.json")
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    sizes = sorted((key for key in size_data if not key.startswith('_')),
                   key=int)
    xs = [int(s) for s in sizes]
    op_s = [get_stat(size_data[s], "order_param") for s in sizes]
    op_ss = [get_ci95_half_width(size_data[s], "order_param") for s in sizes]
    rf_s = [get_stat(size_data[s], "edge_disagreement_penalty") for s in sizes]
    rf_ss = [get_ci95_half_width(size_data[s], "edge_disagreement_penalty") for s in sizes]
    smooth_fill(axes[0, 0], xs, op_s, op_ss, COLORS["blue"], "Order Parameter", smooth=False, clip_y=(0.5, 1.0))
    axes[0, 0].set_xlabel("Network Size N")
    axes[0, 0].set_ylabel("Order Parameter")
    axes[0, 0].set_title("(a) Consensus vs Network Size", fontweight="bold")
    axes[0, 0].set_ylim(0.5, 1.0)
    smooth_fill(axes[0, 1], xs, rf_s, rf_ss, COLORS["orange"], "Mean Edge-disagreement Penalty", smooth=False)
    axes[0, 1].set_xlabel("Network Size N")
    axes[0, 1].set_ylabel("Mean Edge-disagreement Penalty")
    axes[0, 1].set_title("(b) Edge Disagreement vs Network Size", fontweight="bold")

    ms = sorted((key for key in m_data if not key.startswith('_')), key=int)
    xm = [int(m) for m in ms]
    op_m = [get_stat(m_data[m], "order_param") for m in ms]
    op_ms = [get_ci95_half_width(m_data[m], "order_param") for m in ms]
    ft_m = [get_stat(m_data[m], "f_triangle") for m in ms]
    ft_ms = [get_ci95_half_width(m_data[m], "f_triangle") for m in ms]
    smooth_fill(axes[1, 0], xm, op_m, op_ms, COLORS["purple"], "Order Parameter", marker="D", clip_y=(0.5, 1.0))
    axes[1, 0].set_xlabel("Holme-Kim parameter m")
    axes[1, 0].set_ylabel("Order Parameter")
    axes[1, 0].set_title("(c) Consensus vs m", fontweight="bold")
    axes[1, 0].set_ylim(0.5, 1.0)
    smooth_fill(axes[1, 1], xm, ft_m, ft_ms, COLORS["teal"], r"$f_\Delta$", marker="^")
    axes[1, 1].set_xlabel("Holme-Kim parameter m")
    axes[1, 1].set_ylabel(r"Triangle Fraction $f_\Delta$")
    axes[1, 1].set_title(r"(d) Triangle Fraction vs m", fontweight="bold")
    fig.suptitle("Figure 9: Robustness to Network Size and Density", fontweight="bold", fontsize=14, y=1.01)
    fig.tight_layout()
    save_fig(fig, "fig9_robustness.png")


# ============================================================
# Figure 10: mu sweep
# ============================================================
def plot_mu_sweep():
    data = load_json("exp8_mu_sweep.json")
    keys = sorted((key for key in data if not key.startswith('_')), key=float)
    x = [float(k) for k in keys]
    op = [get_stat(data[k], "order_param") for k in keys]
    op_s = [get_ci95_half_width(data[k], "order_param") for k in keys]
    cl = [get_stat(data[k], "n_clusters") for k in keys]
    cl_s = [get_ci95_half_width(data[k], "n_clusters") for k in keys]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    smooth_fill(ax1, x, op, op_s, COLORS["blue"], "Order Parameter", marker="o", clip_y=(0, 1))
    ax1.set_xlabel(r"$\mu$ (convergence speed)")
    ax1.set_ylabel("Order Parameter")
    ax1.set_title(r"(a) Consensus vs $\mu$", fontweight="bold")
    smooth_fill(ax2, x, cl, cl_s, COLORS["red"], "Number of Clusters", marker="s")
    ax2.set_xlabel(r"$\mu$ (convergence speed)")
    ax2.set_ylabel("Number of Clusters")
    ax2.set_title(r"(b) Fragmentation vs $\mu$", fontweight="bold")
    fig.suptitle(r"Figure 10: Sensitivity to Convergence Speed $\mu$", fontweight="bold", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig10_mu_sweep.png")


# ============================================================
# Figure 11: joint p-beta
# ============================================================
def plot_joint_p_beta():
    data = load_json("exp8_joint_p_beta.json")
    p_vals = np.asarray(data.get("p_values", []), dtype=float)
    beta_vals = np.asarray(data.get("beta_values", []), dtype=float)
    sub = data.get("data", {})
    if p_vals.size == 0 or beta_vals.size == 0:
        raise ValueError("exp8_joint_p_beta.json 缺少 p_values 或 beta_values")
    op = np.full((len(beta_vals), len(p_vals)), np.nan)
    ft = np.full_like(op, np.nan)
    for i, beta in enumerate(beta_vals):
        for j, p in enumerate(p_vals):
            key = get_combo_key(sub, float(p), float(beta))
            if key:
                op[i, j] = get_stat(sub[key], "order_param")
                ft[i, j] = get_stat(sub[key], "f_triangle")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    cmap_op = LinearSegmentedColormap.from_list("order", ["#FEF3C7", "#16A34A", "#052E16"])
    cmap_ft = LinearSegmentedColormap.from_list("ftri", ["#F0F9FF", "#7C3AED", "#2E1065"])
    for ax, mat, title, cmap, cbar_label, vmin, vmax in [
        (ax1, op, "(a) Order Parameter", cmap_op, "Order Parameter", 0, 1),
        (ax2, ft, r"(b) Triangle Fraction $f_\Delta$", cmap_ft, r"$f_\Delta$", 0, 1),
    ]:
        im = ax.imshow(mat, aspect="auto", origin="lower", cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
        xtick_step = max(1, len(p_vals) // 10)
        ytick_step = max(1, len(beta_vals) // 10)
        ax.set_xticks(range(0, len(p_vals), xtick_step))
        ax.set_xticklabels([f"{p:.2g}" for p in p_vals[::xtick_step]])
        ax.set_yticks(range(0, len(beta_vals), ytick_step))
        ax.set_yticklabels([f"{b:.2g}" for b in beta_vals[::ytick_step]])
        ax.set_xlabel(r"$p_{HK}$")
        ax.set_ylabel(r"$\beta$")
        ax.set_title(title, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.82, label=cbar_label)
        if mat.size <= 400:
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    if not np.isnan(mat[i, j]):
                        ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7,
                                color="white" if mat[i, j] > 0.55 else "black", fontweight="bold")
    fig.suptitle(r"Figure 11: Joint $p_{HK} \times \beta$ Phase Diagram", fontweight="bold", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig11_joint_p_beta.png")


# ============================================================
# Figure 12: degree distribution
# ============================================================
def plot_degree_distribution():
    data = load_json("exp7_network_validation.json")
    selected = [0.0, 0.4, 0.8, 1.0]
    fig, ax = plt.subplots(figsize=(10, 6))
    for p in selected:
        key = get_p_key(data, p)
        if key is None:
            continue
        hist = np.asarray(data[key].get("degree_histogram", []), dtype=float)
        if hist.size == 0:
            continue
        degree = np.arange(hist.size)
        mask = hist > 1e-8
        if not np.any(mask):
            continue
        color = PHK_COLORS.get(f"{p:.1f}", COLORS["gray"])
        xd, yd = smooth_curve(degree[mask], hist[mask], num_points=160, k=2)
        yd = np.maximum(yd, 0)
        ax.plot(xd, yd, color=color, linewidth=2, label=rf"$p_{{HK}}={key}$")
        ax.fill_between(xd, 0, yd, color=color, alpha=0.08)
    ax.set_xlabel("Degree k")
    ax.set_ylabel("Probability P(k)")
    ax.set_title(r"Figure 12: Degree Distribution for Different $p_{HK}$", fontweight="bold")
    ax.set_xlim(left=-1)
    ax.set_ylim(bottom=0)
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    save_fig(fig, "fig12_degree_distribution.png")


# ============================================================
# Figure 13: summary dashboard
# ============================================================
def plot_summary_dashboard():
    net = load_json("exp7_network_validation.json")
    hyst = load_json("exp_hysteresis_sweep.json")
    abl = load_json("exp6_ablation.json")
    beta = load_json("exp8_beta_sweep.json")
    init = load_json("exp8_init_distribution.json")

    fig = plt.figure(figsize=(17, 11))
    gs = gridspec.GridSpec(3, 3, hspace=0.45, wspace=0.35)

    p_net = sorted((key for key in net if not key.startswith('_')),
                   key=float)
    x_net = [float(p) for p in p_net]
    ax1 = fig.add_subplot(gs[0, 0])
    smooth_fill(ax1, x_net, [get_stat(net[p], "f_triangle") for p in p_net],
                [get_ci95_half_width(net[p], "f_triangle") for p in p_net], COLORS["blue"], None)
    ax1.set_title(r"(a) Triangle coverage", fontweight="bold")
    ax1.set_xlabel(r"$p_{HK}$")
    ax1.set_ylabel(r"$f_\Delta$")

    ax2 = fig.add_subplot(gs[0, 1])
    p_h = sorted((key for key in hyst if not key.startswith('_')), key=float)
    x_h = [float(p) for p in p_h]
    smooth_fill(ax2, x_h, [_as_float(hyst[p].get("hysteresis_area"), np.nan) for p in p_h],
                np.zeros(len(p_h)), COLORS["red"], None)
    ax2.set_title("(b) Hysteresis area", fontweight="bold")
    ax2.set_xlabel(r"$p_{HK}$")
    ax2.set_ylabel("Area")

    ax3 = fig.add_subplot(gs[0, 2])
    smooth_fill(ax3, x_h, [_as_float(hyst[p].get("backward_low_eps_order_param"), np.nan) for p in p_h],
                np.zeros(len(p_h)), COLORS["green"], None, clip_y=(0, 1))
    ax3.axhline(0.9, color=COLORS["gray"], linestyle=":")
    ax3.set_title(r"(c) Perturbed backward resilience", fontweight="bold")
    ax3.set_xlabel(r"$p_{HK}$")
    ax3.set_ylabel(r"Low-$\epsilon_0$ OP")
    ax3.set_ylim(0, 1.02)

    ax4 = fig.add_subplot(gs[1, 0])
    p_abl = sorted((key for key in abl if not key.startswith('_')), key=float)
    x = [float(p) for p in p_abl]
    for model_key in [normalize_model_key(m) for m in MODEL_DISPLAY_ORDER]:
        if model_key not in MODEL_TYPES:
            continue
        style = MODEL_TYPES[model_key]
        y = [get_stat(abl[p].get(model_key, {}), "order_param") for p in p_abl]
        smooth_fill(ax4, x, y, np.zeros(len(x)), style.get("color", COLORS["gray"]),
                    style.get("label", model_key), marker=style.get("marker", "o"), smooth=False, clip_y=(0, 1))
    ax4.set_title("(d) Ablation: order parameter", fontweight="bold")
    ax4.set_xlabel(r"$p_{HK}$")
    ax4.set_ylabel("OP")
    ax4.legend(fontsize=7, framealpha=0.9)

    ax5 = fig.add_subplot(gs[1, 1])
    for p in sorted((key for key in beta if not key.startswith('_')),
                    key=float):
        bkeys = sorted(beta[p].keys(), key=float)
        xb = [float(b) for b in bkeys]
        yb = [get_stat(beta[p][b], "edge_disagreement_penalty") for b in bkeys]
        smooth_fill(ax5, xb, yb, np.zeros(len(xb)), PHK_COLORS.get(f"{float(p):.1f}", COLORS["gray"]),
                    rf"$p={p}$", smooth=False, clip_y=(0, 1))
    ax5.set_title(r"(e) Mean edge-disagreement penalty", fontweight="bold")
    ax5.set_xlabel(r"$\beta$")
    ax5.set_ylabel("Edge penalty")
    ax5.legend(fontsize=8, framealpha=0.9)

    ax6 = fig.add_subplot(gs[1, 2])
    dists = [d for d in ["uniform", "normal", "bimodal"] if d in init]
    vals = [get_stat(init[d], "order_param") for d in dists]
    ax6.bar(range(len(dists)), vals, color=[COLORS["blue"], COLORS["red"], COLORS["green"]][:len(dists)], alpha=0.85)
    ax6.set_xticks(range(len(dists)))
    ax6.set_xticklabels([d.capitalize() for d in dists])
    ax6.set_ylabel("OP")
    ax6.set_ylim(0, 1.02)
    ax6.set_title("(f) Initial distribution", fontweight="bold")

    # 剩余三格给关键曲线局部放大：clustering、sensitivity、clusters
    ax7 = fig.add_subplot(gs[2, 0])
    smooth_fill(ax7, x_net, [get_stat(net[p], "avg_clustering") for p in p_net],
                [get_ci95_half_width(net[p], "avg_clustering") for p in p_net], COLORS["purple"], None)
    ax7.set_title("(g) Average clustering", fontweight="bold")
    ax7.set_xlabel(r"$p_{HK}$")
    ax7.set_ylabel(r"$\bar C$")

    ax8 = fig.add_subplot(gs[2, 1])
    smooth_fill(ax8, x_h, [_as_float(hyst[p].get("peak_sensitivity"), np.nan) for p in p_h],
                np.zeros(len(p_h)), COLORS["orange"], None)
    ax8.set_title("(h) Peak sensitivity", fontweight="bold")
    ax8.set_xlabel(r"$p_{HK}$")
    ax8.set_ylabel("Sensitivity")

    ax9 = fig.add_subplot(gs[2, 2])
    smooth_fill(ax9, x_h, [_as_float(hyst[p].get("backward_low_eps_clusters"), np.nan) for p in p_h],
                np.zeros(len(p_h)), COLORS["teal"], None, smooth=False)
    ax9.set_title(r"(i) Backward low-$\epsilon_0$ clusters", fontweight="bold")
    ax9.set_xlabel(r"$p_{HK}$")
    ax9.set_ylabel("Clusters")

    fig.suptitle("Figure 13: Summary Dashboard", fontweight="bold", fontsize=16, y=1.01)
    fig.tight_layout()
    save_fig(fig, "fig13_summary_dashboard.png")


def plot_convergence_check():
    """Long-horizon diagnostic with paired Full-minus-No-feedback contrast."""
    data = load_json("exp_convergence_check.json")
    models = data["models"]
    styles = {
        "no_triadic_feedback": (COLORS["teal"], "No Triadic Feedback", "v"),
        "full_model": (COLORS["red"], "Full Model", "o"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    for model_key, (color, label, marker) in styles.items():
        block = models[model_key]["by_sweep"]
        keys = sorted(block, key=float)
        x = [float(key) for key in keys]
        y = [block[key]["order_param"]["mean"] for key in keys]
        e = [ci95_half_width(block[key]["order_param"]) for key in keys]
        smooth_fill(axes[0], x, y, e, color, label, marker=marker,
                    smooth=False, clip_y=(0, 1))
        dx = [block[key]["mean_dx_since_last"]["mean"] for key in keys]
        axes[1].plot(x, dx, color=color, marker=marker, label=label)

    axes[0].set_title("(a) Order parameter", fontweight="bold")
    axes[0].set_xlabel("Monte Carlo sweeps")
    axes[0].set_ylabel("Order parameter")
    axes[0].legend(framealpha=0.9)

    axes[1].set_yscale("log")
    axes[1].set_title("(b) Recent-window activity", fontweight="bold")
    axes[1].set_xlabel("Monte Carlo sweeps")
    axes[1].set_ylabel("Mean absolute update")
    axes[1].legend(framealpha=0.9)

    paired = data["paired_comparisons"]["by_sweep"]
    keys = sorted(paired, key=float)
    x = [float(key) for key in keys]
    y = [paired[key]["order_param"]["mean"] for key in keys]
    e = [ci95_half_width(paired[key]["order_param"]) for key in keys]
    axes[2].errorbar(x, y, yerr=e, color=COLORS["black"], marker="o",
                     capsize=4, linewidth=1.6)
    axes[2].axhline(0.0, color=COLORS["gray"], linestyle="--", linewidth=1)
    axes[2].set_title("(c) Paired terminal contrast", fontweight="bold")
    axes[2].set_xlabel("Monte Carlo sweeps")
    axes[2].set_ylabel("Full − No-feedback order parameter")

    fig.suptitle("Long-horizon convergence and paired mechanism check",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig_convergence_check.png")


PLOT_REGISTRY: list[tuple[str, str, Callable[[], None]]] = [
    ("exp_convergence_check.json", "convergence", plot_convergence_check),
    ("exp_perturbation_sweep.json", "perturbation", plot_perturbation_sweep),
    ("exp_update_mode_control.json", "update_mode", plot_update_mode_control),
    ("exp_sbm_community_conflict.json", "community", plot_sbm_community_conflict),
    ("exp_empirical_topologies.json", "empirical", plot_empirical_topologies),
    ("exp_vector_opinions.json", "vector", plot_vector_opinions),
    ("exp_epsilon_sweep.json", "epsilon", plot_epsilon_sweep),
    ("exp_hysteresis_sweep.json", "hysteresis", plot_hysteresis_sweep),
    ("exp6_ablation.json", "ablation", plot_ablation),
    ("exp7_network_validation.json", "network", plot_network_validation),
    ("exp8_alpha_sweep.json", "alpha", plot_alpha_sweep),
    ("exp8_beta_sweep.json", "beta", plot_beta_sweep),
    ("exp8_joint_eps_beta.json", "joint_eps_beta", plot_joint_eps_beta),
    ("exp8_init_distribution.json", "init", plot_init_distribution),
    ("exp8_size_robustness.json", "robustness", plot_robustness),
    ("exp8_mu_sweep.json", "mu", plot_mu_sweep),
    ("exp8_joint_p_beta.json", "joint_p_beta", plot_joint_p_beta),
    ("exp7_network_validation.json", "degree", plot_degree_distribution),
]


def available_plots() -> list[str]:
    return [name for _, name, _ in PLOT_REGISTRY] + ["summary"]


def main(argv: Optional[Iterable[str]] = None):
    parser = argparse.ArgumentParser(description="Generate all publication-quality figures.")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Only run selected plots, e.g. --only hysteresis ablation summary")
    parser.add_argument("--list", action="store_true", help="List available plot names and exit")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.list:
        print("Available plots:")
        for name in available_plots():
            print(f"  - {name}")
        return

    selected = set(args.only) if args.only else None
    print("=" * 60)
    print("  Generating publication-quality figures")
    print(f"  data: {DATA_PATH.resolve()}")
    print(f"  out : {OUT_DIR.resolve()}")
    print("=" * 60)

    for required_file, name, func in PLOT_REGISTRY:
        if selected is not None and name not in selected:
            continue
        try:
            # 部分图共用同一个 required_file，这里只做友好的跳过判断
            if not (DATA_PATH / required_file).exists() and not Path(required_file).exists():
                print(f"  [SKIP] {name}: {required_file} not found")
                continue
            func()
        except FileNotFoundError as e:
            print(f"  [SKIP] {name}: {e}")
        except Exception as e:
            import traceback
            print(f"  [ERR] {name}: {e}")
            traceback.print_exc()

    if selected is None or "summary" in selected:
        required = [
            "exp7_network_validation.json",
            "exp_hysteresis_sweep.json",
            "exp6_ablation.json",
            "exp8_beta_sweep.json",
            "exp8_init_distribution.json",
        ]
        if all((DATA_PATH / f).exists() or Path(f).exists() for f in required):
            try:
                plot_summary_dashboard()
            except Exception as e:
                import traceback
                print(f"  [ERR] summary: {e}")
                traceback.print_exc()
        else:
            missing = [f for f in required if not ((DATA_PATH / f).exists() or Path(f).exists())]
            print(f"  [SKIP] summary: missing {missing}")

    print("=" * 60)
    print(f"  Figures saved to: {OUT_DIR.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
