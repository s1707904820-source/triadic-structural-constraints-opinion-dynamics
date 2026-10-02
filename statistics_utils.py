"""Small, dependency-free statistical helpers for simulation summaries."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


# Two-sided 95% Student-t critical values for df=1..30.
_T975 = (
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306,
    2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120,
    2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064,
    2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
)


def _t975(df: int) -> float:
    if df <= 0:
        return 0.0
    if df <= len(_T975):
        return _T975[df - 1]
    if df <= 60:
        return 2.000
    if df <= 120:
        return 1.980
    return 1.960


def summarize(values: Iterable[float], keep_values: bool = True) -> dict:
    """Return n, mean, sample SD, SE, and a two-sided 95% t interval."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        raise ValueError("cannot summarize an empty sequence")
    n = int(arr.size)
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 0 else float("nan")
    half_width = _t975(n - 1) * se if n > 1 else 0.0
    out = {
        "n": n,
        "mean": mean,
        "sd": sd,
        # Retain the historical key for old plotting code; it now means sample SD.
        "std": sd,
        "se": float(se),
        "ci95_low": float(mean - half_width),
        "ci95_high": float(mean + half_width),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }
    if keep_values:
        out["values"] = [float(v) for v in arr]
    return out


def paired_difference(first: Sequence[float], second: Sequence[float]) -> dict:
    """Summarize paired ``first - second`` differences and standardized dz."""
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired samples must have identical shapes")
    out = summarize(a - b)
    out["definition"] = "first_minus_second"
    out["cohen_dz"] = (
        float(out["mean"] / out["sd"]) if out["sd"] > 0 else None
    )
    return out


def summarize_aligned_series(series: Sequence[Sequence[float]]) -> dict:
    """Summarize equally aligned repeated time series at every time point."""
    if not series:
        raise ValueError("cannot summarize an empty series collection")
    lengths = {len(row) for row in series}
    if len(lengths) != 1:
        raise ValueError("time series must be aligned and have equal lengths")
    rows = np.asarray(series, dtype=float)
    point_stats = [summarize(rows[:, idx], keep_values=False)
                   for idx in range(rows.shape[1])]
    keys = ("mean", "sd", "std", "se", "ci95_low", "ci95_high")
    return {key: [float(item[key]) for item in point_stats] for key in keys}
