"""Small, dependency-free statistical helpers for simulation summaries."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

try:  # SciPy is pinned for the formal analyses, but keep a fallback below.
    from scipy.stats import t as _student_t
except Exception:  # pragma: no cover - exercised only in minimal installs
    _student_t = None


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
    if _student_t is not None:
        value = float(_student_t.ppf(0.975, df))
        if math.isfinite(value):
            return value

    # Cornish-Fisher expansion around the normal 97.5% quantile.  This keeps
    # confidence intervals accurate when SciPy is unavailable instead of
    # jumping prematurely to the normal 1.96 approximation.
    z = 1.959963984540054
    inv_df = 1.0 / float(df)
    z2 = z * z
    z3 = z * z2
    z5 = z3 * z2
    z7 = z5 * z2
    return (z + (z3 + z) * inv_df / 4.0
            + (5.0 * z5 + 16.0 * z3 + 3.0 * z) * inv_df ** 2 / 96.0
            + (3.0 * z7 + 19.0 * z5 + 17.0 * z3 - 15.0 * z)
            * inv_df ** 3 / 384.0)


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


def summarize_proportion(values: Iterable[float]) -> dict:
    """Summarize binary outcomes with a two-sided 95% Wilson interval."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        raise ValueError("cannot summarize an empty sequence")
    if np.any((arr != 0.0) & (arr != 1.0)):
        raise ValueError("proportion outcomes must be binary")
    n = int(arr.size)
    successes = int(np.sum(arr))
    proportion = float(successes / n)
    z = 1.959963984540054
    z2 = z * z
    denominator = 1.0 + z2 / n
    center = (proportion + z2 / (2.0 * n)) / denominator
    half_width = z * math.sqrt(
        proportion * (1.0 - proportion) / n + z2 / (4.0 * n * n)
    ) / denominator
    return {
        "n": n,
        "successes": successes,
        "mean": proportion,
        "se": float(math.sqrt(proportion * (1.0 - proportion) / n)),
        "ci95_low": float(max(0.0, center - half_width)),
        "ci95_high": float(min(1.0, center + half_width)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "values": [float(value) for value in arr],
        "interval": "95% Wilson score interval",
    }


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
