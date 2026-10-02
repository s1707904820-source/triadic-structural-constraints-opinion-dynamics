"""Reproducibility metadata written beside every experiment result."""

from __future__ import annotations

import hashlib
import os
import platform
import sys
from datetime import datetime, timezone

from config import NETWORK, TRUST, DYNAMICS, SIMULATION, METRICS


REVISION_VERSION = "revision-audit-v2"


def standard_seeds(n_runs: int) -> list[int]:
    return [run_i * 997 + 7 for run_i in range(int(n_runs))]


def _source_manifest() -> dict[str, str]:
    root = os.path.dirname(os.path.abspath(__file__))
    manifest = {}
    for name in sorted(os.listdir(root)):
        if not name.endswith(".py"):
            continue
        path = os.path.join(root, name)
        with open(path, "rb") as handle:
            manifest[name] = hashlib.sha256(handle.read()).hexdigest()
    return manifest


def build_metadata(experiment: str | None = None, n_runs: int | None = None,
                   extra: dict | None = None) -> dict:
    metadata = {
        "revision_version": REVISION_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment,
        "time_unit": "Monte Carlo sweeps; one sweep = N asynchronous unilateral updates",
        "boundary_rule": "opinion difference <= applicable threshold",
        "python": sys.version,
        "platform": platform.platform(),
        "config": {
            "NETWORK": dict(NETWORK),
            "TRUST": dict(TRUST),
            "DYNAMICS": dict(DYNAMICS),
            "SIMULATION": dict(SIMULATION),
            "METRICS": dict(METRICS),
        },
        "source_sha256": _source_manifest(),
    }
    if n_runs is not None:
        metadata["n_runs"] = int(n_runs)
        metadata["seeds"] = standard_seeds(n_runs)
    if extra:
        metadata.update(extra)
    return metadata
