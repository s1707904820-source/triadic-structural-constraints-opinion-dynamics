"""Reproducibility metadata written beside every experiment result."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone

from config import NETWORK, TRUST, DYNAMICS, SIMULATION, METRICS


REVISION_VERSION = "revision-audit-v2"
_EXCLUDED_SOURCE_DIRS = {"__pycache__", "output"}


def _python_source_manifest(include_tests=True) -> dict[str, str]:
    """Hash source files recursively using stable POSIX relative paths."""
    root = os.path.dirname(os.path.abspath(__file__))
    manifest = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            name for name in dirnames
            if name not in _EXCLUDED_SOURCE_DIRS
            and (include_tests or name != "tests"))
        for filename in sorted(filenames):
            if not filename.endswith(".py"):
                continue
            path = os.path.join(dirpath, filename)
            relative = os.path.relpath(path, root).replace(os.sep, "/")
            with open(path, "rb") as handle:
                manifest[relative] = hashlib.sha256(handle.read()).hexdigest()
    return manifest


def standard_seeds(n_runs: int) -> list[int]:
    return [run_i * 997 + 7 for run_i in range(int(n_runs))]


def _source_manifest() -> dict[str, str]:
    return _python_source_manifest(include_tests=True)


def execution_source_manifest() -> dict[str, str]:
    """Hash all runtime and analysis sources, excluding tests and outputs."""
    return _python_source_manifest(include_tests=False)


def execution_source_signature() -> str:
    """Return a stable digest used to reject checkpoints from other code."""
    payload = json.dumps(
        execution_source_manifest(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
        "execution_source_sha256": execution_source_manifest(),
    }
    if n_runs is not None:
        metadata["n_runs"] = int(n_runs)
        metadata["seeds"] = standard_seeds(n_runs)
    if extra:
        metadata.update(extra)
    return metadata
