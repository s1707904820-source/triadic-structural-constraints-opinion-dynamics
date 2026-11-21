"""Merge independent p_HK worker checkpoints into the formal data file."""

from __future__ import annotations

import json
from pathlib import Path

from provenance import build_metadata, execution_source_manifest


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "output" / "formal_revision" / "data"
WORKER_DIR = DATA_DIR / "perturbation_workers"
TARGET = DATA_DIR / "exp_formal_perturbation_sweep_revision_v2.json"
P_VALUES = [0.0, 0.4, 0.8]
SIGMA_VALUES = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]
EPS_VALUES = [round(0.10 + index * 0.025, 3) for index in range(19)]
WORKER_PREFIX = "exp_formal_perturbation_revision_v2_p_"


def _write_atomic(payload, target):
    temp_path = target.with_name(target.name + ".tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
        temp_path.replace(target)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _load_worker(path, p_value, current_sources):
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    metadata_path = path.with_suffix(".meta.json")
    if not metadata_path.exists():
        raise ValueError("worker metadata is missing: {}".format(path.name))
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("execution_source_sha256") != current_sources:
        raise ValueError("worker execution source does not match: {}".format(
            path.name))
    control = payload.get("control", {})
    if control.get("p_values") != [p_value]:
        raise ValueError("worker p_HK control does not match: {}".format(
            path.name))
    for key, expected in (
            ("sigma_values", SIGMA_VALUES), ("eps_values", EPS_VALUES),
            ("sweeps_per_epsilon", 100.0), ("perturb_each_step", False)):
        if control.get(key) != expected:
            raise ValueError("worker control {} mismatch: {}".format(
                key, path.name))
    if int(control.get("num_runs", -1)) != 20:
        raise ValueError("worker run count mismatch: {}".format(path.name))
    seeds = [run_i * 997 + 7 for run_i in range(20)]
    p_key = "{:.2f}".format(p_value)
    for sigma in SIGMA_VALUES:
        sigma_key = "{:.3f}".format(sigma)
        p_blocks = payload.get("data", {}).get(sigma_key, {})
        block = p_blocks.get(p_key, {})
        if block.get("seeds") != seeds:
            raise ValueError("worker seed schedule mismatch: {}".format(
                path.name))
        if block.get("num_runs") != 20 or block.get("p_hk") != p_value:
            raise ValueError("worker result block incomplete: {}".format(
                path.name))
    return payload


def merge_payloads(payloads, worker_files):
    if len(payloads) != len(P_VALUES):
        raise ValueError("expected one worker payload per p_HK value")
    first_control = dict(payloads[0]["control"])
    common_control = dict(first_control)
    common_control.pop("p_values", None)
    for p_value, payload in zip(P_VALUES, payloads):
        control = payload["control"]
        candidate = dict(control)
        candidate.pop("p_values", None)
        if candidate != common_control:
            raise ValueError("worker experiment controls do not match")
        if control.get("p_values") != [p_value]:
            raise ValueError("worker p_HK order or control mismatch")

    first_control["p_values"] = list(P_VALUES)
    first_control["parallel_worker_checkpoints"] = [
        "perturbation_workers/{}".format(path.name)
        for path in worker_files]
    data = {}
    for payload in payloads:
        for sigma_key, p_blocks in payload["data"].items():
            data.setdefault(sigma_key, {}).update(p_blocks)
    return {"control": first_control, "data": data}


def main():
    current_sources = execution_source_manifest()
    files = [WORKER_DIR / "{}{:0.2f}.json".format(WORKER_PREFIX, p)
             for p in P_VALUES]
    payloads = [
        _load_worker(path, p_value, current_sources)
        for path, p_value in zip(files, P_VALUES)]
    merged = merge_payloads(payloads, files)
    _write_atomic(merged, TARGET)
    metadata = build_metadata(
        experiment="exp_formal_perturbation_sweep_revision_v2",
        n_runs=merged["control"]["num_runs"],
        extra={
            "experiment_design": "perturbation_strength_gradient_shared_forward",
            "p_values": list(P_VALUES),
            "sigma_values": list(SIGMA_VALUES),
            "eps_values": list(EPS_VALUES),
            "sweeps_per_epsilon": 100.0,
            "worker_execution_source_sha256": current_sources,
        },
    )
    _write_atomic(metadata, TARGET.with_suffix(".meta.json"))
    print(TARGET)


if __name__ == "__main__":
    main()
