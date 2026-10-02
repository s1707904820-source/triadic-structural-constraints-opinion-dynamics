"""Merge independent p_HK worker checkpoints into the formal data file."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "output" / "formal_revision" / "data"
WORKER_DIR = DATA_DIR / "perturbation_workers"
TARGET = DATA_DIR / "exp_formal_perturbation_sweep.json"
META_TARGET = DATA_DIR / "exp_formal_perturbation_sweep.meta.json"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_atomic(payload, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    tmp.replace(path)


def main():
    files = [WORKER_DIR / "exp_formal_perturbation_p_{:.2f}.json".format(p)
             for p in (0.0, 0.4, 0.8)]
    payloads = []
    for path in files:
        with path.open(encoding="utf-8") as handle:
            payloads.append(json.load(handle))
    control = dict(payloads[0]["control"])
    control["p_values"] = [0.0, 0.4, 0.8]
    # Store portable paths relative to the formal data directory, plus hashes
    # that identify the exact worker checkpoints used in the merge.
    control["parallel_worker_checkpoints"] = [
        path.relative_to(DATA_DIR).as_posix() for path in files]
    control["parallel_worker_sha256"] = {
        path.relative_to(DATA_DIR).as_posix(): sha256(path) for path in files}
    data = {}
    for payload in payloads:
        for sigma_key, p_blocks in payload["data"].items():
            data.setdefault(sigma_key, {}).update(p_blocks)
    merged = {"control": control, "data": data}
    write_atomic(merged, TARGET)
    write_atomic({"experiment": "formal_perturbation_gradient",
                  "control": control}, META_TARGET)
    print(TARGET)


if __name__ == "__main__":
    main()
