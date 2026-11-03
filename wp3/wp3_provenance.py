#!/usr/bin/env python3
"""WP3 provenance record: links the stored out-of-fold predictions to the code, input features and TabPFN checkpoint.

Written after the WP3 runs (3 Nov 2026), so it cannot prove what was on disk during the runs. It checks what can be
checked: (1) current script hashes against the hashes the run manifests recorded at run time, (2) current FEAT_50 v2
files against the output hashes recorded by the extraction manifests when the files were written, (3) record identifiers
and labels of every stored prediction file against the input features, and (4) the checkpoint file (hash, size, mtime;
an mtime before the runs means the file has not been rewritten since). Any failed check stops the script.

Usage (project root):  python revizyon_v22/wp3/wp3_provenance.py
Writes revizyon_v22/results/wp3/PROVENANCE_wp3.json
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wp3_common import COHORTS, OUT_ROOT, V2, load_cohort  # noqa: E402
from wp3_teacher_mlp import TABPFN_CKPT  # noqa: E402

W = Path(__file__).resolve().parent
RUN_SCRIPT = W / "run_wp3_all.sh"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def mtime(p: Path) -> str:
    return dt.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")


def main():
    out = {"written": dt.datetime.now().isoformat(timespec="seconds"), "checks": {}}

    # (1) scripts: hash recorded at run time == current file
    recorded = {}
    for m in sorted(OUT_ROOT.glob("*/run_manifest_*.json")):
        for name, h in json.load(open(m))["script_sha256"].items():
            assert recorded.setdefault(name, h) == h, f"{name}: different hashes across run manifests"
    out["scripts"] = {n: {"sha256_at_run": h, "sha256_now": sha(W / n)} for n, h in recorded.items()}
    out["checks"]["scripts_unchanged_since_runs"] = all(v["sha256_at_run"] == v["sha256_now"] for v in out["scripts"].values())
    out["analysis_scripts_sha256_now"] = {p.name: sha(p) for p in sorted(W.glob("wp3_*.py")) if p.name not in recorded}
    out["run_commands"] = [l.strip() for l in RUN_SCRIPT.read_text().splitlines() if "$PY" in l and not l.startswith("PY=")]
    out["run_script_sha256"] = sha(RUN_SCRIPT)

    # (2) FEAT_50 v2 inputs: hash written by the extraction step == current file
    ext = json.load(open(V2 / "extraction_v2_manifest.json"))
    chap = json.load(open(V2 / "chapman_extraction_v2_manifest.json"))
    expected = dict(ext["outputs"]) | {"chapman_eligible_features_v2.csv": chap["output_sha256"]}
    out["inputs"] = {n: {"sha256_at_extraction": h, "sha256_now": sha(V2 / n)} for n, h in expected.items()}
    out["checks"]["inputs_unchanged_since_extraction"] = all(v["sha256_at_extraction"] == v["sha256_now"]
                                                             for v in out["inputs"].values())

    # (3) every stored prediction file: same records and labels as the cohort built from the inputs
    out["predictions"] = {}
    ok = True
    for c in COHORTS:
        d, seeds = load_cohort(c)
        uid, y = d["uid"].astype(str).to_numpy(), d["y"].astype(int).to_numpy()
        for f in sorted((OUT_ROOT / c).glob("seed*_*.npz")):
            z = np.load(f, allow_pickle=True)  # files written by this project's own scripts
            same = np.array_equal(z["uid"].astype(str), uid) and np.array_equal(z["y"].astype(int), y)
            ok &= same
            out["predictions"][f"{c}/{f.name}"] = {"sha256": sha(f), "records_and_labels_match_inputs": bool(same)}
        out["predictions"][f"{c}/cohort_records.csv"] = {"sha256": sha(OUT_ROOT / c / "cohort_records.csv")}
    out["checks"]["predictions_match_inputs"] = bool(ok)

    # (4) checkpoint identity
    ck = Path(TABPFN_CKPT)
    first_run = min(mtime(p) for p in OUT_ROOT.glob("*/run_manifest_*.json"))
    out["tabpfn_checkpoint"] = {"file": ck.name, "path_used_by_scripts": str(ck), "bytes": ck.stat().st_size,
                                "sha256": sha(ck), "mtime": mtime(ck), "earliest_run_manifest_mtime": first_run,
                                "source": "TabPFN v3 multiclass classifier checkpoint for the tabpfn package (Prior Labs); "
                                          "pre-training data provenance not verified here"}
    out["checks"]["checkpoint_not_modified_after_runs"] = mtime(ck) < first_run

    (OUT_ROOT / "PROVENANCE_wp3.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["checks"], indent=1))
    assert all(out["checks"].values()), "a provenance check failed"


if __name__ == "__main__":
    main()
